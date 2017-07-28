import numpy as np
from ..ft import fftshift, ifftshift, irfftn


class Wavefunction:
    """Container class for Kohn-Sham orbitals
    
    Physically, wavefunction is uniquely labeled by a 2-tuple of band index (int)
    and spin ("up" or "down"). Internally, each wavefunction is labeled by an integer index.
    Several maps are defined to describe related transformations.
    
    Attributes:
        norbs (int): total number of KS orbitals to be considered
        nuorbs/ndorbs (int): number of spin up/down orbitals
        sb_iorb_map (dict): (spin, band index) -> orb index map
        iorb_sb_map (list): orb index -> (spin, band index) map
        iorb_psir_map (dict): orb index -> orb object (3D array) map

        cell (Cell): defines cell size, R and G vectors
        ft (FourierTransform): defines grid size for fourier transform

    Right now only consider ground state, insulating, spin-polarized case.
    No occupation number considerations are implemented yet.
    """
    def __init__(self, cell, ft, nuorbs, ndorbs, iorb_sb_map, iorb_fname_map,
                 dft=None, gamma=True, gvecs=None):

        self.cell = cell
        self.ft = ft
        self.dft = dft

        self.nuorbs = nuorbs
        self.ndorbs = ndorbs
        self.norbs = self.nuorbs + self.ndorbs

        self.iorb_sb_map = iorb_sb_map
        self.sb_iorb_map = {
            self.iorb_sb_map[iorb]: iorb for iorb in range(self.norbs)
        }

        self.iorb_fname_map = iorb_fname_map

        self.iorb_psig_arr_map = {}
        self.iorb_psir_map = {}
        self.iorb_rhog_map = {}

        self.gvecs = gvecs
        self.gamma = gamma

        if self.gamma:
            yzplane = np.zeros((self.ft.n2, self.ft.n3))
            yzplane[self.ft.n2 // 2 + 1:, :] = 1
            yzplane[0, self.ft.n3 // 2 + 1:] = 1
            self.yzlowerplane = zip(*np.nonzero(yzplane))

    def compute_psir_from_psig_arr(self, psig_arr):
        """Compute psi(r) based on psi(G) defined on self.gvecs"""

        assert self.gamma, "Only gamma-point is implemented"

        dn1, dn2, dn3 = self.dft.n1, self.dft.n2, self.dft.n3
        n1, n2, n3 = self.ft.n1, self.ft.n2, self.ft.n3

        # Read orbital in density grid
        # x and z axes are switched for convenience of rFFT
        psig_zyx = np.zeros((dn3, dn2, dn1 // 2 + 1), dtype=np.complex_)
        psig_zyx[self.gvecs[:, 2], self.gvecs[:, 1], self.gvecs[:, 0]] = psig_arr

        # If a smoother grid is required, crop high frequency components
        if (n1, n2, n3) != (dn1, dn2, dn3):
            psig_zyx = ifftshift(
                fftshift(psig_zyx, axes=(0, 1))[
                (dn3 - n3 - 1) // 2 + 1:(dn3 - n3 - 1) // 2 + 1 + n3,
                (dn2 - n2 - 1) // 2 + 1:(dn2 - n2 - 1) // 2 + 1 + n2,
                0: n1 // 2 + 1,
                ], axes=(0, 1)
            )
        assert psig_zyx.shape == (n3, n2, n1 // 2 + 1)

        # Complete psig in yz plane
        for ig2, ig3 in self.yzlowerplane:
            psig_zyx[ig3, ig2, 0] = psig_zyx[-ig3, -ig2, 0].conjugate()

        # rFFT in x direction (x has been switched to last axes), FFT in y, z direction
        psir_zyx = irfftn(psig_zyx, s=(n3, n2, n1))

        # Switch back x, z axes
        psir = psir_zyx.swapaxes(0, 2)
        return self.normalize(psir)

    def set_psig_arr(self, iorb, psig_arr):
        if iorb in self.iorb_psig_arr_map:
            raise ValueError("psig_arr {} already set".format(iorb))
        self.iorb_psig_arr_map[iorb] = psig_arr

    def set_psir(self, iorb, psir):
        if iorb in self.iorb_psir_map:
            raise ValueError("psir {} already set".format(iorb))
        self.iorb_psir_map[iorb] = self.normalize(psir)

    def get_psir(self, iorb):
        """Get psi(r) of certain index"""
        if iorb in self.iorb_psir_map:
            return self.iorb_psir_map[iorb]
        else:
            assert iorb in self.iorb_psig_arr_map, "Impossible to get psir: orbital is not loaded"
            return self.compute_psir_from_psig_arr(self.iorb_psig_arr_map[iorb])

    def get_rhog(self, iorb):
        """Get rho(G) of certain index"""
        if iorb in self.iorb_rhog_map:
            return self.iorb_rhog_map[iorb]
        else:
            psir = self.get_psir(iorb)
            return self.ft.forward(psir * np.conj(psir))

    def normalize(self, psir):
        """Normalize psir."""
        assert psir.shape == (self.ft.n1, self.ft.n2, self.ft.n3)
        norm = np.sqrt(np.sum(np.abs(psir) ** 2) * self.cell.omega / self.ft.N)
        return psir / norm

    def compute_all_psir(self):
        for iorb in self.iorb_psir_map:
            self.iorb_psir_map[iorb] = self.get_psir(iorb)

    def clear_all_psig_arr(self):
        self.iorb_psig_arr_map.clear()

    def compute_all_rhog(self):
        for iorb in self.iorb_psir_map:
            self.iorb_rhog_map[iorb] = self.get_rhog(iorb)
