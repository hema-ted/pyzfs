import numpy as np
from ..ft import fftshift, ifftshift, irfftn
from mpi4py import MPI


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
            self.yzlowerplane = list(zip(*np.nonzero(yzplane)))

    def compute_psir_from_psig_arr(self, psig_arr):
        """Compute psi(r) based on psi(G) defined on self.gvecs"""

        assert self.gamma, "Only gamma-point is implemented"

        # d stands for dense, s stands for smooth
        nd = np.array([self.dft.n1, self.dft.n2, self.dft.n3])
        ns = np.array([self.ft.n1, self.ft.n2, self.ft.n3])
        idxs = np.zeros(3, dtype=int)

        psigzyxd = np.zeros((nd[2], nd[1], nd[0] // 2 + 1), dtype=np.complex_)
        psigzyxd[self.gvecs[:, 2], self.gvecs[:, 1], self.gvecs[:, 0]] = psig_arr

        for i in range(1, 3):  # idxs[0] = 0 for gamma trick
            d = nd[i] - ns[i]
            if d % 2 == 0:
                idxs[i] = d / 2
            else:
                if nd[i] % 2 == 0:
                    idxs[i] = d // 2 + 1
                else:
                    idxs[i] = d // 2

        psigzyxs = ifftshift(
            (fftshift(psigzyxd, axes=(0, 1)))[
                idxs[2]:idxs[2] + ns[2],
                idxs[1]:idxs[1] + ns[1],
                0: ns[0] // 2 + 1,
            ], axes=(0, 1)
        )

        for ig2, ig3 in self.yzlowerplane:
            psigzyxs[ig3, ig2, 0] = psigzyxs[-ig3, -ig2, 0].conjugate()

        psirzyxs = irfftn(psigzyxs, s=(ns[2], ns[1], ns[0]))

        return self.normalize(psirzyxs.T)

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
            rank = MPI.COMM_WORLD.Get_rank()
            s = "{} Impossible to get psir: orbital {} is not loaded".format(rank, iorb)
            assert iorb in self.iorb_psig_arr_map, s
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
