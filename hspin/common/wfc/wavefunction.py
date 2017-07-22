import numpy as np

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
    def __init__(self, cell, ft, nuorbs, ndorbs, iorb_sb_map, iorb_fname_map):

        self.cell = cell
        self.ft = ft

        self.nuorbs = nuorbs
        self.ndorbs = ndorbs
        self.norbs = self.nuorbs + self.ndorbs

        self.iorb_sb_map = iorb_sb_map
        self.sb_iorb_map = {
            self.iorb_sb_map[iorb]: iorb for iorb in range(self.norbs)
        }

        self.iorb_fname_map = iorb_fname_map

        self.iorb_psir_map = {}
        self.iorb_rhog_map = {}

    def compute_rho(self):
        for iorb, psir in self.iorb_psir_map.items():
            self.iorb_rhog_map[iorb] = self.ft.forward(psir * np.conj(psir))
