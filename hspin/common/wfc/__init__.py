

class Wavefunction:
    """Container class for Kohn-Sham orbitals
    
    Physically, wavefunction is uniquely labeled by a 2-tuple of band index (int)
    and spin ("up" or "down"). Internally, each wavefunction is labeled by an integer index.
    Several maps are defined to describe related transformations.
    
    Attributes:
        norbs (int): total number of KS orbitals to be considered
        nuorbs/ndorbs (int): number of spin up/down orbitals
        sbidxmap (dict): (spin, band index) -> orb index map
        idxsbmap (list): orb index -> (spin, band index) map
        idxdatamap (dict): orb index -> orb object (3D array) map

        cell (Cell): defines cell size, R and G vectors
        ft (FourierTransform): defines grid size for fourier transform

    Right now only consider ground state, insulating, spin-polarized case.
    No occupation number considerations are implemented yet.
    """
    def __init__(self, cell, ft, nuorbs, ndorbs, idxsbmap):

        self.cell = cell
        self.ft = ft

        self.nuorbs = nuorbs
        self.ndorbs = ndorbs
        self.norbs = self.nuorbs + self.ndorbs

        self.idxsbmap = idxsbmap
        self.sbidxmap = {
            self.idxsbmap[iorb]: iorb for iorb in range(self.norbs)
        }

        self.idxdatamap = {}