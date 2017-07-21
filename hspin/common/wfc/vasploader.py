from __future__ import absolute_import, division, print_function
import os
from time import time
from glob import glob
import numpy as np
from mpi4py import MPI
from pprint import pprint
import resource

from .loader import WavefunctionLoader
from ..cell import Cell
from ..ft import FourierTransform
from . import Wavefunction
from ..parallel import mpiroot

class VaspWavefunctionLoader(WavefunctionLoader):

    def __init__(self):
        super(VaspWavefunctionLoader, self).__init__()

        from ase.io import read
        from sunyata.parsers.vasp import vaspwfc

        # Read cell from POSCAR file
        ase_cell = read("POSCAR")
        cell = Cell(ase_cell)

        # Read wfc from WAVECAR file
        self.wavecar = vaspwfc()
        ft = FourierTransform(*self.wavecar._ngrid)
        nspin, nkpts, nbands = self.wavecar._occs.shape
        assert nspin == 2 and nkpts == 1

        # Get band indices (starting from 1) witt significant occupations
        iuorbs = np.where(self.wavecar._occs[0, 0] > 0.8)[0] + 1
        idorbs = np.where(self.wavecar._occs[1, 0] > 0.8)[0] + 1

        nuorbs = len(iuorbs)
        ndorbs = len(idorbs)
        norbs = nuorbs + ndorbs

        idxsbmap = list(
            ("up", iuorbs[iwfc]) if iwfc < nuorbs
            else ("down", idorbs[iwfc - nuorbs])
            for iwfc in range(norbs)
        )

        self.wfc = Wavefunction(cell=cell, ft=ft, nuorbs=nuorbs, ndorbs=ndorbs, idxsbmap=idxsbmap)

    def normalize(self, psir):
        """Normalize VASP pseudo wavefunction."""
        assert psir.shape == (self.wfc.ft.n1, self.wfc.ft.n2, self.wfc.ft.n3)
        norm = np.sqrt(np.sum(np.abs(psir) ** 2) * self.wfc.cell.omega / self.wfc.ft.N)
        psir /= norm

    def load(self, iorbs):
        """Load KS orbitals to memory, store in wfc.idxdatamap."""
        counter = 0
        for iorb in iorbs:
            spin, band = self.wfc.idxsbmap(iorb)
            psir = self.wavecar.wfc_r(
                ispin=1 if spin == "up" else 2, iband=band, gamma=True
            )
            self.normalize(psir)
            self.wfc.idxdatamap[iorb] = psir

            counter += 1
            if counter >= iorbs // 10:
                if mpiroot:
                    print("........")
                counter = 0
