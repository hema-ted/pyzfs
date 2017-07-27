from __future__ import absolute_import, division, print_function
import os
from time import time
from glob import glob
import numpy as np
from mpi4py import MPI
from pprint import pprint
import resource

from .baseloader import WavefunctionLoader
from ..cell import Cell
from ..ft import FourierTransform
from .wavefunction import Wavefunction
from ..parallel import mpiroot

class VaspWavefunctionLoader(WavefunctionLoader):

    def scan(self):
        super(VaspWavefunctionLoader, self).scan()

        from ase.io import read
        from ...common import vaspwfc

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

        iorb_sb_map = list(
            ("up", iuorbs[iorb]) if iorb < nuorbs
            else ("down", idorbs[iorb - nuorbs])
            for iorb in range(norbs)
        )

        iorb_fname_map = ["WAVECAR"] * norbs
        self.wfc = Wavefunction(cell=cell, ft=ft, nuorbs=nuorbs, ndorbs=ndorbs,
                                iorb_sb_map=iorb_sb_map, iorb_fname_map=iorb_fname_map)

    def load(self, iorbs):
        super(VaspWavefunctionLoader, self).load()

        counter = 0
        for iorb in iorbs:
            spin, band = self.wfc.iorb_sb_map[iorb]
            psir = self.wavecar.wfc_r(
                ispin=1 if spin == "up" else 2, iband=band, gamma=True
            )
            psir = self.normalize(psir)
            self.wfc.iorb_psir_map[iorb] = psir

            counter += 1
            if counter >= len(iorbs) // 10:
                if mpiroot:
                    print("........")
                counter = 0
