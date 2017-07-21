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

class CubeWavefunctionLoader(WavefunctionLoader):
    def __init__(self, density=False):
        super(CubeWavefunctionLoader, self).__init__()
        from sunyata.parsers.text import parse_one_value
        from ase.io.cube import read_cube_data

        self.density = density

        uorbfiles = sorted(glob("*up*.cube"))
        dorbfiles = sorted(glob("*down*.cube"))

        nuorbs = len(uorbfiles)
        ndorbs = len(dorbfiles)

        self.idxfilemap = uorbfiles + dorbfiles
        idxsbmap = map(
            lambda fname: ("up" if "up" in fname else "down",
                           parse_one_value(int, fname, -1)),
            self.idxfilemap
        )

        # Define cell and Fourier transform grid from first cube file
        fname0 = self.idxfilemap[0]
        psir, ase_cell = read_cube_data(fname0)
        cell = Cell(ase_cell)
        if self.density:
            self.dft = FourierTransform(*psir.shape)
            ft = FourierTransform(*map(lambda x: x // 2, psir.shape))
            if mpiroot:
                print("Charge density grid {} will be interpolated to wavefunction grid {}.\n".format(
                    map(lambda x: x // 2, psir.shape), psir.shape
                ))
        else:
            ft = FourierTransform(*psir.shape)

        self.wfc = Wavefunction(cell=cell, ft=ft, nuorbs=nuorbs, ndorbs=ndorbs, idxsbmap=idxsbmap)


    def normalize(self, psir):
        """Normalize cube wavefunction."""
        assert psir.shape == (self.wfc.ft.n1, self.wfc.ft.n2, self.wfc.ft.n3)
        if self.density:
            norm = np.sqrt(np.sum(np.abs(psir)) * self.wfc.cell.omega / self.wfc.ft.N)
        else:
            norm = np.sqrt(np.sum(np.abs(psir) ** 2) * self.wfc.cell.omega / self.wfc.ft.N)
        psir /= norm

    def load(self, iorbs):
        """Load KS orbitals to memory, store in wfc.idxdatamap."""
        from ase.io.cube import read_cube_data

        counter = 0
        for iorb in range(self.wfc.norbs):
            # Iterate over all orbitals, because ASE use global communicator
            fname = self.idxfilemap[iorb]
            if self.density:
                rho = read_cube_data(fname)[0]
                psird = np.sign(rho) * np.sqrt(np.abs(rho))
                psir = self.dft.interp(psird, self.wfc.ft.n1, self.wfc.ft.n2, self.wfc.ft.n3)
            else:
                psir = read_cube_data(fname)[0]

            self.normalize(psir)
            self.wfc.idxdatamap[iorb] = psir

            counter += 1
            if counter >= self.nwfcs // 10:
                if mpiroot:
                    print("........")
                counter = 0
