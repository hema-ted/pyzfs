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

class CubeWavefunctionLoader(WavefunctionLoader):
    def __init__(self, density=False):
        self.density = density
        super(CubeWavefunctionLoader, self).__init__()

    def scan(self):
        from sunyata.parsers.text import parse_one_value
        from ase.io.cube import read_cube_data

        ufnames = sorted(glob("*up*.cube"))
        dfnames = sorted(glob("*down*.cube"))

        nuorbs = len(ufnames)
        ndorbs = len(dfnames)

        iorb_fname_map = ufnames + dfnames
        idxsbmap = map(
            lambda fname: ("up" if "up" in fname else "down",
                           parse_one_value(int, fname, -1)),
            iorb_fname_map
        )

        # Define cell and Fourier transform grid from first cube file
        fname0 = iorb_fname_map[0]
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

        self.wfc = Wavefunction(cell=cell, ft=ft, nuorbs=nuorbs, ndorbs=ndorbs,
                                iorb_sb_map=idxsbmap, iorb_fname_map=iorb_fname_map)

        self.info()

    def load(self, iorbs):
        """Load KS orbitals to memory, store in wfc.idxdatamap."""
        super(CubeWavefunctionLoader, self).load(iorbs=iorbs)
        from ase.io.cube import read_cube_data

        wfc = self.wfc

        counter = 0
        for iorb in range(self.wfc.norbs):
            # Iterate over all orbitals, because ASE use global communicator
            fname = wfc.iorb_fname_map[iorb]
            if self.density:
                rho = read_cube_data(fname)[0]
                psird = np.sign(rho) * np.sqrt(np.abs(rho))
                psir = self.dft.interp(psird, wfc.ft.n1, wfc.ft.n2, wfc.ft.n3)
            else:
                psir = read_cube_data(fname)[0]

            self.normalize(psir)
            wfc.iorb_psir_map[iorb] = psir

            counter += 1
            if counter >= wfc.norbs // 10:
                if mpiroot:
                    print("........")
                counter = 0
