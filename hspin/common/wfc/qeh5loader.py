from __future__ import absolute_import, division, print_function
import numpy as np
import os
import h5py
from lxml import etree

from .baseloader import WavefunctionLoader
from ..cell import Cell
from ..ft import FourierTransform, fftshift, ifftshift, irfftn, ifftn
from .wavefunction import Wavefunction
from ..counter import Counter

from ...common.external import empty_ase_cell


class QEHDF5WavefunctionLoader(WavefunctionLoader):

    def __init__(self, fftgrid="density"):
        self.fftgrid = fftgrid
        self.dft = None
        self.wft = None
        super(QEHDF5WavefunctionLoader, self).__init__()

    def scan(self):
        super(QEHDF5WavefunctionLoader, self).scan()

        self.root = "."

        pwxml = etree.parse("{}/pwscf.xml".format(self.root))
        self.prefix = pwxml.find("input/control_variables/prefix").text

        # parse cell and FFT grid
        R1 = np.fromstring(pwxml.find("output/atomic_structure/cell/a1").text, sep=" ")
        R2 = np.fromstring(pwxml.find("output/atomic_structure/cell/a2").text, sep=" ")
        R3 = np.fromstring(pwxml.find("output/atomic_structure/cell/a3").text, sep=" ")
        cell = Cell(empty_ase_cell(R1, R2, R3, unit="bohr"))

        n1 = int(pwxml.find("output/basis_set/fft_grid").attrib["nr1"])
        n2 = int(pwxml.find("output/basis_set/fft_grid").attrib["nr2"])
        n3 = int(pwxml.find("output/basis_set/fft_grid").attrib["nr3"])

        grids = np.array([n1, n2, n3], dtype=np.int_)
        self.dft = FourierTransform(grids[0], grids[1], grids[2])
        if self.fftgrid == "density":
            n1, n2, n3 = grids
        elif self.fftgrid == "wave":
            n1, n2, n3 = np.array(grids / 2, dtype=int)
        else:
            raise ValueError
        self.wft = FourierTransform(n1, n2, n3)

        # parse general wfc info
        if pwxml.find("output/magnetization/lsda").text == "true":
            if pwxml.find("output/magnetization/noncolin").text == "true":
                nspin = 4
            else:
                nspin = 2
        else:
            nspin = 1

        if nspin != 2:
            raise NotImplementedError

        self.gamma = bool(pwxml.find("output/basis_set/gamma_only").text)
        assert self.gamma, "Only gamma point calculation is supported now"

        nkpt = int(pwxml.find("input/k_points_IBZ/nk").text)
        if nkpt != 1:
            raise NotImplementedError

        if nspin == 2:
            nbnd = [[int(pwxml.find("output/band_structure/nbnd_up").text)],
                    [int(pwxml.find("output/band_structure/nbnd_dw").text)]]
        else:
            raise NotImplementedError

        occ = np.fromstring(pwxml.find("output/band_structure/ks_energies/occupations").text, sep=" ").reshape(
            nspin, nkpt, nbnd[0][0]
        )

        uoccs = occ[0, 0]
        doccs = occ[1, 0]

        iuorbs = np.where(uoccs > 0.8)[0] + 1
        idorbs = np.where(doccs > 0.8)[0] + 1

        nuorbs = len(iuorbs)
        ndorbs = len(idorbs)
        norbs = nuorbs + ndorbs

        iorb_sb_map = list(
            ("up", iuorbs[iwfc]) if iwfc < nuorbs
            else ("down", idorbs[iwfc - nuorbs])
            for iwfc in range(norbs)
        )
        iorb_fname_map = ["wfcup1.hdf5"] * nuorbs + ["wfcdw1.hdf5"] * ndorbs

        self.wfc = Wavefunction(cell=cell, ft=self.wft, nuorbs=nuorbs, ndorbs=ndorbs,
                                iorb_sb_map=iorb_sb_map, iorb_fname_map=iorb_fname_map,
                                dft=self.dft, gamma=self.gamma, gvecs=None)

    def load(self, iorbs):
        # TODO: first column and row read, then bcast to all processors
        super(QEHDF5WavefunctionLoader, self).load(iorbs)

        c = Counter(len(iorbs), percent=0.1,
                    message="(process 0) {n} orbitals ({percent}%) loaded in {dt}...")

        iuorbs = filter(lambda iorb: self.wfc.iorb_sb_map[iorb][0] == "up", iorbs)
        idorbs = filter(lambda iorb: self.wfc.iorb_sb_map[iorb][0] == "down", iorbs)

        # parse KS orbitals
        for ispin in range(2):
            wfcfile = "wfcup1.hdf5" if ispin == 0 else "wfcdw1.hdf5"
            wfch5 = h5py.File(os.path.join(
                self.root, "{}.save".format(self.prefix), wfcfile))

            gvecs = np.array(wfch5["MillerIndices"])
            self.wfc.gvecs = gvecs

            evc = np.array(wfch5["evc"])

            for ievc in range(evc.shape[0]):
                band = ievc + 1
                iorb = self.wfc.sb_iorb_map.get(("up" if ispin == 0 else "down", band))

                if iorb in (iuorbs if ispin == 0 else idorbs):
                    psig_arr = evc[ievc].view(complex)
                    self.wfc.set_psig_arr(iorb, psig_arr)
                    c.count()

        if self.memory == "high":
            self.wfc.compute_all_psir()
            self.wfc.clear_all_psig_arr()
            self.wfc.compute_all_rhog()
        elif self.memory == "low":
            self.wfc.compute_all_psir()
            self.wfc.clear_all_psig_arr()
        elif self.memory == "critical":
            pass
        else:
            raise ValueError

