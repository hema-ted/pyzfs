from __future__ import absolute_import, division, print_function
import os
from time import time
from glob import glob
import numpy as np
from mpi4py import MPI
from pprint import pprint
import resource

from lxml import etree


from .baseloader import WavefunctionLoader
from ..cell import Cell
from ..ft import FourierTransform
from .wavefunction import Wavefunction
from ..parallel import mpiroot

from sunyata.models.systems.empty import empty_ase_cell
from sunyata.parsers.text import parse_one_value

class QEWavefunctionLoader(WavefunctionLoader):

    def scan(self):
        super(QEWavefunctionLoader, self).scan()

        dxml = etree.parse("data-file.xml").getroot()
        assert dxml.find("CELL/DIRECT_LATTICE_VECTORS/UNITS_FOR_DIRECT_LATTICE_VECTORS").attrib["UNITS"] == "Bohr"
        a1 = np.fromstring(dxml.find("CELL/DIRECT_LATTICE_VECTORS/a1").text, sep=" ", dtype=np.float_)
        a2 = np.fromstring(dxml.find("CELL/DIRECT_LATTICE_VECTORS/a2").text, sep=" ", dtype=np.float_)
        a3 = np.fromstring(dxml.find("CELL/DIRECT_LATTICE_VECTORS/a3").text, sep=" ", dtype=np.float_)
        cell = Cell(empty_ase_cell(a1, a2, a3, unit="bohr"))

        fftgrid = dxml.find("PLANE_WAVES/FFT_GRID").attrib
        ft = FourierTransform(
            int(fftgrid["nr1"]), int(fftgrid["nr2"]), int(fftgrid["nr3"])
        )

        gxml = etree.parse("K00001/gkvectors.xml").getroot()
        self.gamma = True if "T" in gxml.find("GAMMA_ONLY").text else False
        assert self.gamma, "Only gamma point calculation is supported now"
        self.npw = int(gxml.find("NUMBER_OF_GK-VECTORS").text)

        self.gvecs = np.fromstring(gxml.find("GRID").text,
                                   sep=" ", dtype=np.int_).reshape(-1, 3)
        assert self.gvecs.shape == (self.npw, 3)
        assert np.ptp(self.gvecs[:, 0]) <= ft.n1
        assert np.ptp(self.gvecs[:, 1]) <= ft.n2
        assert np.ptp(self.gvecs[:, 2]) <= ft.n3
        # self.gvecs[:, [0, 2]] = self.gvecs[:, [2, 0]]

        euxml = etree.parse("K00001/eigenval1.xml").getroot()
        edxml = etree.parse("K00001/eigenval2.xml").getroot()

        uoccs = np.fromstring(euxml.find("OCCUPATIONS").text, sep="\n", dtype=np.float_)
        doccs = np.fromstring(edxml.find("OCCUPATIONS").text, sep="\n", dtype=np.float_)

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

        iorb_fname_map = ["evc1.xml"] * nuorbs + ["evc2.xml"] * ndorbs

        self.wfc = Wavefunction(cell=cell, ft=ft, nuorbs=nuorbs, ndorbs=ndorbs,
                                iorb_sb_map=iorb_sb_map, iorb_fname_map=iorb_fname_map)

        self.c1xml = etree.parse("K00001/evc1.xml").getroot()
        self.c2xml = etree.parse("K00001/evc2.xml").getroot()

    def load(self, iorbs):
        super(QEWavefunctionLoader, self).load(iorbs)

        iterxml = etree.iterparse("K00001/evc1.xml")
        for event, leaf in iterxml:
            if "evc." in leaf.tag:
                band = parse_one_value(int, leaf.tag)
                if ("up", band) in self.wfc.sb_iorb_map:
                    iorb = self.wfc.sb_iorb_map[("up", band)]
                    if iorb in iorbs:
                        psir = self.parse_psir_from_text(leaf.text)
                        psir = self.normalize(psir)
                        self.wfc.iorb_psir_map[iorb] = psir
            leaf.clear()

        if mpiroot:
            print("........")

        iterxml = etree.iterparse("K00001/evc2.xml")
        for event, leaf in iterxml:
            if "evc." in leaf.tag:
                band = parse_one_value(int, leaf.tag)
                if ("down", band) in self.wfc.sb_iorb_map:
                    iorb = self.wfc.sb_iorb_map[("down", band)]
                    if iorb in iorbs:
                        psir = self.parse_psir_from_text(leaf.text)
                        psir = self.normalize(psir)
                        self.wfc.iorb_psir_map[iorb] = psir
            leaf.clear()

        if mpiroot:
            print("........")


    def parse_psir_from_text(self, text):
        c = np.fromstring(
            text.replace(",", "\n"),
            sep="\n", dtype=np.float_).view(np.complex_)

        n1, n2, n3 = self.wfc.ft.n1, self.wfc.ft.n2, self.wfc.ft.n3

        psig = np.zeros((n1, n2, n3), dtype=np.complex_)
        psig[self.gvecs[:, 0], self.gvecs[:, 1], self.gvecs[:, 2]] = c
        if self.gamma:
            for ig1, ig2, ig3 in np.ndindex(n1, n2, n3):
                rig1 = ig1 if ig1 < n1 // 2 + 1 else ig1 - n1
                rig2 = ig2 if ig2 < n2 // 2 + 1 else ig2 - n2
                rig3 = ig3 if ig3 < n3 // 2 + 1 else ig3 - n3
                if ((rig1 < 0) or (rig1 == 0 and rig2 < 0)
                    or (rig1 == 0 and rig2 == 0 and rig3 < 0)):
                    psig[ig1, ig2, ig3] = psig[-ig1, -ig2, -ig3].conjugate()

        return self.wfc.ft.backward(psig)
