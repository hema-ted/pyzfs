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
from ..ft import FourierTransform, fftshift, ifftshift, irfftn, ifftn
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
        self.dft = FourierTransform(
            int(fftgrid["nr1"]), int(fftgrid["nr2"]), int(fftgrid["nr3"])
        )
        self.wft = FourierTransform(
            int(fftgrid["nr1"]), int(fftgrid["nr2"]), int(fftgrid["nr3"])
            #int(fftgrid["nr1"]) // 45, int(fftgrid["nr2"]) // 45, int(fftgrid["nr3"]) // 45
        )

        gxml = etree.parse("K00001/gkvectors.xml").getroot()
        self.gamma = True if "T" in gxml.find("GAMMA_ONLY").text else False
        assert self.gamma, "Only gamma point calculation is supported now"
        self.npw = int(gxml.find("NUMBER_OF_GK-VECTORS").text)

        self.gvecs = np.fromstring(gxml.find("GRID").text,
                                   sep=" ", dtype=np.int_).reshape(-1, 3)
        assert self.gvecs.shape == (self.npw, 3)
        assert np.ptp(self.gvecs[:, 0]) <= self.dft.n1
        assert np.ptp(self.gvecs[:, 1]) <= self.dft.n2
        assert np.ptp(self.gvecs[:, 2]) <= self.dft.n3
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

        self.wfc = Wavefunction(cell=cell, ft=self.wft, nuorbs=nuorbs, ndorbs=ndorbs,
                                iorb_sb_map=iorb_sb_map, iorb_fname_map=iorb_fname_map)

    def load(self, iorbs):
        # TODO: first column and row read, then bcast to all processors
        super(QEWavefunctionLoader, self).load(iorbs)

        iterxml = etree.iterparse("K00001/evc1.xml")
        for event, leaf in iterxml:
            if "evc." in leaf.tag:
                band = parse_one_value(int, leaf.tag)
                if ("up", band) in self.wfc.sb_iorb_map:
                    iorb = self.wfc.sb_iorb_map[("up", band)]
                    if iorb in iorbs:
                        # psir = self.parse_psir_from_text(leaf.text)
                        psir = self.rparse_psir_from_text(leaf.text)
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
                        psir = self.rparse_psir_from_text(leaf.text)
                        psir = self.normalize(psir)
                        self.wfc.iorb_psir_map[iorb] = psir
            leaf.clear()

        if mpiroot:
            print("........")

    def parse_psir_from_text(self, text):
        print("Total memory usage (on process 0):")
        print("{:.2f} MB".format(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.
        ))
        if mpiroot:
            print("parse_psir_from_text")
        c = np.fromstring(
            text.replace(",", "\n"),
            sep="\n", dtype=np.float_).view(np.complex_)

        n1, n2, n3 = self.wfc.ft.n1, self.wfc.ft.n2, self.wfc.ft.n3

        psig = np.zeros((self.dft.n1, self.dft.n2, self.dft.n3), dtype=np.complex_)
        psig[self.gvecs[:, 0], self.gvecs[:, 1], self.gvecs[:, 2]] = c
        if mpiroot:
            print(n1, n2, n3, self.dft.n1, self.dft.n2, self.dft.n3)
            print("psig loaded")
            print(psig)
            print("~")

        sspsig = fftshift(psig)[
            (self.dft.n1 - n1) // 2:(self.dft.n1 - n1) // 2 + n1,
            (self.dft.n2 - n2) // 2:(self.dft.n2 - n2) // 2 + n2,
            (self.dft.n3 - n3) // 2:(self.dft.n3 - n3) // 2 + n3,
        ]

        if self.gamma:
            sspsig = sspsig + sspsig[::-1, ::-1, ::-1].conjugate()
            spsig1 = ifftshift(sspsig)
            spsig1[0, 0, 0] /= 2.

        spsig = ifftshift(sspsig)

        print("spsig1")
        print(spsig1)

        spsig2 = np.zeros(sspsig.shape, dtype=np.complex_)
        if self.gamma:
            for ig1, ig2, ig3 in np.ndindex(n1, n2, n3):
                spsig2[ig1, ig2, ig3] = spsig[ig1, ig2, ig3] + spsig[-ig1, -ig2, -ig3].conjugate()
            spsig2[0, 0, 0] /= 2

        if mpiroot:
            print("spsig2")
            print(spsig2)
            print(np.isclose(spsig1, spsig2))
            print("----------")

        if self.gamma:
            for ig1, ig2, ig3 in np.ndindex(n1, n2, n3):
                rig1 = ig1 if ig1 < n1 // 2 + 1 else ig1 - n1
                rig2 = ig2 if ig2 < n2 // 2 + 1 else ig2 - n2
                rig3 = ig3 if ig3 < n3 // 2 + 1 else ig3 - n3
                if ((rig1 < 0) or (rig1 == 0 and rig2 < 0)
                    or (rig1 == 0 and rig2 == 0 and rig3 < 0)):
                    spsig[ig1, ig2, ig3] = spsig[-ig1, -ig2, -ig3].conjugate()

        if mpiroot:
            print(spsig)
            print(np.isclose(spsig, spsig1))
            print(np.isclose(spsig, spsig2))
            print("----------")
            print("fft...")

        spsir = self.wfc.ft.backward(spsig)

        if mpiroot:
            print("fft finished...")
        return spsir


    def rparse_psir_from_text(self, text):
        if mpiroot:
            print("rparse_psir_from_text")
        c = np.fromstring(
            text.replace(",", "\n"),
            sep="\n", dtype=np.float_).view(np.complex_)

        n1, n2, n3 = self.wft.n1, self.wft.n2, self.wft.n3

        psig_zyx = np.zeros((n3, n2, n1 // 2 + 1), dtype=np.complex_)
        # fpsig_zyx = np.zeros((n3, n2, n1), dtype=np.complex_)
        psig_zyx[self.gvecs[:, 2], self.gvecs[:, 1], self.gvecs[:, 0]] = c
        # fpsig_zyx[self.gvecs[:, 2], self.gvecs[:, 1], self.gvecs[:, 0]] = c
        # if mpiroot:
        #     print("psig_zyx loaded")
        #     print(psig_zyx)
        #     print("~")
        #
        # spsig_zyx = ifftshift(
        #     fftshift(psig_zyx_half)[
        #          (self.dft.n3 - n3 - 1) // 2 + 1:(self.dft.n3 - n3 - 1) // 2 + 1 + n3,
        #          (self.dft.n2 - n2 - 1) // 2 + 1:(self.dft.n2 - n2 - 1) // 2 + 1 + n2,
        #          (self.dft.n1 - n1 - 1) // 2 + 1:(self.dft.n1 - n1 - 1) // 2 + 1 + n1,
        #          ]
        # )
        #

        # if self.gamma:
        #     for ig1, ig2, ig3 in np.ndindex(n1, n2, n3):
        #         rig1 = ig1 if ig1 < n1 // 2 + 1 else ig1 - n1
        #         rig2 = ig2 if ig2 < n2 // 2 + 1 else ig2 - n2
        #         rig3 = ig3 if ig3 < n3 // 2 + 1 else ig3 - n3
        #         if ((rig1 < 0) or (rig1 == 0 and rig2 < 0)
        #             or (rig1 == 0 and rig2 == 0 and rig3 < 0)):
        #             fpsig_zyx[ig3, ig2, ig1] = fpsig_zyx[-ig3, -ig2, -ig1].conjugate()
        # fpsir_zyx = ifftn(fpsig_zyx)

        if self.gamma:
            yzlowerplane = np.zeros((n2, n3))
            yzlowerplane[n2 // 2 + 1:, :] = 1
            yzlowerplane[0, n3 // 2 + 1:] = 1
            #
            # if mpiroot:
            #     print("yzlowerplane")
            #     print(yzlowerplane)
            for ig2, ig3 in zip(*np.nonzero(yzlowerplane)):
                psig_zyx[ig3, ig2, 0] = psig_zyx[-ig3, -ig2, 0].conjugate()
        psir_zyx = irfftn(psig_zyx, s=(n3, n2, n1))

        if mpiroot:
            print("fft finished...")
        #
        # fpsir = fpsir_zyx.swapaxes(0, 2)
        psir = psir_zyx.swapaxes(0, 2)

        return psir #, fpsir
