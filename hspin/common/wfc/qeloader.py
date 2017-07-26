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
from ..counter import Counter

from sunyata.models.systems.empty import empty_ase_cell
from sunyata.parsers.text import parse_one_value


class QEWavefunctionLoader(WavefunctionLoader):

    def __init__(self, fftgrid="density"):
        self.fftgrid = fftgrid
        self.dft = None
        self.wft = None
        super(QEWavefunctionLoader, self).__init__()


    def scan(self):
        super(QEWavefunctionLoader, self).scan()

        dxml = etree.parse("data-file.xml").getroot()
        assert dxml.find("CELL/DIRECT_LATTICE_VECTORS/UNITS_FOR_DIRECT_LATTICE_VECTORS").attrib["UNITS"] == "Bohr"
        a1 = np.fromstring(dxml.find("CELL/DIRECT_LATTICE_VECTORS/a1").text, sep=" ", dtype=np.float_)
        a2 = np.fromstring(dxml.find("CELL/DIRECT_LATTICE_VECTORS/a2").text, sep=" ", dtype=np.float_)
        a3 = np.fromstring(dxml.find("CELL/DIRECT_LATTICE_VECTORS/a3").text, sep=" ", dtype=np.float_)
        cell = Cell(empty_ase_cell(a1, a2, a3, unit="bohr"))

        fftgrid = dxml.find("PLANE_WAVES/FFT_GRID").attrib
        grids = np.array([fftgrid["nr1"], fftgrid["nr2"], fftgrid["nr3"]], dtype=np.int_)
        self.dft = FourierTransform(grids[0], grids[1], grids[2])
        if self.fftgrid == "density":
            n1, n2, n3 = grids
        elif self.fftgrid == "wave":
            n1, n2, n3 = np.array(grids / np.sqrt(2), dtype=int)
        else:
            assert len(fftgrid) == 3
            n1, n2, n3 = self.fftgrid
        self.wft = FourierTransform(n1, n2, n3)

        gxml = etree.parse("K00001/gkvectors.xml").getroot()
        self.gamma = True if "T" in gxml.find("GAMMA_ONLY").text else False
        assert self.gamma, "Only gamma point calculation is supported now"
        if self.gamma:
            yzplane = np.zeros((n2, n3))
            yzplane[n2 // 2 + 1:, :] = 1
            yzplane[0, n3 // 2 + 1:] = 1
            self.yzlowerplane = zip(*np.nonzero(yzplane))
        self.npw = int(gxml.find("NUMBER_OF_GK-VECTORS").text)

        self.gvecs = np.fromstring(gxml.find("GRID").text,
                                   sep=" ", dtype=np.int_).reshape(-1, 3)
        assert self.gvecs.shape == (self.npw, 3)
        assert np.ptp(self.gvecs[:, 0]) <= self.dft.n1
        assert np.ptp(self.gvecs[:, 1]) <= self.dft.n2
        assert np.ptp(self.gvecs[:, 2]) <= self.dft.n3

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

        c = Counter(len(iorbs),
                    message="{n} orbitals ({percent}%) loaded (on processor 0, norbs = {ntot})...")

        iuorbs = filter(lambda iorb: self.wfc.iorb_sb_map[iorb][0] == "up", iorbs)
        idorbs = filter(lambda iorb: self.wfc.iorb_sb_map[iorb][0] == "down", iorbs)

        iterxml = etree.iterparse("K00001/evc1.xml")
        for event, leaf in iterxml:
            if "evc." in leaf.tag:
                band = parse_one_value(int, leaf.tag)
                iorb = self.wfc.sb_iorb_map.get(("up", band))
                if iorb in iuorbs:
                    psir = self.parse_psir_from_text(leaf.text)
                    self.wfc.iorb_psir_map[iorb] = self.normalize(psir)
                    c.count()
            leaf.clear()

        iterxml = etree.iterparse("K00001/evc2.xml")
        for event, leaf in iterxml:
            if "evc." in leaf.tag:
                band = parse_one_value(int, leaf.tag)
                iorb = self.wfc.sb_iorb_map.get(("down", band))
                if iorb in idorbs:
                    psir = self.parse_psir_from_text(leaf.text)
                    self.wfc.iorb_psir_map[iorb] = self.normalize(psir)
                    c.count()
            leaf.clear()

    def parse_psir_from_text(self, text):
        """Get orbital in real space from QE xml file.

        Args:
            text: text of "evc.X" leafs in QE evc{1/2}.xml file

        Returns:
            Real space orbital defined on grid specified by self.wfc.ft (self.wft)

        """
        assert self.gamma, "Only gamma point is implemented yet"

        c = np.fromstring(
            text.replace(",", "\n"),
            sep="\n", dtype=np.float_).view(np.complex_)

        n1, n2, n3 = self.wft.n1, self.wft.n2, self.wft.n3
        dn1, dn2, dn3 = self.dft.n1, self.dft.n2, self.dft.n3

        # Read orbital in density grid
        # x and z axes are switched for convenience of rFFT
        psig_zyx = np.zeros((dn3, dn2, dn1 // 2 + 1), dtype=np.complex_)
        psig_zyx[self.gvecs[:, 2], self.gvecs[:, 1], self.gvecs[:, 0]] = c

        # If a smoother grid is required, crop high frequency components
        if (n1, n2, n3) != (dn1, dn2, dn3):
            psig_zyx = ifftshift(
                fftshift(psig_zyx, axes=(0, 1))[
                (dn3 - n3 - 1) // 2 + 1:(dn3 - n3 - 1) // 2 + 1 + n3,
                (dn2 - n2 - 1) // 2 + 1:(dn2 - n2 - 1) // 2 + 1 + n2,
                0: n1 // 2 + 1,
                ], axes=(0, 1)
            )
        assert psig_zyx.shape == (n3, n2, n1 // 2 + 1)

        # Complete psig in yz plane
        for ig2, ig3 in self.yzlowerplane:
            psig_zyx[ig3, ig2, 0] = psig_zyx[-ig3, -ig2, 0].conjugate()

        # rFFT in x direction (x has been switched to last axes), FFT in y, z direction
        psir_zyx = irfftn(psig_zyx, s=(n3, n2, n1))

        # Switch back x, z axes
        psir = psir_zyx.swapaxes(0, 2)

        return psir
