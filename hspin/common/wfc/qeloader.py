from __future__ import absolute_import, division, print_function
import numpy as np
from lxml import etree

from .baseloader import WavefunctionLoader
from ..cell import Cell
from ..ft import FourierTransform, fftshift, ifftshift, irfftn, ifftn
from .wavefunction import Wavefunction
from ..counter import Counter

from ...common.external import empty_ase_cell
from ...common.external import parse_one_value


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
            n1, n2, n3 = np.array(grids / 2, dtype=int)
        else:
            assert len(fftgrid) == 3
            n1, n2, n3 = self.fftgrid
        self.wft = FourierTransform(n1, n2, n3)

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
                                iorb_sb_map=iorb_sb_map, iorb_fname_map=iorb_fname_map,
                                dft=self.dft, gamma=self.gamma, gvecs=self.gvecs)

    def load(self, iorbs):
        # TODO: first column and row read, then bcast to all processors
        super(QEWavefunctionLoader, self).load(iorbs)

        c = Counter(len(iorbs), percent=0.1,
                    message="(process 0) {n} orbitals ({percent}%) loaded in {dt}...")

        iuorbs = filter(lambda iorb: self.wfc.iorb_sb_map[iorb][0] == "up", iorbs)
        idorbs = filter(lambda iorb: self.wfc.iorb_sb_map[iorb][0] == "down", iorbs)

        iterxml = etree.iterparse("K00001/evc1.xml", huge_tree=True)
        for event, leaf in iterxml:
            if "evc." in leaf.tag:
                band = parse_one_value(int, leaf.tag)
                iorb = self.wfc.sb_iorb_map.get(("up", band))
                if iorb in iuorbs:
                    psig_arr = np.fromstring(
                        leaf.text.replace(",", "\n"),
                        sep="\n", dtype=np.float_).view(np.complex_)
                    self.wfc.set_psig_arr(iorb, psig_arr)
                    c.count()
            leaf.clear()

        iterxml = etree.iterparse("K00001/evc2.xml", huge_tree=True)
        for event, leaf in iterxml:
            if "evc." in leaf.tag:
                band = parse_one_value(int, leaf.tag)
                iorb = self.wfc.sb_iorb_map.get(("down", band))
                if iorb in idorbs:
                    psig_arr = np.fromstring(
                        leaf.text.replace(",", "\n"),
                        sep="\n", dtype=np.float_).view(np.complex_)
                    self.wfc.set_psig_arr(iorb, psig_arr)
                    c.count()
            leaf.clear()

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

    #
    # def parse_psir_from_text(self, text):
    #     """Get orbital in real space from QE xml file.
    #
    #     Args:
    #         text: text of "evc.X" leafs in QE evc{1/2}.xml file
    #
    #     Returns:
    #         Real space orbital defined on grid specified by self.wfc.ft (self.wft)
    #
    #     """
    #     assert self.gamma, "Only gamma point is implemented yet"
    #
    #     c = np.fromstring(
    #         text.replace(",", "\n"),
    #         sep="\n", dtype=np.float_).view(np.complex_)
    #
    #     n1, n2, n3 = self.wft.n1, self.wft.n2, self.wft.n3
    #     dn1, dn2, dn3 = self.dft.n1, self.dft.n2, self.dft.n3
    #
    #     # Read orbital in density grid
    #     # x and z axes are switched for convenience of rFFT
    #     psig_zyx = np.zeros((dn3, dn2, dn1 // 2 + 1), dtype=np.complex_)
    #     psig_zyx[self.gvecs[:, 2], self.gvecs[:, 1], self.gvecs[:, 0]] = c
    #
    #     # If a smoother grid is required, crop high frequency components
    #     if (n1, n2, n3) != (dn1, dn2, dn3):
    #         psig_zyx = ifftshift(
    #             fftshift(psig_zyx, axes=(0, 1))[
    #             (dn3 - n3 - 1) // 2 + 1:(dn3 - n3 - 1) // 2 + 1 + n3,
    #             (dn2 - n2 - 1) // 2 + 1:(dn2 - n2 - 1) // 2 + 1 + n2,
    #             0: n1 // 2 + 1,
    #             ], axes=(0, 1)
    #         )
    #     assert psig_zyx.shape == (n3, n2, n1 // 2 + 1)
    #
    #     # Complete psig in yz plane
    #     for ig2, ig3 in self.yzlowerplane:
    #         psig_zyx[ig3, ig2, 0] = psig_zyx[-ig3, -ig2, 0].conjugate()
    #
    #     # rFFT in x direction (x has been switched to last axes), FFT in y, z direction
    #     psir_zyx = irfftn(psig_zyx, s=(n3, n2, n1))
    #
    #     # Switch back x, z axes
    #     psir = psir_zyx.swapaxes(0, 2)
    #
    #     return psir

