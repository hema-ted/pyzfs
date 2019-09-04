from __future__ import absolute_import, division, print_function
import numpy as np
import os
from glob import glob
from lxml import etree
import base64

from ase import Atoms, Atom

from ..external import parse_many_values
from ..units import bohr_to_angstrom
from .baseloader import WavefunctionLoader
from ..cell import Cell
from ..ft import FourierTransform
from .wavefunction import Wavefunction
from ..counter import Counter
from ..parallel import mpiroot


class QboxWavefunctionLoader(WavefunctionLoader):
    def __init__(self, filename=None):
        self.xmlfile = filename
        super(QboxWavefunctionLoader, self).__init__()

    def scan(self):
        super(QboxWavefunctionLoader, self).scan()

        if self.xmlfile is None:
            xmllist = sorted(glob("*xml"), key=lambda f: os.path.getsize(f))
            if len(xmllist) == 0:
                raise IOError("No xml file found in current directory: {}".format(os.getcwd()))
            elif len(xmllist) == 1:
                self.xmlfile = xmllist[0]
            else:
                self.xmlfile = xmllist[-1]
                if mpiroot:
                    print("More than one xml files found: {}".format(xmllist))
                    print("Assume wavefunction is in the largest xml file: {} ({} MB)".format(
                        self.xmlfile, os.path.getsize(self.xmlfile) / 1024 ** 2
                    ))
        if mpiroot:
            print("Reading wavefunction from file {}".format(self.xmlfile))

        iterxml = etree.iterparse(self.xmlfile, huge_tree=True, events=("start", "end"))

        for event, leaf in iterxml:
            if event == "end" and leaf.tag == "unit_cell":
                R1 = np.fromstring(leaf.attrib["a"], sep=" ", dtype=np.float_) * bohr_to_angstrom
                R2 = np.fromstring(leaf.attrib["b"], sep=" ", dtype=np.float_) * bohr_to_angstrom
                R3 = np.fromstring(leaf.attrib["c"], sep=" ", dtype=np.float_) * bohr_to_angstrom
                lattice = np.array([R1, R2, R3])
                ase_cell = Atoms(cell=lattice, pbc=True)

            if event == "end" and leaf.tag == "atom":
                species = leaf.attrib["species"]
                position = np.array(parse_many_values(3, float, leaf.find("position").text))
                ase_cell.append(Atom(symbol=species, position=position * bohr_to_angstrom))

            if event == "start" and leaf.tag == "wavefunction":
                nspin = int(leaf.attrib["nspin"])
                assert nspin == 2
                iorb_sb_map = list()
                sb_psir_map = dict()

            if event == "end" and leaf.tag == "grid":
                n1, n2, n3 = int(leaf.attrib["nx"]), int(leaf.attrib["ny"]), int(leaf.attrib["nz"])

            if event == "start" and leaf.tag == "slater_determinant":
                spin = leaf.attrib["spin"]

            if event == "end" and leaf.tag == "density_matrix":
                if spin == "up":
                    uoccs = np.fromstring(leaf.text, sep=" ", dtype=np.float_)
                    iuorbs = np.where(uoccs > 0.8)[0] + 1
                    nuorbs = len(iuorbs)
                    iorb_sb_map.extend(
                        ("up", iuorbs[iorb]) for iorb in range(nuorbs)
                    )
                elif spin == "down":
                    doccs = np.fromstring(leaf.text, sep=" ", dtype=np.float_)
                    idorbs = np.where(doccs > 0.8)[0] + 1
                    ndorbs = len(idorbs)
                    iorb_sb_map.extend(
                        ("down", idorbs[iorb]) for iorb in range(ndorbs)
                    )
                else:
                    raise ValueError

            if event == "end" and leaf.tag == "grid_function":
                leaf.clear()

            if event == "start" and leaf.tag == "wavefunction_velocity":
                break

        norbs = nuorbs + ndorbs
        iorb_fname_map = [self.xmlfile] * norbs

        cell = Cell(ase_cell)
        ft = FourierTransform(n1, n2, n3)

        self.wfc = Wavefunction(cell=cell, ft=ft, nuorbs=nuorbs, ndorbs=ndorbs,
                                iorb_sb_map=iorb_sb_map, iorb_fname_map=iorb_fname_map)

        for (band, spin), psir in sb_psir_map.items():
            iorb = self.wfc.sb_iorb_map[band, spin]
            psir = sb_psir_map[band, spin]
            self.wfc.set_psir(iorb, psir)

    def load(self, iorbs, sdm=None):

        iterxml = etree.iterparse(self.xmlfile, huge_tree=True, events=("start", "end"))
        c = Counter(len(iorbs), percent=0.1,
                    message="(process 0) {n} orbitals ({percent}%) loaded in {dt}...")

        for event, leaf in iterxml:
            if event == "start" and leaf.tag == "slater_determinant":
                spin = leaf.attrib["spin"]
                band = 1

            if event == "end" and leaf.tag == "grid_function":
                iorb = self.wfc.sb_iorb_map.get((spin, band))
                if iorb in iorbs:
                    # switch from for z, for y, for x to for x, for y, for z
                    psir = np.frombuffer(
                        base64.decodestring(leaf.text), dtype=np.float64
                    ).reshape(self.wfc.ft.n3, self.wfc.ft.n2, self.wfc.ft.n1).T
                    self.wfc.set_psir(iorb, psir)
                    c.count()
                band += 1
                leaf.clear()

            if event == "start" and leaf.tag == "wavefunction_velocity":
                break
