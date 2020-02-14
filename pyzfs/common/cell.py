from __future__ import absolute_import, division, print_function
import numpy as np
from ase import Atoms
from .units import *


class Cell:
    """A wrapper class for ASE Atoms that defines R and G vectors."""
    def __init__(self, ase_cell):
        """
        Args:
            ase_cell (ASE Atoms): input ASE cell
        """
        assert isinstance(ase_cell, Atoms)
        self.ase_cell = ase_cell.copy()
        self.omega = ase_cell.get_volume() * angstrom_to_bohr**3
        self.R1, self.R2, self.R3 = ase_cell.get_cell() * angstrom_to_bohr
        self.G1, self.G2, self.G3 = 2 * np.pi * ase_cell.get_reciprocal_cell() / angstrom_to_bohr
