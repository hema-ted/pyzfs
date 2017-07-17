import os
from time import time
from glob import glob
import numpy as np
from mpi4py import MPI
from pprint import pprint

from ..common.parallel import ProcessorGrid, SymmetricDistributedMatrix
from ..common.cell import Cell
from ..common.ft import FourierTransform
from .ddi import compute_ddig
from .prefactor import prefactor
from .rhog import compute_rhog


class ZFSCalculation:
    """Zero field splitting D tensor calculation.

    Generally, calculation of D tensor involves pairwise iteration over many wavefuctions
    (KS orbitals). Physically, wavefunction is uniquely labeled by a 2-tuple of band index (int)
    and spin ("up" or "down"). Internally, each wavefunction is labeled by an integer index.
    Several maps are defined to describe related transformations.

    Attributes:
        nwfcs (int): total number of wavefunctions (KS orbitals) to be considered
        nuwfcs/ndwfcs (int): number of spin up/down wavefunctions
        idxmap (dict): (band index, spin) -> wfc index map
        bandspinmap (list): wfc index -> (band index, spin) map
        fnamemap (list): wfc index -> wfc filename map
        wfcobjmap (dict): wfc index -> wfc object (3D array) map
        rhogmap (dict): wfc index -> rhog object (3D array) map
        normalizer (function): normalize a wavefunction

        cell (Cell): defines cell size, R and G vectors
        ft (FourierTransform): defines grid size for fourier transform

        ddig (ndarray): dipole-dipole interaction tensor in G space. Shape = (6, n1, n2, n3),
            where first index labels cartisian directions (xx, xy, xz, yy, yz, zz), last 3
            indices iterate over G space

        Iglobal (ndarray): global I array of shape (nwfcs, nwfcs, 6)
            first two indices iterate over wavefunctions, last index labels catesian directions
            in xx, xy, xz, yy, yz, xz manner
        I (ndarray): local I matrix, first two dimensions are distributed among processors
        D (ndarray): 3 by 3 matrix, total D tensor
        ev, evc (ndarray): eigenvalues and eigenvectors of D tensor
        Dvalue, Evalue (float): scalar D and E parameters for triplet

    """

    def __init__(self, path, wfcfmt, memory="low", comm=MPI.COMM_WORLD):
        """Initialize ZFS calculation.

        Args:
            path (str): working directory for this calculation. Python will first change
                the working dir to path before reading wavefunctions.
            wfcfmt (str): format of input wavefunction. Supported values:
                "vasp": VASP WAVECAR and POSCAR file.
                "cube-wfc": cube files of (real) wavefunctions (Kohn-Sham orbitals).
                "cube-density": cube files of (signed) wavefunction squared, mainly used to
                    support pp.x output with plot_num = 7 and lsign = .TRUE.
                when cube-wfc or cube-density is used, it is assumed that all files in
                the working directory with "up"/"down" in name are wavefunctions in
                spin up/down channel.
            comm (MPI.comm): MPI communicator on which ZFS calculation will be distributed.
            memory (str): memory mode. Supported values:
                "high": high memory usage, better performance
                "low": low memory usage, some intermediate quantities will not be stored and
                    will be computed every time when needed
        """

        # Initialize control parameters
        self.path, self.wfcfmt, self.memory = path, wfcfmt, memory
        os.chdir(self.path)
        assert self.wfcfmt in ["cube-wfc", "cube-density", "vasp"]
        assert self.memory in ["high", "low"]

        # Define a 2D processor grid to parallelize summation over pairs of orbitals.
        self.pgrid = ProcessorGrid(comm, square=True)
        if self.pgrid.onroot:
            print "\n\nZero Field Splitting Calculation Created...\n\n"
        self.pgrid.print_info()

        # Parse wavefunctions, define cell and ft
        self.nwfcs = self.nuwfcs = self.ndwfcs = None
        self.idxmap = self.bandspinmap = self.fnamemap = None
        self.normalizer = None
        self.cell = self.ft = None
        self.parse_wfcs()

        self.wfcobjmap = {}
        self.rhogmap = {}

        # Declare ddig, I arrays and D arrays
        self.ddig = None

        self.I = SymmetricDistributedMatrix(self.pgrid, (self.nwfcs, self.nwfcs, 6), np.float_)
        self.I.print_info("I")
        self.Iglobal = None

        self.D = np.zeros((3, 3))
        self.ev = np.zeros(3)
        self.evc = np.zeros((3, 3))
        self.Dvalue = 0
        self.Evalue = 0

    def parse_wfcs(self):
        if self.wfcfmt in ["cube-wfc", "cube-density"]:
            self.parse_cube()
        elif self.wfcfmt == "vasp":
            self.parse_vasp()
        if self.pgrid.onroot:
            print "\n  Reading input wavefunctions...\n"
            print "     nuwfcs = {}, ndwfcs = {}, nwfcs = {}".format(
                self.nuwfcs, self.ndwfcs, self.nwfcs
            )
            for iwfc in range(self.nwfcs):
                print "       band = {}     spin = {}     file = {}".format(
                    self.bandspinmap[iwfc][0], self.bandspinmap[iwfc][1], self.fnamemap[iwfc])
            print "\n  System Overview:"
            print "    Cell: "
            pprint(self.cell.__dict__, indent=6)
            print "    FFT Grid: "
            pprint(self.ft.__dict__, indent=6)

    def parse_cube(self):
        """Parse all cube files in current path that follows filename convention defined above."""
        from sunyata.parsers.text import parse_one_value
        from ase.io.cube import read_cube_data

        uwfcs = sorted(glob("*up*"))
        dwfcs = sorted(glob("*down*"))
        self.nuwfcs = len(uwfcs)
        self.ndwfcs = len(dwfcs)
        self.nwfcs = self.nuwfcs + self.ndwfcs
        self.fnamemap = uwfcs + dwfcs
        self.bandspinmap = map(
            lambda fname: (parse_one_value(int, fname), "up" if "up" in fname else "down"),
            self.fnamemap
        )
        self.idxmap = {
            self.bandspinmap[iwfc]: iwfc for iwfc in range(self.nwfcs)
        }

        # Use ASE to parse the cell and grid from the first cube file
        fname0 = self.fnamemap[0]
        psir, ase_cell = read_cube_data(fname0)
        self.cell = Cell(ase_cell)
        self.ft = FourierTransform(*psir.shape)
        if self.pgrid.onroot:
            print "\n    Cell and FFT grid are parsed from {}, it is assumed that all " \
                  "wavefunctions are defined on the same cell and grid\n".format(fname0)

        # Compute the integrated charge density from the first cube file,
        # generate a normalizer function that will be called each time when
        # a wavefunction (density) is read.
        # A nested lambda function trick is used to change the deferred evaluation
        # behavior of Python and store the norm inside the normalizer
        if self.wfcfmt == "cube-wfc":
            norm = np.sum(np.abs(psir) ** 2) * self.cell.omega / self.ft.N
            self.normalizer = (lambda c: lambda f: c * f)(1 / np.sqrt(norm))
        elif self.wfcfmt == "cube-density":
            norm = np.sum(np.abs(psir)) * self.cell.omega / self.ft.N
            self.normalizer = (lambda c: lambda f: c * np.sqrt(np.abs(f)))(1 / np.sqrt(norm))
        else:
            raise ValueError
        if self.pgrid.onroot:
            print "    Integrated charge density for {} = {}\n" \
                  "    It is assumed that all wavefunctions follows the " \
                  "same normalization convention".format(
                    fname0, norm
            )

    def parse_vasp(self):
        """Parse VASP WAVECAR and POSCAR files

        Local wfc index is assigned in the manner:
        0, 1, 2, ..., self.nuwfcs-1, ..., self.nwfcs-1  <==>
        (1, "up"), (2, "up"), ... (self.nuwfcs, "up"), (1, "down"), ... (self.ndwfcs, "down")

        NOTE: it is assumed that the occupations of KS orbitals is monotonically decreasing,
        and therefore all occupied states index are contineous, thus the current implementation
        does not work with excited state calculations!!!
        NOTE: Gamma-only caculation is assumed!!
        """
        from ase.io import read
        from sunyata.parsers.vasp import vaspwfc

        ase_cell = read("POSCAR")
        self.cell = Cell(ase_cell)
        self._wavecar = vaspwfc()
        self.ft = FourierTransform(*self._wavecar._ngrid)

        nspin, nkpts, nbands = self._wavecar._occs.shape
        assert nspin == 2 and nkpts == 1
        # Get band indices (starting from 1) with significant occupations
        iuwfcs = np.where(self._wavecar._occs[0, 0] > 0.8)[0] + 1
        idwfcs = np.where(self._wavecar._occs[1, 0] > 0.8)[0] + 1

        self.nuwfcs = len(iuwfcs)
        self.ndwfcs = len(idwfcs)
        self.nwfcs = self.nuwfcs + self.ndwfcs
        self.fnamemap = ["WAVECAR"] * self.nwfcs
        self.bandspinmap = list(
            (iuwfcs[iwfc], "up") if iwfc < self.nuwfcs else (idwfcs[iwfc - self.nuwfcs], "down")
            for iwfc in range(self.nwfcs)
        )
        self.idxmap = {
            self.bandspinmap[iwfc]: iwfc for iwfc in range(self.nwfcs)
        }

        # Here it is assumed that VASP pseudo wavefunctions are not normalized
        # with the same convention, so each wavefunction need to be normalized separately
        self.normalizer = lambda f: f / np.sqrt(np.sum(np.abs(f)**2)*self.cell.omega/self.ft.N)


    def load_wfcs(self):
        """Load KS orbitals required for evaluating the local block of I.

        If self.memory == "high", compute and store charge densities in G space.
        """
        if self.pgrid.onroot:
            print "\n  Loading wavefunctions to memory...\n"

        iwfcs_needed = set(
            list(range(self.I.mstart, self.I.mend))
            + list(range(self.I.nstart, self.I.nend))
        )

        if self.wfcfmt in ["cube-wfc", "cube-density"]:
            self.load_cube(iwfcs_needed)
        elif self.wfcfmt == "vasp":
            self.load_vasp(iwfcs_needed)
        else:
            raise ValueError

        if self.memory == "--high":
            for iwfc, psir in self.wfcobjmap.items():
                self.rhogmap[iwfc] = self.ft.forward(psir * np.conj(psir))


    def load_cube(self, iwfcs_needed):
        """Load KS orbitals from cube files."""
        from ase.io.cube import read_cube_data

        counter = 0
        for iwfc in range(self.nwfcs):
            fname = self.fnamemap[iwfc]
            wfcdata = read_cube_data(fname)[0]
            if iwfc in iwfcs_needed:
                psir = self.normalizer(wfcdata)
                self.wfcobjmap[iwfc] = psir

            counter += 1
            if counter >= self.nwfcs // 10:
                if self.pgrid.onroot:
                    print "......"
                counter = 0

    def load_vasp(self, iwfcs_needed):
        """Load KS orbitals from VASP WAVECAR file."""

        counter = 0
        for iwfc in range(self.nwfcs):
            band, spin = self.bandspinmap[iwfc]
            wfcdata = self._wavecar.wfc_r(
                ispin=1 if spin == "up" else 2, iband=band, gamma=True
            )
            if iwfc in iwfcs_needed:
                psir = self.normalizer(wfcdata)
                self.wfcobjmap[iwfc] = psir

            counter += 1
            if counter >= self.nwfcs // 10:
                if self.pgrid.onroot:
                    print "......"
                counter = 0

    def solve(self):
        """Compute and gather local block of I in each processor.

        TODO: all processor do summation of I first, then MPI allreduce to get D
        """
        tssolve = time()

        # Load wavefunctions from files
        self.load_wfcs()

        # Compute dipole-dipole interaction tensor. Due to symmetry we only need the
        # upper triangular part of ddig
        if self.pgrid.onroot:
            print "\n  Computing dipole-dipole interaction tensor in G space...\n"
        ddig = compute_ddig(self.cell, self.ft)
        self.ddig = ddig[np.triu_indices(3)]

        # Compute contribution to D tensor from every pair of electrons
        if self.pgrid.onroot:
            print "\n  Iteration over pairs...\n"
        csloop = 0
        tsloop = time()
        npairs = len(self.I.get_triu_iterator())
        interval = npairs // 100 + 1
        for counter, (iloc, jloc) in enumerate(self.I.get_triu_iterator()):
            # Load two wavefunctions
            i, j = self.I.ltog(iloc, jloc)
            if i == j:
                csloop += 1
                continue  # skip diagonal terms
            if self.bandspinmap[i][1] == self.bandspinmap[j][1]:
                chi = 1
            else:
                chi = -1
            psi1r = self.wfcobjmap[i]
            psi2r = self.wfcobjmap[j]
            rho1g = self.rhogmap.get(i)
            rho2g = self.rhogmap.get(j)
            rhog = compute_rhog(psi1r, psi2r, self.ft, rho1g=rho1g, rho2g=rho2g)

            # Factor to be multiplied with I:
            #   chi comes from spin direction
            #   prefactor comes from physical constants and unit conversions
            #   omega**2 comes from convention of FT used here
            fac = chi * prefactor * self.cell.omega ** 2

            self.I[iloc, jloc, ...] = np.real(fac * np.tensordot(self.ddig, rhog, axes=3))
            # TODO: check if it is safe to only use real apart

            # Update progress in output
            if counter % interval == 0:
                if self.pgrid.onroot:
                    print "{:.0f}% finished ({} FFTs), time = {}s......".format(
                        float(counter) / npairs * 100,
                        9 * (counter - csloop),  # TODO: change to 6 after optimization
                        time() - tsloop
                    )
                csloop = counter
                tsloop = time()

        self.I.symmetrize()

        # All processor sync local matrix to get global matrix
        self.Iglobal = self.I.collect()

        # Sum over G vectors to get D tensor
        self.D[np.triu_indices(3)] = np.sum(self.Iglobal, axis=(0, 1))
        self.D = self.D + self.D.T - np.diag(self.D.diagonal())
        self.ev, self.evc = np.linalg.eig(self.D)

        # For triplet states, compute D and E parameters:
        # D = 3/2 Dz, E = 1/2(Dx - Dy)
        args = np.abs(self.ev).argsort()
        dy, dx, dz = np.abs(self.ev)[args]
        self.Dvalue = 1.5 * dz
        self.Evalue = 0.5 * (dx - dy)

        if self.pgrid.onroot:
            print "\n\nTotal D tensor (MHz): "
            pprint(self.D)
            print "D eigenvalues (MHz): "
            print self.ev
            print "D eigenvectors: "
            print self.evc[0]
            print self.evc[1]
            print self.evc[2]
            print "D = {:.2f} MHz, E = {:.2f} MHz".format(self.Dvalue, self.Evalue)

            print "\nMemory usage (on process 0):"
            for obj in ["wfcobjmap", "rhogmap", "ddig", "I", "Iglobal"]:
                if isinstance(self.__dict__[obj], dict):
                    nbytes = np.sum(value.nbytes for value in self.__dict__[obj].values())
                else:
                    nbytes = self.__dict__[obj].nbytes
                print "{:10} {:.2f} MB".format(obj, nbytes/1024.**2)
            print "Total memory usage (on process 0):"
            import resource
            print "{:.2f} MB".format(
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.
            )

            print "Time elapsed: {:.0f}s".format(time() - tssolve)

    def get_xml(self):
        """Generate an xml to store information of this calculation.

        Returns:
            A string containing xml.

        """
        from lxml import etree
        from .. import __code__, __version__
        root = etree.Element("root")
        etree.SubElement(root, "code").text = __code__
        etree.SubElement(root, "version").text = __version__
        etree.SubElement(root, "object").text = self.__class__.__name__
        etree.SubElement(root, "DTensor", unit="MHz").text = np.array2string(self.D)
        etree.SubElement(root, "D", unit="MHz").text = "{:.2f}".format(self.Dvalue)
        etree.SubElement(root, "E", unit="MHz").text = "{:.2f}".format(self.Evalue)

        tree = etree.ElementTree(root)
        return etree.tostring(tree, pretty_print=True)
