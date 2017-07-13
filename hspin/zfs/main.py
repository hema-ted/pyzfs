import os
from time import time
from glob import glob
import numpy as np
from mpi4py import MPI
from pprint import pprint

from ase.io.cube import read_cube_data

from ..common.parallel import ProcessorGrid, SymmetricDistributedMatrix
from ..common.cell import Cell
from ..common.ft import FourierTransform
from .ddi import compute_ddig
from .prefactor import prefactor
from .rhog import compute_rhog


class ZFSCalculation:
    def __init__(self, path, wfcfmt, comm=MPI.COMM_WORLD, memory="--low"):

        # Define a 2D processor grid to parallelize summation over pairs of orbitals.
        self.pgrid = ProcessorGrid(comm, square=True)
        self.push("Zero Field Splitting Calculation Created...", 0, "\n\n", "\n\n")
        self.pgrid.print_info()

        self.path, self.flag = path, wfcfmt
        os.chdir(self.path)

        # If memory is "low", store less intermediate quantities and recompute when needed
        self.memory = memory

        # Search all filenames containing "up" or "down", assuming they store
        # wavefunctions in spin up channel and spin down channel respectively
        uwfcs = sorted(glob("*up*"))
        dwfcs = sorted(glob("*down*"))
        self.nuwfcs = len(uwfcs)
        self.ndwfcs = len(dwfcs)
        self.wfcs = uwfcs + dwfcs
        self.nwfcs = len(self.wfcs)
        self.spinmap = ["up"] * self.nuwfcs + ["down"] * self.ndwfcs
        self.wfcmap = {}  # index -> actual wavefunction (3D array)
        self.rhogmap = {}  # index -> rho(G) (3D array)
        if self.pgrid.onroot:
            print "\n  Finding input wavefunctions...\n"
            print "     nuwfcs = {}, ndwfcs = {}, nwfcs = {}".format(
                self.nuwfcs, self.ndwfcs, self.nwfcs
            )
            for wfc, spin in zip(self.wfcs, self.spinmap):
                print "       {}     {}".format(wfc, spin)

        if self.flag == "--vasp":
            raise NotImplementedError

        elif self.flag == "--cube-wfc" or "--cube-density":
            # Using ASE to parse the cell and grid from the first cube file
            # It is assumed that all wavefunctions (density) are defined
            # on the same cell and grid
            psir, ase_cell = read_cube_data(self.wfcs[0])
            self.cell = Cell(ase_cell)
            self.ft = FourierTransform(*psir.shape)

            # Compute the integrated charge density from the first cube file,
            # generate a normalizer function that will be called each time when
            # a wavefunction (density) is read. It is assume that all wavefunctions
            # (density) follows the same normalization convention
            # A nested lambda function trick is used to change the deferred evaluation
            # behavior of Python and store the norm inside the normalizer
            if self.flag == "--cube-wfc":
                norm = np.sum(np.abs(psir) ** 2) * self.cell.omega / self.ft.N
                self.normalize = (lambda c: lambda f: c * f)(1 / np.sqrt(norm))
            elif self.flag == "--cube-density":
                norm = np.sum(np.abs(psir)) * self.cell.omega / self.ft.N
                self.normalize = (lambda c: lambda f: c * np.sqrt(np.abs(f)))(1 / np.sqrt(norm))
            else:
                raise ValueError
            if self.pgrid.onroot:
                print "Integrated charge density for first wavefunction ({}) = {}".format(
                    self.wfcs[0], norm
                )

        else:
            raise ValueError

        if self.pgrid.onroot:
            print "\n  System Overview:"
            print "    Cell: "
            pprint(self.cell.__dict__, indent=6)
            print "    FFT Grid: "
            pprint(self.ft.__dict__, indent=6)

        self.ddig = None  # dipole-dipole energy tensor in G space

        # Define ( nwfcs * nwfcs * 3 * 3 ) dimensional array I to store integral of
        # dipole-ddipole energy tensor against charge density. Firt two dimensions
        # are orbital indices (distributed over the 2D grid); last two dimensions are
        # Cartesian indices
        self.I = SymmetricDistributedMatrix(self.pgrid, (self.nwfcs, self.nwfcs, 6), np.float_)
        self.I.print_info("I")
        self.Iglobal = None  # Used to store global I

        # Define matrix D to store final D tensor as a summation of I over first two indices
        self.D = np.zeros((3, 3))
        self.ev = np.zeros(3)
        self.evc = np.zeros((3, 3))
        self.Dvalue = 0
        self.Evalue = 0


    def loadwfcs(self):
        """
        Load KS orbitals that is required for evaluating local block of I,
        compute charge density in G space
        """

        iwfcs_needed = set(
              list(range(self.I.mstart, self.I.mend))
            + list(range(self.I.nstart, self.I.nend))
        )

        counter = 0
        for iwfc in range(self.nwfcs):
            name = self.wfcs[iwfc]
            if self.flag == "--cube-wfc" or "--cube-density":
                wfcdata = read_cube_data(name)[0]
            else:
                raise NotImplementedError
            if iwfc in iwfcs_needed:
                psir = self.normalize(wfcdata)
                self.wfcmap[iwfc] = psir

            counter += 1
            if counter >= self.nwfcs // 10:
                if self.pgrid.onroot:
                    print "......"
                counter = 0

        if self.memory == "--high":
            for iwfc, psir in self.wfcmap.items():
                self.rhogmap[iwfc] = self.ft.forward(psir * np.conj(psir))

    def solve(self):
        """
        Compute local block of I in each processor, processor 0 gather and print
        TODO: all processor do summation of I first, then MPI_all reduce to get D
        """
        # Compute dipole-dipole interaction tensor. Due to symmetry, in later calculations
        # we only need to consider upper triangular part of ddig
        if self.pgrid.onroot:
            print "\n  Computing dipole-dipole interaction tensor in G space...\n"
        ddig = compute_ddig(self.cell, self.ft)
        self.ddig = ddig[np.triu_indices(3)]

        # Load wavefunctions from files
        if self.pgrid.onroot:
            print "\n  Loading wavefunction...\n"
        self.loadwfcs()


        # Compute contribution to D tensor from every pair of electrons
        if self.pgrid.onroot:
            print "\n  Iteration over pairs...\n"
        cstart = 0
        tstart = time()
        npairs = len(self.I.get_triu_iterator())
        interval = npairs // 100 + 1
        for counter, (iloc, jloc) in enumerate(self.I.get_triu_iterator()):
            # Load two wavefunctions
            i, j = self.I.ltog(iloc, jloc)
            if i == j:
                cstart += 1
                continue  # skip diagonal terms
            if self.spinmap[i] == self.spinmap[j]:
                chi = 1
            else:
                chi = -1
            psi1r = self.wfcmap[i]
            psi2r = self.wfcmap[j]
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
                    print "{:.0f}% finished ({} FFTs), time = {}......".format(
                        float(counter) / npairs * 100,
                        9 * (counter - cstart),  # TODO: change to 6 after optimization
                        time() - tstart
                    )
                cstart = counter
                tstart = time()

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
            print "Total D tensor (MHz): "
            pprint(self.D)
            print "D eigenvalues (MHz): "
            print self.ev
            print "D eigenvectors: "
            print self.evc[0]
            print self.evc[1]
            print self.evc[2]
            print "D = {:.2f} MHz, E = {:.2f} MHz".format(self.Dvalue, self.Evalue)

            print "Memory usage:"
            for obj in ["wfcmap", "rhogmap", "ddig", "I", "Iglobal"]:
                if isinstance(self.__dict__[obj], dict):
                    nbytes = np.sum(value.nbytes for value in self.__dict__[obj].values())
                else:
                    nbytes = self.__dict__[obj].nbytes
                print "{:10} {:.2f} MB".format(obj, nbytes/1024.**2)
            print "Total memory usage:"
            import resource
            print "{:.2f} MB".format(
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.
            )

    def get_wfc_index(self, band, spin):
        """
        Find the index of a given orbital in self.wfcs, it is assumed that
        the band index appear as the first integer in filename, and spin appear
        in filename explicitly as "up" or "down"
        :param band: int
        :param spin: "up" or "down"
        :return: int
        """
        from sunyata.parsers.text import parse_one_value
        for i, wfc in enumerate(self.wfcs):
            if parse_one_value(int, wfc) == band and spin in wfc:
                return i

    def get_wfc_info(self, i):
        """
        Return the band index and spin of a given orbital in self.wfcs, it is assumed that
        the band index appear as the first integer in filename, and spin appear
        in filename explicitly as "up" or "down"
        :param i: index of orbital in self.wfcs
        :return: band index, spin ("up" or "down")
        """
        from sunyata.parsers.text import parse_one_value
        return parse_one_value(int, self.wfcs[i]), self.spinmap[i]

    def push(self, message, hanging, prefix="\n", suffix="\n"):
        if self.pgrid.onroot:
            print "{}{}{}{}".format(
                prefix, " " * hanging, message.replace("\n", " " * hanging), suffix
            )

    def get_xml(self):
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
