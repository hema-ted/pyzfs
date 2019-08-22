from __future__ import absolute_import, division, print_function
import numpy as np
import os
import h5py
from lxml import etree
from mpi4py import MPI

from .baseloader import WavefunctionLoader
from ..cell import Cell
from ..ft import FourierTransform, fftshift, ifftshift, irfftn, ifftn
from .wavefunction import Wavefunction
from ..counter import Counter

from ...common.external import empty_ase_cell


class QEHDF5WavefunctionLoader(WavefunctionLoader):

    def __init__(self, fftgrid="density", comm=MPI.COMM_WORLD):
        self.fftgrid = fftgrid
        self.dft = None
        self.wft = None
        self.comm = comm
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
        super(QEHDF5WavefunctionLoader, self).load(iorbs)

        # define varibles for MPI communications
        comm = self.comm
        rank = comm.Get_rank()
        size = comm.Get_size()
        onroot = rank == 0

        iorbs_of_rank = None
        if onroot:
            iorbs_of_rank = {0: iorbs}
            for r in range(1, size):
                iorbs_of_rank[r] = comm.recv(source=r)
        else:
            comm.send(iorbs, dest=0)

        comm.barrier()

        # processor 0 parse wavefunctions
        gvecs = ngvecs = psig_arrs_all = None
        if onroot:
            iorbs_all = set.union(*iorbs_of_rank.values())
            psig_arrs_all = {}

            # read wavefunctions
            c = Counter(len(iorbs_all), percent=0.1,
                        message="(process 0) {n} orbitals ({percent}%) loaded in {dt}...")
            for ispin in range(2):
                wfcfile = "wfcup1.hdf5" if ispin == 0 else "wfcdw1.hdf5"
                wfch5 = h5py.File(os.path.join(
                    self.root, "{}.save".format(self.prefix), wfcfile))

                gvecs = np.array(wfch5["MillerIndices"], dtype=int)
                ngvecs = gvecs.shape[0]
                assert gvecs.shape == (ngvecs, 3)

                evc = np.array(wfch5["evc"])

                for ievc in range(evc.shape[0]):
                    band = ievc + 1
                    iorb = self.wfc.sb_iorb_map.get(("up" if ispin == 0 else "down", band))

                    if iorb in iorbs_all:
                        psig_arrs_all[iorb] = evc[ievc].view(complex).copy()
                        c.count()

        # broadcast G vectors
        ngvecs = comm.bcast(ngvecs, root=0)
        if onroot:
            self.wfc.gvecs = gvecs
        if not onroot:
            self.wfc.gvecs = np.zeros([ngvecs, 3], dtype=int)
        comm.Bcast(self.wfc.gvecs, root=0)

        # scatter wavefunctions
        if rank == 0:
            psig_arrs = {iorb: psig_arrs_all[iorb] for iorb in iorbs}
            for r in range(1, size):
                for iorb in iorbs_of_rank[r]:
                    comm.Send(psig_arrs_all[iorb], dest=r, tag=iorb)

        else:
            psig_arrs = {iorb: np.zeros(ngvecs, complex) for iorb in iorbs}
            for iorb in iorbs:
                comm.Recv(psig_arrs[iorb], source=0, tag=iorb)

        comm.barrier()

        for iorb in iorbs:
            self.wfc.set_psig_arr(iorb, psig_arrs[iorb])

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

