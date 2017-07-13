import glob
import os

import ase.io
import numpy as np
from ase.io.cube import read_cube_data
from mpi4py import MPI
from sunyata.hspin.common.cell import Cell

from ..common.ft import FourierTransform

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
onroot = rank == 0

def read_input(argv):

    os.chdir(argv[1])

    if argv[2] == "--vasp":

        cell = Cell(ase.io.read("POSCAR"))

        if onroot:
            fups = sorted(glob.glob("*up"))
            fdowns = sorted(glob.glob("*down"))

            wfcs = []

            N = len(np.loadtxt(open(fups[0]), dtype=np.complex))
            n = int(np.cbrt(N))
            ft = FourierTransform(n, n, n)

            for f in fups:
                psir = np.loadtxt(open(f), dtype=np.complex).reshape(n, n, n)
                psir /= np.sqrt(np.sum(np.abs(psir) ** 2) * cell.omega / ft.N)
                wfcs.append((f, "up", psir))

            for f in fdowns:
                psir = np.loadtxt(open(f), dtype=np.complex).reshape(n, n, n)
                psir /= np.sqrt(np.sum(np.abs(psir) ** 2) * cell.omega / ft.N)
                wfcs.append((f, "down", psir))
        else:
            ft = None
            wfcs = None

        ft = comm.bcast(ft, root=0)
        wfcs = comm.bcast(wfcs, root=0)

    elif argv[2] == "--cube-wfc":
        # ASE read automatically bcast data

        cubes = glob.glob("*.cube")

        wfcs = []

        counter = 0
        for cube in cubes:
            if "up" in cube:
                spin = "up"
            elif "down" in cube:
                spin = "down"
            else:
                print "Unrecognized cube file: ", cube
                continue

            psir, ase_cell = read_cube_data(cube)
            wfcs.append((cube, spin, psir))

            counter += 1
            if counter > len(cubes) / 5:
                print "    ......"

        cell = Cell(ase_cell)
        ft = FourierTransform(*psir.shape)

        for _, _, psir in wfcs:
            psir /= np.sqrt(cell.omega)

    if onroot:
        print "\n  System Overview:"
        print "    cell: "
        print cell.__dict__
        print "    grid: "
        print ft.__dict__

        print "\n  Wavefunction input:"
        for name, spin, _ in wfcs:
            print "    {}({})".format(name, spin)

    return cell, ft, wfcs
