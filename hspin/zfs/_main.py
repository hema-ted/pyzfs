import os
import sys
from glob import glob
from itertools import combinations
from time import sleep

import numpy as np
from ase.io.cube import read_cube_data, read_cube
from mpi4py import MPI

from .ddi import compute_ddig
from .prefactor import prefactor
from .readinput import read_input
from .rhog import compute_rhog
from ..common.cell import Cell
from ..common.ft import FourierTransform
from ..common.parallel import ProcessorGrid, DistributedMatrix


comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
onroot = rank == 0

if onroot:
    print "\n  Zero Field Splitting Calculation\n"
    print "\n  Reading input wavefunctions...\n"

cell, ft, wfcs = read_input(sys.argv)
omega = cell.omega
N = ft.N

pairs = list(combinations(wfcs, 2))
npairs = len(pairs)

ipairstart = rank * (npairs // size)
ipairend = (rank + 1) * (npairs // size) if rank < size - 1 else npairs
locpairs = pairs[ipairstart:ipairend]
nlocpairs = ipairend - ipairstart

if onroot:
    print "\n  MPI loadbalance: "
    print "    npairs = ", npairs, "\n"
comm.barrier()
print "    rank {} has {} pairs ranging from {} to {}\n".format(
    rank, nlocpairs, ipairstart, ipairend
)
sys.stdout.flush()
comm.barrier()

if onroot:
    print "\n  Computing dipole-dipole interaction tensor in G space...\n"
ddig = compute_ddig(cell, ft)
# ddir = compute_ddir(cell, ft)
# ddig = np.zeros(ddir.shape)
# for ia, ib in np.ndindex(3, 3):
#     ddig[ia, ib] = ft.forward(ddir[ia, ib])

if onroot:
    print "\n  Summation over pairs...\n"
I = np.zeros([npairs, 3, 3], dtype=float)
Itot = np.zeros([npairs, 3, 3], dtype=float)

#print rank, " looping over: ", list(enumerate(locpairs))

counter = 0
for ipair, ((name1, spin1, psi1r), (name2, spin2, psi2r)) in enumerate(locpairs):

    chi = 1 if spin1 == spin2 else -1

    rhog = compute_rhog(psi1r, psi2r, ft)

    #  unit: bohr^-3
    I[rank * (npairs // size) + ipair, ...] = chi * prefactor * cell.omega**2 * np.real(np.tensordot(ddig, rhog, axes=((2,3,4), (0,1,2))))

    counter += 1
    if counter > nlocpairs / 20:
        print "      ........."
        counter = 0

sleep(rank)
print rank, " reporting: I = "
print I

comm.Allreduce(I, Itot, op=MPI.SUM)

# sleep(rank)
# print rank, " reporting: Itot = "
# print Itot

D =  np.sum(Itot, axis=0) * 2 # from double counting
ev, evc = np.linalg.eig(D)
if onroot:
    print "Total D tensor (MHz): ", D
    print "D eigenvalues (MHz): ", ev
    print "D eigenvectors (MHz): ", evc
