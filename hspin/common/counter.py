from __future__ import absolute_import, division, print_function
from mpi4py import MPI
from time import time


class Counter(object):
    def __init__(self, ntot, percent=0.1, message="{percent}% finished ({dt}).........", comm=MPI.COMM_WORLD):
        self.ntot = ntot
        self.dn = ntot * percent
        self.message = message
        self.comm = comm
        self.onroot = self.comm.Get_rank() == 0

        self.n = 0
        self.i = 0

        self.tstart = time()
        self.tlast = self.tstart

    def count(self):
        self.n += 1
        self.i += 1

        if self.i >= self.dn:
            self.i = 0
            t = time()
            if self.onroot:
                print(self.message.format(
                    n=self.n, ntot=self.ntot,
                    percent=(100 * self.n) // self.ntot,
                    dt="{:2f}s".format(t - self.tlast)
                ))
            self.tlast = t
