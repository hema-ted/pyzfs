from __future__ import absolute_import, division, print_function
from unittest import TestCase, TestLoader, TextTestRunner
import numpy as np
from numpy.random import randint, choice
from hspin.common.parallel import ProcessorGrid, DistributedMatrix, SymmetricDistributedMatrix
from mpi4py import MPI


def mpisync(func, comm=MPI.COMM_WORLD):
    """Decorate a function to ensure all processor get the same returns in MPI environment.

    After decoration, only processor 0 runs and then bcast result to all other processors.

    Args:
        func: undecorated function
        comm: MPI communicator

    Returns: decorated function

    """
    def mpifunc(*args, **kwargs):
        if comm.Get_rank() == 0:
            res = func(*args, **kwargs)
        else:
            res = None
        res = comm.bcast(res, root=0)
        return res
    return mpifunc


class DistributedMatrixTest(TestCase):
    """Unit tests for DistributedMatrix class"""

    def setUp(self):
        """Define a distributed matrix of random size"""
        self.pgrid = ProcessorGrid(MPI.COMM_WORLD)

        m = mpisync(randint)(1, 500)
        n = mpisync(randint)(1, 500)
        p = mpisync(randint)(1, 4)
        q = mpisync(randint)(1, 4)
        dtype = mpisync(choice)([np.int_, np.float_, np.complex])
        print("m = {}, n = {}, p = {}, q = {}, dtype = {}".format(
            m, n, p, q, dtype
        ))
        self.m, self.n, self.p, self.q, self.dtype = m, n, p, q, dtype
        self.mat = DistributedMatrix(self.pgrid, [m, n, p, q], dtype)

    def test_partition(self):
        """Test partition of global matrix to local blocks"""
        mat = self.mat
        self.assertSequenceEqual(
            [mat.m, mat.n, mat.shape[2], mat.shape[3], mat.dtype],
            [self.m, self.n, self.p, self.q, self.dtype]
        )
        if not mat.pgrid.is_active():
            self.assertSequenceEqual(
                [mat.mloc, mat.mstart, mat.mend, mat.nloc, mat.nstart, mat.nend],
                [0, 0, 0, 0, 0, 0]
            )
        else:
            pass

    def test_collect(self):
        """Test collection of local matrix to get global matrix"""
        pass

    def test_symmetrization(self):
        """Test symmetrization of local matrix"""
        pass


if __name__ == "__main__":
    import sys
    module = sys.modules[__name__]
    suite = TestLoader().loadTestsFromModule(module)
    TextTestRunner().run(suite)
