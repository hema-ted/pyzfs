from time import sleep
import numpy as np
from mpi4py import MPI

class ProcessorGrid(object):
    """
    2D Grid of processors. Used to wrap MPI communications.
    """
    def __init__(self, comm, square=False):
        assert isinstance(comm, MPI.Intracomm)
        assert isinstance(square, bool)
        self.comm = comm.Clone()
        commsize = self.comm.Get_size()
        self.square = square

        # Compute maximum possible # of rows and columns
        self.rank = self.comm.Get_rank()
        self.onroot = self.rank == 0
        self.nrow = int(np.sqrt(commsize))
        if self.square:
            self.ncol = self.nrow
        else:
            self.ncol = commsize // self.nrow
        self.size = self.nrow * self.ncol
        assert self.size <= commsize

        # Map processors to a 2D grid
        #   self.pmap: rank -> (irow, icol)
        #   self.invpmap: (irow, icol) -> rank
        self.invpmap = np.arange(self.size).reshape(self.nrow, self.ncol)
        self.pmap = np.array(
            list(
                [
                    np.where(self.invpmap == rank)[0][0],  # row index of given rank
                    np.where(self.invpmap == rank)[1][0],  # col index of given rank
                ]
                for rank in range(self.size)
            )
        )
        self.irow, self.icol = self.pmap[self.rank] if self.rank < self.size else (-1, -1)
        assert self.irow < self.nrow and self.icol < self.ncol
        self.is_active = self.irow >= 0 and self.icol >= 0

        # Create row communicators and column communicators
        # All processors within the same row (column) will be in the same row (column) comm
        self.rowcomm = self.comm.Split(color=self.irow, key=self.icol)
        self.colcomm = self.comm.Split(color=self.icol, key=self.irow)

        # If the grid is a square grid, create communicators connecting symmetric elements
        if self.square:
            self.diagonal = self.irow == self.icol
            self.upper = self.irow < self.icol
            self.lower = self.irow > self.icol
            i, j = sorted([self.irow, self.icol], reverse=True)
            color = i * (i + 1) / 2 + j
            self.symmcomm = self.comm.Split(color=color, key=self.irow)  # upper processor has rank 0


    def sleep(self, nsec=None):
        sleep(nsec if nsec else self.rank)

    def report(self, tag="", sleep=True):
        if sleep:
            self.sleep()
        print "rank {} reporting {}".format(self.rank, tag)

    def print_info(self):
        columns = "rank  onroot  nrow  ncol  size  irow  icol  " \
                  "is_active  rowcomm_rank  colcomm_rank"
        message = "  {}    {}      {}    {}    {}    {}    {}    {}    {}    {}".format(
            self.rank, self.onroot, self.nrow, self.ncol,
            self.size, self.irow, self.icol, self.is_active,
            self.rowcomm.Get_rank(), self.colcomm.Get_rank()
        )
        allmessage = self.comm.gather(message, root=0)
        if self.onroot:
            print "ProcessGrid{} info:".format(" (square)" if self.square else "")
            print "rank -> (irow, icol) mapping:"
            print self.pmap
            print "(irow, icol) -> rank mapping:"
            print self.invpmap
            print columns
            for m in allmessage:
                print m


class DistributedMatrix(object):

    def __init__(self, pgrid, shape, dtype):
        """
        :param pgrid: ProcessorGrid instance, defines how the matrix will be distributed
        :param shape: Tuple of integers, defines shape of matrix.
                      First two dimensions are distributed on processor grid.

        Convention: a variable indexing local block of a distributed matrix should have
                    trailing "loc" in its name, otherwise it is considered a global index

        """
        assert isinstance(pgrid, ProcessorGrid)
        self.pgrid = pgrid
        self.is_active = self.pgrid.is_active
        self.comm = self.pgrid.comm
        self.rowcomm, self.colcomm = self.pgrid.rowcomm, self.pgrid.colcomm
        self.onroot = self.pgrid.onroot
        self.irow, self.icol = self.pgrid.irow, self.pgrid.icol
        self.nrow, self.ncol = self.pgrid.nrow, self.pgrid.ncol

        # Compute index range for the first two dimensions on each processor
        assert isinstance(shape, tuple) and len(shape) >= 2
        assert all(isinstance(shape[i], int) for i in range(len(shape)))
        self.m, self.n = shape[0], shape[1]

        # First dimension (row)
        if self.is_active:
            self.mloc = self.m // self.nrow
            if self.irow < self.m % self.nrow:
                self.mloc += 1
        else:
            self.mloc = 0
        msum = self.colcomm.allreduce(self.mloc, op=MPI.SUM)
        if self.is_active:
            assert msum == self.m

        self.mstart = self.colcomm.exscan(self.mloc, op=MPI.SUM)
        if self.irow <= 0:
            self.mstart = 0  # Exclusive scan result for 0th processor is undefined
        self.mend = self.colcomm.scan(self.mloc, op=MPI.SUM)
        assert self.mstart + self.mloc == self.mend

        # Second dimension (column)
        if self.is_active:
            self.nloc = self.n // self.ncol
            if self.icol < self.n % self.ncol:
                self.nloc += 1
        else:
            self.nloc = 0
        nsum = self.rowcomm.allreduce(self.nloc, op=MPI.SUM)
        if self.is_active:
            assert nsum == self.n

        self.nstart = self.rowcomm.exscan(self.nloc, op=MPI.SUM)
        if self.icol <= 0:
            self.nstart = 0  # Exclusive scan result for 0th processor is undefined
        self.nend = self.rowcomm.scan(self.nloc, op=MPI.SUM)
        assert self.nstart + self.nloc == self.nend

        # Build index map: irow, icol -> mstart, mloc, mend, nstart, nloc, nend
        indexmap = np.zeros([self.nrow, self.ncol, 6], dtype=np.int_)
        indexmap[self.irow, self.icol] = [
            self.mstart, self.mloc, self.mend, self.nstart, self.nloc, self.nend
        ]
        self.indexmap = self.comm.allreduce(indexmap, op=MPI.SUM)

        # Build local matrix (self.val)
        self.shape = shape
        self.ndim = len(shape)
        self.locshape = (self.mloc, self.nloc) + shape[2:]
        self.dtype = dtype
        self.val = np.zeros(self.locshape, dtype=self.dtype)

    def print_info(self, name=""):
        columns = "irow  icol  nrow  ncol  m  mloc  mstart  mend  n  nloc  nstart  nend  val.shape"
        message = "  {}    {}    {}    {}    {}    {}    {}    {}    {}    {}    {}    {}     {}".format(
            self.pgrid.irow, self.pgrid.icol, self.pgrid.nrow, self.pgrid.ncol,
            self.m, self.mloc, self.mstart, self.mend,
            self.n, self.nloc, self.nstart, self.nend,
            self.val.shape
        )
        allmessage = self.comm.gather(message, root=0)
        if self.onroot:
            print "DistributedMatrix info: {}".format(name)
            print columns
            for m in allmessage:
                print m
            print "Index map:"
            print self.indexmap

    # Overload [] operator
    def __getitem__(self, item):
        return self.val[item]

    def __setitem__(self, key, value):
        self.val[key] = value

    def gtol(self, i, j=None):
        """
        global -> local index map
        """
        assert self.mstart <= i < self.mend
        if j is not None:
            assert self.nstart <= j < self.nend
            return i - self.mstart, j - self.nstart
        else:
            return i - self.mstart

    def ltog(self, iloc, jloc=None):
        """
        local -> global index map
        """
        assert 0 <= iloc < self.mloc
        if jloc is not None:
            assert 0 <= jloc < self.nloc
            return iloc + self.mstart, jloc + self.nstart
        else:
            return iloc + self.mstart

    def collect(self):
        """
        Gather the distributed matrix to all processor
        :return: global matrix

        TODO: rewrite this function to reduce memory cost
        """
        lmatrix = np.zeros(self.shape, dtype=self.dtype)
        lmatrix[self.mstart:self.mend, self.nstart:self.nend] = self.val[0:self.mloc, 0:self.nloc]

        gmatrix = np.zeros(self.shape, dtype=self.dtype)
        self.comm.Allreduce(lmatrix, gmatrix, op=MPI.SUM)
        return gmatrix


class SymmetricDistributedMatrix(DistributedMatrix):

    def __init__(self, pgrid, shape, dtype):
        super(SymmetricDistributedMatrix, self).__init__(pgrid, shape, dtype)
        assert pgrid.square
        assert shape[0] == shape[1]
        self.symmcomm = self.pgrid.symmcomm

        # Resize local matrix on boundary to mlocx (maximum mloc) by nlocx (maximum nloc),
        # which facilitates symmetrization
        self.mlocx = self.colcomm.allreduce(self.mloc, op=MPI.MAX)
        self.nlocx = self.rowcomm.allreduce(self.nloc, op=MPI.MAX)
        assert self.mlocx == self.nlocx
        if self.is_active and (self.mloc < self.mlocx or self.nloc < self.nlocx):
            # self.pgrid.report("expanded array from {}x{} to {}x{}".format(
            #     self.mloc, self.nloc, self.mlocx, self.nlocx
            # ))
            self.val.resize((self.mlocx, self.nlocx) + self.shape[2:])
        self.nbytes = self.val.nbytes

    def get_triu_iterator(self):
        """
        Get a list of 2D indices to iterate over upper triangular part of the local matrix
        :return: list of 2-tuples of integer
        """
        if self.is_active:
            if self.mloc < self.mlocx or self.nloc < self.nlocx:
                # boundary case
                return [
                    (i,j) for (i,j) in zip(*np.triu_indices(self.mlocx))
                    if i < self.mloc and j < self.nloc
                    ]
            else:
                return zip(*np.triu_indices(self.mloc))
        else:
            return []

    def symmetrize(self):
        """
        Compute lower triangular part of the matrix from upper triangular part
        """
        # lower trangular indices
        tril = np.tril_indices(self.mlocx)

        # order of axes to be used when transpose first two dimension of self.val
        transpose_axes = (1, 0) + tuple(range(2, self.ndim))

        if self.pgrid.diagonal:
            # Diagonal processor symmetrize in-place
            for iloc, jloc in zip(*tril):
               self.val[iloc, jloc, ...] = self.val[jloc, iloc, ...]

        else:
            # Off-diagonal processors communicate with its symmetric counterparts
            # to symmetrize

            if self.pgrid.upper:
                send = self.val.copy()
            else:
                send = self.val.transpose(transpose_axes).copy()
            recv = np.zeros(self.val.shape, dtype=self.dtype)

            self.symmcomm.Allreduce(send, recv, op=MPI.SUM)

            if self.pgrid.upper:
                self.val = recv
            else:
                self.val = recv.transpose(transpose_axes).copy()

            for i in range(self.mlocx):
                self.val[i, i, ...] /= 2.
