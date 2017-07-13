import numpy as np

from ..common.parallel import ProcessorGrid, DistributedMatrix, SymmetricDistributedMatrix

from mpi4py import MPI

pgrid = ProcessorGrid(MPI.COMM_WORLD, square=True)
pgrid.print_info()

dm = SymmetricDistributedMatrix(pgrid, (13, 13), np.complex_)
dm.print_info()

for iloc, jloc in dm.get_triu_iterator():
    i, j = dm.ltog(iloc, jloc)
    dm[iloc, jloc] = max(i, j) + min(i, j) * 1j

pgrid.report("dm before symm")
print dm[...]

dmg = dm.collect()
if dm.onroot:
    print "dmg"
    print dmg

dm.symmetrize()

pgrid.report("dm after symm")
print dm[...]

dmg = dm.collect()
if dm.onroot:
    print "dmg"
    print dmg