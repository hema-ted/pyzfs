import sys
import numpy as np
from mpi4py import MPI

from ..zfs.main import ZFSCalculation

path, flag = sys.argv[1], sys.argv[2]

if len(sys.argv) > 3:
    memory = sys.argv[3]
else:
    memory = "--high"

zfscalc = ZFSCalculation(path, flag, MPI.COMM_WORLD, memory)
zfscalc.solve()

if zfscalc.pgrid.onroot:
    np.save("Iijab.npy", zfscalc.Iglobal)
    open("zfs.xml", "w").write(zfscalc.get_xml())
