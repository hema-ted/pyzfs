from __future__ import absolute_import, division, print_function
import numpy as np
from mpi4py import MPI
from pprint import pprint

from ..zfs.main import ZFSCalculation
from . import parse_sys_argv

# Default arguments
kwargs = {
    "path": ".",
    "wfcfmt": "cube-wfc",
    "comm": MPI.COMM_WORLD,
    "memory": "low"
}

# Override default arguments with sys.argv
kwargs.update(parse_sys_argv())

if MPI.COMM_WORLD.Get_rank() == 0:
    print("exec.runzfs: instantializing ZFSCalculation with following arguments...")
    pprint(kwargs)

# ZFS calculation
zfscalc = ZFSCalculation(**kwargs)
zfscalc.solve()

# Write global I matrix and xml file
if zfscalc.pgrid.onroot:
    np.save("Iijab.npy", zfscalc.Iglobal)
    open("zfs.xml", "w").write(zfscalc.get_xml())
