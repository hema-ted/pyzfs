from __future__ import print_function, division
import pkg_resources
from .common.io import indent
from .common.parallel import mpiroot

if mpiroot:
    version = pkg_resources.require("PyZFS")[0].version
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print("PyZFS version {}".format(version))
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
