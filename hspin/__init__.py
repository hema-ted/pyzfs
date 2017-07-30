from __future__ import print_function, division
from subprocess import Popen, PIPE
import os
from common.io import indent
from common.parallel import mpiroot

__code__ = "hpsi"
__version__ = "0.0.0"

try:
    p = Popen(["git", "show", "--summary"], stdout=PIPE,
              cwd=os.path.dirname(os.path.abspath(__file__)))
    __git_summary__ = p.communicate()[0]
except KeyError:
    __git_summary__ = ""

if mpiroot:
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print("Code: {}".format(__code__))
    print("Version: {}".format(__version__))
    print("Git summary:")
    indent(4).indented_print(__git_summary__)
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
