from __future__ import absolute_import, division, print_function
import numpy as np
import os
from mpi4py import MPI
from pprint import pprint
from ..common.external import parse_sys_argv
from ..common.external import parse_many_values
from ..zfs.main import ZFSCalculation
from ..common.parallel import mpiroot

"""Run Zero Field Splitting calculation

Example:
    mpirun -n 256 python -m hspin.exec.runzfs --wfcfmt vasp
    mpirun -n 256 python -m hspin.exec.runzfs --path /home/mahe/zfs/nv/pwscf.save --wfcfmt qe
    mpirun -n 256 python -m hspin.exec.runzfs --wfcfmt qe --fftgrid 60,60,60

Acceptable kwargs are:
    --path: working directory for this calculation. Python will first change
        the working dir before any calculations. Default is ".".

    --wfcfmt: format of input wavefunction. Supported values are
        "qe": Quantum Espresso save file. path should contains "data-files.xml" etc.
        "vasp": VASP WAVECAR and POSCAR file.
        "cube-wfc": cube files of (real) wavefunctions (Kohn-Sham orbitals).
        "cube-density": cube files of (signed) wavefunction squared, mainly used to
            support pp.x output with plot_num = 7 and lsign = .TRUE.
        file name convention for cube file:
            1. must end with ".cube".
            2. must contains either "up" or "down", intepreted as spin channel.
            3. the LAST integer value found the file name is interpreted as band index.
        Default is "qe"

    --fftgrid: "density" or "wave", currently only works for QE wavefunction. If "wave"
        is specified, orbitals will use a reduced grid for FFT. Default is "wave".

    --memory: "high" or "low", see ZFSCalculation documentation. Default is "low".
"""

# Default arguments
kwargs = {
    "path": ".",
    "wfcfmt": "qe",
    "fftgrid": "wave",
    "comm": MPI.COMM_WORLD,
    "memory": "low"
}

# Override default arguments with sys.argv
kwargs.update(parse_sys_argv()[1])

# Change directory
path = kwargs.pop("path")
if mpiroot:
    print("hspin.exec.runzfs: setting working directory as \"{}\"...".format(path))
os.chdir(path)

# Construct proper wavefunction loader
wfcfmt = kwargs.pop("wfcfmt")
fftgrid = kwargs.pop("fftgrid")
if fftgrid not in ["density", "wave"]:
    fftgrid = np.array(parse_many_values(3, int, fftgrid))
if wfcfmt == "qe":
    from ..common.wfc.qeloader import QEWavefunctionLoader
    wfcloader = QEWavefunctionLoader(fftgrid=fftgrid)
elif wfcfmt in ["cube-wfc", "cube-density"]:
    from ..common.wfc.cubeloader import CubeWavefunctionLoader
    wfcloader = CubeWavefunctionLoader(
        density=True if wfcfmt == "cube-density" else False
    )
elif wfcfmt == "qbox":
    from ..common.wfc.qboxloader import QboxWavefunctionLoader
    wfcloader = QboxWavefunctionLoader()
elif wfcfmt == "vasp":
    from ..common.wfc.vasploader import VaspWavefunctionLoader
    wfcloader = VaspWavefunctionLoader()
else:
    raise ValueError("Unsupported wfcfmt: {}".format(wfcfmt))

kwargs["wfcloader"] = wfcloader

# ZFS calculation
if mpiroot:
    print("\n\nhspin.exec.runzfs: instantializing ZFSCalculation with following arguments...")
    pprint(kwargs, indent=2)

zfscalc = ZFSCalculation(**kwargs)
zfscalc.solve()

# Write global I matrix and xml file
if zfscalc.pgrid.onroot:
    np.save("Iijab.npy", zfscalc.Iglobal)
    open("zfs.xml", "w").write(zfscalc.get_xml())
