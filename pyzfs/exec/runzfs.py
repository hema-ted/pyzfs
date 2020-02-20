from __future__ import absolute_import, division, print_function
import numpy as np
import os
from datetime import datetime
from mpi4py import MPI
from pprint import pprint
import pkg_resources
from ..common.misc import parse_sys_argv
from ..common.misc import parse_many_values
from ..zfs.main import ZFSCalculation
from ..common.parallel import mpiroot

"""Run Zero Field Splitting calculation

Example:
    mpirun python -m pyzfs.exec.runzfs --wfcfmt qeh5 --prefix pwscf
    mpirun python -m pyzfs.exec.runzfs --wfcfmt qbox --filename gs.xml

Acceptable kwargs are:
    --path: working directory for this calculation. Python will first change
        the working dir before any calculations. Default is ".".

    --wfcfmt: format of input wavefunction. Supported values are
        "qeh5": Quantum Espresso HDF5 save file. path should contains "prefix.xml" and save folder.
        "qe": Quantum Espresso (v6.1) save file. path should be the save folder that contains "data-files.xml", etc.
              The gvector and evc files have to be converted to xml through iotk.
        "qbox": Qbox xml file
        "cube-wfc": cube files of (real) wavefunctions (Kohn-Sham orbitals).
        "cube-density": cube files of (signed) wavefunction squared, mainly used to
            support pp.x output with plot_num = 7 and lsign = .TRUE.
        File name convention for cube file:
            1. Must end with ".cube".
            2. Must contains either "up" or "down" to indicate spin channel.
            3. The last integer value found the file name is interpreted as band index.
        Default is "qeh5"

    --filename: name for input wavefunction. Only used for Qbox wavefunction

    --fftgrid: "density" or "wave", currently only works for QE wavefunction. If "wave"
        is specified, orbitals will use a reduced grid for FFT. Default is "wave".

    --memory: "high", "low" or "critical". See ZFSCalculation documentation. Default is "critical".
"""

if __name__ == "__main__":
    if mpiroot:
        try:
            version = pkg_resources.require("PyZFS")[0].version
        except Exception:
            version = ""

        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        print("PyZFS code {}".format(version))
        print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

    # Default arguments
    kwargs = {
        "path": ".",
        "wfcfmt": "qeh5",
        "prefix": "pwscf",
        "fftgrid": "wave",
        "comm": MPI.COMM_WORLD,
        "memory": "low"
    }

    # Override default arguments with sys.argv
    kwargs.update(parse_sys_argv()[1])

    # Change directory
    path = kwargs.pop("path")
    if mpiroot:
        print("pyzfs.exec.runzfs: setting working directory as \"{}\"...".format(path))
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
        filename = kwargs.pop("filename", None)
        wfcloader = QboxWavefunctionLoader(filename=filename)
    # elif wfcfmt == "vasp":
    #     from ..common.wfc.vasploader import VaspWavefunctionLoader
    #     wfcloader = VaspWavefunctionLoader()
    elif wfcfmt == "qeh5":
        from ..common.wfc.qeh5loader import QEHDF5WavefunctionLoader
        prefix = kwargs.pop("prefix", "pwscf")
        wfcloader = QEHDF5WavefunctionLoader(fftgrid=fftgrid, prefix=prefix)
    else:
        raise ValueError("Unsupported wfcfmt: {}".format(wfcfmt))

    kwargs["wfcloader"] = wfcloader

    # ZFS calculation
    if mpiroot:
        print("\n\npyzfs.exec.runzfs: instantializing ZFSCalculation with following arguments...")
        pprint(kwargs, indent=2)

    zfscalc = ZFSCalculation(**kwargs)
    zfscalc.solve()

    # Write global I matrix and xml file
    if zfscalc.pgrid.onroot:
        np.save("Iijab.npy", zfscalc.Iglobal)
        xml = zfscalc.get_xml()
        try:
            open("zfs.xml", "w").write(xml)
        except TypeError:
            open("zfs.xml", "wb").write(xml)
