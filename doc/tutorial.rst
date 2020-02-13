.. _tutorial:

============
Tutorial
============

After installation, **hspin** can be executed in two manners:

1. Construct WavefunctionLoader and ZFSCalculation loader from within Python terminal or Jupyter notebook, and call ZFSCalculation.solve to perform the calculation.

2. Directly execute hspin.exec.runzfs with Python from a terminal.

We will mainly focus on approch 2, which works more smoothly with MPI.

For serial execution, simply type the following command in the folder that contains DFT wavefunction file(s)

.. code:: bash

   $ python -m hspin.exec.runzfs [--flags]

For parallel execution, use the following command
   
.. code:: bash

   $ mpiexec [-n num_of_processes] python -m hspin.exec.runzfs [--flags]

where `num_of_processors` is the number of processes.

Acceptable flags [`--flags`] are listed below, for detailed explanation see `hspin.exec.runzfs.py`.

- `path`: working directory for this calculation. Python will first change the working dir before any calculations. Default is ".".

- `wfcfmt`: format of input wavefunction. Default is "qeh5". Supported values are:

   - "qeh5": Quantum Espresso HDF5 save file. path should contains "prefix.xml" and save folder.
   - "qe": Quantum Espresso (v6.1) save file. path should be the save folder that contains "data-files.xml", etc.
   - "qbox": Qbox xml file.
   - "cube-wfc": cube files of (real) wavefunctions (Kohn-Sham orbitals).
   - "cube-density": cube files of (signed) squared wavefunction, this option is to support `pp.x` output with `plot_num = 7` and `lsign = .TRUE.`.

- `filename`: name of the Qbox sample XML file that contains input wavefunction. Only used if `wfcfmt = "qbox"`.

- `fftgrid`: FFT grid used. Supported values are "density" or "wave". If "wave" is specified, use a reduced FFT grid to perform calculations. Default is "wave".

- `memory`: "high", "low" or "critical". See ZFSCalculation documentation. Default is "critical".


