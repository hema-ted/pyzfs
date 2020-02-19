.. _installation:

============
Installation
============

**PyZFS** uses the **mpi4py** package for parallelization. An existing MPI implementation (e.g. **MPICH** or **OpenMPI**) is required to install **mpi4py** and **PyZFS**. Many supercomputers provide modules for pre-compiled MPI implementations. To install MPI manually (taking **MPICH** as example), execute the following command on Linux

.. code:: bash

   $ sudo apt-get install mpich libmpich-dev

or execute the following command on Mac

.. code:: bash

   $ brew install mpich

**PyZFS** can be installed with **pip** by executing the following command in the project root folder containing **setup.py**

.. code:: bash

   $ pip install .

To install **PyZFS** in editable mode

.. code:: bash

   $ pip install -e .

**PyZFS** is designed to be compatible with both Python 2 and Python 3, and depends on the following packages:

   - ``numpy``
   - ``scipy``
   - ``mpi4py``
   - ``h5py``
   - ``ase``
   - ``lxml``

The dependencies will all be installed automatically if installed through **pip**.

