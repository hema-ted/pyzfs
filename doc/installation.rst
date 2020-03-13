.. _installation:

============
Installation
============

**PyZFS** uses the **mpi4py** package for parallelization. An existing MPI implementation (e.g. **MPICH** or **OpenMPI**) is required to install **mpi4py** and **PyZFS**. Many supercomputers provide modules for pre-compiled MPI implementations. To install MPI manually (taking **MPICH** as example), execute the following command on Linux

.. code:: bash

   $ sudo apt-get install mpich libmpich-dev

or the following command on Mac

.. code:: bash

   $ brew install mpich

**PyZFS** can be executed with Python 2.7 and Python 3.5+. However, to run **PyZFS** with Python 2.7 one may need to build certain legacy versions of dependencies such as **ase** (**ase** v3.17.0 is tested to work with **PyZFS** in Python 2).

It is recommended to install **PyZFS** using **pip**. First, clone the git repository into a local directory

.. code:: bash

   $ git clone https://github.com/hema-ted/pyzfs.git

Then, execute **pip** in the folder containing **setup.py**

.. code:: bash

   $ pip install .

**PyZFS** depends on the following packages, which will be installed automatically if installed through **pip**

   - ``numpy``
   - ``scipy``
   - ``mpi4py``
   - ``h5py``
   - ``ase``
   - ``lxml``

If using **pip** is not possible, one can manually install the above dependencies, and then include the directory of **PyZFS** repository to the **PYTHONPATH** by appending the following command to the **.bashrc** file

.. code:: bash

   $ export PYTHONPATH=$PYTHONPATH:path/to/pyzfs
