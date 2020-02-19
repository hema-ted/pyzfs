.. _documentation:

Code documentation
==================

**PyZFS** can be extended to support more wavefunction formats by defining subclasses of **WavefunctionLoader** abstract class.
The abstract method **scan** and **load** have to be override to parse and read the wavefunction data into memory and store as a **Wavefunction** object.

**PyZFS** API documentation:

ZFS
------

.. automodule:: pyzfs.zfs.main
    :members:
    :show-inheritance:

Common
------

.. automodule:: pyzfs.common.wfc.baseloader
    :members:

.. automodule:: pyzfs.common.wfc.wavefunction
    :members:

.. automodule:: pyzfs.common.cell
    :members:

.. automodule:: pyzfs.common.ft
    :members:

.. automodule:: pyzfs.common.parallel
    :members:



