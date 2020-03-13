#!/bin/bash

# QE save file is too large to be included in Git repo. Therefore, to run this test, an actual QE calculation will be performed to obtain the wavefunction.
# This example was tested with QE 6.4.1.

# To install QE, please follow instructions on https://www.quantum-espresso.org/Doc/user_guide/
# Note that the compilation of QE can be technical and users unfamiliar with the process may find it helpful to consult experts first.
# QE must be compiled with HDF5 flag enabled, i.e. when configuring QE one needs to specify HDF5 library (--with-hdf5=/path/to/hdf5/lib/).

mpirun pw.x -i pw.in > pw.out

mpirun pyzfs --wfcfmt qeh5 > zfs.out
# equivalently: mpirun python -m pyzfs.exec.runzfs --wfcfmt qeh5 > zfs.out

D=`grep --color=never "D unit" zfs.xml | grep --color=never -Eoh '[+-]?[0-9]+([.][0-9]+)?'`
Dref=`grep --color=never "D unit" zfs_ref.xml | grep --color=never -Eoh '[+-]?[0-9]+([.][0-9]+)?'`

E=`grep --color=never "E unit" zfs.xml | grep --color=never -Eoh '[+-]?[0-9]+([.][0-9]+)?'`
Eref=`grep --color=never "E unit" zfs_ref.xml | grep --color=never -Eoh '[+-]?[0-9]+([.][0-9]+)?'`

echo "D = " $D
echo "Ref D = " $Dref
echo "E = " $E
echo "Ref E = " $Eref

if [ `python -c "print(int(abs($D - $Dref) < 1 and abs($E - $Eref) < 1))"` -ne 0 ]
then 
    exit 0
else
    exit 1
fi

