#!/bin/bash

# QE save file is too large to be included in Git repo.
# To run this test, QE calculation will be performed.
# QE must be compiled with HDF5 flag enabled.

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

