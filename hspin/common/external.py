"""
Following contents are copied from sunyata package,
they are put here in order to make the hspin package standalone
"""

import sys
import numpy as np
from collections import OrderedDict
import re
from ase import Atoms
from .units import bohr_to_angstrom
from .ft import ifftn


def parse_sys_argv(flags=None):
    """Parse sys.argv into a list of unnamed arguments and a dict of named arguments

    It is assumed that a flag starts with "--", and is followed by a certain number
    of values (can be 0)
    Args:
        flags: a dict of flag name: nvalue (how many values are associated with that flag),
            default to None, in which case all elements in sys.argv starting with "--" are
            interpreted as flags with nvalue = 1

    Returns:
        args (list): a list representing unnamed arguments
        kwargs (dict): a dict of named arguments. If nvalue == 0, then the dict value is True;
            if nvalue == 1, the dict value is the following element in sys.argv; if nvalue > 1,
            the dict value is a list of the following nvalue elements in sys.argv.

        elements of args and values of kwargs are always strings, no type conversions
        are performed here

    """

    argv = np.array(sys.argv[1:], dtype=object)
    argc = len(argv)
    unparsed = np.ones(argc, dtype=bool)

    if flags is None:
        flags = {
            flag[2:]: 1
            for flag in argv if "--" in flag
        }

    kwargs = OrderedDict()
    for flag, nvalue in flags.items():
        indices = [i for i in range(argc) if argv[i] == "--{}".format(flag)]
        if len(indices) == 0:
            continue
        elif len(indices) == 1:
            index = indices[0]
            try:
                if nvalue == 0:
                    kwargs[flag] = True
                elif nvalue == 1:
                    kwargs[flag] = argv[index + 1]
                else:
                    kwargs[flag] = argv[index + 1: index + nvalue + 1]
            except IndexError:
                print("cannot parse enough values for flag --{}".format(flag))
            unparsed[index: index + nvalue + 1] = False
        else:
            raise ValueError("flag --{} appeared multiple times".format(flag))

    args = argv[unparsed]

    return args, kwargs


def regex(dtype):
    """Returns the regular expression required by re package

    :param dtype: int, float or str
    :return: string of regular expression
    """

    if dtype is int:
        return r"-*\d+"
    elif dtype is float:
        return r"-*\d+\.\d*[DeEe]*[+-]*\d*"
    elif dtype is str:
        return r".*"
    else:
        raise ValueError("unsupported type")


def parse_one_value(dtype, content, index=0):
    """Parse one value of type dtype from content

    :param dtype: type of value wanted
    :param content: a string to be parsed
    :param index: index of parsed value
    :return: first (if index not specified) value found in content
    """

    results = re.findall(regex(dtype), content)
    if results:
        return dtype(results[index])


def parse_many_values(n, dtype, content):
    """Parse n values of type dtype from content

    :param n: # of values wanted
    :param dtype: type of values wanted
    :param content: a string or a list of strings,
        it is assumed that n values exist in continues
        lines of content starting from the first line
    :return: a list of n values
    """

    if isinstance(content, basestring) or isinstance(content, np.string_):
        results = re.findall(regex(dtype), content)
        return [dtype(value) for value in results[0:n]]

    results = list()
    started = False
    for i in xrange(len(content)):
        found = re.findall(regex(dtype), content[i])
        if found:
            started = True
        else:
            if started:
                raise ValueError("cannot parse {} {} variables in content {}".format(
                    n, dtype, content
                ))
        results.extend(found)
        assert len(results) <= n
        if len(results) == n:
            return [dtype(result) for result in results]


def empty_ase_cell(a, b, c, unit="angstrom"):
    if unit == "angstrom":
        s = 1
    elif unit == "bohr":
        s = bohr_to_angstrom
    else:
        raise ValueError
    return Atoms(cell=s * np.array([a, b, c]), pbc=[1, 1, 1])


"""Following codes are adapted from https://github.com/QijingZheng/pyvaspwfc
These codes are needed to support VASP wavefunction"""

AUTOA    = 0.529177249
RYTOEV   = 13.605826
CLIGHT   =  137.037          # speed of light in a.u.
EVTOJ    = 1.60217733E-19
AMTOKG   = 1.6605402E-27
BOLKEV   = 8.6173857E-5
BOLK     = BOLKEV * EVTOJ
EVTOKCAL = 23.06

# FELECT    =  (the electronic charge)/(4*pi*the permittivity of free space)
#         in atomic units this is just e^2
# EDEPS    =  electron charge divided by the permittivity of free space
#         in atomic units this is just 4 pi e^2
# HSQDTM    =  (plancks CONSTANT/(2*PI))**2/(2*ELECTRON MASS)
#
PI     = 3.141592653589793238
TPI    = 2 * PI
CITPI  = 1j * TPI
FELECT = 2 * AUTOA * RYTOEV
EDEPS  = 4 * PI * 2 * RYTOEV * AUTOA
HSQDTM = RYTOEV * AUTOA * AUTOA

# vector field A times momentum times e/ (2 m_e c) is an energy
# magnetic moments are supplied in Bohr magnetons
# e / (2 m_e c) A(r) p(r)    =  energy
# e / (2 m_e c) m_s x ( r - r_s) / (r-r_s)^3 hbar nabla    =
# e^2 hbar^2 / (2 m_e^2 c^2) 1/ lenght^3    =  energy
# conversion factor from magnetic moment to energy
# checked independently in SI by Gilles de Wijs

MAGMOMTOENERGY  = 1 / CLIGHT**2 * AUTOA**3 * RYTOEV

# dimensionless number connecting input and output magnetic moments
# AUTOA e^2 (2 m_e c^2)
MOMTOMOM   = AUTOA / CLIGHT / CLIGHT / 2
AUTOA2   = AUTOA * AUTOA
AUTOA3   = AUTOA2 * AUTOA
AUTOA4   = AUTOA2 * AUTOA2
AUTOA5   = AUTOA3 * AUTOA2

# dipole moment in atomic units to Debye
AUTDEBYE = 2.541746

class vaspwfc():
    '''
    Class for VASP Pseudowavefunction stored in WAVECAR
    The format of VASP WAVECAR, as shown in
        http://www.andrew.cmu.edu/user/feenstra/wavetrans/
    is:
        Record-length #spin components RTAG(a value specifying the precision)
        #k-points #bands ENCUT(maximum energy for plane waves)
        LatVec-A
        LatVec-B
        LatVec-C
        Loop over spin
           Loop over k-points
              #plane waves, k vector
              Loop over bands
                 band energy, band occupation
              End loop over bands
              Loop over bands
                 Loop over plane waves
                    Plane-wave coefficient
                 End loop over plane waves
              End loop over bands
           End loop over k-points
        End loop over spin
    '''

    def __init__(self, fnm='WAVECAR'):
        '''
        Initialization.
        '''

        self._fname = fnm
        try:
            self._wfc = open(self._fname, 'rb')
        except:
            raise IOError('Failed to open %s' % self._fname)

        # read the basic information
        self.readWFHeader()
        # read the band information
        self.readWFBand()

    def readWFHeader(self):
        '''
        Read the system information from WAVECAR, which is written in the first
        two record.
        rec1: recl, nspin, rtag
        rec2: nkpts, nbands, encut, ((cell(i,j) i=1, 3), j=1, 3)
        '''

        # goto the start of the file and read the first record
        self._wfc.seek(0)
        self._recl, self._nspin, self._rtag = np.array(
            np.fromfile(self._wfc, dtype=np.float, count=3),
            dtype=int
        )
        self._WFPrec = self.setWFPrec()
        # the second record
        self._wfc.seek(self._recl)
        dump = np.fromfile(self._wfc, dtype=np.float, count=12)

        self._nkpts = int(dump[0])  # No. of k-points
        self._nbands = int(dump[1])  # No. of bands
        self._encut = dump[2]  # Energy cutoff
        self._Acell = dump[3:].reshape((3, 3))  # real space supercell basis
        self._Omega = np.linalg.det(self._Acell)  # real space supercell volume
        self._Bcell = np.linalg.inv(self._Acell).T  # reciprocal space supercell volume

        # Minimum FFT grid size
        Anorm = np.linalg.norm(self._Acell, axis=1)
        CUTOF = np.ceil(
            np.sqrt(self._encut / RYTOEV) / (TPI / (Anorm / AUTOA))
        )
        self._ngrid = np.array(2 * CUTOF + 1, dtype=int)

    def setWFPrec(self):
        '''
        Set wavefunction coefficients precision:
            TAG = 45200: single precision complex, np.complex64, or complex(qs)
            TAG = 45210: double precision complex, np.complex128, or complex(q)
        '''
        if self._rtag == 45200:
            return np.complex64
        elif self._rtag == 45210:
            return np.complex128
        elif self._rtag == 53300:
            raise ValueError("VASP5 WAVECAR format, not implemented yet")
        elif self._rtag == 53310:
            raise ValueError("VASP5 WAVECAR format with double precision "
                             + "coefficients, not implemented yet")
        else:
            raise ValueError("Invalid TAG values: {}".format(self._rtag))

    def readWFBand(self, ispin=1, ikpt=1, iband=1):
        '''
        Extract KS energies and Fermi occupations from WAVECAR.
        '''

        self._npw = np.zeros(self._nkpts, dtype=int)
        self._kvecs = np.zeros((self._nkpts, 3), dtype=float)
        self._bands = np.zeros((self._nspin, self._nkpts, self._nbands), dtype=float)
        self._occs = np.zeros((self._nspin, self._nkpts, self._nbands), dtype=float)

        for ispin in range(self._nspin):
            for ikpt in range(self._nkpts):
                rec = self.whereRec(ispin + 1, ikpt + 1, 1) - 1
                self._wfc.seek(rec * self._recl)
                dump = np.fromfile(self._wfc, dtype=np.float, count=4 + 3 * self._nbands)
                if ispin == 0:
                    self._npw[ikpt] = int(dump[0])
                    self._kvecs[ikpt] = dump[1:4]
                dump = dump[4:].reshape((-1, 3))
                self._bands[ispin, ikpt, :] = dump[:, 0]
                self._occs[ispin, ikpt, :] = dump[:, 2]

        if self._nkpts > 1:
            tmp = np.linalg.norm(
                np.dot(np.diff(self._kvecs, axis=0), self._Bcell), axis=1)
            self._kpath = np.concatenate(([0, ], np.cumsum(tmp)))
        else:
            self._kpath = None
        return self._kpath, self._bands

    def gvectors(self, ikpt=1, gamma=False):
        '''
        Generate the G-vectors that satisfies the following relation
            (G + k)**2 / 2 < ENCUT
        '''
        assert 1 <= ikpt <= self._nkpts, 'Invalid kpoint index!'

        kvec = self._kvecs[ikpt - 1]
        # fx, fy, fz = [fftfreq(n) * n for n in self._ngrid]
        # fftfreq in scipy.fftpack is a little different with VASP frequencies
        fx = [ii if ii < self._ngrid[0] / 2 + 1 else ii - self._ngrid[0]
              for ii in range(self._ngrid[0])]
        fy = [jj if jj < self._ngrid[1] / 2 + 1 else jj - self._ngrid[1]
              for jj in range(self._ngrid[1])]
        fz = [kk if kk < self._ngrid[2] / 2 + 1 else kk - self._ngrid[2]
              for kk in range(self._ngrid[2])]
        if gamma:
            # parallel gamma version of VASP WAVECAR exclude some planewave
            # components, -DwNGZHalf
                kgrid = np.array([(fz[kk], fy[jj], fx[ii])
                    for kk in range(self._ngrid[2])
                    for jj in range(self._ngrid[1])
                    for ii in range(self._ngrid[0])
                    if (
                        (fx[ii] > 0) or
                        (fx[ii] == 0 and fy[jj] > 0) or
                        (fx[ii] == 0 and fy[jj] == 0 and fz[kk] >= 0)
                )], dtype=float)
        else:
            kgrid = np.array([(fx[ii], fy[jj], fz[kk])
                              for kk in range(self._ngrid[2])
                              for jj in range(self._ngrid[1])
                              for ii in range(self._ngrid[0])], dtype=float)

        # Kinetic_Energy = (G + k)**2 / 2
        # HSQDTM    =  hbar**2/(2*ELECTRON MASS)
        KENERGY = HSQDTM * np.linalg.norm(
            np.dot(kgrid + kvec[np.newaxis, :], TPI * self._Bcell), axis=1
        ) ** 2
        # find Gvectors where (G + k)**2 / 2 < ENCUT
        Gvec = kgrid[np.where(KENERGY < self._encut)[0]]

        assert Gvec.shape[0] == self._npw[ikpt - 1], 'No. of planewaves not consistent! %d %d %d' % \
                                                     (Gvec.shape[0], self._npw[ikpt - 1], np.prod(self._ngrid))
        return np.asarray(Gvec, dtype=int)

    def save2vesta(self, phi=None, poscar='POSCAR', prefix='wfc', gamma=False):
        '''
        Save the real space pseudo-wavefunction as vesta format.
        '''
        nx, ny, nz = phi.shape
        try:
            pos = open(poscar, 'r')
            head = ''
            for line in pos:
                if line.strip():
                    head += line
                else:
                    break
            head += '\n%5d%5d%5d\n' % (nx, ny, nz)
        except:
            raise IOError('Failed to open %s' % poscar)

        with open(prefix + '_r.vasp', 'w') as out:
            out.write(head)
            nwrite = 0
            for kk in range(nz):
                for jj in range(ny):
                    for ii in range(nx):
                        nwrite += 1
                        out.write('%16.8E ' % phi.real[ii, jj, kk])
                        if nwrite % 10 == 0:
                            out.write('\n')
        if not gamma:
            with open(prefix + '_i.vasp', 'w') as out:
                out.write(head)
                nwrite = 0
                for kk in range(nz):
                    for jj in range(ny):
                        for ii in range(nx):
                            nwrite += 1
                            out.write('%16.8E ' % phi.imag[ii, jj, kk])
                            if nwrite % 10 == 0:
                                out.write('\n')

    def wfc_r(self, ispin=1, ikpt=1, iband=1,
              gvec=None, ngrid=None, norm=False,
              gamma=False):
        '''
        Obtain the pseudo-wavefunction of the specified KS states in real space
        by performing FT transform on the reciprocal space planewave
        coefficients.  The 3D FT grid size is determined by ngrid, which
        defaults to self._ngrid if not given.  Gvectors of the KS states is used
        to put 1D planewave coefficients back to 3D grid.
        '''
        self.checkIndex(ispin, ikpt, iband)

        if ngrid is None:
            ngrid = self._ngrid.copy()
        else:
            ngrid = np.array(ngrid, dtype=int)
            assert ngrid.shape == (3,)
            assert np.alltrue(ngrid >= self._ngrid), \
                "Minium FT grid size: (%d, %d, %d)" % \
                (self._ngrid[0], self._ngrid[1], self._ngrid[2])
        if gvec is None:
            gvec = self.gvectors(ikpt, gamma)

        # if gamma:
        #     phi_k = np.zeros((ngrid[0], ngrid[1], ngrid[2] / 2 + 1), dtype=np.complex128)
        # else:
        phi_k = np.zeros(ngrid, dtype=np.complex128)

        gvec %= ngrid[np.newaxis, :]
        phi_k[gvec[:, 0], gvec[:, 1], gvec[:, 2]] = self.readBandCoeff(ispin, ikpt, iband, norm)

        if gamma:
            # add some components that are excluded and perform c2r FFT
            for ii in range(ngrid[0]):
                for jj in range(ngrid[1]):
                    for kk in range(ngrid[2]):
                        fx = ii if ii < ngrid[0] / 2 + 1 else ii - ngrid[0]
                        fy = jj if jj < ngrid[1] / 2 + 1 else jj - ngrid[1]
                        fz = kk if kk < ngrid[2] / 2 + 1 else kk - ngrid[2]
                        if (fz > 0) or (fz == 0 and fy > 0) or (fz == 0 and fy == 0 and fx >= 0):
                            continue
                        phi_k[ii, jj, kk] = phi_k[-ii, -jj, -kk].conjugate()
            #phi_k /= np.sqrt(2.)
            #phi_k[0, 0, 0] *= np.sqrt(2.)
            #return np.fft.irfftn(phi_k, s=ngrid), phi_k
            return ifftn(phi_k)#, phi_k
        else:
            # perform complex2complex FFT
            return ifftn(phi_k)

    def readBandCoeff(self, ispin=1, ikpt=1, iband=1, norm=False):
        '''
        Read the planewave coefficients of specified KS states.
        '''

        self.checkIndex(ispin, ikpt, iband)

        rec = self.whereRec(ispin, ikpt, iband)
        self._wfc.seek(rec * self._recl)

        npw = self._npw[ikpt - 1]
        dump = np.fromfile(self._wfc, dtype=self._WFPrec, count=npw)

        cg = np.asarray(dump, dtype=np.complex128)
        if norm:
            cg /= np.linalg.norm(cg)
        return cg

    def whereRec(self, ispin=1, ikpt=1, iband=1):
        '''
        Return the rec position for specified KS state.
        '''

        self.checkIndex(ispin, ikpt, iband)

        rec = 2 + (ispin - 1) * self._nkpts * (self._nbands + 1) + \
              (ikpt - 1) * (self._nbands + 1) + \
              iband
        return rec

    def checkIndex(self, ispin, ikpt, iband):
        '''
        Check if the index is valid!
        '''
        assert 1 <= ispin <= self._nspin, 'Invalid spin index!'
        assert 1 <= ikpt <= self._nkpts, 'Invalid kpoint index!'
        assert 1 <= iband <= self._nbands, 'Invalid band index!'
