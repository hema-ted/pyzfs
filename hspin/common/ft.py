from __future__ import absolute_import, division, print_function
from mpi4py import MPI
try:
    from pyfftw.interfaces.numpy_fft import fftn, ifftn
    if MPI.COMM_WORLD.Get_rank() == 0:
        print("common.ft: using PyFFTW library...")
except ImportError:
    from numpy.fft import fftn, ifftn
    if MPI.COMM_WORLD.Get_rank() == 0:
        print("common.ft: using numpy.fft library...")
from numpy.fft import fftshift, ifftshift


class FourierTransform:

    # Define 3D FFT with following conventions:
    # f(G) = 1/omega * int{ f(r) exp(-iGr) dr }
    # f(r) = sigma{ f(G) exp(iGr) }
    def __init__(self, n1, n2, n3):
        self.n1 = n1
        self.n2 = n2
        self.n3 = n3
        self.N = n1 * n2 * n3

    def forward(self, fr):
        assert fr.ndim == 3 and fr.shape == (self.n1, self.n2, self.n3)
        fg = (1./self.N) * fftn(fr)
        return fg

    def backward(self, fg):
        assert fg.ndim == 3 and fg.shape == (self.n1, self.n2, self.n3)
        fr = self.N * ifftn(fg)
        return fr

    def fftintep(self, fr, n1, n2, n3):
        assert fr.ndim == 3 and fr.shape == (self.n1, self.n2, self.n3)
        assert n1 <= self.n1 and n2 <= self.n2 and n3 <= n3
        if (n1, n2, n3) == (self.n1, self.n2, self.n3):
            return fr
        fg = fftn(fr)
        sfg = fftshift(fg)
        newsfg = sfg[(self.n1-n1)//2:(self.n1-n1)//2+n1,
                     (self.n2-n2)//2:(self.n2-n2)//2+n2,
                     (self.n3-n3)//2:(self.n3-n3)//2+n3,
                    ]
        newfg = ifftshift(newsfg)
        newfr = (float(n1*n2*n3)/float(self.N)) * ifftn(newfg)
        return newfr

