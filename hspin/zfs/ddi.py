# Define functions to compute dipole-dipole interactions (abbr. "ddi")
# The dipole-dipole interaction is a 3 by 3 tensor (with unit bohr^-3)
# as a function of r (labeled as ddir) or G (labeled as ddig)
# ddir is defined as eq. 4 in PRB 77, 035119 (2008), without the leading 1/2
# ddig can be computed by Fourier transform ddir, or computed analytically in G space

from __future__ import absolute_import, division, print_function
import numpy as np


def compute_ddig(cell, ft):
    """Compute dipole-dipole interaction in G space.

    ddi(G)_{ab} = 4 * pi * [ Ga * Gb / G^2 - delta(a,b) / 3 ]
    a, b are cartesian indices.

    Args:
        cell (..common.cell.Cell): Cell on which to compute ddig.
        ft (..common.ft.FourierTransform): FT which defines grid size.

    Returns:
        np.ndarray of shape (3, 3, ft.N, ft.N). First two indices iterate
            over cartesian coordinates, last two indices iterate over G space.
    """

    n1, n2, n3 = ft.n1, ft.n2, ft.n3
    G1, G2, G3 = cell.G1, cell.G2, cell.G3
    omega = cell.omega

    from numpy.fft import fftfreq
    ddig = np.zeros([3, 3, n1, n2, n3])

    G1_arr = np.outer(G1, fftfreq(n1, d=1 / n1))
    G2_arr = np.outer(G2, fftfreq(n2, d=1 / n2))
    G3_arr = np.outer(G3, fftfreq(n3, d=1 / n3))

    Gx = (  G1_arr[0, :, np.newaxis, np.newaxis]
          + G2_arr[0, np.newaxis, :, np.newaxis]
          + G3_arr[0, np.newaxis, np.newaxis, :])
    Gy = (  G1_arr[1, :, np.newaxis, np.newaxis]
          + G2_arr[1, np.newaxis, :, np.newaxis]
          + G3_arr[1, np.newaxis, np.newaxis, :])
    Gz = (  G1_arr[2, :, np.newaxis, np.newaxis]
          + G2_arr[2, np.newaxis, :, np.newaxis]
          + G3_arr[2, np.newaxis, np.newaxis, :])

    Gxx = Gx ** 2
    Gyy = Gy ** 2
    Gzz = Gz ** 2
    Gxy = Gx * Gy
    Gxz = Gx * Gz
    Gyz = Gy * Gz
    Gsquare = Gxx + Gyy + Gzz
    Gsquare[0, 0, 0] = 1  # avoid runtime error message, G = 0 term will be excluded later

    ddig[0, 0, ...] = Gxx / Gsquare - 1. / 3.
    ddig[1, 1, ...] = Gyy / Gsquare - 1. / 3.
    ddig[2, 2, ...] = Gzz / Gsquare - 1. / 3.
    ddig[0, 1, ...] = ddig[1, 0, ...] = Gxy / Gsquare
    ddig[0, 2, ...] = ddig[2, 0, ...] = Gxz / Gsquare
    ddig[1, 2, ...] = ddig[2, 1, ...] = Gyz / Gsquare

    ddig[..., 0, 0, 0] = 0
    ddig *= 4 * np.pi / omega

    return ddig

def compute_ddir(cell, ft):
    """Compute dipole-dipole interaction in R space.

    ddi(r)_{ab} = ( r^2 * delta(a,b) - 3 * ra * rb ) / r^5
    a, b are cartesian indices.

    Args:
        cell (..common.cell.Cell): Cell on which to compute ddig.
        ft (..common.ft.FourierTransform): FT which defines grid size.

    Returns:
        np.ndarray of shape (3, 3, ft.N, ft.N). First two indices iterate
            over cartesian coordinates, last two indices iterate over R space.
    """

    n1, n2, n3 = ft.n1, ft.n2, ft.n3
    R1, R2, R3 = cell.R1, cell.R2, cell.R3

    ddir = np.zeros([3, 3, n1, n2, n3])

    for ir1, ir2, ir3 in np.ndindex(n1, n2, n3):
        if ir1 == ir2 == ir3 == 0:
            continue  # neglect r = 0 component

        r = ((ir1 - n1 * int(ir1 > n1 / 2)) * R1 / n1
             + (ir2 - n2 * int(ir2 > n2 / 2)) * R2 / n2
             + (ir3 - n3 * int(ir3 > n3 / 2)) * R3 / n3)

        rnorm = np.linalg.norm(r)
        ddir[..., ir1, ir2, ir3] = (rnorm**2 * np.eye(3)
                                    - 3 * np.outer(r, r)) / rnorm**5
    return ddir
