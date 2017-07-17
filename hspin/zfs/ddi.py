# Define functions to compute dipole-dipole interactions (abbr. "ddi")
# The dipole-dipole interaction is a 3 by 3 tensor (with unit bohr^-3)
# as a function of r (labeled as ddir) or G (labeled as ddig)
# ddir is defined as eq. 4 in PRB 77, 035119 (2008), without the leading 1/2
# ddig can be computed by Fourier transform ddir, or computed analytically in G space

from __future__ import absolute_import, division, print_function
import numpy as np

def compute_ddig(cell, ft):
    """
    Compute dipole-dipole interaction in G space by
    ddi(G)_{ab} = 4 * pi * [ Ga * Gb / G^2 - delta(a,b) / 3 ]
    """

    n1, n2, n3 = ft.n1, ft.n2, ft.n3
    G1, G2, G3 = cell.G1, cell.G2, cell.G3
    omega = cell.omega

    ddig = np.zeros([3, 3, n1, n2, n3])

    for ig1, ig2, ig3 in np.ndindex(n1, n2, n3):
        if ig1 == ig2 == ig3 == 0:
            continue  # neglect G = 0 component

        G = ((ig1 - n1 * int(ig1 > n1 / 2)) * G1
             + (ig2 - n2 * int(ig2 > n2 / 2)) * G2
             + (ig3 - n3 * int(ig3 > n3 / 2)) * G3)

        ddig[..., ig1, ig2, ig3] = ((4*np.pi/omega)
                                    * (np.outer(G, G) / np.linalg.norm(G)**2
                                      - np.eye(3)/3))
    return ddig


def compute_ddir(cell, ft):
    """
    Compute dipole-dipole interaction in R space by
    ddi(r)_{ab} = ( r^2 * delta(a,b) - 3 * ra * rb ) / r^5
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
