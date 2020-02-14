from __future__ import absolute_import, division, print_function
import numpy as np


def compute_rhog(psi1r, psi2r, ft, rho1g=None, rho2g=None):
    """Compute rho(G, -G) for two electrons occupying two (KS) orbitals.

    rho(G, -G) is defined as f1(G) * f2(-G) - |f3(G)|^2,
    which is equal to f1(G) * conj(f2(G)) - f3(G) * conj(f3(G))
    f1, f2 and f3 are defined following PRB 77, 035119 (2008):
      f1(r) = |psi1(r)|^2
      f2(r) = |psi2(r)|^2
      f3(r) = conj(psi1(r)) * psi2(r)
    f1(G), f2(G) and f3(G) are obtained by Fourier Transform of f1(r), f2(r) and f3(r)
    rho(r) is computed for debug purpose as inverse FT of rho(G, -G) and returned as well

    Args:
        psi1r, psi2r (np.ndarray): R space wavefunction for electron 1 and 2.
        ft (..common.ft.FourierTransform): FT which defines grid size.
        rho1g, rho2g (np.ndarray): R space charge density for electron 1 and 2.
            If not provided, will be computed from psi1r and psi2r.

    Returns:
        rho(G, -G) as a np.ndarray of shape (ft.N, ft.N).

    """

    if rho1g is not None:
        assert rho1g.shape == psi1r.shape
        f1g = rho1g
    else:
        f1r = psi1r * np.conj(psi1r)
        f1g = ft.forward(f1r)

    if rho2g is not None:
        assert rho2g.shape == psi2r.shape
        f2g = rho2g
    else:
        f2r = psi2r * np.conj(psi2r)
        f2g = ft.forward(f2r)

    f3r = psi1r * np.conj(psi2r)
    f3g = ft.forward(f3r)

    #rhoj = f1g * np.conj(f2g)
    #rhok = f3g * np.conj(f3g)

    rhog = f1g * np.conj(f2g) - f3g * np.conj(f3g)
    #rhor = ft.backward(rhog)

    return rhog #, rhor, rhoj, rhok


def compute_delta_model_rhog(cell, ft, d1, d2, d3, s=1):
    """Compute rho(G, -G) for two point dipoles.

    Two spin dipoles are approximated homogeneious dipole gas in small boxes

    Args:
        cell (..common.cell.Cell): Cell on which to compute ddig.
        ft (..common.ft.FourierTransform): FT which defines grid size.
        d1, d2, d3 (float): distance between two dipoles in 3 dimensions
        s (float): box size

    Returns:
        rho(G, -G) as a np.ndarray of shape (ft.N, ft.N)
    """

    n1, n2, n3, N = ft.n1, ft.n2, ft.n3, ft.N
    R1, R2, R3 = cell.R1, cell.R2, cell.R3
    omega = cell.omega

    ns1 = int(n1 * s / R1[0])
    ns2 = int(n2 * s / R2[1])
    ns3 = int(n3 * s / R3[2])

    nd1 = int(n1 * d1 / R1[0])
    nd2 = int(n2 * d2 / R2[1])
    nd3 = int(n3 * d3 / R3[2])

    print(ns1, ns2, ns3)
    print(nd1, nd2, nd3)
    print("effective d1, d2, d3: ", nd1 * R1[0] / n1, nd2 * R2[1] / n2, nd3 * R3[2] / n3)

    psi1r = np.zeros([n1, n2, n3])
    psi2r = np.zeros([n1, n2, n3])

    for ir1, ir2, ir3 in np.ndindex(ns1, ns2, ns3):
        psi1r[ir1, ir2, ir3] = 1.
        psi2r[nd1 + ir1, nd2 + ir2, nd3 + ir3] = 1.

    psi1r /= np.sqrt(np.sum(psi1r ** 2) * omega / N)
    psi2r /= np.sqrt(np.sum(psi2r ** 2) * omega / N)

    rhog = compute_rhog(psi1r, psi2r, ft)
    return rhog

    #rhog, rhor, rhoj, rhok = compute_rhog(psi1r, psi2r, ft)
    #return rhog, rhor, rhoj, rhok, psi1r, psi2r