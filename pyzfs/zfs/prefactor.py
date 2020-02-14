from __future__ import absolute_import, division, print_function
import numpy as np
from scipy.constants import physical_constants

from ..common.units import *

gamma = physical_constants["electron gyromag. ratio"][0]
hbar = physical_constants["Planck constant over 2 pi"][0]
mu0 = physical_constants["mag. constant"][0]
ge = physical_constants["electron g factor"][0]
mub = physical_constants["Bohr magneton"][0]

prefactor = np.prod(
    [
        # -1,  # sign convention for D tensor
        # 1. / 2,            # eq. 2 from PRB paper
        1. / 4,  # eq. 2 and eq. 8 from PRB paper
        mu0 / (4 * np.pi),  # magnetic constant
        (gamma * hbar) ** 2,  # conversion factor from unitless spin to magnetic moment

        # at this point, unit is J m^3
        m_to_bohr ** 3,
        joule_to_mhz,
        # at this point, unit is MHz bohr^3
    ]
)
