"""Module for calculating 'coherent enhancement factor' of Stohr, Scherz

See Stohr&Scherz PRL (2015) for details
"""

import numpy as np

# constants taken from prl:
G_COH = 360.0  # dimensionless
WAVELENGTH = 1.59e-9  # meters
GAMMA_x = 0.96e-3  # eV
GAMMA = 0.43  # eV
C = 2.997925e8  # m/s

# factor for converting eV to J:
eV_to_J = 1.60218e-19
# factor for converting cm to m:
cm_to_m = 1e-2


def calculate_factors(intensity: float) -> float:
    """Calculate s&s stimulated elastic enhancement factor

    Parameters:
    -----------
    intensity: float
        Intensity of X-ray pulses (W/cm)

    Returns:
    --------
    factor: float
        Enhancement factor
    """
    # convert intensities to needed units:
    intensities_w_units = intensity / (eV_to_J * cm_to_m ** 2)
    stuff = GAMMA_x * G_COH * WAVELENGTH ** 3 / (np.pi ** 2 * C * GAMMA ** 2)
    factor = 1 + G_COH / (1 + intensities_w_units * stuff)
    return factor
