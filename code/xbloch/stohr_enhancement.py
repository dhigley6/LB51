"""Module for calculating 'coherent enhancement factor' of Stohr, Scherz
"""

import numpy as np 
import matplotlib.pyplot as plt

# constants taken from prl:
G_COH = 360.0      # dimensionless
WAVELENGTH = 1.59E-9    # meters
GAMMA_x = 0.96E-3    # eV
GAMMA = 0.43    # eV
C = 2.997925E8    # m/s

# factor for converting eV to J:
eV_to_J = 1.60218E-19
# factor for converting cm to m:
cm_to_m = 1E-2


def calculate_factors(intensities):
    # convert intensities to needed units:
    intensities_w_units = intensities/(eV_to_J*cm_to_m**2)
    stuff = GAMMA_x*G_COH*WAVELENGTH**3/(np.pi**2*C*GAMMA**2)
    factors = 1+G_COH/(1+intensities_w_units*stuff)
    return factors

def example_plot():
    intensities = np.logspace(9, 17, int(1E3))
    factors = calculate_factors(intensities)
    plt.figure()
    plt.semilogx(intensities, factors)
    plt.xlabel('Intensities (W/cm$^2)')
    plt.ylabel('Stohr Enhancement Factor')
    