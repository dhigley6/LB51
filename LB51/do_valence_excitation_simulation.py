"""Simulate X-ray-induced XAS changes from valence excitation

This code borrows from LK30 analysis code
"""

import numpy as np 
import matplotlib.pyplot as plt 

from scipy import constants

from LB51 import LB51_get_cal_data

SHORT_DATA = LB51_get_cal_data.get_short_pulse_data()
PHOT = SHORT_DATA['99']['sum_intact']['phot']
SSRL_ABSORPTION_COPD = SHORT_DATA["99"]["sum_intact"]["ssrl_absorption"]
SSRL_RESONANT_ABSORPTION = SSRL_ABSORPTION_COPD-SSRL_ABSORPTION_COPD[0]
DEFAULT_NO_SAM_SPEC = SHORT_DATA['99']['sum_intact']['no_sam_spec']

FWHMFLUENCE_2_AVERAGEFLUENCE = 0.4412712003053032    # From LK30 code

CORE_2_FERMI = 777.5 # Photon energy for excitations from 2p core to Fermi level (eV)

# conversion factor from absorbed fluence to absorbed energy density
FLUENCE2ABSORBEDENERGY = 17.3     # (mJ/cm^2 to meV/atom, I think)

# 3TM Electronic heat capacity constant for Co (J*M^3/K^2)
GAMMA = 405

# Boltzmann's constant (eV/K)
BOLTZMANN = constants.value(u'Boltzmann constant in eV/K')

def calculate_valence_energy(pulse_length=5.0, cascade_duration=13.0):
    """Calculate energy of electrons within 2 eV of Fermi level from LK30 model
    """
    time = np.arange(-1000, 2000, 0.1)
    pulse = (time > 0) & (time <= pulse_length)    # flat top pulse
    cascade_time = np.arange(-400, 400, 0.1)
    cascade = (cascade_time > 0) & (cascade_time <= cascade_duration)
    cascade = cascade/np.sum(cascade)
    response = np.convolve(pulse, cascade, mode='same')
    integrated_response = np.sum(response*pulse)/np.sum(pulse)
    return integrated_response

def convert_absorbed_fluence_to_energy_density(absorbed_fluence):
    return absorbed_fluence*FLUENCE2ABSORBEDENERGY

def calculate_electronic_temperature(energy_per_atom):
    pass

def calculate_absorption_changes(electronic_temperature=10000):
    energy = np.linspace(-5, 5, 1000)
    initial_occupations = energy < 0
    final_occupations = 1/(np.exp(energy/(BOLTZMANN*electronic_temperature))+1)
    change_occupations = final_occupations-initial_occupations
    sigma = 0.43
    broadening = np.exp(-1*(energy)**2/(sigma**2))
    broadening = broadening/np.sum(broadening)
    change_absorption = np.convolve(change_occupations, broadening, mode='same')
    plt.figure()
    plt.plot(change_absorption)
    return change_absorption

def calculate_absorption_loss(electronic_temperature=7500, no_sam_spec=DEFAULT_NO_SAM_SPEC):
    change_absorption = calculate_absorption_changes(electronic_temperature)
    change_absorption = np.interp(PHOT, np.linspace(-5+CORE_2_FERMI, 5+CORE_2_FERMI, 1000), change_absorption)
    plt.figure()
    plt.plot(PHOT, SSRL_RESONANT_ABSORPTION)
    nonlinear_absorption = SSRL_RESONANT_ABSORPTION-change_absorption
    plt.plot(PHOT, nonlinear_absorption)
    plt.plot(PHOT, change_absorption)
    linear_transmission = np.exp(-1*SSRL_RESONANT_ABSORPTION)
    nonlinear_transmission = np.exp(-1*nonlinear_absorption)
    plt.figure()
    plt.plot(PHOT, linear_transmission)
    plt.plot(PHOT, nonlinear_transmission)
    plt.figure()
    plt.plot(PHOT, no_sam_spec)
    linear_transmitted = linear_transmission*no_sam_spec
    nonlinear_transmitted = nonlinear_transmission*no_sam_spec
    in_range = PHOT > 777.5
    linear_integral = np.sum(1-linear_transmitted[in_range])
    nonlinear_integral = np.sum(1-nonlinear_transmitted[in_range])
    print(nonlinear_integral/linear_integral)
    return nonlinear_integral/linear_integral