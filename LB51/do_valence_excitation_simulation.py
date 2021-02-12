"""Simulate X-ray-induced XAS changes from valence excitation

This code borrows from LK30 analysis code
"""

import numpy as np 
import matplotlib.pyplot as plt 

from scipy import constants

from LB51 import LB51_get_cal_data

EMISSION_SHIFT = 0.4

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


def calculate_absorption_changes(incident_fluence=1E3, no_sample_spectrum=DEFAULT_NO_SAM_SPEC):
    absorbed_fluence = calculate_total_absorbed_fluence(incident_fluence, no_sample_spectrum)
    pulse_averaged_valence_fraction = calculate_pulse_averaged_valence_fraction()
    pulse_averaged_valence_energy = absorbed_fluence*pulse_averaged_valence_fraction
    electronic_temperature = calculate_electronic_temperature(pulse_averaged_valence_energy)
    print(electronic_temperature)
    change_absorption = calculate_absorption_changes_for_temperature(electronic_temperature)
    return PHOT, change_absorption

def calculate_total_absorbed_fluence(incident_fluence: float, no_sample_spectrum: np.ndarray) -> float:
    """Calculate fluence absorbed in sample, assuming linear sample response
    """
    sample_transmission = np.exp(-1*SSRL_ABSORPTION_COPD)
    transmitted_xrays = no_sample_spectrum*sample_transmission
    absorbed_xrays = no_sample_spectrum-transmitted_xrays
    absorbed_fraction = np.sum(absorbed_xrays)/np.sum(no_sample_spectrum)
    absorbed_fluence = incident_fluence*absorbed_fraction
    return absorbed_fluence

def calculate_pulse_averaged_valence_fraction(pulse_length: float = 5.0, cascade_duration: float = 13.0) -> float:
    """Calculate pulse-averaged fraction of deposited X-ray energy that
    is stored as valence excitations within 2 eV of the Fermi level, using
    LK30 model

    Returns:
    --------
    averaged_response: float
        (between 0 and 1)
    """
    time = np.arange(-1000, 2000, 0.1)
    pulse = (time > 0) & (time <= pulse_length)    # flat top pulse
    cascade_time = np.arange(-400, 400, 0.1)
    cascade = cascade_time/cascade_duration
    cascade[cascade_time < 0] = 0
    cascade[cascade_time >= cascade_duration] = 1
    response = np.convolve(pulse, cascade, mode='same')/np.sum(pulse)
    averaged_response = np.sum(response*pulse)/np.sum(pulse)
    plt.figure()
    plt.plot(time, pulse, label='pulse')
    plt.plot(cascade_time, cascade, label='cascade')
    plt.plot(time, response, label='response')
    plt.legend()
    return averaged_response

def calculate_electronic_temperature(averaged_valence_fluence: float) -> float:
    """Calculate sample elctronic temperature for a given absorbed X-ray fluence
    """
    THICKNESS = 49.5E-9
    FLUENCECONVERSION = 10    # mJ/cm^2 to J/m^2
    T0 = 293     # room temperature in K
    temperature = np.sqrt(2*FLUENCECONVERSION*averaged_valence_fluence/(THICKNESS*GAMMA)+T0)
    return temperature

def calculate_absorption_changes_for_temperature(electronic_temperature=10000):
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

def testing(electronic_temperature=7500, no_sam_spec=DEFAULT_NO_SAM_SPEC):
    emission = get_emission()
    emission['y'] = np.interp(PHOT, emission['x'], emission['y'], right=emission['y'][0])-0.2
    emission['x'] = PHOT
    change_absorption = calculate_absorption_changes_for_temperature(electronic_temperature)
    change_absorption = np.interp(PHOT, np.linspace(-5+CORE_2_FERMI, 5+CORE_2_FERMI, 1000), change_absorption)
    plt.figure() 
    plt.plot(PHOT, SSRL_ABSORPTION_COPD)
    nonlinear_absorption = SSRL_ABSORPTION_COPD-change_absorption
    nonlinear_absorption = nonlinear_absorption-SSRL_RESONANT_ABSORPTION*0.1
    nonlinear_absorption = nonlinear_absorption-emission['y']*0.4
    plt.plot(PHOT, nonlinear_absorption)
    plt.plot(PHOT, emission['y'])
    linear_sam_spec = no_sam_spec*np.exp(-1*SSRL_ABSORPTION_COPD)
    sam_spec = no_sam_spec*np.exp(-1*nonlinear_absorption)
    nonlinear_spec = sam_spec-linear_sam_spec
    f, axs = plt.subplots(1, 2, sharex=True, sharey=True)
    axs[0].plot(PHOT, no_sam_spec)
    axs[0].plot(PHOT, 5*nonlinear_spec)
    axs[1].plot(PHOT, no_sam_spec)
    axs[1].plot(PHOT, SHORT_DATA['99']['sum_intact']['exc_sam_spec']*5)
    axs[0].set_xlim((770, 785))
    place_vlines(axs[0])
    place_vlines(axs[1])

def get_emission():
    data = np.genfromtxt("data/measuredEmissionPoints3.txt")
    emission = {"x": data[:, 0]-EMISSION_SHIFT, "y": data[:, 1]}
    return emission

def place_vlines(ax):
    vline_loc_list = [774.5, 776.5, 778]
    for vline_loc in vline_loc_list:
        ax.axvline(vline_loc, linestyle='--', color='k')