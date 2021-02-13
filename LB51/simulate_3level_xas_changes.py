"""Simulate XAS changes from 3-level dynamics
"""

import numpy as np 
import matplotlib.pyplot as plt 

from LB51.xbloch import do_xbloch_sim
from LB51 import LB51_get_cal_data
from LB51 import do_valence_excitation_simulation

SHORT_DATA = LB51_get_cal_data.get_short_pulse_data()
PHOT = SHORT_DATA['99']['sum_intact']['phot']
SSRL_ABSORPTION_COPD = SHORT_DATA["99"]["sum_intact"]["ssrl_absorption"]
SSRL_RESONANT_ABSORPTION = SSRL_ABSORPTION_COPD-SSRL_ABSORPTION_COPD[0]
DEFAULT_NO_SAM_SPEC = SHORT_DATA['99']['sum_intact']['no_sam_spec']

ABSORPTION_RANGE = (777.5, 778.5)

RIXS_RANGE = (775.25, 776.75)

EMISSION_SHIFT = 0.4

def testing3():
    emission = get_emission()
    plt.figure()
    plt.plot(emission['x'], emission['y'])
    plt.plot(PHOT, SSRL_ABSORPTION_COPD)


def testing2():
    PHOT, valence_xas_changes = do_valence_excitation_simulation.calculate_absorption_changes(1.5E3)
    three_level_xas_changes = calculate_absorption_changes(5.0, 1.5)
    total_xas_changes = valence_xas_changes+three_level_xas_changes
    plt.figure()
    plt.plot(PHOT, valence_xas_changes)
    plt.plot(PHOT, total_xas_changes)
    plt.plot(PHOT, three_level_xas_changes)
    #plt.plot(PHOT, SSRL_ABSORPTION_COPD)
    no_sam_spec = DEFAULT_NO_SAM_SPEC
    linear_sam_spec = no_sam_spec*np.exp(-1*SSRL_ABSORPTION_COPD)
    sam_spec = no_sam_spec*np.exp(total_xas_changes-SSRL_ABSORPTION_COPD)
    nonlinear_spec = sam_spec-linear_sam_spec
    f, axs = plt.subplots(1, 2, sharex=True, sharey=True)
    axs[0].plot(PHOT, no_sam_spec)
    axs[0].plot(PHOT, 5*nonlinear_spec)
    axs[1].plot(PHOT, no_sam_spec)
    axs[1].plot(PHOT, SHORT_DATA['99']['sum_intact']['exc_sam_spec']*5)
    axs[0].set_xlim((770, 785))
    #place_vlines(axs[0])
    #place_vlines(axs[1])

def calculate_absorption_reduction_at_fluence(pulse_duration=5.0, fluence=1.5):
    fluences, absorption_magnitudes = calculate_absorption_magnitudes()
    absorption_magnitude = np.interp(fluence, fluences, absorption_magnitudes)
    absorption_reduction = (absorption_magnitude-absorption_magnitudes[0])/absorption_magnitudes[0]
    return absorption_reduction

def calculate_absorption_magnitudes(pulse_duration=5.0):
    sim_results = do_xbloch_sim.load_multipulse_data(5.0)
    phot = sim_results['phot']
    absorption_magnitudes = []
    for i in range(len(sim_results['fluences'])):
        absorption = -np.log(sim_results['summed_transmitted_intensities'][i]/sim_results['summed_incident_intensities'][i])
        in_range = (phot > ABSORPTION_RANGE[0]) & (phot < ABSORPTION_RANGE[1])
        integrated_absorption = np.trapz(absorption[in_range])
        absorption_magnitudes.append(integrated_absorption)
    plt.figure()
    plt.plot(sim_results['fluences'], absorption_magnitudes)
    return sim_results['fluences'], absorption_magnitudes

def calculate_rixs_gain_at_fluence(pulse_duration=5.0, fluence=1000.0):
    fluences, rixs_magnitudes = calculate_rixs_magnitudes()
    rixs_magnitude = np.interp(fluence, fluences, rixs_magnitudes)
    fluences, absorption_magnitudes = calculate_absorption_magnitudes(pulse_duration)
    rixs_gain = rixs_magnitude/absorption_magnitudes[0]
    return rixs_gain

def calculate_rixs_magnitudes(pulse_duration=5.0):
    sim_results = do_xbloch_sim.load_multipulse_data(5.0)
    phot = sim_results['phot']
    rixs_magnitudes = []
    plt.figure()
    for i in range(len(sim_results['fluences'])):
        absorption = -np.log(sim_results['summed_transmitted_intensities'][i]/sim_results['summed_incident_intensities'][i])
        plt.plot(phot, absorption)
        in_range = (phot > RIXS_RANGE[0]) & (phot < RIXS_RANGE[1])
        integrated_rixs = np.trapz(absorption[in_range])
        rixs_magnitudes.append(integrated_rixs)
    rixs_magnitudes = rixs_magnitudes-rixs_magnitudes[0]
    rixs_magnitudes = np.abs(rixs_magnitudes)
    plt.figure()
    plt.plot(sim_results['fluences'], rixs_magnitudes)
    return sim_results['fluences'], rixs_magnitudes

def calculate_absorption_changes(pulse_duration=5.0, fluence=1000):
    absorption_reduction = calculate_absorption_reduction_at_fluence(pulse_duration, fluence)
    absorption_changes = -1*absorption_reduction*SSRL_RESONANT_ABSORPTION
    emission = get_emission()
    emission['y'] = np.interp(PHOT, emission['x'], emission['y'], right=emission['y'][0])
    emission['y'] = emission['y']-emission['y'][0]
    emission['x'] = PHOT
    in_range = (PHOT > 770) & (PHOT < 785)
    absorption_integral = np.sum(SSRL_RESONANT_ABSORPTION[in_range])
    in_range = (PHOT > 773.5) & (PHOT < 780)
    emission_integral = np.sum(emission['y'][in_range])
    rixs_gain = calculate_rixs_gain_at_fluence(pulse_duration, fluence)
    print('RIXS gain:')
    print(rixs_gain)
    print(rixs_gain*absorption_integral/emission_integral)
    absorption_changes = absorption_changes+10*emission['y']*rixs_gain*absorption_integral/emission_integral
    #absorption_changes = absorption_changes+emission['y']*0.4
    return absorption_changes

def get_emission():
    data = np.genfromtxt("data/measuredEmissionPoints3.txt")
    emission = {"x": data[:, 0]-EMISSION_SHIFT, "y": data[:, 1]}
    return emission


def testing():
    sim_results_5fs = do_xbloch_sim.load_multipulse_data(5.0)
    plt.figure()
    inds_to_plot = [0, -7, -5, -3]
    for i in inds_to_plot:
        absorption = -np.log(sim_results_5fs['summed_transmitted_intensities'][i]/sim_results_5fs['summed_incident_intensities'][i])
        plt.plot(sim_results_5fs['phot'], absorption)