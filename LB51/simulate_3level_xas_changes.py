"""Simulate XAS changes from 3-level dynamics
"""

import numpy as np 
import matplotlib.pyplot as plt 

from LB51.xbloch import do_xbloch_sim
from LB51 import LB51_get_cal_data

SHORT_DATA = LB51_get_cal_data.get_short_pulse_data()
PHOT = SHORT_DATA['99']['sum_intact']['phot']
SSRL_ABSORPTION_COPD = SHORT_DATA["99"]["sum_intact"]["ssrl_absorption"]
SSRL_RESONANT_ABSORPTION = SSRL_ABSORPTION_COPD-SSRL_ABSORPTION_COPD[0]
DEFAULT_NO_SAM_SPEC = SHORT_DATA['99']['sum_intact']['no_sam_spec']

ABSORPTION_RANGE = (777.5, 778.5)

RIXS_RANGE = (775.25, 776.75)

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

def calculate_rixs_magnitudes(pulse_duration=5.0):
    sim_results = do_xbloch_sim.load_multipulse_data(5.0)
    phot = sim_results['phot']
    rixs_magnitudes = []
    for i in range(len(sim_results['fluences'])):
        absorption = -np.log(sim_results['summed_transmitted_intensities'][i]/sim_results['summed_incident_intensities'][i])
        in_range = (phot > RIXS_RANGE[0]) & (phot < RIXS_RANGE[1])
        integrated_rixs = np.trapz(absorption[in_range])
        rixs_magnitudes.append(integrated_rixs)
    plt.figure()
    plt.plot(sim_results['fluences'], rixs_magnitudes)
    return sim_results['fluences'], rixs_magnitudes

def calculate_absorption_changes(pulse_duration=5.0, fluence=1000):
    pass


def testing():
    sim_results_5fs = do_xbloch_sim.load_multipulse_data(5.0)
    plt.figure()
    inds_to_plot = [0, -7, -5, -3]
    for i in inds_to_plot:
        absorption = -np.log(sim_results_5fs['summed_transmitted_intensities'][i]/sim_results_5fs['summed_incident_intensities'][i])
        plt.plot(sim_results_5fs['phot'], absorption)