"""
@author: dhigley

Run X-ray Maxwell-Bloch 3-level simulations for SASE pulses
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle

from xbloch import xbloch2020
from xbloch import sase_sim

FIELD_1E15 = 8.68E10   # Electric field strength in V/m that corresponds to
# 10^15 W/cm^2 or 1 J/cm^2/fs

# Simulation parameters:
STRENGTHS = np.logspace(-3, 0, 10)     # Intensities of incident X-ray pulses (in 10^15 W/cm^2)
STRENGTHS = np.append(1E-8, STRENGTHS)

ABS_LIMITS = (777, 779)        # Region of absorption (in eV above 778 eV)
STIM_LIMITS = (775, 777)    # Region of stimulated inelastic scattering (in eV above 778 eV)

def simulate_sase_series():
    """Simulate and save series of SASE X-ray pulses interacting with 3-level atoms
    """
    sim_results = []
    times, E_in = sase_sim.simulate_gaussian()
    plt.figure()
    plt.plot(times, np.abs(E_in)**2)
    for strength in STRENGTHS:
        sim_result = run_sase_sim(times, E_in, strength)
        sim_results.append(sim_result)
        print(f'Completed {str(strength)}')
    data = {'strengths': STRENGTHS,
            'sim_results': sim_results}
    with open('xbloch/results/sase.pickle', 'wb') as f:
        pickle.dump(data, f)

def load_sase_series():
    with open('xbloch/results/sase.pickle', 'rb') as f:
        data = pickle.load(f)
    return data

def run_sase_sim(times, E_in, strength=1E-3, duration=5):
    system = xbloch2020.make_model_system()
    E_in = FIELD_1E15*np.sqrt(strength)*E_in
    sim_result = system.run_simulation(times, E_in)
    return sim_result

def calculate_stim_efficiencies():
    data = load_sase_series()
    strengths = data['strengths']
    sim_results = data['sim_results']
    phot0 = sim_results[0].phot_results_    # lowest fluence result in photon energy domain
    linear_difference = (np.abs(phot0['E_out'])**2-np.abs(phot0['E_in'])**2)/strengths[0]
    abs_region = (phot0['phots'] > ABS_LIMITS[0]) & (phot0['phots'] < ABS_LIMITS[1])
    abs_strength = -1*np.trapz(linear_difference[abs_region])
    stim_efficiencies = []
    for i, sim_result in enumerate(sim_results):
        phot_result = sim_result.phot_results_
        spec_difference = (np.abs(phot_result['E_out'])**2-np.abs(phot_result['E_in'])**2)/(strengths[i])
        plt.figure()
        plt.plot(phot_result['phots'], np.abs(phot_result['E_in'])**2)
        plt.plot(phot_result['phots'], spec_difference)
        stim_region = (phot_result['phots'] > STIM_LIMITS[0]) & (phot_result['phots'] < STIM_LIMITS[1])
        change_from_linear = spec_difference-linear_difference
        stim_strength = np.trapz(change_from_linear[stim_region])
        stim_efficiency = stim_strength/abs_strength
        stim_efficiencies.append(stim_efficiency)
    return strengths, stim_efficiencies