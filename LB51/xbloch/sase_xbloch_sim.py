"""
@author: dhigley

Run X-ray Maxwell-Bloch 3-level simulations for SASE pulses
"""

import numpy as np
import pickle
from joblib import Parallel, delayed

from LB51.xbloch import xbloch2020
from LB51.xbloch import sase_sim

# Simulation parameters:
STRENGTHS = np.logspace(-3, 1, 3)  # Fluences of incident X-ray pulses (in 10^15 W/cm^2)
STRENGTHS = np.append(1e-8, STRENGTHS)

ABS_LIMITS = (777, 779)  # Region of absorption (in eV above 778 eV)
STIM_LIMITS = (
    775,
    777,
)  # Region of stimulated inelastic scattering (in eV above 778 eV)


def simulate_multipulse_sase_series(n_pulses=2):
    """Simulate and save series of n_pulses SASE pulses interacting with 3-level atoms
    """
    results = []
    for _ in range(n_pulses):
        result = _simulate_sase_series()
        results.append(result)
        strengths, stim_efficiencies = calculate_multipulse_stim(results)
    with open("LB51/xbloch/results/multipulse_sase.pickle", "wb") as f:
        pickle.dump(results, f)


def load_multipulse_sase_series():
    with open("LB51/xbloch/results/multipulse_sase.pickle", "rb") as f:
        data = pickle.load(f)
    return data


def _simulate_sase_series():
    """Simulate increasing intensity series of SASE X-ray pulses interacting with 3-level atoms
    """
    sim_results = []
    times, E_in = sase_sim.simulate_gaussian()
    sim_results = Parallel(n_jobs=-1)(
        delayed(_run_sase_sim)(times, E_in, strength) for strength in STRENGTHS
    )
    data = {"strengths": STRENGTHS, "sim_results": sim_results}
    return data


def _run_sase_sim(times, E_in, strength=1e-3):
    """Run single pulse SASE simulation
    """
    system = xbloch2020.make_model_system()
    E_in = E_in * np.sqrt(strength)
    sim_result = system.run_simulation(times, E_in)
    print(f"Completed {strength}")
    return sim_result


def calculate_stim_single_pulse(data):
    strengths = data["strengths"]
    sim_results = data["sim_results"]
    phot0 = sim_results[
        0
    ].phot_results_  # lowest fluence result in photon energy domain
    linear_difference = (
        np.abs(phot0["E_out"]) ** 2 - np.abs(phot0["E_in"]) ** 2
    ) / strengths[0]
    abs_region = (phot0["phots"] > ABS_LIMITS[0]) & (phot0["phots"] < ABS_LIMITS[1])
    abs_strength = -1 * np.trapz(linear_difference[abs_region])
    stim_efficiencies = []
    for i, sim_result in enumerate(sim_results):
        phot_result = sim_result.phot_results_
        spec_difference = (
            np.abs(phot_result["E_out"]) ** 2 - np.abs(phot_result["E_in"]) ** 2
        ) / (strengths[i])
        stim_region = (phot_result["phots"] > STIM_LIMITS[0]) & (
            phot_result["phots"] < STIM_LIMITS[1]
        )
        change_from_linear = spec_difference - linear_difference
        stim_strength = np.trapz(change_from_linear[stim_region])
        stim_efficiency = stim_strength / abs_strength
        stim_efficiencies.append(stim_efficiency)
    return strengths, stim_efficiencies


def calculate_multipulse_stim(data_set):
    #data_set = load_multipulse_sase_series()
    stim_efficiencies_list = []
    for data in data_set:
        strengths, stim_efficiencies = calculate_stim_single_pulse(data)
        stim_efficiencies_list.append(stim_efficiencies)
    stim_efficiencies = np.mean(stim_efficiencies_list, axis=0)
    return strengths, stim_efficiencies

def load_multipulse_stim():
    data = 
    with open("LB51/xbloch/results/multipulse_sase.pickle", "rb") as f:
        data = pickle.load(f)