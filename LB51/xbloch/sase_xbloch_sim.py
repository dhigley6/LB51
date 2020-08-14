"""
@author: dhigley

Run X-ray Maxwell-Bloch 3-level simulations for SASE pulses
"""

import numpy as np
import pickle
from typing import List, Dict, Tuple, Union
from joblib import Parallel, delayed

from LB51.xbloch import xbloch2020
from LB51.xbloch import sase_sim

# Simulation parameters:
STRENGTHS = np.logspace(-3, 1, 3)  # Fluences of incident X-ray pulses (in J/cm^2)
STRENGTHS = np.append(1e-8, STRENGTHS)

ABS_LIMITS = (777, 779)  # Region of absorption (in eV above 778 eV)
STIM_LIMITS = (
    775,
    777,
)  # Region of stimulated inelastic scattering (in eV above 778 eV)

# where to save simulation results:
RESULTS_FILE = "LB51/xbloch/results/multipulse_sase.pickle"


def simulate_multipulse_sase_series(n_pulses: int = 4):
    """Simulate interaction of SASE pulses with 3-level system vs fluence

    Results are saved in RESULTS_FILE as a dictionary
        'fluences' key: list[float]
            Fluences of simulations (J/cm^2)
        'stim_efficiencies' key: list[float]
            Efficiency of stimulated inelastic scattering in %
        'phot': 1d np.ndarray
            Photon energies of saved spectra (eV)
        'summed_incident_fields': 2d np.ndarray (complex)
            Incident spectral field strengths. Rows are different fluences
            and columns are different photon energies.
        'summed_transmitted_fields': 2d np.ndarray (complex)
            Transmitted spectral field strengths. Rows are different fluences
            and columns are different photon energies.

    Parameters:
    -----------
    n_pulses: int
        The number of different SASE pulse realizations to simulate for each
        fluence.
    """
    times = sase_sim.TIMES  # use default times
    E_in_list = [sase_sim.simulate_gaussian()[1] for _ in range(n_pulses)]
    summed_incident_fields_list = []
    summed_transmitted_fields_list = []
    stim_efficiencies = []
    for i, strength in enumerate(STRENGTHS):
        sim_results = Parallel(n_jobs=-1)(
            delayed(_run_sase_sim)(times, E_in, strength) for E_in in E_in_list
        )
        (
            phot,
            summed_incident_fields,
            summed_transmitted_fields,
            intensity_difference,
        ) = _get_summed_result(sim_results, strength)
        if i == 0:
            # this is lowest fluence, ~linear case -> calculate linear
            # absorption strength
            linear_intensity_difference = intensity_difference
        stim_efficiency = _get_stim_efficiency(
            phot, intensity_difference, linear_intensity_difference
        )
        summed_incident_fields_list.append(summed_incident_fields)
        summed_transmitted_fields_list.append(summed_transmitted_fields)
        stim_efficiencies.append(stim_efficiency)
    summary_result = {
        "fluences": STRENGTHS,
        "stim_efficiencies": stim_efficiencies,
        "phot": sim_results[0]["phot"],
        "summed_incident_fields": summed_incident_fields_list,
        "summed_transmitted_fields": summed_transmitted_fields_list,
    }
    with open(RESULTS_FILE, "wb") as f:
        pickle.dump(summary_result, f)


def _get_stim_efficiency(
    phot: np.ndarray,
    intensity_difference: np.ndarray,
    linear_intensity_difference: np.ndarray,
) -> float:
    """Return stimulated efficiency in %

    Parameters:
    -----------
    phot: 1d np.ndarray
        Photon energies (eV)
    intensity_difference: 1d np.ndarray
        Transmitted X-ray intensity minus incident X-ray intensity
        (arbitrary units)
    liner_intensity_difference: 1d np.ndarray
        Transmitted X-ray intensity minus incident X-ray intensity
        for low fluence ~linear case (arbitrary units)

    Returns:
    --------
    stim_efficiency: float
        Efficiency of stimulated inelastic scattering (%)
    """
    abs_region = (phot > ABS_LIMITS[0]) & (phot < ABS_LIMITS[1])
    abs_strength = -1 * np.trapz(linear_intensity_difference[abs_region])
    change_from_linear = intensity_difference - linear_intensity_difference
    stim_region = (phot > STIM_LIMITS[0]) & (phot < STIM_LIMITS[1])
    stim_strength = np.trapz(change_from_linear[stim_region])
    stim_efficiency = 100 * stim_strength / abs_strength
    return stim_efficiency


def _get_summed_result(
    sim_results: List[Dict[str, np.ndarray]], fluence: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Get summed spectra from simulations

    Parameters:
    -----------
    sim_results: List[Dict[str, np.ndarray]]
        'phot' key: List[float]
            Photon energies (eV)
        'E_phot_in': 1d np.ndarray
            Incident spectral field strength
        'E_phot_out': 1d np.ndarray
            Transmitted spectral field strength
        
    Returns:
    --------
    phot: 1d np.ndarray
        Photon energies (eV)
    summed_incident_fields: 1d np.ndarray (complex)
        Summed incident spectral fields
    summed_transmitted_fields: 1d np.ndarray (complex)
        Summed transmitted spectral fields
    intensity_difference: 1d np.ndarray
        Normalized intensity difference of transmitted and incident fields
    """
    phot = sim_results[0]["phot"]  # photon energies
    E_phot_ins = [sim_result["E_phot_in"] for sim_result in sim_results]
    summed_incident_fields = np.sum(E_phot_ins, axis=0)
    E_phot_outs = [sim_result["E_phot_out"] for sim_result in sim_results]
    summed_transmitted_fields = np.sum(E_phot_outs, axis=0)
    intensity_difference = (
        np.abs(summed_transmitted_fields) ** 2 - np.abs(summed_incident_fields) ** 2
    ) / fluence
    return phot, summed_incident_fields, summed_transmitted_fields, intensity_difference


def _run_sase_sim(
    times: np.ndarray, E_in: np.ndarray, strength: float = 1e-3
) -> Dict[str, np.ndarray]:
    """Run single pulse SASE simulation

    Parameters:
    -----------
    times: np.ndarray
        times of simulation (fs)
    E_in: np.ndarray (complex)
        Incident electric field strengths (V/m)
    strength: float
        Fluence of incident X-ray pulse (J/cm^2)
    
    Returns:
    --------
    simplified_result: Dict[str, np.ndarray]
        'phot' key: np.ndarray
            Photon energies (eV)
        'E_phot_in' key: np.ndarray (complex)
            Spectral field strength of incident elecric field
        'E_phot_out' key: np.ndarray (complex)
            Spectral field strength of transmitted electric field
    """
    system = xbloch2020.make_model_system()
    E_in = E_in * np.sqrt(strength)
    sim_result = system.run_simulation(times, E_in)
    simplified_result = {
        "phot": sim_result.phot_results_["phots"],
        "E_phot_in": sim_result.phot_results_["E_in"],
        "E_phot_out": sim_result.phot_results_["E_out"],
    }
    print(f"Completed simulation with strength {strength}")
    return simplified_result


def load_multipulse_data() -> Dict[str, Union[List[float], np.ndarray]]:
    """Load saved simulation results

    Returns:
    --------
    data: dictionary
        'fluences' key: list[float]
            Fluences of simulations (J/cm^2)
        'stim_efficiencies' key: list[float]
            Efficiency of stimulated inelastic scattering in %
        'phot': 1d np.ndarray
            Photon energies of saved spectra (eV)
        'summed_incident_fields': 2d np.ndarray (complex)
            Incident field strengths (V/m). Rows are different fluences
            and columns are different photon energies.
        'summed_transmitted_fields': 2d np.ndarray (complex)
            Transmitted field strengths (V/m). Rows are different fluences
            and columns are different photon energies.
    """
    with open(RESULTS_FILE, "rb") as f:
        data = pickle.load(f)
    return data
