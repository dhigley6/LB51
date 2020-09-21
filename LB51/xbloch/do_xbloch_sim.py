"""
@author: dhigley

Run X-ray Maxwell-Bloch 3-level simulations for SASE and Gaussian pulses
"""

import numpy as np
import pickle
from typing import List, Dict, Tuple, Union
from joblib import Parallel, delayed

from LB51.xbloch import xbloch2020
from LB51.xbloch import enhancement_xbloch2020
from LB51.xbloch import sase_sim

# Simulation parameters:
STRENGTHS = np.logspace(-1, 1, 10)  # Fluences of incident X-ray pulses (in J/cm^2)
STRENGTHS = np.append(1e-8, STRENGTHS)

# Below two lines only for experimenting
# STRENGTHS = np.logspace(-1, 1, 3)  # Fluences of incident X-ray pulses (in J/cm^2)
# STRENGTHS = np.append(1e-8, STRENGTHS)

ABS_LIMITS = (777, 779)  # Region of absorption (in eV above 778 eV)
STIM_LIMITS = (
    775,
    777,
)  # Region of stimulated inelastic scattering (in eV above 778 eV)

# where to save simulation results:
SASE_RESULTS_FILE_START = "LB51/xbloch/results/multipulse_sase_"
GAUSS_RESULTS_FILE_START = "LB51/xbloch/results/gauss_"

N_PULSES_MANUSCRIPT = 20    # number of pulses to simulate for manuscript plots

def run_manuscript_simulations():
    """Run simulations to be used in manuscript
    """
    times_5fs = np.linspace(-25, 50, int(5e4))
    simulate_multipulse_sase_series(5.0, N_PULSES_MANUSCRIPT, times_5fs)
    print('Completed 5 fs simulations')
    times_25fs = np.linspace(-50, 100, int(5e4))
    simulate_multipulse_sase_series(25.0, N_PULSES_MANUSCRIPT, times_25fs)
    print('Completed 25 fs simulations')

def simulate_gaussian_case(duration: float = 0.5):
    """Simulate interaction of Gaussian pulses with 3-level-system vs fluence

    Results are saved in GAUSS_RESULTS_FILE as a dictionary with the
    same format as the dictionary returned by simulate_multipulse_series

    Parameters:
    -----------
    duration: float
        Duration of pulse to simulate (fs)
    """
    times = sase_sim.TIMES
    E_in_list = [sase_sim._normalize_pulse(times, gauss(times, 0, sigma=duration), 1)]
    summary_result = simulate_multipulse_series(times, E_in_list)
    results_file = GAUSS_RESULTS_FILE_START+str(duration)+'.pickle'
    with open(results_file, "wb") as f:
        pickle.dump(summary_result, f)


def gauss(
    x: np.ndarray, 
    t0: float, 
    sigma: float
) -> np.ndarray:
    """Calculate Gaussian pulse with t0 center and sigma width

    Parameters:
    -----------
    x: 1d np.ndarray
        Locations to calculate Gaussian intensity at
    t0: float
        Location of Gaussian peak
    sigma: float
        Width of Gaussian
    
    Returns:
    --------
    gaussian: 1d np.ndarray
        Gaussian intensities
    """
    gaussian = np.exp((-1 / 2) * ((x - t0) / sigma) ** 2)
    integral = np.trapz(np.abs(gaussian) ** 2, x=x)
    gaussian = gaussian / np.sqrt(integral)
    return gaussian


def simulate_multipulse_sase_series(duration: float = 5.0, n_pulses: int = 4, times: np.ndarray = sase_sim.TIMES):
    """Simulate interaction of SASE pulses with 3-level-system vs fluence

    Results are saved in SASE_RESULTS_FILE as a dictionary with the
    same format as the dictionary returned by simulate_multipulse_series

    Parameters:
    -----------
    n_pulses: int
        Number of pulses to simulate
    """
    E_in_list = [sase_sim.simulate_gaussian(duration, times=times)[1] for _ in range(n_pulses)]
    summary_result = simulate_multipulse_series(times, E_in_list)
    results_file = SASE_RESULTS_FILE_START+str(duration)+'.pickle'
    with open(results_file, "wb") as f:
        pickle.dump(summary_result, f)


def simulate_multipulse_series(
    times: np.ndarray,
    E_in_list: List[np.ndarray],
    enhanced: bool = False
) -> Dict[str, Union[List[float], np.ndarray]]:
    """Simulate interaction of pulses with 3-level system vs fluence

    Parameters:
    -----------
    times: 1d np.ndarray
        Times of pulses (fs)
    E_in_list: List[1d np.ndarray]
        Electric field strengths of pulses (V/m)
    enhanced: bool
        True if we want to use s&s enhancement factor

    Returns:
    --------
    summary_result: Dictionary
        'fluences' key: List[float]
            Fluences of simulations (J/cm^2)
        'stim_efficiencies' key: List[float]
            Efficiency of stimulated inelastic scattering in %
        'phot': 1d np.ndarray
            Photon energies of saved spectra (eV)
        'summed_incident_fields': 2d np.ndarray (complex)
            Incident spectral field strengths. Rows are different fluences
            and columns are different photon energies.
        'summed_transmitted_fields': 2d np.ndarray (complex)
            Transmitted spectral field strengths. Rows are different fluences
            and columns are different photon energies.
    """
    summed_incident_intensities_list = []
    summed_transmitted_intensities_list = []
    stim_efficiencies = []
    for i, strength in enumerate(STRENGTHS):
        sim_results = Parallel(n_jobs=-1)(
            delayed(_run_single_pulse_sim)(times, E_in, strength, enhanced) for E_in in E_in_list
        )
        (
            phot,
            summed_incident_intensities,
            summed_transmitted_intensities,
            intensity_difference,
        ) = _get_summed_result(sim_results, strength)
        if i == 0:
            # this is lowest fluence, ~linear case -> calculate linear
            # absorption strength
            linear_intensity_difference = intensity_difference
        stim_efficiency = _get_stim_efficiency(
            phot, intensity_difference, linear_intensity_difference
        )
        summed_incident_intensities_list.append(summed_incident_intensities)
        summed_transmitted_intensities_list.append(summed_transmitted_intensities)
        stim_efficiencies.append(stim_efficiency)
    summary_result = {
        "fluences": STRENGTHS,
        "stim_efficiencies": stim_efficiencies,
        "phot": sim_results[0]["phot"],
        "summed_incident_intensities": summed_incident_intensities_list,
        "summed_transmitted_intensities": summed_transmitted_intensities_list,
    }
    return summary_result


def _get_stim_efficiency(
    phot: np.ndarray,
    intensity_difference: np.ndarray,
    linear_intensity_difference: np.ndarray
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
    change_from_linear[change_from_linear < 0] = 0     # clip negative values to zero (shouldn't change result significantly for simulation)
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
    summed_incident_intensities: 1d np.ndarray
        Summed incident spectral intensities
    summed_transmitted_fields: 1d np.ndarray
        Summed transmitted spectral intensities
    intensity_difference: 1d np.ndarray
        Normalized intensity difference of transmitted and incident pulses
    """
    phot = sim_results[0]["phot"]  # photon energies
    I_phot_ins = [np.abs(sim_result["E_phot_in"]) ** 2 for sim_result in sim_results]
    summed_incident_intensities = np.sum(I_phot_ins, axis=0)
    I_phot_outs = [np.abs(sim_result["E_phot_out"]) ** 2 for sim_result in sim_results]
    summed_transmitted_intensities = np.sum(I_phot_outs, axis=0)
    intensity_difference = (
        summed_transmitted_intensities - summed_incident_intensities
    ) / fluence
    return (
        phot,
        summed_incident_intensities,
        summed_transmitted_intensities,
        intensity_difference,
    )


def _run_single_pulse_sim(
    times: np.ndarray, E_in: np.ndarray, strength: float = 1e-3, enhanced: bool = False
) -> Dict[str, np.ndarray]:
    """Run single pulse simulation

    Parameters:
    -----------
    times: np.ndarray
        times of simulation (fs)
    E_in: np.ndarray (complex)
        Incident electric field strengths (V/m)
    strength: float
        Fluence of incident X-ray pulse (J/cm^2)
    enhanced: bool
        If True, use s&s enhancement factor
    
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
    assert len(times) == len(E_in)
    if enhanced:
        system = enhancement_xbloch2020.make_model_system()
    else:
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


def load_multipulse_data(duration: float) -> Dict[str, Union[List[float], np.ndarray]]:
    """Load saved SASE simulation results

    Returns:
    --------
    data: Dictionary
        Format is the same as that returned by simulate_multipulse_series
    """
    results_file = SASE_RESULTS_FILE_START+str(duration)+'.pickle'
    with open(results_file, "rb") as f:
        data = pickle.load(f)
    return data


def load_gauss_data(duration: float) -> Dict[str, Union[List[float], np.ndarray]]:
    """Load saved Gaussian pulse simulation results

    Returns:
    --------
    data: Dictionary
        Format is the same as that returned by simulate_multipulse_series
    """
    results_file = GAUSS_RESULTS_FILE_START+str(duration)+'.pickle'
    with open(results_file, "rb") as f:
        data = pickle.load(f)
    return data
