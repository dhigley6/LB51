"""Simulate SASE pulses according to the method described in
T. Pfeiffer et al., "Partial-coherence method to model experimental
free-electron laser pulse statistics" (2010)
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple

from LB51.xbloch import phot_fft_utils

FWHM2SIGMA = 1.0 / 2.3548  # Gaussian conversion factor (from Wolfram Mathworld)
# time points over which to simulate gaussian envelope SASE pulses (fs):
TIMES = np.linspace(-25, 50, int(10e4))
FIELD_1E15 = 8.68e10  # Electric field strength in V/m that corresponds to
# 10^15 W/cm^2 or 1 J/cm^2/fs

np.random.seed(42)   # if we want reproducible r


def simulate_gaussian(
    pulse_duration: float = 5.0,
    E0: float = 777.0,
    bw: float = 4.0,
    pulse_fluence: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Simulate a SASE pulse with Gaussian spectral and temporal envelopes

    Parameters
    ----------
    pulse_duration: float
        FWHM pulse duration in fs
    E0: float
        central photon energy in eV
    bw: float
        FWHM pulse bandwidth in eV
    pulse_fluence: float
        integrated fluence of pulse in J/cm^2

    Returns
    -------
    t: one-dimensional np.ndarray
        Time points of simulated pulse
    t_y: one-dimensional np.ndarray (complex)
        Complex electric field envelope of simulated pulse (V/m)
    """
    E = phot_fft_utils.convert_times_to_phots(TIMES)
    spectral_envelope = _calculate_envelope(E, bw, E0)
    temporal_envelope = _calculate_envelope(TIMES, pulse_duration, 0)
    t, t_y = _simulate(E, spectral_envelope, TIMES, temporal_envelope, pulse_fluence)
    return t, t_y


def _calculate_envelope(x: np.ndarray, fwhm: float, x0: float = 0) -> np.ndarray:
    """Calculate Gaussian envelope for a given fwhm and peak (x0)

    Parameters:
    -----------
    x: 1d np.ndarray
        locations at which to calculate envelope
    fwhm: float
        Full-width-at-half-maximum of envelope
    x0: float
        Peak of envelope

    Returns:
    --------
    envelope: 1d np.ndarray
        strength of envelope as a function of x
    """
    sigma = fwhm * FWHM2SIGMA
    envelope = np.exp(-((x - x0) ** 2) / (2 * sigma ** 2))
    return envelope


def _simulate(
    E: np.ndarray,
    E_intensity_envelope: np.ndarray,
    t: np.ndarray,
    t_intensity_envelope: np.ndarray,
    pulse_fluence: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Simulate a SASE pulse with input spectral and temporal envelopes

    Parameters:
    -----------
    E: one-dimensional np.ndarray
        Photon energies of X-ray pulse (eV)
    E_intensity_envelope: one-dimensional np.ndarray
        Spectral intensity envelope (average over different realizations)
        of SASE pulse
    t: one-dimensional np.ndarray
        Times of X-ray pulse (fs)
    t_intensity_envelope: one-dimensional np.ndarray
        Temporal intensity envelope (average over different realizations)
    pulse_fluence: float
        Fluence of pulse (J/cm^2)

    Returns
    -------
    t: 1d np.ndarray
        Time points of simulated pulse
    t_y: 1d np.ndarray (complex)
        Complex electric field envelope of simulated pulse (V/m)
    """
    spectral_phases = np.random.uniform(-np.pi, np.pi, len(E))
    spectrum = np.sqrt(E_intensity_envelope) * np.exp(spectral_phases * 1j)
    t_y = phot_fft_utils.convert_phot_signal_to_time_signal(spectrum)
    t_y = t_y * np.sqrt(t_intensity_envelope)
    t_y = _normalize_pulse(t, t_y, pulse_fluence)
    return t, t_y


def _normalize_pulse(
    t: np.ndarray, t_y: np.ndarray, pulse_fluence: float = 1000
) -> np.ndarray:
    """Normalize pulse to specified fluence

    Parameters:
    -----------
    t: 1d np.ndarray
        Times of pulse
    t_y: 1d np.ndarray
        Unnormalized temporal envelope of pulse
    pulse_fluence: float
        Fluene of output normalized pulse (J/cm^2)

    Returns:
    --------
    t_y: 1d np.ndarray (complex)
        Complex electric field envelope of simulated pulse (V/m)
    """
    integral = np.trapz(np.abs(t_y) ** 2, x=t)
    # normalize amplitude by roots since intensity is amplitude squared
    t_y = t_y * np.sqrt(pulse_fluence) / np.sqrt(integral)
    t_y = t_y * FIELD_1E15
    return t_y


def demo_gauss_simulations():
    """Demonstrate Gaussian envelope SASE pulse simulations
    """
    t_y_list = []
    E_y_list = []
    for _ in range(1000):
        t, t_y = simulate_gaussian()
        t_y_list.append(t_y)
        E = phot_fft_utils.convert_times_to_phots(t)
        E_y = phot_fft_utils.convert_time_signal_to_phot_signal(t_y)
        E_y_list.append(E_y)
    t_y_array = np.vstack(t_y_list)
    E_y_array = np.vstack(E_y_list)
    _, axs = plt.subplots(2, 1)
    axs[0].plot(t, np.mean(np.abs(t_y_array) ** 2, axis=0))
    axs[1].plot(E, np.mean(np.abs(E_y_array) ** 2, axis=0))
    _, axs = plt.subplots(1, 2)
    for i in range(4):
        t_norm = np.amax(np.abs(t_y_array[i] ** 2))
        axs[0].plot(t, np.abs(t_y_array[i] ** 2) / t_norm + i)
        E_norm = np.amax(np.abs(E_y_array[i] ** 2))
        axs[1].plot(E, np.abs(E_y_array[i] ** 2) / E_norm + i)
