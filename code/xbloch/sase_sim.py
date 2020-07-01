"""Simulate SASE pulses according to the method described in
T. Pfeiffer et al., "Partial-coherence method to model experimental
free-electron laser pulse statistics" (2010)
"""

import numpy as np
import matplotlib.pyplot as plt

from xbloch import phot_fft_utils

HBAR = 6.582E-1 # Planck's constant (eV*fs) (from Wikipedia)
FWHM2SIGMA = 1.0/2.3548 # Gaussian conversion factor (from Wolfram Mathworld)

# time points over which to simulate gaussian envelope SASE pulses (fs):
TIMES = np.linspace(-50, 50, int(1E4))

def simulate_gaussian(pulse_duration=5.0, E0=777.0, bw=4.0, pulse_fluence=1.0):
    """Simulate a SASE pulse with Gaussian spectral and temporal envelopes

    Parameters
    ----------
    pulse_duration: float
        pulse duration in fs
    E0: float
        central photon energy in eV
    bw: float
        FWHM pulse bandwidth in eV
    pulse_fluence: float
        integrated fluence of pulse in J/cm^2

    Returns
    -------
    array: time points of simulated pulse
    array: complex electric field of simulated pulse
    """
    duration_sigma = pulse_duration*FWHM2SIGMA
    bw_sigma = bw*FWHM2SIGMA
    E = phot_fft_utils.convert_times_to_phots(TIMES)
    frequency_envelope = np.exp(-(E-E0)**2/(2*bw_sigma**2))
    frequency_envelope = frequency_envelope/np.sum(frequency_envelope)
    temporal_envelope = np.exp(-1*(TIMES)**2/(2*duration_sigma**2))
    temporal_envelope = temporal_envelope*len(TIMES)/np.sum(temporal_envelope)
    t, t_y = _simulate(E, frequency_envelope, TIMES, temporal_envelope, pulse_fluence)
    return t, t_y

def _simulate(E, E_intensity_envelope, t, t_intensity_envelope, pulse_fluence=1):
    """Simulate a SASE pulse with input spectral and temporal envelopes
    """
    spectral_amplitude_envelope = np.sqrt(E_intensity_envelope)
    spectral_phases = np.random.uniform(-np.pi, np.pi, len(E))
    spectrum = spectral_amplitude_envelope*np.exp(spectral_phases*1j)
    t = phot_fft_utils.convert_phots_to_times(E)
    t_y = phot_fft_utils.convert_phot_signal_to_time_signal(spectrum)
    t_amplitude_envelope = np.sqrt(t_intensity_envelope)   
    t_y = t_y*t_amplitude_envelope
    t_y = _normalize_pulse(t, t_y, pulse_fluence)
    return t, t_y

def _normalize_pulse(t, t_y, pulse_fluence=1000):
    """Normalize pulse to specified fluence
    """
    integral = np.trapz(np.abs(t_y)**2, x=t)
    t_y = t_y/integral
    spacing = t[1]-t[0]
    t_y = t_y*np.sqrt(pulse_energy/spacing)
    return t_y

def test_gauss_simulations():
    """plot average pulse envelopes in time and photon energy domains
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
    axs[0].plot(t, np.mean(np.abs(t_y_array)**2, axis=0))
    axs[1].plot(E, np.mean(np.abs(E_y_array)**2, axis=0))
    pulse_energies = np.trapz(np.abs(t_y_array)**2, dx=(t[2]-t[1]), axis=1)
    plt.figure()
    plt.hist(pulse_energies)
    plt.axvline(np.mean(pulse_energies), color='k', linestyle='--')
    _, axs = plt.subplots(1, 2)
    for i in range(4):
        t_norm = np.amax(np.abs(t_y_array[i]**2))
        axs[0].plot(t, np.abs(t_y_array[i]**2)/t_norm+i)
        E_norm = np.amax(np.abs(E_y_array[i]**2))
        axs[1].plot(E, np.abs(E_y_array[i]**2)/E_norm+i)