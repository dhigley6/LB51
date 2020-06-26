"""Simulate SASE pulses according to the method described in
T. Pfeiffer et al., "Partial-coherence method to model experimental
free-electron laser pulse statistics" (2010)
"""

import numpy as np
import matplotlib.pyplot as plt

HBAR = 4.1357 # Planck's constant (eV*fs) (from Wikipedia)
FWHM2SIGMA = 1.0/2.3548 # Gaussian conversion factor (from Wolfram Mathworld)

def simulate_gaussian(pulse_duration=5, E0=778, bw=8, pulse_energy=1):
    """Simulate a SASE pulse with Gaussian spectral and temporal envelopes
    Photon energy units: eV
    time units: fs
    Pulse energy units: J
    Intensity units: J/cm^2/s (for time domain)
    Intensity units: J/cm^2/eV (for frequency domain)
    """
    duration_sigma = pulse_duration*FWHM2SIGMA
    times = np.linspace(-100, 100, int(1E4))
    E, _ = _convert_time_to_phot(times, times)
    intensity_envelope = np.exp(-(E-E0)**2/(2*bw**2))
    intensity_envelope = intensity_envelope/np.sum(intensity_envelope)
    t, _ = _convert_phot_to_time(E, E)
    temporal_window = np.exp(-1*(t)**2/(2*duration_sigma**2))
    temporal_window = temporal_window*len(t)/np.sum(temporal_window)
    t, t_y = simulate(E, intensity_envelope, t, temporal_window, pulse_energy)
    return t, t_y

def simulate(E, E_intensity_envelope, t, t_intensity_envelope, pulse_energy=1):
    spectral_amplitude_envelope = np.sqrt(E_intensity_envelope)
    spectral_phases = np.random.uniform(-np.pi, np.pi, len(E))
    spectrum = spectral_amplitude_envelope*np.exp(spectral_phases*1j)
    t, t_y = _convert_phot_to_time(E, spectrum)
    t_y = t_y*np.sqrt(len(t_y))  # normalize so that sum of intensities equals 1
    t_amplitude_envelope = np.sqrt(t_intensity_envelope)   
    t_y = t_y*t_amplitude_envelope
    t_y = _normalize_pulse(t, t_y, pulse_energy)
    return t, t_y

def _plot_pulse(t, t_y):
    E, E_y = _convert_time_to_phot(t, t_y)
    _, axs = plt.subplots(2, 1)
    axs[0].plot(t, np.abs(t_y))
    axs[1].plot(E, np.abs(E_y))

def _normalize_pulse(t, t_y, pulse_energy=1000):
    """Normalize pulse so that average integrated intensity equals pulse energy
    (assumes that input average summed intensity samples equals 1)
    """
    spacing = t[1]-t[0]
    t_y = t_y*np.sqrt(pulse_energy/spacing)
    return t_y


def _convert_phot_to_time(E, y, t_0=0):
    """Convert a photon energy-domain signal to the time domain
    """
    freqs = E/(HBAR*2*np.pi)
    sample_rate = freqs[2]-freqs[1]
    t = np.fft.fftshift(np.fft.fftfreq(len(E), sample_rate))
    t_y = np.fft.ifft(np.fft.ifftshift(y))
    return t+t_0, t_y


def _convert_time_to_phot(t, y, E_0=778):
    """Convert a time-domain signal to the photon energy domain
    """
    sample_rate = t[2]-t[1]
    freqs = np.fft.fftshift(np.fft.fftfreq(len(t), sample_rate))
    phot = HBAR*2*np.pi*freqs
    phot_y = np.fft.fftshift(np.fft.fft(y))
    return phot+E_0, phot_y

def _test_time_phot_conversion():
    times = np.arange(1000)
    middle = np.mean(times)
    times = times-middle
    signal = np.exp(-(times)**2/(2*20**2))
    phot, phot_y = _convert_time_to_phot(times, signal)
    times_2, signal_2 = _convert_phot_to_time(phot, phot_y)
    plt.figure()
    plt.plot(times, signal)
    plt.plot(times_2, signal_2)

def test_gauss_simulations():
    # plot average pulse envelope in time and photon energy domains
    # plot distribution of integrated pulse energies
    # plot 4 pulse envelopes in time domain
    # plot 4 spectra
    t_y_list = []
    E_y_list = []
    for _ in range(1000):
        t, t_y = simulate_gaussian()
        t_y_list.append(t_y)
        E, E_y = _convert_time_to_phot(t, t_y)
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