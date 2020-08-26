"""Some utility functions for transforms between time and photon energy
"""

import numpy as np 

HBAR = 6.582E-1 # Planck's constant (eV*fs) (from Wikipedia)

def convert_times_to_phots(t, E_0=778):
    """Convert times to photon energies
    """
    sample_rate = t[2]-t[1]
    freqs = np.fft.fftshift(np.fft.fftfreq(len(t), sample_rate))
    phots = HBAR*2*np.pi*freqs+E_0
    return phots

def convert_time_signal_to_phot_signal(t_signal):
    """Convert time domain signal to photon energy domain
    """
    windowed_signal = np.hanning(len(t_signal))*t_signal
    phot_signal = np.fft.fftshift(np.fft.fft(windowed_signal))
    return phot_signal

def convert_phots_to_times(phots, t_0=0):
    """Convert photon energies to times
    """
    freqs = phots/(HBAR*2*np.pi)
    sample_rate = freqs[2]-freqs[1]
    t = np.fft.fftshift(np.fft.fftfreq(len(phots), sample_rate))+t_0
    return t

def convert_phot_signal_to_time_signal(phot_signal):
    """Convert photon energy domain signal to time domain signal
    """
    t_signal = np.fft.ifft(np.fft.ifftshift(phot_signal))
    return t_signal

def _test_time_phot_conversion():
    """currently not working
    """
    times = np.arange(1000)
    middle = np.mean(times)
    times = times-middle
    signal = np.exp(-(times)**2/(2*20**2))
    phot, phot_y = _convert_time_to_phot(times, signal)
    times_2, signal_2 = _convert_phot_to_time(phot, phot_y)
    plt.figure()
    plt.plot(times, signal)
    plt.plot(times_2, signal_2)