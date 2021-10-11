"""Very approximate expectations for change in Co L3 absorption from ground
state at different states
"""

import numpy as np
import matplotlib.pyplot as plt 

from LB51 import LB51_get_cal_data
from LB51.manuscript_plots import set_plot_params

set_plot_params.init_paper_small()
plt.rcParams["axes.linewidth"] = 0.25


EMISSION_SHIFT = 0.4  # photon energy shift from reference data

FERMI_ENERGY = 777.5

def make_plots():
    make_linear_plot()
    make_core_excited_plot()
    make_temp_change_plot()

def make_linear_plot():
    long_data = LB51_get_cal_data.get_long_pulse_data()
    short_data = LB51_get_cal_data.get_short_pulse_data()
    #binned_603 = LB51_get_cal_data.bin_burst_data(long_data["603"]["total"], 603)
    #binned_99 = LB51_get_cal_data.bin_burst_data(short_data["99"]["total"], 99)
    sum_data359 = short_data['359']['sum_intact']
    plt.figure(figsize=(1.7,1.25))
    ax = plt.gca()
    ssrl_absorption = sum_data359['ssrl_absorption']
    ssrl_absorbed = 1 - np.exp(-1 * ssrl_absorption)
    ax.plot(sum_data359['phot'], sum_data359['ssrl_absorption']-0.6, label='Absorption', color='k')
    format_cartoon()
    ax.set_ylabel('Absorption')
    plt.savefig('plots/cartoon_linear.eps', dpi=600)

def make_temp_change_plot():
    short_data = LB51_get_cal_data.get_short_pulse_data()
    sum_data359 = short_data['359']['sum_intact']
    plt.figure(figsize=(1.7, 1.25))
    plt.plot(sum_data359['phot'], sum_data359['ssrl_absorption']-0.6, label='Ground State Absorption', color='k', linestyle='--')
    abs_change = fermi_dirac_change(sum_data359['phot'], FERMI_ENERGY, 7000)
    plt.plot(sum_data359['phot'], sum_data359['ssrl_absorption']+abs_change-0.6, color='k', label='Current State Absorption')
    format_cartoon()
    plt.savefig('plots/cartoon_temp_change.eps', dpi=600)


def make_core_excited_plot():
    short_data = LB51_get_cal_data.get_short_pulse_data()
    sum_data359 = short_data['359']['sum_intact']
    plt.figure(figsize=(1.7, 1.25))
    plt.plot(sum_data359['phot'], sum_data359['ssrl_absorption']-0.6, label='Ground', color='k', linestyle='--')
    emission = get_emission()
    elastic = lorentzian(emission['x']-778-EMISSION_SHIFT, 0.43)
    plt.plot(emission['x'] - EMISSION_SHIFT, -2*(elastic/2+emission['y']-0.2), color='k', label='Current')
    format_cartoon()
    plt.gca().set_xlabel('Photon Energy')
    plt.legend(loc='upper left', frameon=False)
    plt.savefig('plots/cartoon_core_excited.eps', dpi=600)


def format_cartoon():
    ax = plt.gca()
    ax.set_xlim((770, 782))
    ax.set_ylim((-1.5, 1.2))
    ax.axes.get_xaxis().set_ticks([])
    ax.axes.get_yaxis().set_ticks([])

def fermi_dirac_change(phot, mu, T, T0=300.0, broad=0.43):
    """Adapted from LK30 code
    """
    # Make sample vector which is at least 10*broad longer than phot on each side
    if np.size(phot) < 2:
        diff = 0.01
    else:
        diff = np.amin(np.diff(phot))
    sample_phot = np.arange(-21*broad+np.amin(phot), 21*broad+np.amax(phot)+diff/10, diff/10)
    print(len(sample_phot))
    # Co L3 lifetime broadening is 0.43 eV
    k = 8.617E-5
    fermi_dirac_start = 1/(np.exp((sample_phot-mu)/(k*T0))+1)
    fermi_dirac_end = 1/(np.exp((sample_phot-mu)/(k*T))+1)
    fermi_dirac_change = fermi_dirac_start-fermi_dirac_end
    lorentz_points = np.arange(-10*broad, 10*broad+diff, diff)
    lorentz_points = sample_phot-np.mean(sample_phot)
    lorentz_broad = lorentzian(lorentz_points, broad)
    lorentz_broad = lorentz_broad/np.sum(lorentz_broad)
    fermi_dirac_change = np.convolve(fermi_dirac_change, lorentz_broad, mode='same')
    # 260 meV broadened Gaussian to account for experimental resolving power of 3000
    gauss_broad_points = np.arange(-20*broad, 20*broad+sample_phot[1]-sample_phot[0], sample_phot[1]-sample_phot[0])
    gauss_broad = np.exp(-1*(gauss_broad_points)**2/(2*0.11)**2)
    gauss_broad = gauss_broad/np.sum(gauss_broad)
    fermi_dirac_change = np.convolve(fermi_dirac_change, gauss_broad, mode='same')
    fermi_dirac_change = np.interp(phot, sample_phot, fermi_dirac_change)
    return fermi_dirac_change

def linear_plot(ax, sum_data359):
    emission = get_emission()
    ssrl_absorption = sum_data359["ssrl_absorption"]
    ssrl_absorbed = 1 - np.exp(-1 * ssrl_absorption)
    ax.plot(sum_data359['phot'], ssrl_absorbed, label='Absorption')
    ax.plot([], [], label='Emission', color='tab:orange')
    emission_ax = ax.twinx()
    emission_ax.plot(
        emission['x'] - EMISSION_SHIFT,
        emission['y'],
        label='Emission',
        color='tab:orange',
    )
    return emission_ax

def get_emission():
    data = np.genfromtxt("data/measuredEmissionPoints3.txt")
    emission = {"x": data[:, 0], "y": data[:, 1]}
    return emission

def lorentzian(lorentz_points, broad):
    return 1.0/(1.0+(lorentz_points/(broad/2))**2)