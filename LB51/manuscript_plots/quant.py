"""Plot of stim efficiency vs X-ray intensity
"""

import numpy as np 
import matplotlib.pyplot as plt
import pickle

from LB51 import LB51_get_cal_data
from LB51.xbloch import do_xbloch_sim
from LB51.manuscript_plots import set_plot_params
set_plot_params.init_paper_small()

MEASURED_STIM_FILE = 'data/proc/stim_efficiency.pickle'

def quant():
    measured = get_measured_stim_efficiency()
    #sim_results = do_xbloch_sim.load_gauss_data()   # gaussian pulse case
    sim_results_5fs = do_xbloch_sim.load_multipulse_data(5.0)    # SASE pulses case
    sim_results_25fs = do_xbloch_sim.load_multipulse_data(25.0)
    markus = get_markus_simulation()
    _diagnostic_figure(sim_results_5fs)
    f, axs = plt.subplots(2, 1, figsize=(3.37, 4))
    axs[0].scatter(measured['short_fluences']*1E-12, measured['short_efficiencies'], label='5 fs Pulses\nExpt.')
    axs[1].scatter(measured['long_fluences']*1E-12, measured['long_efficiencies'], label='25 fs Pulses\nExpt.')
    axs[0].plot(markus['5fs']['fluence'], markus['5fs']['stim']*100, label='Rate Eqs.')
    axs[1].plot(markus['25fs']['fluence']*5, markus['25fs']['stim']*100, label='Rate Eqs.')
    axs[0].plot(sim_results_5fs['fluences']*1E3, np.array(sim_results_5fs['stim_efficiencies']), color='k', label='Three Level\nSimulation')
    axs[1].plot(sim_results_25fs['fluences']*1E3, np.array(sim_results_25fs['stim_efficiencies']), color='k', label='Three Level\nSimulation')
    format_quant_plot(axs)
    #plt.savefig('../plots/2019_02_03_quant.eps', dpi=600)
    #plt.savefig('../plots/2019_02_03_quant.png', dpi=600)

def _diagnostic_figure(sim_results):
    """Show spectra on separate figure
    """
    f, axs = plt.subplots(2, 1, sharex=True)
    axs[0].plot(sim_results['phot'], sim_results['summed_incident_intensities'][0]/sim_results['fluences'][0])
    for i in range(len(sim_results['fluences'])):
        intensity_difference = (sim_results['summed_transmitted_intensities'][i]-sim_results['summed_incident_intensities'][i])/sim_results['fluences'][i]
        axs[1].plot(sim_results['phot'], intensity_difference, label=sim_results['fluences'][i])
    axs[0].set_xlim((770, 784))
    axs[0].set_ylabel('Intensity')
    axs[1].set_ylabel('Intensity')
    axs[1].set_ylabel('Photon Energy (eV)')
    axs[1].legend(loc='best')

def get_measured_stim_efficiency():
    with open(MEASURED_STIM_FILE, 'rb') as f:
        measured = pickle.load(f)
    return measured

def format_quant_plot(axs):
    axs[0].set_xlabel('Fluence (mJ/cm$^2$)')
    axs[0].set_ylabel('Stim. Scattering\nEfficiency (%)')
    axs[1].set_xlabel('Fluence (mJ/cm$^2$)')
    axs[1].set_ylabel('Stim. Scattering\nEfficiency (%)')
    #plt.legend(loc='best', frameon=True)
    plt.tight_layout()

def run_quant_ana():
    """Calculate stim. strength of expt. data
    """
    short_data = LB51_get_cal_data.get_short_pulse_data()
    long_data = LB51_get_cal_data.get_long_pulse_data()
    short_run_sets_list = ['99', '290', '359', '388']
    short_fluences = []
    short_stim_strengths = []
    for run_set in short_run_sets_list:
        fluence = short_data[run_set]['sum_intact']['fluence']
        if run_set == '99':
            fluence = 1898
        stim_strength = get_stim_efficiency(short_data[run_set])
        short_fluences.append(fluence)
        short_stim_strengths.append(stim_strength)
    long_run_sets_list = ['641', '554', '603']
    long_fluences = []
    long_stim_strengths = []
    for run_set in long_run_sets_list:
        fluence = long_data[run_set]['sum_intact']['fluence']
        stim_strength = get_stim_efficiency(long_data[run_set])
        long_fluences.append(fluence)
        long_stim_strengths.append(stim_strength)
    quant_data = {'short_fluences': np.array(short_fluences)*1E12,
                  'long_fluences': np.array(long_fluences)*1E12,
                  'short_efficiencies': 100*np.array(short_stim_strengths),
                  'long_efficiencies': 100*np.array(long_stim_strengths)}
    save_quant_data(quant_data)
    
def save_quant_data(quant_data):
    """Save quantified data
    """
    with open(MEASURED_STIM_FILE, 'wb') as f:
        pickle.dump(quant_data, f)

def get_stim_efficiency(data):
    ssrl_res_absorption = data['sum_intact']['ssrl_absorption']-data['sum_intact']['ssrl_absorption'][0]
    ssrl_res_trans = np.exp(-1*ssrl_res_absorption)
    res_transmitted = data['sum_intact']['no_sam_spec']*ssrl_res_trans
    res_absorbed = data['sum_intact']['no_sam_spec']-res_transmitted
    phot = data['sum_intact']['phot']
    abs_region = (phot > 774) & (phot < 780)
    res_absorbed_sum = np.trapz(res_absorbed[abs_region])
    stim_region = (phot > 773.5) & (phot < 775)
    stim = data['sum_intact']['exc_sam_spec']
    stim[stim < 0] = 0     # clip negative values to zero
    stim_sum = np.trapz(stim[stim_region])
    stim_efficiency = stim_sum/res_absorbed_sum
    return stim_efficiency

def get_markus_simulation():
    markus_5fs_data = np.genfromtxt('data/proc/Markus_5fs.txt', skip_header=1)
    markus_25fs_data = np.genfromtxt('data/proc/Markus_25fs.txt', skip_header=1)
    markus_5fs_result = convert_markus_data(markus_5fs_data)
    markus_25fs_result = convert_markus_data(markus_25fs_data)
    return {
        '5fs': markus_5fs_result,
        '25fs': markus_25fs_result
    }

def convert_markus_data(data):
    peak_intensity = data[:, 0]
    stim_efficiency = data[:, 1]
    fluence = peak_intensity*5.3223351989345264/1E12    # mJ/cm^2
    return {
        'fluence': fluence,
        'stim': stim_efficiency
    }