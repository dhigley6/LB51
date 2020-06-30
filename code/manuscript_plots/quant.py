"""Plot of stim efficiency vs X-ray intensity
"""

import numpy as np 
import matplotlib.pyplot as plt
import pickle

import LB51_get_cal_data
from xbloch import gaussian_xbloch_sim, sase_xbloch_sim
from manuscript_plots import set_plot_params
set_plot_params.init_paper_small()

def quant():
    measured = get_measured_stim_efficiency()
    strengths, stim_efficiencies = sase_xbloch_sim.calculate_multipulse_stim_efficiencies()
    plt.figure(figsize=(3.37, 2.5))
    plt.scatter(measured['short_fluences']/5, measured['short_efficiencies'], label='5 fs Pulses\nExpt.')
    plt.scatter(measured['long_fluences']/25, measured['long_efficiencies'], label='25 fs Pulses\nExpt.')
    plt.semilogx(strengths*1E15, np.array(stim_efficiencies)*100, color='k', label='Three Level\nSimulation')
    plt.xlabel('Intensity (W/cm$^2$)')
    plt.ylabel('Inelastic Stimulated Scattering Efficiency')
    plt.xlim((1E10, 1E15))
    plt.legend(loc='best')
    format_quant_plot()
    #plt.savefig('../plots/2019_02_03_quant.eps', dpi=600)
    #plt.savefig('../plots/2019_02_03_quant.png', dpi=600)

def get_measured_stim_efficiency():
    with open('../data/proc/stim_efficiency.pickle', 'rb') as f:
        measured = pickle.load(f)
    return measured

def format_quant_plot():
    plt.xlabel('Intensity (10$^{12}$ W/cm$^2$)')
    plt.ylabel('Stim. Scattering\nEfficiency (%)')
    plt.legend(loc='best', frameon=True)
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
    with open('../data/proc/stim_efficiency.pickle', 'wb') as f:
        pickle.dump(quant_data, f)

def get_stim_efficiency(data):
    ssrl_res_absorption = data['sum_intact']['ssrl_absorption']-data['sum_intact']['ssrl_absorption'][0]
    ssrl_res_trans = np.exp(-1*ssrl_res_absorption)
    res_transmitted = data['sum_intact']['no_sam_spec']*ssrl_res_trans
    res_absorbed = data['sum_intact']['no_sam_spec']-res_transmitted
    phot = data['sum_intact']['phot']
    abs_region = (phot > 774) & (phot < 780)
    res_absorbed_sum = np.sum(res_absorbed[abs_region])
    stim_region = (phot > 773.5) & (phot < 775)
    stim = data['sum_intact']['exc_sam_spec']
    stim_sum = np.sum(stim[stim_region])
    stim_efficiency = stim_sum/res_absorbed_sum
    return stim_efficiency