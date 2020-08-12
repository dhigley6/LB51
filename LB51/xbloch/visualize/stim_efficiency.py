"""Make plot of simulated stimulated inelastic scattering efficiency
"""

import matplotlib.pyplot as plt
import pickle

from xbloch import gaussian_xbloch_sim

def make_figure():
    strengths, stim_efficiencies = gaussian_xbloch_sim.calculate_stim_efficiencies()
    measured = get_measured_stim_efficiency()
    plt.figure(figsize=(3.37, 3.37))
    plt.scatter(measured['short_fluences']/5, measured['short_efficiencies']/100, label='5 fs Pulses\nExpt.')
    plt.scatter(measured['long_fluences']/25, measured['long_efficiencies']/100, label='25 fs Pulses\nExpt.')
    plt.semilogx(strengths*1E15, stim_efficiencies, color='k', label='Three Level\nSimulation')

def format_figure():
    plt.xlabel('Peak Intensity (W/cm$^2$)')
    plt.ylabel('Inelastic Stimulated Scattering Efficiency')
    plt.xlim((1E10, 1E15))
    plt.legend(loc='best')
    plt.tight_layout()

def get_measured_stim_efficiency():
    file_path = 'data/stim_efficiency.pickle'
    with open(file_path, 'rb') as f:
        measured = pickle.load(f)
    return measured