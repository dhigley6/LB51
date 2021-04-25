"""Plot of absorption loss vs X-ray fluence
"""

EXPERIMENT_FILE = 'data/proc/stim_efficiency.pickle'

import numpy as np 
import matplotlib.pyplot as plt
import pickle

from LB51.xbloch import do_xbloch_sim

def quant2():
    measured = get_measurements()
    sim_results_5fs = do_xbloch_sim.load_multipulse_data(5.0)
    sim_results_25fs = do_xbloch_sim.load_multipulse_data(25.0)
    with open('LB51/xbloch/results/enhanced_5fs.pickle', 'rb') as f:
        sim_results_5fs_enhanced = pickle.load(f)
    with open('LB51/xbloch/results/enhanced_25fs.pickle', 'rb') as f:
        sim_results_25fs_enhanced = pickle.load(f)
    f, axs = plt.subplots(1, 2)
    axs[0].errorbar(
        measured['short_fluences'],
        measured['short_absorption_losses'],
        yerr=measured['short_absorption_stds'],
    )
    axs[1].errorbar(
        measured['long_fluences'],
        measured['long_absorption_losses'],
        yerr=measured['long_absorption_stds'],
    )
    axs[0].plot(
        sim_results_5fs['fluences'] * 1e3,
        np.array(sim_results_5fs['absorption_losses']),
    )
    axs[0].plot(
        sim_results_5fs_enhanced['fluences'] * 1e3,
        np.array(sim_results_5fs_enhanced['absorption_losses']),
    )
    axs[1].plot(
        sim_results_25fs['fluences'] * 1e3,
        np.array(sim_results_25fs['absorption_losses'])
    )
    axs[1].plot(
        sim_results_25fs_enhanced['fluences'] * 1e3,
        np.array(sim_results_25fs_enhanced['absorption_losses']),
    )
    _format(axs)

def _format(axs):
    axs[0].set_xlabel('Fluence (mJ/cm$^2$)')
    axs[0].set_ylabel('Absorption Loss (Percent)')

def get_measurements():
    with open(EXPERIMENT_FILE, 'rb') as f:
        measured = pickle.load(f)
    return measured