"""Log-log plot of RIXS gain vs X-ray fluence
"""

EXPERIMENT_FILE = "data/proc/stim_efficiency.pickle"

FLUENCE_MEASUREMENT_ACCURACY = 0.3  # fraction within which fluence was measured

SPON_EFFICIENCY = 0.8
ACCEPTANCE_ANGLE_FRACTION = 1e5
EFFICIENCY_TO_AMPLIFICATION = (
    ACCEPTANCE_ANGLE_FRACTION/SPON_EFFICIENCY
)
# starting efficiency for above factor is in %

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

from LB51.xbloch import do_xbloch_sim
from LB51.manuscript_plots import set_plot_params

set_plot_params.init_paper_small()

def plot():
    measured = get_measurements()
    sim_results_5fs = do_xbloch_sim.load_multipulse_data(5.0)
    sim_results_25fs = do_xbloch_sim.load_multipulse_data(25.0)
    with open("LB51/xbloch/results/enhanced_5fs.pickle", "rb") as f:
        sim_results_5fs_enhanced = pickle.load(f)
    with open("LB51/xbloch/results/enhanced_25fs.pickle", "rb") as f:
        sim_results_25fs_enhanced = pickle.load(f)
    f = plt.figure(figsize=(3.37, 3.37))
    ax = plt.gca()
    l1 = ax.errorbar(
        measured["short_fluences"],
        measured["short_efficiencies"]/100,
        xerr=measured["short_fluences"] * FLUENCE_MEASUREMENT_ACCURACY,
        yerr=measured["short_stds"]/100,
        color="m",
        label="5 fs Expt.",
        linestyle="",
        marker="o",
    )
    (l2,) = ax.plot(
        sim_results_5fs["fluences"] * 1e3,  # convert from J/cm^2 to mJ/cm^2
        np.array(sim_results_5fs["stim_efficiencies"])/100,
        color="m",
        label="5 fs\nSimulation",
    )
    ax.set_xscale('log')
    ax.set_yscale('log')

    # only plot high fluence point right now
    ax.errorbar(
        measured["long_fluences"][-1],
        measured["long_efficiencies"][-1]/100,
        xerr=measured["long_fluences"][-1] * FLUENCE_MEASUREMENT_ACCURACY,
        yerr=measured["long_stds"][-1]/100,
        color="g",
        label="25 fs Expt.",
        linestyle="",
        marker="o",
    )
    ax.plot(
        sim_results_25fs["fluences"] * 1e3,  # convert from J/cm^2 to mJ/cm^2
        np.array(sim_results_25fs["stim_efficiencies"])/100,
        color="g",
        label="25 fs\nSimulation",
    )

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Fluence (mJ/cm$^2$)')
    ax.set_ylabel('Projected Stimulated RIXS Gain')
    ax.legend(loc='right', frameon=False)
    ax.set_xlim((10**(-2.5), 10**(4.5)))
    ax.set_ylim((10**(-8.2), 1))
    ax.axhline(0.008, color='k', linestyle='--')
    ax.axhline((0.008*10**(-5)), color='k', linestyle='--')
    ax.text(
        10**(-1.55),  # 769.5
        10**(-7.85),  # 3.2
        "Spontaneous RIXS\nIn Spectrometer",
        transform=ax.transData,
        fontsize=8,
    )
    ax.text(
        10**(-1.7),  # 769.5
        10**(-2.8),  # 3.2
        "Total Spontaneous\nRIXS",
        transform=ax.transData,
        fontsize=8,
    )
    plt.tight_layout()
    plt.savefig("manuscript_plots/2021_10_22_rixs_gain.eps", dpi=600)
    plt.savefig("manuscript_plots/2021_10_22_rixs_gain.png", dpi=600)
    plt.savefig("manuscript_plots/2021_10_22_rixs_gain.svg", dpi=600)

def get_measurements():
    with open(EXPERIMENT_FILE, "rb") as f:
        measured = pickle.load(f)
    return measured

def save_data():
    measured = get_measurements()
    sim_results_5fs = do_xbloch_sim.load_multipulse_data(5.0)
    sim_results_25fs = do_xbloch_sim.load_multipulse_data(25.0)
    short_exp_dict = {
        'fluences': measured['short_fluences'],
        'efficiencies': measured['short_efficiencies']
    }
    short_exp_pd = pd.DataFrame(data=short_exp_dict)
    short_exp_pd.to_csv('../data_for_jo_gain_plot/short_experiment.csv')
    long_exp_dict = {
        'fluences': measured['long_fluences'],
        'efficiencies': measured['long_efficiencies']
    }
    long_exp_pd = pd.DataFrame(data=long_exp_dict)
    long_exp_pd.to_csv('../data_for_jo_gain_plot/long_experiment.csv')
    short_simulation_dict = {
        'fluences': sim_results_5fs['fluences'][1:],
        'efficiencies': sim_results_5fs['stim_efficiencies'][1:]
    }
    short_simulation_pd = pd.DataFrame(data=short_simulation_dict)
    short_simulation_pd.to_csv('../data_for_jo_gain_plot/short_simulation.csv')
    long_simulation_dict = {
        'fluences': sim_results_25fs['fluences'][1:],
        'efficiencies': sim_results_25fs['stim_efficiencies'][1:]
    }
    long_simulation_pd = pd.DataFrame(data=long_simulation_dict)
    long_simulation_pd.to_csv('../data_for_jo_gain_plot/long_simulation.csv')