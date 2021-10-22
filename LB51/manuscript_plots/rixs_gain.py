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
        measured["short_efficiencies"] * EFFICIENCY_TO_AMPLIFICATION,
        xerr=measured["short_fluences"] * FLUENCE_MEASUREMENT_ACCURACY,
        yerr=measured["short_stds"] * EFFICIENCY_TO_AMPLIFICATION,
        color="m",
        label="5 fs Expt.",
        linestyle="",
        marker="o",
    )
    (l2,) = ax.plot(
        sim_results_5fs["fluences"] * 1e3,  # convert from J/cm^2 to mJ/cm^2
        np.array(sim_results_5fs["stim_efficiencies"]) * EFFICIENCY_TO_AMPLIFICATION,
        color="m",
        label="5 fs Three Level\nSimulation",
    )
    ax.set_xscale('log')
    ax.set_yscale('log')

    ax.errorbar(
        measured["long_fluences"],
        measured["long_efficiencies"] * EFFICIENCY_TO_AMPLIFICATION,
        xerr=measured["long_fluences"] * FLUENCE_MEASUREMENT_ACCURACY,
        yerr=measured["long_stds"] * EFFICIENCY_TO_AMPLIFICATION,
        color="g",
        label="25 fs Expt.",
        linestyle="",
        marker="o",
    )
    ax.plot(
        sim_results_25fs["fluences"] * 1e3,  # convert from J/cm^2 to mJ/cm^2
        np.array(sim_results_25fs["stim_efficiencies"]) * EFFICIENCY_TO_AMPLIFICATION,
        color="g",
        label="25 fs Three Level\nSimulation",
    )

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Fluence (mJ/cm$^2$)')
    ax.set_ylabel('Projected Stimulated RIXS Gain')
    ax.legend(loc='best')
    ax.set_xlim((10**(-2.5), 10**(4.5)))
    plt.tight_layout()

def get_measurements():
    with open(EXPERIMENT_FILE, "rb") as f:
        measured = pickle.load(f)
    return measured