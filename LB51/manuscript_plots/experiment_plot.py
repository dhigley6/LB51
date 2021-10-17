"""Not formally used, just for experiment
"""


EXPERIMENT_FILE = "data/proc/stim_efficiency.pickle"

FLUENCE_MEASUREMENT_ACCURACY = 0.3  # fraction within which fluence was measured

SPON_EFFICIENCY = 0.8
ACCEPTANCE_ANGLE_FRACTION = 1e5
EFFICIENCY_TO_AMPLIFICATION = (
    SPON_EFFICIENCY * ACCEPTANCE_ANGLE_FRACTION / 1e6
)  # (millions)

import numpy as np
import matplotlib.pyplot as plt
import pickle

from LB51.xbloch import do_xbloch_sim
from LB51.manuscript_plots import set_plot_params

set_plot_params.init_paper_small()


def quant2():
    measured = get_measurements()
    sim_results_5fs = do_xbloch_sim.load_multipulse_data(5.0)
    sim_results_25fs = do_xbloch_sim.load_multipulse_data(25.0)
    with open("LB51/xbloch/results/enhanced_5fs.pickle", "rb") as f:
        sim_results_5fs_enhanced = pickle.load(f)
    with open("LB51/xbloch/results/enhanced_25fs.pickle", "rb") as f:
        sim_results_25fs_enhanced = pickle.load(f)
    f, axs = plt.subplots(2, 2, figsize=(3.37, 4), sharex="col", sharey="row")
    _short_stim_plot(axs[0, 0], measured, sim_results_5fs, sim_results_5fs_enhanced, f)
    _long_stim_plot(axs[0, 1], measured, sim_results_25fs, sim_results_25fs_enhanced)
    _short_absorption_plot(
        axs[1, 0], measured, sim_results_5fs, sim_results_5fs_enhanced
    )
    _long_absorption_plot(
        axs[1, 1], measured, sim_results_25fs, sim_results_25fs_enhanced
    )
    _format(f, axs)
    # plt.savefig("plots/2021_10_03_quant.eps", dpi=600)
    # plt.savefig("plots/2021_10_03_quant.png", dpi=600)


def _short_stim_plot(ax, measured, sim_results_5fs, sim_results_5fs_enhanced, f):
    l1 = ax.errorbar(
        measured["short_fluences"],
        measured["short_efficiencies"] * EFFICIENCY_TO_AMPLIFICATION,
        xerr=measured["short_fluences"] * FLUENCE_MEASUREMENT_ACCURACY,
        yerr=measured["short_stds"] * EFFICIENCY_TO_AMPLIFICATION,
        color="k",
        label="Expt.",
        linestyle="",
        marker="o",
    )
    (l2,) = ax.plot(
        sim_results_5fs["fluences"] * 1e3,  # convert from J/cm^2 to mJ/cm^2
        np.array(sim_results_5fs["stim_efficiencies"]) * EFFICIENCY_TO_AMPLIFICATION,
        color="tab:blue",
        label="Three Level\nSimulation",
    )
    (l3,) = ax.plot(
        sim_results_5fs_enhanced["fluences"] * 1e3,  # convert from J/cm^2 to mJ/cm^2
        np.array(sim_results_5fs_enhanced["stim_efficiencies"])
        * EFFICIENCY_TO_AMPLIFICATION,
        color="tab:orange",
        label="Enhancement\nSimulation",
    )
    f.legend(
        (l2, l3, l1),
        ("Three Level Simulation", "Enhancement Simulation", "Experiment"),
        "upper center",
        ncol=2,
    )


def _long_stim_plot(ax, measured, sim_results_25fs, sim_results_25fs_enhanced):
    ax.errorbar(
        measured["long_fluences"],
        measured["long_efficiencies"] * EFFICIENCY_TO_AMPLIFICATION,
        xerr=measured["long_fluences"] * FLUENCE_MEASUREMENT_ACCURACY,
        yerr=measured["long_stds"] * EFFICIENCY_TO_AMPLIFICATION,
        color="k",
        label="Expt.",
        linestyle="",
        marker="o",
    )
    ax.plot(
        sim_results_25fs["fluences"] * 1e3,  # convert from J/cm^2 to mJ/cm^2
        np.array(sim_results_25fs["stim_efficiencies"]) * EFFICIENCY_TO_AMPLIFICATION,
        color="tab:blue",
        label="Three Level\nSimulation",
    )
    ax.plot(
        sim_results_25fs_enhanced["fluences"] * 1e3,  # convert from J/cm^2 to mJ/cm^2
        np.array(sim_results_25fs_enhanced["stim_efficiencies"])
        * EFFICIENCY_TO_AMPLIFICATION,
        color="tab:orange",
        label="Enhancement\nSimulation",
    )


def _short_absorption_plot(ax, measured, sim_results_5fs, sim_results_5fs_enhanced):
    ax.errorbar(
        measured["short_fluences"],
        measured["short_absorption_losses"],
        xerr=measured["short_fluences"] * FLUENCE_MEASUREMENT_ACCURACY,
        yerr=measured["short_absorption_stds"],
        color="k",
        label="Experiment",
        linestyle="",
        marker="o",
    )
    ax.plot(
        sim_results_5fs["fluences"] * 1e3,
        np.array(sim_results_5fs["absorption_losses"]),
        label="3-Level\nSimulation",
    )
    ax.plot(
        sim_results_5fs_enhanced["fluences"] * 1e3,
        np.array(sim_results_5fs_enhanced["absorption_losses"]),
        label="Enhancement\nSimulation",
    )


def _long_absorption_plot(ax, measured, sim_results_25fs, sim_results_25fs_enhanced):
    ax.errorbar(
        measured["long_fluences"],
        measured["long_absorption_losses"],
        xerr=measured["long_fluences"] * FLUENCE_MEASUREMENT_ACCURACY,
        yerr=measured["long_absorption_stds"],
        color="k",
        label="Expt.",
        linestyle="",
        marker="o",
    )
    ax.plot(
        sim_results_25fs["fluences"] * 1e3,
        np.array(sim_results_25fs["absorption_losses"]),
    )
    ax.plot(
        sim_results_25fs_enhanced["fluences"] * 1e3,
        np.array(sim_results_25fs_enhanced["absorption_losses"]),
    )


def _format(f, axs):
    axs[0, 0].set_ylabel("RIXS Amplification (Millions)")
    axs[1, 0].set_ylabel("Absorption Loss (%)")
    # axs[0].set_xlabel('Fluence (mJ/cm$^2$)')
    # axs[0].set_ylabel('Absorption Loss (Percent)')
    f.text(0.52, 0.02, "Fluence (mJ/cm$^2$)", ha="center")
    plt.tight_layout(w_pad=0, h_pad=0.1, rect=(0, 0.03, 0.98, 0.93))
    # axs[0, 0].set_xlim((-100, 1600))
    # axs[0, 0].set_ylim((-0.1, 0.65))
    # axs[0, 1].set_xlim((-300, 10000))
    # axs[1, 0].set_ylim((-5, 100))
    # axs[1].set_ylabel("Inelastic Stim.\nScattering Efficiency (%)")
    # axs[1, 0].legend(loc="upper left", frameon=False)
    axs[0, 0].text(
        0.9,
        0.1,
        "a",
        fontsize=10,
        weight="bold",
        horizontalalignment="center",
        transform=axs[0, 0].transAxes,
    )
    axs[0, 1].text(
        0.9,
        0.1,
        "b",
        fontsize=10,
        weight="bold",
        horizontalalignment="center",
        transform=axs[0, 1].transAxes,
    )
    axs[1, 0].text(
        0.9,
        0.9,
        "c",
        fontsize=10,
        weight="bold",
        horizontalalignment="center",
        transform=axs[1, 0].transAxes,
    )
    axs[1, 1].text(
        0.1,
        0.9,
        "d",
        fontsize=10,
        weight="bold",
        horizontalalignment="center",
        transform=axs[1, 1].transAxes,
    )
    axs[0, 0].text(
        0.5,
        0.1,
        "5 fs",
        fontsize=10,
        horizontalalignment="center",
        transform=axs[0, 0].transAxes,
    )
    axs[0, 1].text(
        0.5,
        0.1,
        "25 fs",
        fontsize=10,
        horizontalalignment="center",
        transform=axs[0, 1].transAxes,
    )
    """
    for ax in axs:
        second_ax = ax.secondary_yaxis(
            "right",
            functions=(efficiency_to_amplification, amplification_to_efficiency),
        )
        second_ax.set_ylabel(" ")
    plt.tight_layout()
    big_ax = plt.gcf().add_subplot(111, frameon=False)
    plt.tick_params(labelcolor="none", top=False, bottom=False, left=False, right=False)
    big_ax.set_ylabel("Inelastic Stim. Scattering Efficiency (%)")
    second_big_ax = big_ax.secondary_yaxis(
        "right",
        functions=(efficiency_to_amplification, amplification_to_efficiency),
        frameon=False,
    )
    second_big_ax.tick_params(
        labelcolor="none", top=False, bottom=False, left=False, right=False
    )
    second_big_ax.set_ylabel("RIXS Signal Amplification (Millions)")
    """
    axs[0, 0].set_xscale("log")
    axs[0, 1].set_xscale("log")
    # axs[0, 0].set_yscale('log')


def get_measurements():
    with open(EXPERIMENT_FILE, "rb") as f:
        measured = pickle.load(f)
    return measured


def efficiency_to_amplification(x):
    percent_to_decimal = 0.01
    spectrometer_transmission = 0.1
    traditional_efficiency = 1e-8
    millions = 1e6
    return (
        x
        * percent_to_decimal
        * spectrometer_transmission
        / traditional_efficiency
        / millions
    )
