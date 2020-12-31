"""Plot of stim efficiency vs X-ray intensity
"""

import numpy as np

np.random.seed(0)
import matplotlib.pyplot as plt
import pickle

from LB51 import LB51_get_cal_data
from LB51.xbloch import do_xbloch_sim
from LB51.manuscript_plots import set_plot_params

set_plot_params.init_paper_small()

MEASURED_STIM_FILE = "data/proc/stim_efficiency.pickle"


def quant():
    measured = get_measured_stim_efficiency()
    sim_results_5fs = do_xbloch_sim.load_multipulse_data(5.0)
    sim_results_25fs = do_xbloch_sim.load_multipulse_data(25.0)
    markus = get_markus_simulation()
    f, axs = plt.subplots(2, 1, figsize=(3.37, 4))
    axs[0].errorbar(
        measured["short_fluences"],
        measured["short_efficiencies"],
        xerr=measured["short_fluences"] * 0.3,
        yerr=measured["short_stds"],
        color="k",
        label="Expt.",
        linestyle="",
        marker="o",
    )
    axs[1].errorbar(
        measured["long_fluences"],
        measured["long_efficiencies"],
        xerr=measured["long_fluences"] * 0.3,
        yerr=measured["long_stds"],
        color="k",
        label="Expt.",
        linestyle="",
        marker="o",
    )
    axs[0].loglog(
        sim_results_5fs["fluences"] * 1e3,  # convert from J/cm^2 to mJ/cm^2
        np.array(sim_results_5fs["stim_efficiencies"]),
        color="tab:blue",
        label="Three Level\nSimulation",
    )
    axs[1].loglog(
        sim_results_25fs["fluences"] * 1e3,  # convert from J/cm^2 to mJ/cm^2
        np.array(sim_results_25fs["stim_efficiencies"]),
        color="tab:blue",
        label="Three Level\nSimulation",
    )
    #
    #axs[0].plot(
    #    markus["5fs"]["fluence"],
    #    markus["5fs"]["stim"] * 100,
    #    color="tab:orange",
    #    label="Rate Eqs.",
    #)
    #axs[1].plot(
    #    markus["25fs"]["fluence"],
    #    markus["25fs"]["stim"] * 100,
    #    color="tab:orange",
    #    label="Rate Eqs.",
    #)
    format_quant_plot(axs)
    #plt.savefig("plots/2020_12_28_quant.eps", dpi=600)
    #plt.savefig("plots/2020_12_28_quant.png", dpi=600)


def get_measured_stim_efficiency():
    with open(MEASURED_STIM_FILE, "rb") as f:
        measured = pickle.load(f)
    return measured


def format_quant_plot(axs):
    axs[0].set_xlabel("Fluence (mJ/cm$^2$)")
    axs[0].set_ylabel(" ")
    axs[0].set_xlim((-100, 1600))
    axs[0].set_ylim((-1, 10))
    axs[1].set_xlim((-300, 10000))
    axs[1].set_ylim((-1, 10))
    axs[1].set_xlabel("Fluence (mJ/cm$^2$)")
    # axs[1].set_ylabel("Inelastic Stim.\nScattering Efficiency (%)")
    axs[0].legend(loc="upper left", frameon=False)
    axs[0].text(
        0.9,
        0.1,
        "a",
        fontsize=10,
        weight="bold",
        horizontalalignment="center",
        transform=axs[0].transAxes,
    )
    axs[1].text(
        0.9,
        0.1,
        "b",
        fontsize=10,
        weight="bold",
        horizontalalignment="center",
        transform=axs[1].transAxes,
    )
    axs[0].text(
        0.5,
        0.1,
        "5 fs",
        fontsize=10,
        horizontalalignment="center",
        transform=axs[0].transAxes,
    )
    axs[1].text(
        0.5,
        0.1,
        "25 fs",
        fontsize=10,
        horizontalalignment="center",
        transform=axs[1].transAxes,
    )
    # plt.legend(loc='best', frameon=True)
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


def amplification_to_efficiency(x):
    percent_to_decimal = 0.01
    spectrometer_transmission = 0.1
    traditional_efficiency = 1e-8
    millions = 1e6
    return (
        x
        * millions
        * traditional_efficiency
        / (percent_to_decimal * spectrometer_transmission)
    )


def run_quant_ana():
    """Calculate stim. strength of expt. data"""
    short_data = LB51_get_cal_data.get_short_pulse_data()
    long_data = LB51_get_cal_data.get_long_pulse_data()
    short_run_sets_list = ["99", "290", "359", "388"]
    short_fluences = []
    short_stim_strengths = []
    for run_set in short_run_sets_list:
        fluence = short_data[run_set]["sum_intact"]["fluence"]
        # if run_set == "99":
        # fluence = 1898
        stim_strength = get_stim_efficiency(short_data[run_set])
        short_fluences.append(fluence)
        short_stim_strengths.append(stim_strength)
    long_run_sets_list = ["641", "554", "603"]
    long_fluences = []
    long_stim_strengths = []
    for run_set in long_run_sets_list:
        fluence = long_data[run_set]["sum_intact"]["fluence"]
        stim_strength = get_stim_efficiency(long_data[run_set])
        long_fluences.append(fluence)
        long_stim_strengths.append(stim_strength)
    quant_data = {
        "short_fluences": np.array(short_fluences),
        "long_fluences": np.array(long_fluences),
        "short_efficiencies": 100 * np.array(short_stim_strengths),
        "long_efficiencies": 100 * np.array(long_stim_strengths),
    }
    _save_quant_data(quant_data)


def _save_quant_data(quant_data):
    """Save quantified data"""
    with open(MEASURED_STIM_FILE, "wb") as f:
        pickle.dump(quant_data, f)


def get_stim_efficiency(data):
    ssrl_res_absorption = (
        data["sum_intact"]["ssrl_absorption"] - data["sum_intact"]["ssrl_absorption"][0]
    )
    a = 1 / 0
    ssrl_res_trans = np.exp(-1 * ssrl_res_absorption)
    res_transmitted = data["sum_intact"]["no_sam_spec"] * ssrl_res_trans
    res_absorbed = data["sum_intact"]["no_sam_spec"] - res_transmitted
    phot = data["sum_intact"]["phot"]
    abs_region = (phot > 774) & (phot < 780)
    res_absorbed_sum = np.trapz(res_absorbed[abs_region])
    stim_region = (phot > 773) & (phot < 775)
    stim = data["sum_intact"]["exc_sam_spec"]
    stim[stim < 0] = 0  # clip negative values to zero
    stim_sum = np.trapz(stim[stim_region])
    stim_efficiency = stim_sum / res_absorbed_sum
    return stim_efficiency


def get_markus_simulation():
    markus_5fs_data = np.genfromtxt("data/proc/Markus_5fs.txt", skip_header=1)
    markus_25fs_data = np.genfromtxt("data/proc/Markus_25fs.txt", skip_header=1)
    markus_5fs_result = convert_markus_data(markus_5fs_data, 5)
    markus_25fs_result = convert_markus_data(markus_25fs_data, 25)
    return {"5fs": markus_5fs_result, "25fs": markus_25fs_result}


def convert_markus_data(data, fwhm=5):
    peak_intensity = data[:, 0]
    stim_efficiency = data[:, 1]
    # Gaussian with fwhm of 2*np.sqrt(2*np.log(2)) has integral of np.sqrt(np.pi)
    fluence = (
        peak_intensity * np.sqrt(np.pi) * fwhm / (2 * np.sqrt(2 * np.log(2))) / 1e12
    )  # mJ/cm^2
    return {"fluence": fluence, "stim": stim_efficiency}
