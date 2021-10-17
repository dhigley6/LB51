"""Make first part of experimental data summary figure
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects

from LB51 import LB51_get_cal_data
from LB51.manuscript_plots import set_plot_params

set_plot_params.init_paper_small()

EMISSION_SHIFT = 0.4  # photon energy shift from reference data

NO_SAM = 1  # Factor to multiply no sample spectra by when plotting



def make_figure():
    long_data = LB51_get_cal_data.get_long_pulse_data()
    short_data = LB51_get_cal_data.get_short_pulse_data()
    binned_603 = LB51_get_cal_data.bin_burst_data(long_data["603"]["total"], 603)
    binned_99 = LB51_get_cal_data.bin_burst_data(short_data["99"]["total"], 99)
    f, axs = plt.subplots(2, 1, figsize=(3, 4), sharex=True)
    emission_ax = linear_plot(axs[0], short_data['359']['sum_intact'])
    summed_spectra_plot(axs[1], binned_99[0])
    format_figure(f, axs, emission_ax)
    

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

def format_figure(f, axs, emission_ax):
    place_vlines([axs[0]])
    emission_ax.spines["right"].set_color("tab:orange")
    emission_ax.set_ylabel("I$_{emission}$/I$_0$ $\sim 10^{-8}$")
    emission_ax.yaxis.label.set_color("tab:orange")
    emission_ax.yaxis.set_ticklabels([])
    emission_ax.yaxis.set_ticks([])
    axs[0].xaxis.set_ticks([770, 775, 780, 785])
    axs[0].set_xlim((769, 787))
    axs[1].set_xlabel('Photon Energy (eV)')
    axs[0].set_ylabel("I$_{absorbed}$/I$_0$ $\sim$ 0.3")
    axs[1].set_ylabel("Intensity")
    handles, labels = axs[0].get_legend_handles_labels()
    axs[0].legend(handles, labels, frameon=False, loc='upper right')
    emission_ax.spines["left"].set_color("tab:blue")
    axs[0].tick_params(axis="y", color="tab:blue")
    axs[0].yaxis.label.set_color("tab:blue")
    axs[0].yaxis.set_ticklabels([])
    axs[0].yaxis.set_ticks([])
    axs[1].yaxis.set_ticklabels([])
    axs[1].yaxis.set_ticks([])
    axs[1].legend(loc='upper right')
    axs[0].text(
        0.1, 0.9, "a", transform=axs[0].transAxes, fontsize=10, fontweight="bold"
    )
    axs[1].text(
        0.1, 0.9, "b", transform=axs[1].transAxes, fontsize=10, fontweight="bold"
    )
    axs[1].text(
        0.65,
        0.3,
        '5 fs,\n1600 mJ/cm$^2$',
        #"780 eV,\n9490\nmJ/cm$^2$",
        transform=axs[1].transAxes,
        fontsize=8,
    )
    plt.tight_layout()
    plt.savefig("manuscript_plots/2021_10_03_summary_p1.eps", dpi=600)
    plt.savefig("manuscript_plots/2021_10_03_summary_p1.png", dpi=600)


def summed_spectra_plot(ax, data):
    phot = data["sum_intact"]["phot"]
    norm = np.amax(data["sum_intact"]["no_sam_spec"])
    sam_spec = data["sum_intact"]["sam_spec"] / norm
    no_sam_spec = data["sum_intact"]["no_sam_spec"] / norm
    exc_spec = data["sum_intact"]["exc_sam_spec"] / norm
    ssrl_trans = np.exp(-1 * data["sum_intact"]["ssrl_absorption"])
    lin_sam_spec = no_sam_spec * ssrl_trans
    ax.plot(phot, no_sam_spec * NO_SAM, "k--", label="Incident")
    ax.plot(phot, sam_spec, "k", label="Co/Pd")
    ax.plot(phot, lin_sam_spec, "k:", label="Estimated\nLinear Co/Pd")
    ax.fill_between(
        phot,
        lin_sam_spec,
        sam_spec,
        where=(exc_spec > 0),
        facecolor="b",
        edgecolor="w",
    )
    ax.fill_between(
        phot,
        lin_sam_spec,
        sam_spec,
        where=(exc_spec < 0),
        facecolor="r",
        edgecolor="w",
    )

def place_vlines(axs):
    vline_loc_list = [774.5, 776.5, 778]
    for ax in axs:
        for vline_loc in vline_loc_list:
            ax.axvline(vline_loc, linestyle=":", color="k")