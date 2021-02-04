"""Summarize experimental data
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects

from LB51 import LB51_get_cal_data
from LB51.manuscript_plots import set_plot_params

set_plot_params.init_paper_small()

EMISSION_SHIFT = 0.4  # photon energy shift from reference data


def summary():
    long_data = LB51_get_cal_data.get_long_pulse_data()
    short_data = LB51_get_cal_data.get_short_pulse_data()
    binned_603 = LB51_get_cal_data.bin_burst_data(long_data["603"]["total"], 603)
    binned_99 = LB51_get_cal_data.bin_burst_data(short_data["99"]["total"], 99)
    short_summary_data = [short_data["359"]]
    short_summary_data.extend(binned_99)
    long_summary_data = [long_data["641"]]
    long_summary_data.extend(binned_603)
    f, axs = plt.subplots(
        2,
        2,
        sharex=True,
        sharey="row",
        figsize=(3.37, 5),
        gridspec_kw={"height_ratios": [2, 6]},
    )
    spectra_series_plot(axs[1, 0], short_summary_data)
    spectra_series_plot(axs[1, 1], long_summary_data)
    emission_axs = linear_plots([axs[0, 0], axs[0, 1]], short_data["359"]["sum_intact"])
    format_summary_plot(f, axs, emission_axs)
    plt.savefig("plots/2021_02_03_summary.eps", dpi=600)
    plt.savefig("plots/2021_02_03_summary.png", dpi=600)


def linear_plots(ax_list, sum_data359):
    emission = get_emission()
    ssrl_absorption = sum_data359["ssrl_absorption"]
    ssrl_absorbed = 1 - np.exp(-1 * ssrl_absorption)
    emission_axs = []
    for ind, ax in enumerate(ax_list):
        ax.plot(sum_data359["phot"], ssrl_absorbed, label="Absorption")
        ax.plot([], [], label="Emission", color="tab:orange")
        emission_ax = ax.twinx()
        emission_ax.plot(
            emission["x"] - EMISSION_SHIFT,
            emission["y"],
            label="Emission",
            color="tab:orange",
        )
        emission_axs.append(emission_ax)
    return emission_axs


def format_summary_plot(f, axs, emission_axs):
    def place_vlines():
        ax_list = [axs[0, 0], axs[0, 1], axs[1, 0], axs[1, 1]]
        vline_loc_list = [774.5, 776.5, 778]
        for ax in ax_list:
            for vline_loc in vline_loc_list:
                ax.axvline(vline_loc, linestyle="--", color="k")

    emission_axs[0].tick_params(labelright=False)
    emission_axs[0].spines["left"].set_color("tab:blue")
    axs[0, 0].tick_params(axis="y", color="tab:blue")
    axs[0, 0].yaxis.label.set_color("tab:blue")
    axs[0, 0].yaxis.set_ticklabels([])
    axs[0, 0].yaxis.set_ticks([])
    emission_axs[1].set_ylabel("I$_{emission}$/I$_0$ $\sim 10^{-8}$")
    emission_axs[1].yaxis.label.set_color("tab:orange")
    emission_axs[1].tick_params(axis="y", colors="tab:orange")
    emission_axs[1].yaxis.set_ticklabels([])
    emission_axs[1].yaxis.set_ticks([])
    place_vlines()
    axs[1, 0].set_xlim((769, 787))
    axs[1, 0].set_ylim((-0.5, 4.75))
    f.text(0.52, 0.02, "Photon Energy (eV)", ha="center")
    axs[0, 0].set_ylabel("I$_{absorbed}$/I$_0$ $\sim$ 0.3")
    axs[1, 0].set_ylabel("Intensity (a.u.)")
    handles, labels = axs[0, 0].get_legend_handles_labels()
    f.legend(handles, labels, loc=(0.37, 0.88), frameon=True)
    texts = []
    texts.append(
        axs[1, 0].text(
            769.5,
            3.2,
            "779 eV,\n1080\nmJ/cm$^2$",
            transform=axs[1, 0].transData,
            fontsize=8,
        )
    )
    texts.append(
        axs[1, 0].text(
            780.5,
            1.8,
            "777 eV,\n1600\nmJ/cm$^2$",
            transform=axs[1, 0].transData,
            fontsize=8,
        )
    )
    texts.append(
        axs[1, 0].text(
            769.5,
            0.25,
            "779 eV,\n12\nmJ/cm$^2$",
            transform=axs[1, 0].transData,
            fontsize=8,
        )
    )
    texts.append(
        axs[1, 1].text(
            769.5,
            3.2,
            "780 eV,\n9490\nmJ/cm$^2$",
            transform=axs[1, 1].transData,
            fontsize=8,
        )
    )
    texts.append(
        axs[1, 1].text(
            780.5,
            1.8,
            "777 eV,\n8950\nmJ/cm$^2$",
            transform=axs[1, 1].transData,
            fontsize=8,
        )
    )
    texts.append(
        axs[1, 1].text(
            769.5,
            0.25,
            "781 eV,\n30\nmJ/cm$^2$",
            transform=axs[1, 1].transData,
            fontsize=8,
        )
    )
    texts.append(
        axs[1, 0].text(
            0.3,
            0.93,
            "5 fs",
            transform=axs[1, 0].transAxes,
            fontsize=10,
            backgroundcolor="w",
        )
    )
    texts.append(
        axs[1, 1].text(
            0.3,
            0.93,
            "25 fs",
            transform=axs[1, 1].transAxes,
            fontsize=10,
            backgroundcolor="w",
        )
    )
    texts.append(
        axs[1, 0].text(
            774.5,
            -0.35,
            r"$\alpha$",
            transform=axs[1, 0].transData,
            fontsize=8,
            horizontalalignment="center",
        )
    )
    texts.append(
        axs[1, 0].text(
            776.5,
            -0.35,
            r"$\beta$",
            transform=axs[1, 0].transData,
            fontsize=8,
            horizontalalignment="center",
        )
    )
    texts.append(
        axs[1, 0].text(
            778,
            -0.35,
            r"$\gamma$",
            transform=axs[1, 0].transData,
            fontsize=8,
            horizontalalignment="center",
        )
    )
    texts.append(
        axs[1, 1].text(
            774.5,
            -0.35,
            r"$\alpha$",
            transform=axs[1, 1].transData,
            fontsize=8,
            horizontalalignment="center",
        )
    )
    texts.append(
        axs[1, 1].text(
            776.5,
            -0.35,
            r"$\beta$",
            transform=axs[1, 1].transData,
            fontsize=8,
            horizontalalignment="center",
        )
    )
    texts.append(
        axs[1, 1].text(
            778,
            -0.35,
            r"$\gamma$",
            transform=axs[1, 1].transData,
            fontsize=8,
            horizontalalignment="center",
        )
    )
    for text in texts:
        text.set_path_effects(
            [
                path_effects.Stroke(linewidth=5, foreground="white"),
                path_effects.Normal(),
            ]
        )
    axs[0, 0].text(
        0.1, 0.7, "a", transform=axs[0, 0].transAxes, fontsize=10, fontweight="bold"
    )
    axs[1, 0].text(
        0.1, 0.94, "c", transform=axs[1, 0].transAxes, fontsize=10, fontweight="bold"
    )
    axs[0, 1].text(
        0.8, 0.7, "b", transform=axs[0, 1].transAxes, fontsize=10, fontweight="bold"
    )
    axs[1, 1].text(
        0.8, 0.94, "d", transform=axs[1, 1].transAxes, fontsize=10, fontweight="bold"
    )
    plt.tight_layout(w_pad=0, h_pad=0.1, rect=(0, 0.03, 1, 1))


def get_emission():
    data = np.genfromtxt("data/measuredEmissionPoints3.txt")
    emission = {"x": data[:, 0], "y": data[:, 1]}
    return emission


def spectra_series_plot(ax, data_list):
    for ind, data in enumerate(data_list):
        phot = data["sum_intact"]["phot"]
        norm = np.amax(data["sum_intact"]["no_sam_spec"])
        no_sam_spec = data["sum_intact"]["no_sam_spec"] / norm
        exc_spec = data["sum_intact"]["exc_sam_spec"] / norm
        offset = ind * 1.5
        ax.plot(phot, offset + exc_spec * 5, color="k", label="5 X Nonlin.")
        ax.plot(phot, offset + no_sam_spec, "k--", label="Incident")
        ax.fill_between(
            phot,
            offset + np.zeros_like(phot),
            offset + exc_spec * 5,
            where=(exc_spec > 0),
            facecolor="b",
            edgecolor="w",
        )
        ax.fill_between(
            phot,
            offset + np.zeros_like(phot),
            offset + exc_spec * 5,
            where=(exc_spec < 0),
            facecolor="r",
            edgecolor="w",
        )
        if ind == 0:
            handles, labels = ax.get_legend_handles_labels()
            plt.gcf().legend(handles, labels, loc=(0.35, 0.66), frameon=True)
