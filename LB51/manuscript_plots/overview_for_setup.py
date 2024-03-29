"""
Similar to overview plot, but basically just used 
to get some single shot spectra for putting into the
experimental schematic.
"""

import numpy as np
import matplotlib.pyplot as plt

from LB51 import LB51_get_cal_data
from LB51.manuscript_plots import set_plot_params

set_plot_params.init_paper_small()

NO_SAM = 1  # Factor to multiply no sample spectra by when plotting


def overview_plot():
    """Make plots summarizing the obtained data"""

    def individual_shots_plot(ax, data, shots_to_plot):
        phot = data["sum_intact"]["phot"]
        for ind, shot in enumerate(shots_to_plot):
            sam_spec = data["blown"]["sam_spec"][shot]
            no_sam_spec = data["blown"]["no_sam_spec"][shot]
            norm_sam_spec = sam_spec / np.amax(sam_spec) / 1.5
            norm_no_sam_spec = no_sam_spec * NO_SAM / np.amax(sam_spec) / 1.5
            if ind == 0:
                sam_label = "Co/Pd Path"
                no_sam_label = "Blank SiN Path"
            else:
                sam_label = "_nolegend_"
                no_sam_label = "_nolegend_"
            ax.plot(phot, norm_sam_spec + ind, "r", label=sam_label)
            ax.plot(phot, norm_no_sam_spec + ind, "k--", label=no_sam_label)
            if ind == 0:
                ax.legend(loc="best")

    def summed_spectra_plot(ax, data):
        phot = data["sum_intact"]["phot"]
        norm = np.amax(data["sum_intact"]["no_sam_spec"])
        sam_spec = data["sum_intact"]["sam_spec"] / norm
        no_sam_spec = data["sum_intact"]["no_sam_spec"] / norm
        exc_spec = data["sum_intact"]["exc_sam_spec"] / norm
        ssrl_trans = np.exp(-1 * data["sum_intact"]["ssrl_absorption"])
        lin_sam_spec = no_sam_spec * ssrl_trans
        ax.plot(phot, no_sam_spec * NO_SAM, "k--", label="Ref.")
        ax.plot(phot, sam_spec, "k", label="Sam.")
        ax.plot(phot, lin_sam_spec, "k:", label="Linear Sam.")
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

    def nonlinear_spectra_plot(ax, data):
        phot = data["sum_intact"]["phot"]
        norm = np.amax(data["sum_intact"]["no_sam_spec"])
        no_sam_spec = data["sum_intact"]["no_sam_spec"] / norm
        exc_spec = data["sum_intact"]["exc_sam_spec"] / norm
        ax.plot(phot, exc_spec * 5, color="k", label="5 X Diff.")
        ax.plot(phot, no_sam_spec, "k--", label="Ref.")
        ax.fill_between(
            phot,
            np.zeros_like(phot),
            exc_spec * 5,
            where=(exc_spec > 0),
            facecolor="b",
            edgecolor="w",
        )
        ax.fill_between(
            phot,
            np.zeros_like(phot),
            exc_spec * 5,
            where=(exc_spec < 0),
            facecolor="r",
            edgecolor="w",
        )

    def xas_plot(ax, low_data, high_data, threshold=0.15):
        high_phot = high_data["sum_intact"]["phot"]
        high_threshold = threshold * np.amax(high_data["sum_intact"]["no_sam_spec"])
        high_in_range = high_data["sum_intact"]["no_sam_spec"] > high_threshold
        ax.plot(
            high_phot[high_in_range],
            high_data["sum_intact"]["abs"][high_in_range],
            color="tab:orange",
            label="1335\nmJ/cm$^2$",
        )
        low_phot = low_data["sum_intact"]["phot"]
        low_threshold = threshold * np.amax(low_data["sum_intact"]["no_sam_spec"])
        low_in_range = low_data["sum_intact"]["no_sam_spec"] > low_threshold
        ax.plot(
            low_phot[low_in_range],
            low_data["sum_intact"]["abs"][low_in_range],
            "g",
            label="12\nmJ/cm$^2$",
        )
        ax.plot(
            low_phot, low_data["sum_intact"]["ssrl_absorption"], "k--", label="Sync."
        )

    short_data = LB51_get_cal_data.get_short_pulse_data()
    long_data = LB51_get_cal_data.get_long_pulse_data()
    plt.figure(figsize=(3.3, 2.5))
    f = plt.gcf()
    ax = plt.gca()
    shots_to_plot = [
        20,
        23,
        26,
    ]  # last shots on some samples (blown up) after the first few
    individual_shots_plot(ax, long_data["603"], shots_to_plot)
    new_format_plot(f, ax)

    # summed_spectra_plot(axs[1, 0], short_data["99"])
    # nonlinear_spectra_plot(axs[0, 1], short_data["99"])
    # xas_plot(axs[1, 1], short_data["359"], short_data["99"])
    # format_data_overview_plot(f, axs)
    plt.savefig("manuscript_plots/2021_09_25_overview_for_setup.eps", dpi=600)
    plt.savefig("manuscript_plots/2021_09_25_overview_for_setup.png", dpi=600)


def new_format_plot(f, ax):
    ax.set_xlim((770, 785))
    ax.set_xticks((770, 775, 780, 785))
    ax.set_xlabel("Photon Energy (eV)")
    ax.set_yticks([])
    ax.set_ylabel("Intensity")
    ax.legend(loc="lower left", frameon=False, ncol=2)
    ax.set_ylim((-0.5, 3))
    plt.title("Spectra Recorded with Samples Removed")
    plt.tight_layout()


def format_data_overview_plot(f, axs):
    axs[0, 0].set_xlim((770, 788))
    axs[0, 0].set_xticks((770, 775, 780, 785))
    axs[0, 0].set_xlabel("Photon Energy (eV)")
    axs[1, 1].set_xlim((770, 788))
    axs[1, 1].set_xticks((770, 775, 780))
    axs[0, 1].yaxis.tick_right()
    axs[1, 1].yaxis.tick_right()
    axs[1, 0].set_xlabel(".", color=(0, 0, 0, 0))
    axs[0, 0].set_yticks([])
    axs[1, 0].set_yticks([])
    axs[0, 1].set_yticks([])
    axs[1, 1].set_yticks([])
    axs[0, 0].set_ylabel("Intensity")
    axs[1, 0].set_ylabel("Intensity")
    axs[0, 1].set_ylabel("Intensity")
    axs[1, 1].set_ylabel("XAS")
    axs[0, 1].yaxis.set_label_position("right")
    axs[1, 1].yaxis.set_label_position("right")
    # axs[0, 0].text(
    #    0.9,
    #    0.9,
    #    "a",
    #    fontsize=10,
    #    weight="bold",
    #    horizontalalignment="center",
    #    transform=axs[0, 0].transAxes,
    # )
    axs[1, 0].text(
        0.9,
        0.9,
        "b",
        fontsize=10,
        weight="bold",
        horizontalalignment="center",
        transform=axs[1, 0].transAxes,
    )
    axs[0, 1].text(
        0.9,
        0.9,
        "c",
        fontsize=10,
        weight="bold",
        horizontalalignment="center",
        transform=axs[0, 1].transAxes,
    )
    axs[1, 1].text(
        0.9,
        0.9,
        "d",
        fontsize=10,
        weight="bold",
        horizontalalignment="center",
        transform=axs[1, 1].transAxes,
    )
    f.text(0.5, 0.04, "Photon Energy (eV)", va="center", ha="center")
    axs[0, 0].legend(loc="lower left", frameon=False, ncol=2)
    axs[0, 1].legend(loc="upper left", frameon=False)
    axs[1, 0].legend(loc="lower center", frameon=False)
    axs[1, 1].legend(loc="best", frameon=False, fontsize=8)
    axs[0, 1].axhline(linestyle=":", color="k")
    axs[0, 0].set_ylim((-0.5, 3))
    axs[1, 0].set_ylim((-0.5, 1.1))
    axs[0, 1].set_ylim((-0.3, 1.25))
    axs[0, 1].annotate(
        r"$\alpha$",
        xy=(774.5, 0.1),
        xycoords="data",
        xytext=(772, 0.6),
        textcoords="data",
        arrowprops=dict(facecolor="black", shrink=0.05, width=1, headwidth=4),
        fontsize=10,
    )
    axs[0, 1].annotate(
        r"$\beta$",
        xy=(776.25, -0.05),
        xycoords="data",
        xytext=(780, -0.2),
        textcoords="data",
        arrowprops=dict(facecolor="black", shrink=0.05, width=1, headwidth=4),
        fontsize=10,
        verticalalignment="center",
    )
    axs[0, 1].annotate(
        r"$\gamma$",
        xy=(778, 0.15),
        xycoords="data",
        xytext=(780.5, 0.8),
        textcoords="data",
        arrowprops=dict(facecolor="black", shrink=0.05, width=1, headwidth=4),
        fontsize=10,
    )
    # plt.tight_layout(pad=0.8, w_pad=0, h_pad=0)
    plt.tight_layout()
