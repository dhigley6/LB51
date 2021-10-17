"""Make raw data plots corresponding to summary figure of main text
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects

from LB51 import LB51_get_cal_data
from LB51.manuscript_plots import set_plot_params

set_plot_params.init_paper_small()

EMISSION_SHIFT = 0.4  # photon energy shift from reference data

def make_figure():
    long_data = LB51_get_cal_data.get_long_pulse_data()
    short_data = LB51_get_cal_data.get_short_pulse_data()
    binned_603 = LB51_get_cal_data.bin_burst_data(long_data["603"]["total"], 603)
    binned_99 = LB51_get_cal_data.bin_burst_data(short_data["99"]["total"], 99)
    short_summary_data = [short_data["359"]]
    short_summary_data.extend(binned_99)
    # The below line changes the order to be in order of ascending fluence
    short_summary_data = [short_summary_data[0], short_summary_data[2], short_summary_data[1]]
    long_summary_data = [long_data["641"]]
    long_summary_data.extend(binned_603)
    f, axs = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(4, 4))
    spectra_series_plot(axs[0], short_summary_data)
    spectra_series_plot(axs[1], long_summary_data)
    format_figure(f, axs)
    plt.savefig("manuscript_plots/2021_10_11_supplement_raw_specs.eps", dpi=600)
    plt.savefig("manuscript_plots/2021_10_11_supplement_raw_specs.png", dpi=600)

def spectra_series_plot(ax, data_list):
    for ind, data in enumerate(data_list):
        phot = data["sum_intact"]["phot"]
        norm = np.amax(data["sum_intact"]["no_sam_spec"])
        sam_spec = data["sum_intact"]["sam_spec"] / norm
        no_sam_spec = data["sum_intact"]["no_sam_spec"] / norm
        exc_spec = data["sum_intact"]["exc_sam_spec"] / norm
        ssrl_trans = np.exp(-1 * data["sum_intact"]["ssrl_absorption"])
        lin_sam_spec = no_sam_spec * ssrl_trans
        offset = ind * 1.5
        ax.plot(phot, offset + no_sam_spec, "k--", label="Incident")
        ax.plot(phot, offset + sam_spec, 'k', label='Co/Pd')
        ax.plot(phot, offset + lin_sam_spec, "k:", label="Estimated\nLinear Co/Pd")
        ax.fill_between(
            phot,
            lin_sam_spec+offset,
            sam_spec+offset,
            where=(exc_spec > 0),
            facecolor='b',
            edgecolor='w',
        )
        ax.fill_between(
            phot,
            lin_sam_spec+offset,
            sam_spec+offset,
            where=(exc_spec < 0),
            facecolor='r',
            edgecolor='w',
        )
        """
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
        """
        if ind == 0:
            handles, labels = ax.get_legend_handles_labels()
            plt.gcf().legend(handles, labels, loc=(0.38, 0.82), frameon=True)

def format_figure(f, axs):
    axs[0].yaxis.set_ticks([])
    axs[0].xaxis.set_ticks([770, 775, 780, 785])
    axs[0].set_xlim((769, 787))
    axs[0].set_ylim((-0.5, 4.75))
    f.text(0.52, 0.02, "Photon Energy (eV)", ha="center")
    axs[0].set_ylabel("Intensity")
    texts = []
    texts.append(
        axs[0].text(
            780.5,      #769.5
            3.3,        # 3.2
            '1600\nmJ/cm$^2$',
            #"779 eV,\n1080\nmJ/cm$^2$",
            transform=axs[0].transData,
            fontsize=8,
        )
    )
    texts.append(
        axs[0].text(
            769.5,
            1.7,
            '1080\nmJ/cm$^2$',
            #"777 eV,\n1600\nmJ/cm$^2$",
            transform=axs[0].transData,
            fontsize=8,
        )
    )
    texts.append(
        axs[0].text(
            769.5,
            0.25,
            '12\nmJ/cm$^2$',
            #"779 eV,\n12\nmJ/cm$^2$",
            transform=axs[0].transData,
            fontsize=8,
        )
    )
    texts.append(
        axs[1].text(
            769.5,
            3.2,
            '9490\nmJ/cm$^2$',
            #"780 eV,\n9490\nmJ/cm$^2$",
            transform=axs[1].transData,
            fontsize=8,
        )
    )
    texts.append(
        axs[1].text(
            780.5,
            1.8,
            '8950\nmJ/cm$^2$',
            #"777 eV,\n8950\nmJ/cm$^2$",
            transform=axs[1].transData,
            fontsize=8,
        )
    )
    texts.append(
        axs[1].text(
            769.5,
            0.25,
            '30\nmJ/cm$^2$',
            #"781 eV,\n30\nmJ/cm$^2$",
            transform=axs[1].transData,
            fontsize=8,
        )
    )
    texts.append(
        axs[0].text(
            0.3,
            0.93,
            "5 fs",
            transform=axs[0].transAxes,
            fontsize=10,
            backgroundcolor="w",
        )
    )
    texts.append(
        axs[1].text(
            0.3,
            0.93,
            "25 fs",
            transform=axs[1].transAxes,
            fontsize=10,
            backgroundcolor="w",
        )
    )
    for text in texts:
        text.set_path_effects(
            [
                path_effects.Stroke(linewidth=5, foreground="white"),
                path_effects.Normal(),
            ]
        )
    axs[0].text(
        0.1, 0.94, "a", transform=axs[0].transAxes, fontsize=10, fontweight="bold"
    )
    axs[1].text(
        0.8, 0.94, "b", transform=axs[1].transAxes, fontsize=10, fontweight="bold"
    )
    axs[0].set_xlabel(' ')
    plt.tight_layout()
