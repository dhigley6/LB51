"""Plot simulated incident and transmitted spectra for a series of spectra
"""

import numpy as np
import matplotlib.pyplot as plt


def make_figure(strengths, sim_results):
    _, axs = plt.subplots(3, 1, sharex=True, figsize=(6, 3.37))
    for i, sim_result in enumerate(sim_results):
        phot_result = sim_result.phot_results_
        axs[0].plot(
            phot_result["phots"],
            np.abs(phot_result["E_in"]) ** 2 / (strengths[i]),
            label=np.format_float_scientific(strengths[i] * 1e15, precision=2),
        )
        axs[1].plot(
            phot_result["phots"], np.abs(phot_result["E_out"]) ** 2 / (strengths[i])
        )
        spec_difference = (
            np.abs(phot_result["E_out"]) ** 2 - np.abs(phot_result["E_in"]) ** 2
        ) / (strengths[i])
        axs[2].plot(phot_result["phots"], spec_difference)
    _format_figure(axs)


def _format_figure(axs):
    axs[0].set_title("Incident Spectra")
    axs[1].set_title("Transmitted Spectra")
    axs[2].set_title("Difference of Transmitted and Incident Spectra")
    axs[1].set_xlim((773, 783))
    axs[2].set_xlabel("Photon Energy Above 778 (eV)")
    plt.gcf().legend(loc=7, title="Peak Intensity\n(W/cm$^2$)")
    plt.gcf().tight_layout()
    plt.gcf().subplots_adjust(right=0.8)
