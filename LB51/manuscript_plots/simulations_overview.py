"""Make a figure showing how simulations work
"""

import numpy as np
import matplotlib.pyplot as plt

from LB51.xbloch import do_xbloch_sim
from LB51.manuscript_plots import set_plot_params

set_plot_params.init_paper_small()


def simulations_overview():
    sim_results = do_xbloch_sim.load_multipulse_data(5.0)  # SASE pulses case
    f, axs = plt.subplots(2, 1, sharex=True, figsize=(3.37, 3.5))
    to_plot = sim_results["summed_incident_intensities"][0]
    to_plot = to_plot / np.amax(to_plot)
    axs[0].plot(sim_results["phot"], to_plot)
    inds_to_plot = [0, -7, -5, -3]
    for i in inds_to_plot:
        intensity_difference = (
            sim_results["summed_transmitted_intensities"][i]
            - sim_results["summed_incident_intensities"][i]
        ) / sim_results["fluences"][i]
        if i == 0:
            label = "10 nJ/cm$^2$"
            norm = np.amax(np.abs(intensity_difference))
        else:
            label = str(int(1e3 * sim_results["fluences"][i])) + " mJ/cm$^2$"
        axs[1].plot(sim_results["phot"], intensity_difference / norm, label=label)
    axs[0].set_xlim((772, 782))
    axs[0].set_ylabel("Intensity (a.u.)")
    axs[1].set_ylabel("Intensity (a.u.)")
    axs[1].set_xlabel("Photon Energy (eV)")
    axs[1].legend(loc="best", title="Fluence")
    axs[0].text(
        0.9,
        0.9,
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
    plt.tight_layout()
    plt.savefig("plots/2020_09_09_sim_overview.eps", dpi=600)
    plt.savefig("plots/2020_09_09_sim_overview.png", dpi=600)
