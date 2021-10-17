"""Plot for simulations with elastic scatteing enhancement factor proposed in
Stohr&Scherz
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt

from LB51.manuscript_plots import set_plot_params

set_plot_params.init_paper_small()

RESULTS_FILE = "LB51/xbloch/results/enhanced_5fs.pickle"


def simulations_with_enhancement():
    loaded_result = _load_data()
    f, axs = plt.subplots(2, 1, sharex=True, figsize=(3.37, 3.5))
    incident = loaded_result["summed_incident_intensities"][0]
    incident = incident / np.amax(incident)
    axs[0].plot(loaded_result["phot"], incident)
    inds_to_plot = [0, -7, -5, -3]
    for i in inds_to_plot:
        intensity_difference = (
            loaded_result["summed_transmitted_intensities"][i]
            - loaded_result["summed_incident_intensities"][i]
        ) / loaded_result["fluences"][i]
        if i == 0:
            label = "10 nJ/cm$^2$"
            norm = np.amax(np.abs(intensity_difference))
        else:
            label = str(int(1e3 * loaded_result["fluences"][i])) + " mJ/cm$^2$"
        axs[1].plot(loaded_result["phot"], intensity_difference / norm, label=label)
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
    plt.savefig("manuscript_plots/2020_09_18_enhancement_sim_overview.eps", dpi=600)
    plt.savefig("manuscript_plots/2020_09_18_enhancement_sim_overview.png", dpi=600)


def _load_data():
    with open(RESULTS_FILE, "rb") as f:
        loaded_result = pickle.load(f)
    return loaded_result
