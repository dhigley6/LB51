"""Save some data for Jo
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects

from LB51 import LB51_get_cal_data
from LB51.manuscript_plots import set_plot_params

set_plot_params.init_paper_small()

EMISSION_SHIFT = 0.4  # photon energy shift from reference data

def save_data():
    """Save high fluence, high photon energy, long pusle druation spectra for Jo
    """
    long_data = LB51_get_cal_data.get_long_pulse_data()
    binned_603 = LB51_get_cal_data.bin_burst_data(long_data["603"]["total"], 603)
    data = binned_603[1]
    phot = data["sum_intact"]["phot"]
    norm = np.amax(data["sum_intact"]["no_sam_spec"])
    no_sam_spec = data["sum_intact"]["no_sam_spec"] / norm
    exc_spec = data["sum_intact"]["exc_sam_spec"] / norm
    sam_spec = data["sum_intact"]["sam_spec"] / norm
    lin_sam_spec = sam_spec-exc_spec
    plt.figure()
    plt.plot(phot, no_sam_spec)
    plt.plot(phot, lin_sam_spec)
    plt.plot(phot, sam_spec)
    data_to_save = pd.DataFrame.from_dict(
        {
            'photon energy': phot,
            'no sample spectrum': no_sam_spec,
            'sample spectrum with linear sample response': lin_sam_spec,
            'sample spectrum': sam_spec,
        }
    )
    data_to_save.to_csv('../data_for_jo_spectrum/spectra_data.csv')