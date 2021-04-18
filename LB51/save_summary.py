"""Save summary data for Markus
Created on 2020-02-05
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt

from LB51 import LB51_get_cal_data

MARKUS_FILES_START = "data/markus_data_"


def load_data():
    long_data = LB51_get_cal_data.get_long_pulse_data()
    short_data = LB51_get_cal_data.get_short_pulse_data()
    binned_603 = LB51_get_cal_data.bin_burst_data(long_data["603"]["total"], 603)
    binned_99 = LB51_get_cal_data.bin_burst_data(short_data["99"]["total"], 99)
    data_sets = {
        "short_pulse_low_fluence": short_data["359"]["sum_intact"],
        "short_pulse_high_fluence_low_photon_energy": binned_99[0]["sum_intact"],
        "short_pulse_high_fluence_high_photon_energy": binned_99[0]["sum_intact"],
        "long_pulse_low_fluence": long_data["641"]["sum_intact"],
        "long_pulse_high_fluence_low_photon_energy": binned_603[0]["sum_intact"],
        "long_pulse_high_fluence_high_photon_energy": binned_603[1]["sum_intact"],
    }
    return data_sets


def save_markus_csv_data():
    header = "row 1: photon energy, row 2: no sample spectrum, row 3: difference from linear response"
    data_sets = load_data()
    for data_name, data in data_sets.items():
        file_path = MARKUS_FILES_START + data_name + ".csv"
        to_save = (data["phot"], data["no_sam_spec"], data["exc_sam_spec"])
        np.savetxt(file_path, to_save, delimiter=",", header=header)


def save_markus_data():
    long_data = LB51_get_cal_data.get_long_pulse_data()
    short_data = LB51_get_cal_data.get_short_pulse_data()
    binned_603 = LB51_get_cal_data.bin_burst_data(long_data["603"]["total"], 603)
    binned_99 = LB51_get_cal_data.bin_burst_data(short_data["99"]["total"], 99)
    data_sets_to_save = {
        "short_pulse_low_fluence": short_data["359"]["sum_intact"],
        "short_pulse_high_fluence_low_photon_energy": binned_99[0]["sum_intact"],
        "short_pulse_high_fluence_high_photon_energy": binned_99[0]["sum_intact"],
        "long_pulse_low_fluence": long_data["641"]["sum_intact"],
        "long_pulse_high_fluence_low_photon_energy": binned_603[0]["sum_intact"],
        "long_pulse_high_fluence_high_photon_energy": binned_603[1]["sum_intact"],
    }
    with open("markus_data.pickle", "wb") as f:
        pickle.dump(data_sets_to_save, f)
    make_test_figure(data_sets_to_save)


def load_markus_data():
    with open("markus_data.pickle", "rb") as f:
        data = pickle.load(f)
    make_test_figure(data)
    return data


def make_test_figure(data):
    """Make a test figure to make sure the saved data is correct"""
    _, axs = plt.subplots(3, 2, sharex=True)
    plot_spectra(axs[0, 0], data, "short_pulse_low_fluence")
    plot_spectra(axs[1, 0], data, "short_pulse_high_fluence_low_photon_energy")
    plot_spectra(axs[2, 0], data, "short_pulse_high_fluence_high_photon_energy")
    plot_spectra(axs[0, 1], data, "long_pulse_low_fluence")
    plot_spectra(axs[1, 1], data, "long_pulse_high_fluence_low_photon_energy")
    plot_spectra(axs[2, 1], data, "long_pulse_high_fluence_high_photon_energy")


def plot_spectra(ax, data, k):
    ax.plot(data[k]["phot"], data[k]["no_sam_spec"])
    ax.plot(data[k]["phot"], data[k]["exc_sam_spec"])
