"""Save processed spectra that are plotted in manuscript
"""

import numpy as np
import pandas as pd

from LB51 import LB51_get_cal_data
from LB51 import LB51_get_synchro

EMISSION_SHIFT = 0.4  # photon energy shift from reference data

def save_data():
    save_short_data()
    save_long_data()
    emission = get_emission()
    save_emission(emission)
    save_ssrl_absorption()

def save_ssrl_absorption():
    data = LB51_get_synchro.get_copd_xas()
    to_save = {
        'photon energy': data['phot'],
        'xas': data['xas']
    }
    df_to_save = pd.DataFrame.from_dict(
        to_save
    )
    df_to_save.to_csv('../data_spectra/ssrl_xas.csv')

def get_emission():
    data = np.genfromtxt("data/measuredEmissionPoints3.txt")
    emission = {"x": data[:, 0], "y": data[:, 1]}
    emission['x'] = emission['x']-EMISSION_SHIFT
    return emission

def save_emission(emission):
    emission = get_emission()
    to_save = pd.DataFrame.from_dict(emission)
    to_save.to_csv('../data_spectra/synchrotron_emission.csv')

def save_long_data():
    long_data = LB51_get_cal_data.get_long_pulse_data()
    binned_603 = LB51_get_cal_data.bin_burst_data(long_data["603"]["total"], 603)
    long_summary_data = [long_data['641']]
    long_summary_data.extend(binned_603)
    long_summary_data.extend([long_data['603']])
    file_start = '../data_spectra/25fs_'
    long_summary_filenames = [
        ''.join([file_start, 'Low_Fluence.csv']),
        ''.join([file_start, 'High_Fluence_Low_Photon_Energy.csv']),
        ''.join([file_start, 'High_Fluence_High_Photon_Energy.csv']),
        ''.join([file_start, 'High_Fluence_Total.csv'])
    ]
    for data, filename in zip(long_summary_data, long_summary_filenames):
        save_dataset(data, filename)

def save_short_data():
    short_data = LB51_get_cal_data.get_short_pulse_data()
    binned_99 = LB51_get_cal_data.bin_burst_data(short_data["99"]["total"], 99)
    short_summary_data = [short_data["359"]]
    short_summary_data.extend(binned_99)
    short_summary_data.extend([short_data['99']])
    file_start = '../data_spectra/5fs_'
    short_summary_filenames = [
        ''.join([file_start, 'Low_Fluence.csv']),
        ''.join([file_start, 'High_Fluence_Low_Photon_Energy.csv']),
        ''.join([file_start, 'High_Fluence_High_Photon_Energy.csv']),
        ''.join([file_start, 'High_Fluence_Total.csv'])
    ]
    for data, filename in zip(short_summary_data, short_summary_filenames):
        save_dataset(data, filename)

def save_dataset(data, filename):
    norm = np.amax(data['sum_intact']['no_sam_spec'])
    no_sam_spec = data["sum_intact"]["no_sam_spec"] / norm
    exc_spec = data["sum_intact"]["exc_sam_spec"] / norm
    sam_spec = data["sum_intact"]["sam_spec"] / norm
    lin_sam_spec = sam_spec-exc_spec
    to_save = {
        'photon energy': data["sum_intact"]["phot"],
        'no sample spectrum': no_sam_spec,
        'sample spectrum with linear sample response': lin_sam_spec,
        'sample spectrum': sam_spec,
    }
    df_to_save = pd.DataFrame.from_dict(to_save)
    df_to_save.to_csv(filename)