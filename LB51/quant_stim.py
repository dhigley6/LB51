"""Quantifying experimentally measured stimulated RIXS strength
"""

import numpy as np 
import pickle

from LB51 import LB51_get_cal_data

MEASURED_STIM_FILE = 'data/proc/stim_efficiency.pickle'

def run_quant_ana():
    """Calculate stim. efficiency of expt. data
    """
    short_data = LB51_get_cal_data.get_short_pulse_data()
    long_data = LB51_get_cal_data.get_long_pulse_data()
    short_run_sets_list = ['99', '290', '359', '388']
    short_fluences = []
    short_stim_strengths = []
    short_stim_stds = []
    for run_set in short_run_sets_list:
        fluence = short_data[run_set]["sum_intact"]["fluence"]
        #if run_set == "99":
            #fluence = 1898
        stim_strength, stim_std = get_stim_efficiency_data_set(short_data[run_set])
        short_fluences.append(fluence)
        short_stim_strengths.append(stim_strength)
        short_stim_stds.append(stim_std)
    long_run_sets_list = ["641", "554", "603"]
    long_run_sets_list = ['641', '603']
    long_fluences = []
    long_stim_strengths = []
    long_stim_stds = []
    for run_set in long_run_sets_list:
        fluence = long_data[run_set]["sum_intact"]["fluence"]
        stim_strength, stim_std = get_stim_efficiency_data_set(long_data[run_set])
        long_fluences.append(fluence)
        long_stim_strengths.append(stim_strength)
        long_stim_stds.append(stim_std)
    quant_data = {
        "short_fluences": np.array(short_fluences),
        "long_fluences": np.array(long_fluences),
        "short_efficiencies": 100 * np.array(short_stim_strengths),
        "long_efficiencies": 100 * np.array(long_stim_strengths),
        "short_stds": 100 * np.array(short_stim_stds),
        "long_stds": 100 * np.array(long_stim_stds),
    }
    with open(MEASURED_STIM_FILE, 'wb') as f:
        pickle.dump(quant_data, f)



def get_stim_efficiency_data_set(data):
    """Calculate stim. strength of data set
    """
    print(len(data['intact']['sam_spec']))
    num_splits = len(data['intact']['sam_spec']) // 7
    ssrl_nonres_absorption = data['sum_intact']['ssrl_absorption'][0]
    ssrl_res_absorption = (
        data['sum_intact']['ssrl_absorption']-ssrl_nonres_absorption
    )
    ssrl_trans = np.exp(-1*data['sum_intact']['ssrl_absorption'])
    ssrl_res_trans = np.exp(-1*ssrl_res_absorption)
    inds = np.arange(len(data['intact']['sam_spec']))
    np.random.shuffle(inds)
    print(inds)
    sam_specs = data['intact']['sam_spec'][inds]
    no_sam_specs = data['intact']['no_sam_spec'][inds]
    sam_specs = np.array_split(sam_specs, num_splits)
    no_sam_specs = np.array_split(no_sam_specs, num_splits)
    stim_efficiencies = []
    res_absorbed_sums = []
    for i in range(num_splits):
        exp_lin_sam_spec = np.sum(no_sam_specs[i], axis=0)*ssrl_trans
        exc_sam_spec = np.sum(sam_specs[i], axis=0)-exp_lin_sam_spec
        stim_efficiency = _get_stim_efficiency(
            np.sum(no_sam_specs[i], axis=0),
            np.sum(sam_specs[i], axis=0),
            exc_sam_spec,
            ssrl_res_trans,
            data['sum_intact']['phot'],
        )
        res_absorbed_sum = _get_res_absorbed_sum(
            np.sum(no_sam_specs[i], axis=0),
            np.sum(sam_specs[i], axis=0),
            data['sum_intact']['phot']
        )
        stim_efficiencies.append(stim_efficiency)
        res_absorbed_sums.append(res_absorbed_sum)
    stim_efficiency_avg = np.average(stim_efficiencies, weights=res_absorbed_sums)
    stim_efficiency_std = np.sqrt(np.cov(stim_efficiencies, aweights=res_absorbed_sums))/np.sqrt(num_splits)
    return stim_efficiency_avg, stim_efficiency_std

def _get_stim_efficiency(no_sam_spec, sam_spec, exc_sam_spec, ssrl_res_trans, phot):
    res_transmitted = no_sam_spec*ssrl_res_trans
    res_absorbed = no_sam_spec-res_transmitted
    abs_region = (phot > 774) & (phot < 780)
    res_absorbed_sum = np.trapz(res_absorbed[abs_region])
    stim_region = (phot > 773) & (phot < 775)
    stim = exc_sam_spec
    #stim[stim < 0] = 0    # clip negative values to zero
    stim_sum = np.trapz(stim[stim_region])
    stim_efficiency = stim_sum / res_absorbed_sum
    return stim_efficiency

def _get_res_absorbed_sum(no_sam_spec, ssrl_res_trans, phot):
    res_transmitted = no_sam_spec*ssrl_res_trans
    res_absorbed = no_sam_spec-res_transmitted
    abs_region = (phot > 774) & (phot < 780)
    res_absorbed_sum = np.trapz(res_absorbed[abs_region])
    return -1*res_absorbed_sum

def _get_bootstrap_inds(inds):
    