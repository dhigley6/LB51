"""Quantifying experimentally measured stimulated RIXS strength
"""

import numpy as np 
import pickle
import matplotlib.pyplot as plt
from scipy.stats import percentileofscore

from LB51 import LB51_get_cal_data

MEASURED_STIM_FILE = 'data/proc/stim_efficiency.pickle'

SHORT_DATA = LB51_get_cal_data.get_short_pulse_data()
LONG_DATA = LB51_get_cal_data.get_long_pulse_data()

SHORT_RUN_SETS_LIST = ['99', '290', '359', '388']
SHORT_RUN_SETS_LIST = ['99', '290', '359']   # remove run set 388 since that one has very limited data
LONG_RUN_SETS_LIST = ['641', '603']
SSRL_ABSORPTION_COPD = SHORT_DATA['99']['sum_intact']['ssrl_absorption']

OFFSET_REFERENCE = {
    '99': SHORT_DATA['99'],
    '290': SHORT_DATA['99'],
    '359': SHORT_DATA['99'],
    '388': SHORT_DATA['99'],
    '603': LONG_DATA['603'],
    '641': LONG_DATA['603'],
}

BOOTSTRAPS = 1000

def run_quant_ana():
    """Calculate stim. efficiency of expt. data
    Saves results in MEASURED_STIM_FILE
    """
    short_results = _get_data_set_results(SHORT_RUN_SETS_LIST, SHORT_DATA)
    long_results = _get_data_set_results(LONG_RUN_SETS_LIST, LONG_DATA)
    quant_data = {
        "short_fluences": np.array(short_results['fluences']),
        "long_fluences": np.array(long_results['fluences']),
        "short_efficiencies": 100 * np.array(short_results['stim_efficiencies']),
        "long_efficiencies": 100 * np.array(long_results['stim_efficiencies']),
        "short_stds": 100 * np.array(short_results['stim_stds']),
        "long_stds": 100 * np.array(long_results['stim_stds']),
    }
    with open(MEASURED_STIM_FILE, 'wb') as f:
        pickle.dump(quant_data, f)

def _calculate_bootstrap_std(data):
    if data['sum_intact']['fluence'] > 100:
        _, error = _get_blown_stim_efficiency_data_set(data)
    else:
        _, error = _get_stim_efficiency_data_set(data)
    return error

def _calculate_stim_offset(run_set):
    offset_data = OFFSET_REFERENCE[run_set]
    offset_stim_efficiency, _ = _get_blown_stim_efficiency_data_set(offset_data)
    return offset_stim_efficiency

def _make_model_with_sample_from_blown_up(data_set):
    """Make model expected (linear) data that would be recorded from
    data recorded with blown up samples
    """
    no_sam_specs = data_set['blown']['no_sam_spec']
    ssrl_absorption = data_set['sum_intact']['ssrl_absorption']
    ssrl_transmission = np.exp(-1*ssrl_absorption)
    sam_specs = data_set['blown']['sam_spec']*ssrl_transmission
    phot = data_set['sum_intact']['phot']
    return {
        'no_sam_spec': no_sam_specs,
        'sam_spec': sam_specs,
        'ssrl_absorption': ssrl_absorption,
        'phot': phot
    }

def _get_data_set_results(run_sets_list, data_sets):
    """Return fluences, stim. efficiencies and standard errors for run sets
    """
    to_return = {
        'fluences': [],
        'stim_efficiencies': [],
        'stim_stds': [],
    }
    for run_set in run_sets_list:
        fluence = data_sets[run_set]['sum_intact']['fluence']
        stim_efficiency, stim_std = _get_stim_efficiency_data_set(data_sets[run_set])
        stim_std = _calculate_bootstrap_std(data_sets[run_set])
        stim_offset = _calculate_stim_offset(run_set)
        stim_efficiency = stim_efficiency-stim_offset
        to_return['fluences'].append(fluence)
        to_return['stim_efficiencies'].append(stim_efficiency)
        to_return['stim_stds'].append(stim_std)
    return to_return

def _get_blown_stim_efficiency_data_set(data):
    model_blown = _make_model_with_sample_from_blown_up(
        data
    )
    num_samples = len(data['intact']['no_sam_spec'])
    blown_bootstrap_stim_efficiencies = _bootstrap_stim_efficiencies(
        model_blown['no_sam_spec'],
        model_blown['sam_spec'],
        model_blown['ssrl_absorption'],
        model_blown['phot'],
        num_samples,
    )
    blown_stim_efficiency = _get_stim_efficiency(
        np.sum(model_blown['no_sam_spec'], axis=0),
        np.sum(model_blown['sam_spec'], axis=0),
        model_blown['ssrl_absorption'],
        model_blown['phot'],
    )
    return blown_stim_efficiency, np.std(blown_bootstrap_stim_efficiencies)

def _get_stim_efficiency_data_set(data):
    """Calculate stim. strength of data set
    """
    stim_efficiencies = _bootstrap_stim_efficiencies(
        data['intact']['no_sam_spec'],
        data['intact']['sam_spec'],
        data['sum_intact']['ssrl_absorption'],
        data['sum_intact']['phot'],
    )
    stim_efficiency = _get_stim_efficiency(
        np.sum(data['intact']['no_sam_spec'], axis=0),
        np.sum(data['intact']['sam_spec'], axis=0),
        data['sum_intact']['ssrl_absorption'],
        data['sum_intact']['phot'],
    )
    return stim_efficiency, np.std(stim_efficiencies)

def _get_exc_sam_spec(no_sam_spec, sam_spec, ssrl_absorption):
    """Return output sample spectrum minus that expected for linear sample response
    """
    ssrl_transmission = np.exp(-1*ssrl_absorption)
    exp_lin_sam_spec = no_sam_spec*ssrl_transmission
    exc_sam_spec = sam_spec-exp_lin_sam_spec
    return exc_sam_spec

def _get_ssrl_res_trans():
    """Return resonant component of SSRL absorption
    """
    ssrl_nonres_absorption = SSRL_ABSORPTION_COPD[0]
    ssrl_res_absorption = (
        SSRL_ABSORPTION_COPD-ssrl_nonres_absorption
    )
    ssrl_res_trans = np.exp(-1*ssrl_res_absorption)
    return ssrl_res_trans


def _bootstrap_stim_efficiencies(no_sam_specs, sam_specs, ssrl_absorption, phot, num_samples=None):
    inds = np.arange(len(no_sam_specs))
    if num_samples is None:
        num_samples = len(inds)
    bootstrap_inds_list = [np.random.choice(inds, size=num_samples) for _ in range(BOOTSTRAPS)]
    bootstrapped_stim_efficiencies = []
    for bootstrap_inds in bootstrap_inds_list:
        no_sam_spec = np.sum(no_sam_specs[bootstrap_inds], axis=0)
        sam_spec = np.sum(sam_specs[bootstrap_inds], axis=0)
        bootstrapped_stim_efficiencies.append(_get_stim_efficiency(
            no_sam_spec,
            sam_spec,
            ssrl_absorption,
            phot
        ))
    return bootstrapped_stim_efficiencies

def _get_stim_efficiency(no_sam_spec, sam_spec, ssrl_absorption, phot):
    exc_sam_spec = _get_exc_sam_spec(
        no_sam_spec,
        sam_spec,
        ssrl_absorption,
    )
    res_absorbed_sum = _get_res_absorbed_sum(
        no_sam_spec,
        phot
    )
    stim_region = (phot > 773) & (phot < 775.5)
    stim = exc_sam_spec
    #stim[stim < 0] = 0    # clip negative values to zero
    stim_sum = np.trapz(stim[stim_region])
    res_absorbed_sum = np.sum(no_sam_spec)/5
    stim_efficiency = stim_sum / res_absorbed_sum
    return stim_efficiency

def _get_res_absorbed_sum(no_sam_spec, phot):
    ssrl_res_trans = _get_ssrl_res_trans()
    res_transmitted = no_sam_spec*ssrl_res_trans
    res_absorbed = no_sam_spec-res_transmitted
    abs_region = (phot > 774) & (phot < 780)
    res_absorbed_sum = np.trapz(res_absorbed[abs_region])
    return res_absorbed_sum

def experiment(run_set=LONG_DATA['603']):
    model_blown_data = _make_model_with_sample_from_blown_up(run_set)
    result_blown = _bootstrap_stim_efficiencies(
        model_blown_data['no_sam_spec'],
        model_blown_data['sam_spec'],
        model_blown_data['ssrl_absorption'],
        model_blown_data['phot'],
    )
    result_intact = _bootstrap_stim_efficiencies(
        run_set['intact']['no_sam_spec'],
        run_set['intact']['sam_spec'],
        run_set['sum_intact']['ssrl_absorption'],
        run_set['sum_intact']['phot'],
    )
    plt.figure()
    plt.hist(result_intact, bins=100, alpha=0.5, label='intact')
    plt.hist(result_blown, bins=100, alpha=0.5, label='blown')
    plt.legend()
    print(percentileofscore(result_intact, 0))


    """
    print(_get_stim_efficiency(
        run_set['sum_blown']['no_sam_spec'],
        run_set['sum_blown']['sam_spec'],
        run_set['sum_blown']['ssrl_absorption'],
        run_set['sum_blown']['phot']
    ))
    print(_get_stim_efficiency(
        run_set['sum_intact']['no_sam_spec'],
        run_set['sum_intact']['sam_spec'],
        run_set['sum_intact']['ssrl_absorption'],
        run_set['sum_intact']['phot']
    ))
    """