"""Return calibrated data for LB51 experiment
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import scoreatpercentile

from LB51 import LB51_h52rec, LB51_calibman, LB51_get_synchro

####
BEAMLINE_TRANS = 0.1
# Parameters of XAS spectra (in eV):
Co_L3EDGE_MAX = 778.15
####


def bin_burst_data(data, run):
    percentiles = [0, 50, 100]
    ebeam_bins = scoreatpercentile(data["ebeam"], percentiles)
    ebeam_bin_edges = zip(ebeam_bins[:-1], ebeam_bins[1:])
    binned_data = []
    for bin_edges in ebeam_bin_edges:
        ebeam = data["ebeam"]
        in_range = (ebeam > bin_edges[0]) & (ebeam < bin_edges[1])
        data_in_range = data[in_range]
        processed_data_in_range = process_burst_data(data_in_range, run)
        binned_data.append(processed_data_in_range)
    return binned_data


def get_short_pulse_data():
    """Return most recorded short pulse data"""
    data86 = get_nonburst_data(86)
    data122 = get_nonburst_data(122)
    data99 = get_burst_data(99)
    data136 = get_burst_data(136)
    data331 = get_burst_data(331)
    data99 = combine_burst_data([data99, data136, data331])
    data290 = get_nonburst_data(290)
    data359 = get_nonburst_data(359)
    data388 = get_burst_data(388)
    data510 = get_burst_data(510)
    data416 = get_burst_data(416)
    to_return = {
        "86": data86,
        "99": data99,
        "122": data122,
        "136": data136,
        "290": data290,
        "331": data331,
        "359": data359,
        "388": data388,
        "416": data416,
        "510": data510,
    }
    return to_return


def get_long_pulse_data():
    """Return most recorded long pulse data"""
    data548 = get_nonburst_data(548)
    data641 = get_nonburst_data(641)
    # data583 = get_burst_data(583)
    data554 = get_burst_data(554)
    data603 = get_burst_data(603)
    data647 = get_burst_data(647)
    data603 = combine_burst_data([data603, data647])
    to_return = {"548": data548, "641": data641, "554": data554, "603": data603}
    return to_return


def combine_burst_data(data_sets):
    cals = LB51_calibman.get_cals(603)
    data_total = data_sets[0]["total"]
    data_intact = data_sets[0]["intact"]
    data_blown = data_sets[0]["blown"]
    for data in data_sets[1:]:
        data_total = np.append(data_total, data["total"])
        data_intact = np.append(data_intact, data["intact"])
        data_blown = np.append(data_blown, data["blown"])
    sum_intact = get_summed_spectra(
        data_intact["sam_spec"], data_intact["no_sam_spec"], cals
    )
    sum_intact["fluence"] = np.mean(data_intact["fluence"])
    sum_blown = get_summed_spectra(
        data_blown["sam_spec"], data_blown["no_sam_spec"], cals, blown_sam=True
    )
    to_ret = {
        "total": data_total,
        "intact": data_intact,
        "blown": data_blown,
        "sum_intact": sum_intact,
        "sum_blown": sum_blown,
    }
    return to_ret


def get_burst_data(run, correct_split=True):
    data = get_total_burst_data(run, correct_split)
    to_ret = process_burst_data(data, run)
    return to_ret


def get_total_burst_data(run, correct_split=True):
    spec_data, nonspec_data = LB51_h52rec.load_run_set_data(run)
    cals = LB51_calibman.get_cals(run)
    spec_data = get_good_data(spec_data, cals)
    nonspec_data = get_good_data(nonspec_data, cals)
    blown, unknown = get_when_blown(spec_data, cals)
    spec_data = spec_data[~unknown]
    nonspec_data = nonspec_data[~unknown]
    blown = blown[~unknown]
    if correct_split:
        spec_data = correct_splitting_drift(spec_data, blown, cals)
    # Assemble data into numpy record array
    data = {
        "fluence": get_fluences(nonspec_data["gd2"], cals),
        "ebeam": nonspec_data["fEbeamL3Energy"],
        "blown": blown,
        "sam_spec": spec_data["sam_spec"],
        "no_sam_spec": spec_data["no_sam_spec"],
        "run_num": spec_data["run_num"],
        "shot_num": spec_data["image_num"],
    }
    data = dict_to_rec_array(data)
    return data


def process_burst_data(data, run):
    cals = LB51_calibman.get_cals(run)
    data_intact = data[~data["blown"]]
    data_blown = data[data["blown"]]
    sum_intact = get_summed_spectra(
        data_intact["sam_spec"], data_intact["no_sam_spec"], cals
    )
    sum_blown = get_summed_spectra(
        data_blown["sam_spec"], data_blown["no_sam_spec"], cals, blown_sam=True
    )
    sum_intact["fluence"] = np.mean(data["fluence"])
    to_ret = {
        "total": data,
        "intact": data_intact,
        "blown": data_blown,
        "sum_intact": sum_intact,
        "sum_blown": sum_blown,
    }
    return to_ret


def get_nonburst_data(run):
    spec_data, nonspec_data = LB51_h52rec.load_run_set_data(run)
    cals = LB51_calibman.get_cals(run)
    spec_data = get_good_data(spec_data, cals)
    nonspec_data = get_good_data(nonspec_data, cals)
    # nonspec_data['fluences'] = get_fluences(nonspec_data['gd2'], cals)
    spec_data["no_sam_spec"] = spec_data["no_sam_spec"] * cals["splitting"]
    # Assemble data into numpy record array
    data_intact = {
        "sam_spec": spec_data["sam_spec"],
        "no_sam_spec": spec_data["no_sam_spec"],
        "run_num": spec_data["run_num"],
        "image_num": spec_data["image_num"],
    }
    data_intact = dict_to_rec_array(data_intact)
    sum_intact = get_summed_spectra(
        spec_data["sam_spec"], spec_data["no_sam_spec"], cals
    )
    fluence = get_fluences(nonspec_data["gd2"], cals)
    sum_intact["fluence"] = np.mean(fluence)
    to_ret = {"intact": data_intact, "sum_intact": sum_intact, "nonspec": nonspec_data}
    return to_ret


def get_fluences(gd_vals, cals):
    spot_size = cals["spot_size_x"] * cals["spot_size_y"]  # um^2
    spot_size = spot_size * (10 ** -8)  # cm^2
    fluences = gd_vals * BEAMLINE_TRANS / spot_size
    return fluences


def get_photon_e(cals):
    pixels = np.arange(cals["sam_spec_ROI"][2], cals["sam_spec_ROI"][3])
    phot = (pixels - cals["L3_pixel_max"]) * cals["ev_per_pixel"] + Co_L3EDGE_MAX
    return phot


def correct_splitting_drift(spec_data, sam_blown, cals):
    """Correct spectra for a drifting splitting ratio
    Uses data with blown up sample to determine drifting splitting ratio
    """
    phot = get_photon_e(cals)
    blown_sam_specs = spec_data["sam_spec"][sam_blown]
    blown_no_sam_specs = spec_data["no_sam_spec"][sam_blown]
    blown_sam_specs = blown_sam_specs[:, ((phot > 700) & (phot < 800))]
    blown_no_sam_specs = blown_no_sam_specs[:, ((phot > 700) & (phot < 800))]
    blown_sam_spec_sum = np.sum(blown_sam_specs, axis=1)
    blown_no_sam_spec_sum = np.sum(blown_no_sam_specs, axis=1)
    blown_split = blown_sam_spec_sum / blown_no_sam_spec_sum

    x = spec_data["run_num"][sam_blown]
    y = blown_split
    w = blown_no_sam_spec_sum
    smooth_split = np.poly1d(np.polyfit(x, y, deg=cals["split_poly_deg"], w=w))
    pred_split = smooth_split(spec_data["run_num"])
    # if cals['corr_split_end'] == 1:
    #    pred_split = pred_split*cals['splitting']/pred_split[-1]
    # else:
    #    pred_split = pred_split*cals['splitting']/pred_split[0]
    # split_plot(x, y, pred_split[sam_blown])
    spec_data["no_sam_spec"] = spec_data["no_sam_spec"] * pred_split[:, np.newaxis]
    return spec_data


def split_plot(x, y, pred_split):
    """Make plot to show correction of drifting splitting ratio"""
    plt.figure()
    plt.scatter(x, y, c="b", label="Measured Splitting")
    plt.scatter(x, pred_split, c="r", label="Predicted Splitting")
    plt.legend(loc="best")
    plt.figure()
    for run in np.unique(x):
        plt.plot(y[x == run])


def get_summed_spectra(sam_specs, no_sam_specs, cals, blown_sam=False):
    if blown_sam:
        ssrl = LB51_get_synchro.get_np_coo_xas()
        ssrl["trans"] = np.ones_like(ssrl["trans"])
        ssrl["xas"] = np.zeros_like(ssrl["trans"])
    elif cals["is_np_CoO"]:
        ssrl = LB51_get_synchro.get_np_coo_xas()
    elif cals["is_plain_CoO"]:
        ssrl = LB51_get_synchro.get_plain_coo_xas()
    else:
        ssrl = LB51_get_synchro.get_copd_xas()
    phot = get_photon_e(cals)
    sam_spec_sum = np.sum(sam_specs, axis=0).astype(float)
    no_sam_spec_sum = np.sum(no_sam_specs, axis=0).astype(float)
    ssrl_trans_interp = np.interp(phot, ssrl["phot"], ssrl["trans"])
    exp_lin_sam_spec_sum = no_sam_spec_sum * ssrl_trans_interp
    exc_sam_spec = sam_spec_sum - exp_lin_sam_spec_sum
    absorption = -np.log(sam_spec_sum / no_sam_spec_sum)
    summed_data = {
        "phot": phot,
        "sam_spec": sam_spec_sum,
        "no_sam_spec": no_sam_spec_sum,
        "abs": absorption,
        "ssrl_absorption": np.interp(phot, ssrl["phot"], ssrl["xas"]),
        "exc_sam_spec": exc_sam_spec,
    }
    return summed_data


def get_good_data(data, cals):
    """Return good portion of data"""
    good = get_when_good(data["run_num"], cals)
    return data[good]


def get_when_good(run_nums, cals):
    """Calculate when data is good from calibrations"""
    good = []
    for run_num in run_nums:
        good_sample = run_num in cals["good_sample_runs"]
        good_no_sample = run_num in cals["good_no_sample_runs"]
        if good_sample or good_no_sample:
            good.append(True)
        else:
            good.append(False)
    return np.array(good)


def get_when_blown(spec_data, cals):
    """Calculate when sample is blown up for burst mode data set"""
    blown_sam = []
    unknown_sam = []
    for run_num, shot_num in zip(spec_data["run_num"], spec_data["image_num"]):
        if run_num in cals["good_no_sample_runs"]:
            blown_sam.append(True)
            unknown_sam.append(False)
        elif shot_num == 0:
            # This is the first shot of a run starting with sample
            blown_sam.append(False)
            unknown_sam.append(False)
        elif (shot_num > 0) and (shot_num <= cals["shot_after_blown"]):
            # Sample may have blown up, but not sure
            blown_sam.append(True)
            unknown_sam.append(True)
        else:
            # Assume sample is blown up after cals['shot_after_blown']+1 shots
            blown_sam.append(True)
            unknown_sam.append(False)
    return np.array(blown_sam), np.array(unknown_sam)


def dict_to_rec_array(dict_in):
    """Assemble data in form of dictionary into numpy record array"""
    dtype_args = [
        (key, type(dict_in[key][0]), np.size(dict_in[key][0])) for key in dict_in.keys()
    ]
    dtype = np.dtype(dtype_args)
    rec_out = np.array(list(zip(*dict_in.values())), dtype=dtype)
    return rec_out
