"""Convert data in LCLS HDF5 file to numpy record array and save
"""

# TODO list:
#   - Get pnCCD data as more reliable check of whether sample is blown

import numpy as np
import h5py

from LB51 import LB51_calibman, iccd3

#### CONSTANTS
# Path to LCLS HDF5 files:
HDF5_PATH = "/reg/data/ana01/amo/amob5114/hdf5/amob5114-r"
# Path to some data in HDF5 file:
GD_DATA_STR = "Configure:0000/Run:0000/CalibCycle:0000/Bld::BldDataFEEGasDetEnergy/FEEGasDetEnergy/data"
EBEAM_DATA_STR = (
    "Configure:0000/Run:0000/CalibCycle:0000/Bld::BldDataEBeamV5/EBeam/data"
)
# Path to save spectral data:
SPEC_SAVE_PATH = "data/pre_proc/spec"
# Path to save non-spectral data:
NONSPEC_SAVE_PATH = "data/pre_proc/non_spec"
#### END CONSTANTS


def load_run_set_data(run):
    spec_save_file = SPEC_SAVE_PATH + str(run) + ".npy"
    nonspec_save_file = NONSPEC_SAVE_PATH + str(run) + ".npy"
    spec_data = np.load(spec_save_file)
    nonspec_data = np.load(nonspec_save_file)
    return spec_data, nonspec_data


def save_run_set_data(run):
    spec_data, nonspec_data = get_run_set_data(run)
    spec_save_file = SPEC_SAVE_PATH + str(run) + ".npy"
    nonspec_save_file = NONSPEC_SAVE_PATH + str(run) + ".npy"
    np.save(spec_save_file, spec_data)
    np.save(nonspec_save_file, nonspec_data)


def get_run_set_data(run):
    cals = LB51_calibman.get_cals(run)
    spec_data = iccd3.get_run_set_spectra(run)
    nonspec_data = get_nonspec_data(cals["all_runs"])
    return spec_data, nonspec_data


def get_run_str(run):
    """Return the HDF5 path of run"""
    run_str = HDF5_PATH + str(run).zfill(4) + ".h5"
    return run_str


def get_nonspec_data(runs):
    data_list = []
    for run in runs:
        # Get data file path
        data_path = get_run_str(run)
        data_file = h5py.File(data_path, "r")
        data_dict = get_gd_data(data_file)
        data_dict["run_num"] = [run] * len(data_dict["gd2"])
        data_dict.update(get_ebeam_data(data_file))
        data = np.rec.fromarrays(data_dict.values(), names=data_dict.keys())
        data_list.append(data)
    data = np.hstack(data_list)
    return data


def get_gd_data(data_file):
    """Return gas detector data"""
    gd_data = data_file[GD_DATA_STR]
    gd11 = gd_data["f_11_ENRC"]
    gd12 = gd_data["f_12_ENRC"]
    gd21 = gd_data["f_21_ENRC"]
    gd22 = gd_data["f_22_ENRC"]
    gd1 = (gd11 + gd12) / 2
    gd2 = (gd21 + gd22) / 2
    to_ret_gd_data = {
        "gd11": gd11,
        "gd12": gd12,
        "gd21": gd21,
        "gd22": gd22,
        "gd1": gd1,
        "gd2": gd2,
    }
    return to_ret_gd_data


def get_ebeam_data(data_file):
    """Return electron beam data"""
    ebeam_data = data_file[EBEAM_DATA_STR]
    # Convert from view of HDF5 file data to a dictionary:
    ebeam_data = {field: ebeam_data[field] for field in ebeam_data.dtype.names}
    return ebeam_data
