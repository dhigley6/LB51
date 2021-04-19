"""Calibration manager for the LB51 experiment
"""
import numpy as np

#### FILE PATH ################################################################
CAL_FILE = "cals/cals.csv"
#### END FILE PATH ############################################################


def get_cals(run):
    csv_cals = get_csv_cals(run)
    spec_ROIs = {
        "sam_spec_ROI": (
            csv_cals["sam_spec_ROI_top"],
            csv_cals["sam_spec_ROI_bot"],
            csv_cals["sam_spec_ROI_left"],
            csv_cals["sam_spec_ROI_right"],
        ),
        "no_sam_spec_ROI": (
            csv_cals["no_sam_spec_ROI_top"],
            csv_cals["no_sam_spec_ROI_bot"],
            csv_cals["no_sam_spec_ROI_left"],
            csv_cals["no_sam_spec_ROI_right"],
        ),
    }
    good_runs = get_good_runs(csv_cals["start_run"])
    all_cals = dict(zip(csv_cals.dtype.names, csv_cals))
    all_cals.update(good_runs)
    all_cals.update(spec_ROIs)
    return all_cals


def get_csv_cals(run):
    """Get calibrations stored in csv file"""
    cals = get_csv_cals_list()
    for run_set_cals in cals:
        if (run_set_cals["start_run"] <= run) & (run_set_cals["end_run"] >= run):
            return run_set_cals
    raise ValueError("No Calibrations Found for Specified Run")


def get_csv_cals_list():
    cals = np.genfromtxt(CAL_FILE, names=True, delimiter=",")
    return cals


def get_good_runs(first_run):
    """Return lists of good sample and no sample runs in a run set"""
    good_runs_list = get_good_runs_list()
    for good_runs in good_runs_list:
        if good_runs["all_runs"][0] == first_run:
            return good_runs
    raise ValueError("No Lists of Good Runs Found for Specified Run Set")


def get_good_runs_list():
    """Return list of good sample and no sample runs for all run sets"""
    all_runs = []
    all_runs.append(
        {"all_runs": [17], "good_sample_runs": [17], "good_no_sample_runs": []}
    )
    all_runs.append(
        {
            "all_runs": [20, 22, 23, 24, 25],
            "good_sample_runs": [],
            "good_no_sample_runs": [],
        }
    )
    all_runs.append(
        {"all_runs": [32], "good_sample_runs": [32], "good_no_sample_runs": []}
    )
    all_runs.append(
        {"all_runs": [33], "good_sample_runs": [33], "good_no_sample_runs": []}
    )
    all_runs.append(
        {"all_runs": [34], "good_sample_runs": [34], "good_no_sample_runs": []}
    )
    all_runs.append(
        {"all_runs": [35], "good_sample_runs": [35], "good_no_sample_runs": []}
    )
    all_runs.append(
        {"all_runs": [36], "good_sample_runs": [36], "good_no_sample_runs": []}
    )
    all_runs.append(
        {
            "all_runs": np.arange(86, 97),
            "good_sample_runs": np.arange(86, 97),
            "good_no_sample_runs": [],
        }
    )
    all_runs.append(
        {
            "all_runs": np.arange(99, 121),
            "good_sample_runs": np.arange(99, 120, 2),
            "good_no_sample_runs": np.arange(100, 121, 2),
        }
    )
    all_runs.append(
        {
            "all_runs": np.append(np.arange(122, 125), np.arange(126, 133)),
            "good_sample_runs": np.append(np.arange(122, 125), np.arange(126, 133)),
            "good_no_sample_runs": [],
        }
    )
    all_runs.append(
        {
            "all_runs": np.append(np.arange(136, 147, 2), np.arange(149, 156, 2)),
            "good_sample_runs": np.append(np.arange(136, 147), np.arange(149, 156)),
            "good_no_sample_runs": np.append(
                np.arange(137, 148, 2), np.arange(150, 157, 2)
            ),
        }
    )
    all_runs.append(
        {
            "all_runs": np.arange(166, 175),
            "good_sample_runs": np.arange(166, 175),
            "good_no_sample_runs": [],
        }
    )
    all_runs.append(
        {
            "all_runs": np.arange(177, 198),
            "good_sample_runs": np.append(177, np.arange(180, 197, 2)),
            "good_no_sample_runs": np.append(
                np.arange(178, 180), np.arange(181, 198, 2)
            ),
        }
    )
    all_runs.append(
        {"all_runs": [198], "good_sample_runs": [198], "good_no_sample_runs": []}
    )
    all_runs.append(
        {
            "all_runs": np.arange(211, 287),
            "good_sample_runs": np.arange(211, 260),
            "good_no_sample_runs": [287],
        }
    )
    # below run set is low fluence. Some outlier samples where alignment isn't good were
    # taken out
    all_runs.append(
        {
            "all_runs": np.append(np.arange(290, 307), np.arange(312, 319)),
            "good_sample_runs": np.append(np.arange(290, 307), np.arange(312, 319)),
            "good_no_sample_runs": [],
        }
    )
    all_runs.append(
        {
            "all_runs": np.arange(331, 357),
            "good_sample_runs": np.arange(331, 357),
            "good_no_sample_runs": [],
        }
    )
    all_runs.append(
        {
            "all_runs": np.arange(359, 385),
            "good_sample_runs": np.arange(359, 385),
            "good_no_sample_runs": [],
        }
    )
    all_runs.append(
        {
            "all_runs": np.append([388], np.arange(390, 415)),
            "good_sample_runs": np.append([388], np.arange(390, 415)),
            "good_no_sample_runs": [],
        }
    )
    all_runs.append(
        {
            "all_runs": np.arange(416, 428),
            "good_sample_runs": np.arange(416, 428),
            "good_no_sample_runs": [],
        }
    )
    all_runs.append(
        {
            "all_runs": np.arange(438, 443),
            "good_sample_runs": np.arange(438, 443),
            "good_no_sample_runs": [],
        }
    )
    all_runs.append(
        {
            "all_runs": np.append(np.arange(454, 464), np.arange(465, 473)),
            "good_sample_runs": np.append(np.arange(454, 464), np.arange(465, 473)),
            "good_no_sample_runs": [],
        }
    )
    all_runs.append(
        {"all_runs": [477], "good_sample_runs": [477], "good_no_sample_runs": []}
    )
    all_runs.append(
        {
            "all_runs": np.append(np.arange(481, 493), np.arange(494, 500)),
            "good_sample_runs": np.append(np.arange(481, 493), np.arange(494, 500)),
            "good_no_sample_runs": [],
        }
    )
    all_runs.append(
        {
            "all_runs": [503, 504],
            "good_sample_runs": [503, 504],
            "good_no_sample_runs": [],
        }
    )
    all_runs.append(
        {
            "all_runs": np.append(np.arange(510, 516), np.arange(523, 536)),
            "good_sample_runs": np.append(np.arange(510, 516), np.arange(523, 536)),
            "good_no_sample_runs": [],
        }
    )
    # all runs seem ok in 548-551
    all_runs.append(
        {
            "all_runs": np.arange(548, 552),
            "good_sample_runs": np.arange(548, 552),
            "good_no_sample_runs": [],
        }
    )
    # Below run set has some problems with alignment that are possibly correctable,
    # but aren't used in analysis for now, in case it's not
    test = np.append(np.arange(560, 566), np.arange(570, 575))
    orig = np.append(np.arange(558, 568), np.arange(570, 582))
    test = np.append(np.arange(554, 555), np.arange(558, 568))
    test = np.append(test, np.arange(570, 582))
    test = np.append(np.arange(554, 568), np.arange(570, 582))
    test = np.append(np.arange(560, 568), np.arange(570, 575))
    test_reduced = np.append(np.arange(559, 568), np.arange(570, 576))
    # test = np.arange(558, 570)
    # test = np.arange(570, 582)
    test_all = np.arange(554, 582)
    # test = np.arange(557, 568)
    # Note changing vernier for runs 554-557. Constant vernier in 557 and afterwords
    # Note: run 568 taken on an already blown up sample
    # Note: run 555 taken on an already blown up sample
    # No PnCCD scattering in run 569+splitting of spectra looks like it was taken on an already blown sample. Would be nice to verify this somehow
    all_runs.append(
        {
            "all_runs": np.arange(554, 582),
            "good_sample_runs": test,
            "good_no_sample_runs": [],
        }
    )
    # Note: it seems we performed some alignment during run 584 to correct splitting
    all_runs.append(
        {
            "all_runs": np.arange(583, 596),
            "good_sample_runs": np.arange(585, 596),
            "good_no_sample_runs": [],
        }
    )
    # Note: run 598: Co/Pd not lifted off of no sample part
    # Note: including run 597 here seems to mess this up for some reason. Need to look into this further
    # But these runs aren't used in final analysis right now anyway, since they
    # don't contain sufficient data to extract something meaningful
    good_596 = np.append(np.arange(596, 597), np.arange(599, 602))
    all_runs.append(
        {
            "all_runs": np.arange(596, 602),
            "good_sample_runs": good_596,
            "good_no_sample_runs": [],
        }
    )
    good_sample_runs603 = np.append(np.arange(603, 607), np.arange(610, 614))
    good_sample_runs603 = np.append(good_sample_runs603, [616, 617])
    good_sample_runs603 = np.append(good_sample_runs603, np.arange(618, 639))
    good_sample_runs647 = np.append(np.arange(647, 666), np.arange(670, 671))
    good_sample_runs603_total = np.append(good_sample_runs603, good_sample_runs647)
    # Note: 608 & 609 have basically zero sample spectra, 607 also very low,
    # probably alignment issue or bad samples,
    # so taken out from below dataset. 614&615 have same issue, 666-669 have same issue
    # 616&617 look okay though
    test = np.append(np.arange(603, 608), np.arange(610, 639))
    test = np.append(np.arange(603, 639), np.arange(647, 671))
    all_runs.append(
        {
            "all_runs": np.append(np.arange(603, 639), np.arange(647, 671)),
            "good_sample_runs": good_sample_runs603_total,
            "good_no_sample_runs": [],
        }
    )
    good_641 = np.arange(641, 644)
    # good_641 = np.array([642])
    # Sample seems to have been damaged during run 642. Edge is less prominent in later spectra recorded of that run
    all_runs.append(
        {
            "all_runs": np.arange(641, 644),
            "good_sample_runs": [641, 643],
            "good_no_sample_runs": [],
        }
    )
    # Note: 669 has basically zero sample spectra, 668, 666 and 667 also very low and
    # adjacent, so taken out
    test = np.arange(647, 671)
    all_runs.append(
        {
            "all_runs": np.arange(647, 671),
            "good_sample_runs": good_sample_runs647,
            "good_no_sample_runs": [],
        }
    )
    return all_runs
