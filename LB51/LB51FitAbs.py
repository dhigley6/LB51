"""Module to fit measured LCLS absorption spectrum to that at SSRL

For calibrating photon energy @ LCLS
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import interp
from scipy.optimize import leastsq

import LB51_get_synchro
import set_plot_params

####
CoO_OFFSET = 0.27

Co_L3EDGE_MAX = 778.15
####

def fit_abs_copd(sam_spec, no_sam_spec, splitting_guess=1,
                 L3_guess=535, ev_per_pixel_guess=0.07):
    copd_xas = LB51_get_synchro.get_copd_xas()
    fit_coef_dict = fit_abs(sam_spec, no_sam_spec, copd_xas,
                            thick_ratio_guess=None,
                            splitting_guess=splitting_guess,
                            L3_guess=L3_guess,
                            ev_per_pixel_guess=ev_per_pixel_guess)
    return fit_coef_dict

def fit_abs_plain_coo(sam_spec, no_sam_spec, thick_ratio_guess=1.5,
                      splitting_guess=1, L3_guess=540, ev_per_pixel_guess=0.07):
    """Get Plain CoO cals
    """
    np_coo_xas = LB51_get_synchro.get_np_coo_xas()
    fit_coef_dict = fit_abs(sam_spec, no_sam_spec, np_coo_xas,
                            thick_ratio_guess=thick_ratio_guess,
                            splitting_guess=splitting_guess,
                            L3_guess=L3_guess,
                            ev_per_pixel_guess=ev_per_pixel_guess)
    return fit_coef_dict

def fit_abs_np_coo(sam_spec, no_sam_spec, splitting_guess=1,
                   L3_guess=540, ev_per_pixel_guess=0.07):
    """Get NP CoO cals
    """
    np_coo_xas = LB51_get_synchro.get_np_coo_xas()
    fit_coef_dict = fit_abs(sam_spec, no_sam_spec, np_coo_xas,
                            thick_ratio_guess=None,
                            splitting_guess=splitting_guess,
                            L3_guess=L3_guess,
                            ev_per_pixel_guess=ev_per_pixel_guess)
    return fit_coef_dict

def fit_abs(sam_spec, no_sam_spec, ssrl_xas, thick_ratio_guess=1.5,
            splitting_guess=1, L3_guess=540, ev_per_pixel_guess=0.07, pixels=np.arange(200, 1000)):
    """Get CoO cals
    """
    np_coo_xas = LB51_get_synchro.get_np_coo_xas()
    set_plot_params.init_powerpoint()

    def get_est_sam_spec(fit_coef):
        """Return estimated sample spectrum with input fitting coefficients
        """
        cals = {'thick_ratio': fit_coef[0],
                'splitting': fit_coef[1],
                'L3_pixel': fit_coef[2],
                'ev_per_pixel': fit_coef[3]}
        phot = pixel_to_photon_e(pixels, cals)
        lin_abs = (ssrl_xas['xas']-CoO_OFFSET)*cals['thick_ratio']+CoO_OFFSET
        lin_trans = np.exp(-1*lin_abs)
        lin_trans_interp = interp(phot, ssrl_xas['phot'], lin_trans)
        est_sam_spec = lin_trans_interp*no_sam_spec*cals['splitting']
        return est_sam_spec

    def sam_spec_diffs(fit_coef):
        est_sam_spec = get_est_sam_spec(fit_coef)
        diffs = sam_spec-est_sam_spec
        return diffs

    def get_pars(input_pars, guesses):
        output_pars = [0]*len(guesses)
        inds = list(np.arange(len(guesses)))
        for i, guess in enumerate(guesses):
            if guess is None:
                output_pars[i] = defaults[i]
                inds.remove(i)
        for ind, input_par in zip(inds, input_pars):
            output_pars[ind] = input_par
        return output_pars

    guesses = (thick_ratio_guess, splitting_guess, L3_guess,
               ev_per_pixel_guess)
    defaults = (1, 1, 470, 0.07)

    def optimize_function(input_pars):
         pars = get_pars(input_pars, guesses)
         diffs = sam_spec_diffs(pars)
         return diffs

    start_params = (thick_ratio_guess, splitting_guess, L3_guess,
                    ev_per_pixel_guess)
    start_params = [start_param for start_param in start_params if start_param is not None]
    fit_coef, pcov = leastsq(optimize_function, start_params)
    fit_coef = get_pars(fit_coef, guesses)
    fit_coef_dict = {'thick_ratio': fit_coef[0],
                     'splitting': fit_coef[1],
                     'L3_pixel': fit_coef[2],
                     'ev_per_pixel': fit_coef[3]}
    # Make a plot to see how well the fitting worked:
    phot = pixel_to_photon_e(pixels, fit_coef_dict)
    f, axs = plt.subplots(2, 1, sharex=True)
    axs[0].plot(phot, no_sam_spec*fit_coef_dict['splitting'], 'k--', label='Incident Spectra')
    est_sam_spec = get_est_sam_spec(fit_coef)
    diff_spec = est_sam_spec-sam_spec
    axs[0].plot(phot, -1*diff_spec*5, 'k', label='Difference Spectra')
    axs[0].fill_between(phot, 0, -1*diff_spec*5,
                        where=((-1*diff_spec) > 0),
                        facecolor='r', edgecolor='w',
                        alpha=0.65)
    axs[0].fill_between(phot, 0, -1*diff_spec*5,
                        where=((-1*diff_spec) < 0),
                        facecolor='b', edgecolor='w',
                        alpha=0.65)
    axs[1].plot(phot, -np.log(sam_spec/no_sam_spec/fit_coef_dict['splitting']), label='LCLS Absorption')
    lin_abs = (ssrl_xas['xas']-CoO_OFFSET)*fit_coef_dict['thick_ratio']+CoO_OFFSET
    axs[1].plot(ssrl_xas['phot'], lin_abs, label='Scaled SSRL Absorption')
    axs[1].set_xlabel('Photon Energy (eV)')
    axs[0].legend(loc='best')
    axs[1].legend(loc='best')
    print('Found Calibration Coefficients:')
    print(fit_coef_dict)
    return fit_coef_dict

def pixel_to_photon_e(pixels, cal_data):
    """Convert spectrometer pixels to photon energy
    """
    pix_energy = (pixels-cal_data['L3_pixel'])*cal_data['ev_per_pixel']+Co_L3EDGE_MAX
    return pix_energy

