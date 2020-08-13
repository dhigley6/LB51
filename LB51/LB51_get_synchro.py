"""Get Synchrotron data relevant to LB51 Experiment
"""

import numpy as np
import matplotlib.pyplot as plt

# Absorption spectra collected at SSRL with the same samples:
SSRL_SPECTRA_PATH = 'data/SSRL_NPCoOandCoPdTrans.txt'

# Parameters for converting nanopatterned CoO XAS to plain CoO XAS:
CoO_OFFSET = 0.2
CoO_THICK_RATIO = 1.5726
# Photon energy values which were found by plotting the XAS
JO_L3 = 778
JO_L2 = 793.2
SSRL_L3 = 777.63
SSRL_L2 = 792.5

def get_copd_xas():
    ssrl_spectra = get_ssrl_data()
    data = {'phot': cal_phot(ssrl_spectra[:, 0]),
            'trans': ssrl_spectra[:, 2],
            'xas': -np.log(ssrl_spectra[:, 2])}
    return data

def get_plain_coo_xas():
    plain_coo_xas = get_np_coo_xas()
    plain_coo_xas['xas'] = (plain_coo_xas['xas']-CoO_OFFSET)*CoO_THICK_RATIO+CoO_OFFSET
    plain_coo_xas['trans'] = np.exp(-1*plain_coo_xas['xas'])
    return plain_coo_xas

def get_np_coo_xas():
    ssrl_spectra = get_ssrl_data()
    data = {'phot': cal_phot(ssrl_spectra[:, 0]),
            'trans': ssrl_spectra[:, 1],
            'xas': -np.log(ssrl_spectra[:, 1])}
    return data

def plot_xas():
    """Make test plot of SSRL spectra
    """
    copd_xas = get_copd_xas()
    plain_coo_xas = get_plain_coo_xas()
    np_coo_xas = get_np_coo_xas()
    plt.figure()
    plt.plot(copd_xas['phot'], copd_xas['xas'], label='Co/Pd')
    plt.plot(np_coo_xas['phot'], np_coo_xas['xas'], label='Nanopatterned CoO')
    plt.plot(plain_coo_xas['phot'], plain_coo_xas['xas'], label='Plain CoO')
    plt.xlabel('Photon Energy (eV)')
    plt.ylabel('Absorption (a.u.)')
    plt.legend(loc='best')

def get_ssrl_data():
    return np.genfromtxt(SSRL_SPECTRA_PATH, delimiter=',', skip_header=1)

def cal_phot(phot_in):
    """Convert photon energy to Jo's scale
    """
    phot_scale = (JO_L3-JO_L2)/(SSRL_L3-SSRL_L2)
    new_phot = (phot_in-SSRL_L3)*phot_scale+JO_L3
    return new_phot
