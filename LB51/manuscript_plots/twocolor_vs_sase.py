"""Compare two color stimulated RIXS simulation results with SASE
"""

import matplotlib.pyplot as plt
import numpy as np

from LB51.xbloch import do_xbloch_sim

def run():
    twocolor = do_xbloch_sim.load_twocolor_case()
    sase = do_xbloch_sim.load_multipulse_data(25.0)
    single_flattop = do_xbloch_sim.load_single_flattop_case()
    plt.loglog(
        np.array(sase['fluences'][1:])*1e3/25, 
        np.array(sase['stim_efficiencies'][1:])/100,
        label='25 fs SASE'
    )
    plt.loglog(
        np.array(twocolor['fluences'][1:])*1e3/25, 
        np.array(twocolor['stim_efficiencies'][1:])/100,
        label='25 fs Two Color'
    )
    plt.loglog(
        np.array(single_flattop['fluences'][1:])*1e3/0.41, 
        np.array(twocolor['stim_efficiencies'][1:])/100,
        label='0.41 fs Flat Top Pulse'
    )
    plt.xlim([10**-2, 10**1])
    plt.ylim([10**-5, 10**-1.5])
    plt.legend(loc='best')
    plt.xlabel('Intensity (mJ/cm$^2$/fs)')
    plt.ylabel('Stimulated RIXS/XAS Rate')
