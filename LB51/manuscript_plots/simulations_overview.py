"""Make a figure showing how simulations work
"""

import matplotlib.pyplot as plt

from LB51.xbloch import do_xbloch_sim

def simulations_overview():
    sim_results = do_xbloch_sim.load_multipulse_data(5.0)    # SASE pulses case
    f, axs = plt.subplots(2, 2, sharex='col')
    axs[0, 1].plot(sim_results['phot'], sim_results['summed_incident_intensities'][0]/sim_results['fluences'][0])
    inds_to_plot = [0, -7, -5, -3]
    for i in inds_to_plot:
        intensity_difference = (sim_results['summed_transmitted_intensities'][i]-sim_results['summed_incident_intensities'][i])/sim_results['fluences'][i]
        axs[1, 1].plot(sim_results['phot'], intensity_difference, label=sim_results['fluences'][i])
    axs[0, 1].set_xlim((770, 784))
    axs[0, 1].set_ylabel('Intensity')
    axs[1, 1].set_ylabel('Intensity')
    axs[1, 1].set_ylabel('Photon Energy (eV)')
    axs[1, 1].legend(loc='best')