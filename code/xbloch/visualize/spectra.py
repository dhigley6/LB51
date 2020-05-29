"""Plot spectra from xbloch simulation
"""

import numpy as np
import matplotlib.pyplot as plt

def make_figure(system):
    phot_results = system.phot_results_
    _, axs = plt.subplots(4, 1, sharex=True)
    axs[0].plot(phot_results['phots'], np.abs(phot_results['E_in'])**2)
    axs[0].set_title('Input Intensity')
    axs[1].plot(phot_results['phots'], np.abs(phot_results['E_delta'])**2)
    axs[1].set_title('Change Upon Propagation Through Sample')
    axs[2].plot(phot_results['phots'], np.abs(phot_results['E_out']))
    axs[2].set_title('Output Intensity')
    axs[3].plot(phot_results['phots'], np.abs(phot_results['E_out']**2)-np.abs(phot_results['E_in'])**2)
    axs[3].set_title('?')