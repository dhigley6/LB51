import numpy as np 
import matplotlib.pyplot as plt

from manuscript_plots import quant
from xbloch import gaussian_xbloch_sim

def load_markus_simulations():
    markus_5fs = np.genfromtxt('../data/proc/Markus_5fs.txt', delimiter='', skip_header=1)
    markus_25fs = np.genfromtxt('../data/proc/Markus_25fs.txt', delimiter=' ', skip_header=1)
    return {
        '5': markus_5fs,
        '25': markus_25fs
    }

def plot():
    markus = load_markus_simulations()
    measured = quant.get_measured_stim_efficiency()
    strengths, stim_efficiencies = gaussian_xbloch_sim.calculate_stim_efficiencies()
    plt.figure()
    plt.plot(markus['5'][:, 0], markus['5'][:, 1]*100)
    plt.plot(markus['25'][:, 0], markus['25'][:, 1]*100)
    plt.scatter(measured['short_fluences']/5, measured['short_efficiencies'], label='5 fs Pulses\nExpt.')
    plt.scatter(measured['long_fluences']/25, measured['long_efficiencies'], label='25 fs Pulses\nExpt.')
    plt.semilogx(strengths*1E15, np.array(stim_efficiencies)*100, color='k', label='Three Level\nSimulation')
    plt.xlabel('Intensity (W/cm$^2$)')
    plt.ylabel('Inelastic Stimulated Scattering Efficiency')
    plt.xlim((1E10, 1E15))
    plt.tight_layout()
