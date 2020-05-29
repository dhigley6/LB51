"""Code for doing tests with multilevel Maxwell-Bloch simulations
"""

import numpy as np
import matplotlib.pyplot as plt

from new_xsim import xbloch2020
from new_xsim import gaussian_xbloch_sim, sase_xbloch_sim
from new_xsim.visualize import density_matrix, spectra, stim_efficiency, spectra_series, enhancement_spectra_series

def test():
    system = xbloch2020.make_model_system()
    times = np.linspace(-10, 20, int(1E4))
    strength = 1E0
    duration = 0.33
    E_in = xbloch2020.FIELD_1E15*np.sqrt(strength)*gauss(times, 0, sigma=duration)
    system.run_simulation(times, E_in)
    phot_result = system.phot_results_
    f, axs = plt.subplots(4, 1, sharex=True)
    axs[0].plot(phot_result['phots'], np.abs(phot_result['E_in']))
    axs[1].plot(phot_result['phots'], np.abs(phot_result['E_delta']))
    axs[2].plot(phot_result['phots'], np.abs(phot_result['E_out']))
    axs[3].plot(phot_result['phots'], np.abs(phot_result['E_out']**2)-np.abs(phot_result['E_in'])**2)
    a = 1/0

def test2():
    sim_result = gaussian_xbloch_sim.run_gauss_sim()
    density_matrix.make_figure(sim_result)
    spectra.make_figure(sim_result)
    stim_efficiency.make_figure()
    strengths, sim_results = gaussian_xbloch_sim.load_gauss_series().values()
    spectra_series.make_figure(strengths, sim_results)
    enhancement_spectra_series.make_figure(strengths, sim_results)

def test3():
    sim_result = sase_xbloch_sim.run_sase_sim()
    density_matrix.make_figure(sim_result)
    spectra.make_figure(sim_result)

def gauss(x, t0, sigma):
    gaussian = np.exp((-1/2)*((x-t0)/sigma)**2)
    return gaussian