"""
@author: dhigley

Run X-ray Maxwell-Bloch 3-level simulations for SASE pulses
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle

from new_xsim import xbloch2020
from new_xsim import sase_sim

FIELD_1E15 = 8.68E10   # Electric field strength in V/m that corresponds to
# 10^15 W/cm^2 or 1 J/cm^2/fs

# Simulation parameters:
STRENGTHS = np.logspace(-3, 0, 10)     # Intensities of incident X-ray pulses (in 10^15 W/cm^2)
STRENGTHS = np.append(1E-8, STRENGTHS)

def simulate_sase_series():
    """Simulate and save series of SASE X-ray pulses interacting with 3-level atoms
    """
    sim_results = []
    for strength in STRENGTHS:
        sim_result = run_sase_sim(strength)
        sim_results.append(sim_result)
        print(f'Completed {str(strength)}')
    data = {'strengths': STRENGTHS,
            'sim_results': sim_results}
    with open('sase_data.pickle', 'wb') as f:
        pickle.dump(data, f)

def run_sase_sim(strength=1E-3, duration=5):
    system = xbloch2020.make_model_system()
    times, E_in = sase_sim.simulate_gaussian()
    E_in = FIELD_1E15*np.sqrt(strength)*E_in
    sim_result = system.run_simulation(times, E_in)
    return sim_result