"""Tests for sase_sim.py
"""

import numpy as np

from LB51.xbloch import sase_sim

def test_normalization(pulse_fluence=1.0):
    t, t_y = sase_sim.simulate_gaussian()
    integral = np.trapz(np.abs(t_y)**2, x=t)
    assert np.isclose(integral, 1)