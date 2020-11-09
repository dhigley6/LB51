"""Tests for xbloch2020
"""

import numpy as np

from LB51.xbloch import xbloch2020


def test_evolve_step_zero_field_ground():
    """Field-free evolution from ground state shouldn't change system"""
    model_system = xbloch2020.make_model_system()
    before_s = model_system.s
    model_system._evolve_step(0.1, 0)
    after_s = model_system.s
    assert np.isclose(before_s, after_s).all()


def test_density_hermitian():
    """Test that density matrix is Hermitian after some evolution"""
    model_system = xbloch2020.make_model_system()
    model_system._evolve_step(0.1, 0.1)
    model_system._evolve_step(0.1, 0.1)
    model_system._evolve_step(0.1, 0.1)
    s = model_system.s
    assert np.isclose(np.matrix(s), np.matrix(s).getH()).all()
