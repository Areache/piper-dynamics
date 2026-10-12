"""Friction model: smoothness is the property that matters."""

import numpy as np

from piper_dynamics.estimation.friction import friction_torque, smooth_sign


def test_smooth_sign_is_continuous_through_zero():
    v = np.linspace(-0.2, 0.2, 2001)
    s = smooth_sign(v, eps=0.05)
    # No step: the largest jump between adjacent samples stays small.
    assert np.max(np.abs(np.diff(s))) < 0.05


def test_friction_opposes_motion():
    v = np.array([1.0, -1.0])
    tau = friction_torque(v, fc=np.array([0.3, 0.3]), fv=np.array([0.1, 0.1]))
    assert tau[0] > 0 and tau[1] < 0
    assert np.isclose(tau[0], -tau[1])


def test_viscous_term_scales_with_velocity():
    fc = np.zeros(1)
    fv = np.array([0.2])
    slow = friction_torque(np.array([1.0]), fc, fv)
    fast = friction_torque(np.array([2.0]), fc, fv)
    assert np.isclose(fast[0], 2 * slow[0])


def test_stiction_decays_away_from_standstill():
    fc, fv, fs = np.zeros(1), np.zeros(1), np.array([0.5])
    near = friction_torque(np.array([0.01]), fc, fv, fs, eps=0.05)
    far = friction_torque(np.array([1.0]), fc, fv, fs, eps=0.05)
    assert abs(near[0]) > abs(far[0])
