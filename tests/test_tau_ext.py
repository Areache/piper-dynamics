"""Current-to-torque conversion and the decomposition bookkeeping."""

import numpy as np
from test_params import make_params

from piper_dynamics.estimation.tau_ext import current_to_torque, finite_difference_accel


def test_direction_flips_sign_but_not_magnitude():
    params = make_params(2)
    params.joints[0].direction = 1
    params.joints[1].direction = -1
    params.joints[1].kt = params.joints[0].kt
    tau = current_to_torque(np.array([1.0, 1.0]), params)
    assert tau[0] == -tau[1]
    assert tau[0] > 0


def test_current_offset_is_subtracted():
    params = make_params(1)
    params.joints[0].current_offset = 0.4
    tau = current_to_torque(np.array([0.4]), params)
    assert np.isclose(tau[0], 0.0)


def test_finite_difference_accel_recovers_constant_slope():
    dt = 0.01
    t = np.arange(0, 2, dt)
    v = np.stack([2.0 * t, -0.5 * t], axis=1)
    a = finite_difference_accel(v, dt, smooth=1)
    # Interior samples only: np.gradient's edges are one-sided.
    np.testing.assert_allclose(a[5:-5, 0], 2.0, atol=1e-6)
    np.testing.assert_allclose(a[5:-5, 1], -0.5, atol=1e-6)
