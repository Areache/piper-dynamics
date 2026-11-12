"""Momentum observer on a simulated arm: no acceleration needed, first-order response."""

import numpy as np
import pytest
from test_params import make_params

from piper_dynamics.estimation.momentum import MomentumObserver


def _simulate(model, tau_ext_fn, T=1500, dt=0.002, sub=20):
    """Forward dynamics (sub-stepped, so integration error does not masquerade as a residual) with motor torque
    = gravity + M(q) (light damping + a slow sinusoid), held over each sample; an external torque acts on the arm.
    The excitation is set as an acceleration and mapped through M(q) so that it suits any URDF: a fixed torque
    drove the light wrist links of the real Piper X URDF to NaN on the first step."""
    pin = pytest.importorskip("pinocchio")
    nv = model.nv
    q = np.zeros(nv); v = np.zeros(nv)
    if np.linalg.cond(model.mass_matrix(q)) > 1e10:
        pytest.skip("singular mass matrix (e.g. an unlocked massless gripper joint): forward dynamics undefined")
    Q, V, TAU = [], [], []
    for k in range(T):
        t = k * dt
        acc = -2.0 * v + 0.3 * np.sin(2 * np.pi * 0.8 * t + np.arange(nv))
        tau_m = model.gravity_torque(q) + model.mass_matrix(q) @ acc
        Q.append(q.copy()); V.append(v.copy()); TAU.append(tau_m)
        for i in range(sub):
            # M a + C v + g = tau_m - tau_ext  (conventions: tau_meas = tau_model + tau_ext)
            a = pin.aba(model.model, model.data, q, v, tau_m - tau_ext_fn(t + i * dt / sub))
            v = v + a * dt / sub
            q = q + v * dt / sub
    return np.array(Q), np.array(V), np.array(TAU)


def _params_unit(nq):
    p = make_params(nq)
    for j in p.joints:
        j.kt, j.direction, j.current_offset, j.fc, j.fv, j.fs = 1.0, 1, 0.0, 0.0, 0.0, 0.0
    return p


def test_free_motion_residual_vanishes(model):
    dt = 0.002
    q, v, tau = _simulate(model, lambda t: np.zeros(model.nv), dt=dt)
    r = MomentumObserver(model, _params_unit(model.nv), gain=50.0).batch(q, v, tau, dt)
    assert np.abs(r[200:]).max() < 0.02 * np.abs(tau).max()


def test_step_external_torque_first_order(model):
    dt, K, t0 = 0.002, 40.0, 1.0
    step = np.zeros(model.nv); step[1] = 0.5
    q, v, tau = _simulate(model, lambda t: step if t >= t0 else np.zeros(model.nv), dt=dt)
    r = MomentumObserver(model, _params_unit(model.nv), gain=K).batch(q, v, tau, dt)
    k0 = int(t0 / dt)
    k_tau = k0 + int(round(1.0 / K / dt))                    # one time constant after the step
    assert r[k_tau, 1] == pytest.approx(0.5 * (1 - np.exp(-1)), abs=0.03)
    assert r[-1, 1] == pytest.approx(0.5, abs=0.02)
    assert np.abs(r[-1, [0, 2, 3, 4, 5]]).max() < 0.02


def test_unstable_gain_rejected(model):
    with pytest.raises(ValueError):
        MomentumObserver(model, _params_unit(model.nv), gain=600.0).batch(
            np.zeros((3, model.nv)), np.zeros((3, model.nv)), np.zeros((3, model.nv)), 0.002)
