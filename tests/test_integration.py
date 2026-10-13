"""End-to-end: synthesise telemetry from the model, recover the physics.

This is the test that matters most. Each unit test checks one link in the
chain; this one checks that the whole chain -- current -> torque -> subtract
dynamics -> tau_ext -> payload mass -- composes without a sign flip or a
scaling error hiding in a seam.

No hardware and no real URDF needed: the toy fixture plays the part of the
arm, and we know the ground truth because we generated it.
"""

import numpy as np
import pytest

pin = pytest.importorskip("pinocchio", reason="pinocchio not installed")

from piper_dynamics import ArmParams, PiperModel, TauExtEstimator  # noqa: E402
from piper_dynamics.estimation import estimate_payload_mass  # noqa: E402
from piper_dynamics.params import JointParams, Provenance  # noqa: E402


@pytest.fixture
def rig(urdf_path, ee_frame):
    """A model plus a parameter set with deliberately awkward constants.

    Non-unit kt, mixed signs and a nonzero current offset: if the estimator
    quietly assumes any of those is trivial, this fixture catches it.
    """
    model = PiperModel(urdf_path, ee_frame=ee_frame)
    params = ArmParams(
        joints=[
            JointParams(kt=0.42, direction=1, current_offset=0.3),
            JointParams(kt=0.77, direction=-1, current_offset=-0.2),
            JointParams(kt=0.55, direction=1, current_offset=0.0),
            JointParams(kt=0.31, direction=-1, current_offset=0.1),
            JointParams(kt=0.24, direction=1, current_offset=0.0),
            JointParams(kt=0.19, direction=-1, current_offset=0.05),
        ],
        provenance=Provenance(arm_id="synthetic", calibrated_at="2026-10-12"),
    )
    return model, params


def synth_current(params, tau):
    """Invert current_to_torque: what current would produce this torque?"""
    return tau / (params.direction * params.kt) + params.current_offset


def test_free_space_tau_ext_is_zero(rig):
    """An unloaded arm at rest must produce no external torque. If this is
    nonzero, every downstream estimate inherits the bias."""
    model, params = rig
    est = TauExtEstimator(model, params, quasi_static=True)
    rng = np.random.default_rng(0)
    for _ in range(20):
        q = rng.uniform(-0.6, 0.6, model.nq)
        current = synth_current(params, model.gravity_torque(q))
        np.testing.assert_allclose(est(q, np.zeros(model.nv), current), 0, atol=1e-9)


def test_recovers_known_payload_mass(rig):
    """Hang a known mass on the model, read it back through the full chain."""
    model, params = rig
    est = TauExtEstimator(model, params, quasi_static=True)

    frame = model.model.frames[model.ee_id]
    joint_id = frame.parentJoint if hasattr(frame, "parentJoint") else frame.parent
    m_true = 0.65

    rng = np.random.default_rng(1)
    qs = rng.uniform(-0.6, 0.6, (60, model.nq))

    # Gravity torques WITH the payload attached -> the currents an arm holding
    # that object would report.
    original = model.model.inertias[joint_id].copy()
    model.model.inertias[joint_id] = original + frame.placement.act(
        pin.Inertia.FromSphere(m_true, 1e-4)
    )
    model.data = model.model.createData()
    try:
        tau_loaded = np.stack([model.gravity_torque(q) for q in qs])
    finally:
        model.model.inertias[joint_id] = original
        model.data = model.model.createData()

    currents = np.stack([synth_current(params, t) for t in tau_loaded])
    tau_ext = est.batch(qs, np.zeros_like(qs), currents)

    out = estimate_payload_mass(model, qs, tau_ext)
    assert out.observable
    np.testing.assert_allclose(out.mass_kg, m_true, rtol=1e-6)


def test_payload_estimate_degrades_gracefully_with_noise(rig):
    """Noise should widen the error bar, not bias the estimate."""
    model, params = rig
    est = TauExtEstimator(model, params, quasi_static=True)
    frame = model.model.frames[model.ee_id]
    joint_id = frame.parentJoint if hasattr(frame, "parentJoint") else frame.parent
    m_true = 0.4

    rng = np.random.default_rng(2)
    qs = rng.uniform(-0.6, 0.6, (300, model.nq))
    original = model.model.inertias[joint_id].copy()
    model.model.inertias[joint_id] = original + frame.placement.act(
        pin.Inertia.FromSphere(m_true, 1e-4)
    )
    model.data = model.model.createData()
    try:
        tau_loaded = np.stack([model.gravity_torque(q) for q in qs])
    finally:
        model.model.inertias[joint_id] = original
        model.data = model.model.createData()

    currents = np.stack([synth_current(params, t) for t in tau_loaded])
    currents += rng.normal(0, 0.02, currents.shape)
    tau_ext = est.batch(qs, np.zeros_like(qs), currents)

    out = estimate_payload_mass(model, qs, tau_ext)
    assert abs(out.mass_kg - m_true) < 6 * out.stderr_kg
    assert out.stderr_kg > 0


def test_wrench_recovers_applied_force(rig):
    """tau_ext = -J^T F, so a known F must come back out."""
    from piper_dynamics.estimation import wrench_from_tau_ext

    model, _ = rig
    rng = np.random.default_rng(3)
    # Conditioning depends on the geometry (real URDFs via PIPER_URDF), and
    # this test is about recovery, not about where the singularities are.
    for _ in range(100):
        q = rng.uniform(-0.5, 0.5, model.nq)
        if np.linalg.cond(model.jacobian(q).T) < 1e2:
            break
    F = np.array([2.0, -1.0, 3.5, 0.1, -0.2, 0.05])
    tau_ext = -model.jacobian(q).T @ F
    out = wrench_from_tau_ext(model, q, tau_ext)
    assert out.trustworthy
    np.testing.assert_allclose(out.wrench, F, atol=1e-6)


def test_joint_count_mismatch_is_refused(rig):
    """Mismatched model/params produce plausible-looking nonsense; refuse."""
    model, params = rig
    bad = ArmParams(joints=params.joints[:3], provenance=params.provenance)
    with pytest.raises(ValueError, match="Mismatched|joints"):
        TauExtEstimator(model, bad)
