"""Model layer: kinematics, dynamics, and the sign convention.

Runs against the toy URDF in ``tests/fixtures`` by default. Set ``PIPER_URDF``
(and ``PIPER_EE_FRAME`` if needed) to run the identical suite against a real
Piper description -- worth doing once when you first set up an arm.
"""

import numpy as np
import pytest

pin = pytest.importorskip("pinocchio", reason="pinocchio not installed")


def test_unknown_frame_lists_the_real_ones(urdf_path):
    from piper_dynamics import PiperModel

    with pytest.raises(ValueError, match="not in this URDF"):
        PiperModel(urdf_path, ee_frame="definitely_not_a_frame")


def test_gravity_torque_vanishes_without_gravity(urdf_path, ee_frame):
    from piper_dynamics import PiperModel

    m = PiperModel(urdf_path, ee_frame=ee_frame, gravity=np.zeros(3))
    rng = np.random.default_rng(0)
    q = rng.uniform(-0.5, 0.5, m.nq)
    np.testing.assert_allclose(m.gravity_torque(q), 0, atol=1e-9)


def test_rnea_at_rest_equals_gravity_torque(model):
    rng = np.random.default_rng(0)
    q = rng.uniform(-0.5, 0.5, model.nq)
    z = np.zeros(model.nv)
    np.testing.assert_allclose(model.rnea(q, z, z), model.gravity_torque(q), atol=1e-9)


def test_jacobian_matches_finite_difference(model):
    """Catches a frame or reference-frame mix-up in `jacobian_linear`."""
    rng = np.random.default_rng(1)
    q = rng.uniform(-0.4, 0.4, model.nq)
    Jv = model.jacobian_linear(q)
    eps = 1e-6
    for i in range(model.nq):
        dq = np.zeros(model.nq)
        dq[i] = eps
        dp = (model.fk(q + dq)[:3, 3] - model.fk(q - dq)[:3, 3]) / (2 * eps)
        np.testing.assert_allclose(Jv[:, i], dp, atol=1e-5)


def test_regressor_reproduces_rnea(model):
    """Y(q,v,a) @ phi == rnea(q,v,a). If this fails, identification is built
    on sand."""
    rng = np.random.default_rng(2)
    q = rng.uniform(-0.5, 0.5, model.nq)
    v = rng.uniform(-0.3, 0.3, model.nv)
    a = rng.uniform(-0.3, 0.3, model.nv)
    Y = model.joint_torque_regressor(q, v, a)
    np.testing.assert_allclose(Y @ model.inertial_params(), model.rnea(q, v, a), atol=1e-8)


def test_payload_regressor_matches_added_mass(model):
    """Adding mass m at the EE must change g(q) by exactly m * b(q).

    This pins the sign convention end to end. Get it wrong and every payload
    estimate comes out negative, with nothing else in the suite noticing.
    """
    from piper_dynamics.estimation.payload import payload_regressor

    rng = np.random.default_rng(3)
    q = rng.uniform(-0.5, 0.5, model.nq)
    b = payload_regressor(model, q)

    frame = model.model.frames[model.ee_id]
    joint_id = frame.parentJoint if hasattr(frame, "parentJoint") else frame.parent

    m_payload = 0.5
    original = model.model.inertias[joint_id].copy()
    added = frame.placement.act(pin.Inertia.FromSphere(m_payload, 1e-4))
    before = model.gravity_torque(q)
    model.model.inertias[joint_id] = original + added
    model.data = model.model.createData()
    try:
        after = model.gravity_torque(q)
    finally:
        model.model.inertias[joint_id] = original
        model.data = model.model.createData()

    np.testing.assert_allclose(after - before, m_payload * b, atol=1e-5)


def test_payload_regressor_with_com_offset_matches_offset_mass(model):
    """Same check, with the payload hanging off the flange rather than at it."""
    from piper_dynamics.estimation.payload import payload_regressor

    rng = np.random.default_rng(4)
    q = rng.uniform(-0.5, 0.5, model.nq)
    offset = np.array([0.0, 0.0, 0.08])
    b = payload_regressor(model, q, com_offset=offset)

    frame = model.model.frames[model.ee_id]
    joint_id = frame.parentJoint if hasattr(frame, "parentJoint") else frame.parent
    m_payload = 0.4
    placement = frame.placement * pin.SE3(np.eye(3), offset)
    original = model.model.inertias[joint_id].copy()
    before = model.gravity_torque(q)
    model.model.inertias[joint_id] = original + placement.act(
        pin.Inertia.FromSphere(m_payload, 1e-4)
    )
    model.data = model.model.createData()
    try:
        after = model.gravity_torque(q)
    finally:
        model.model.inertias[joint_id] = original
        model.data = model.model.createData()

    np.testing.assert_allclose(after - before, m_payload * b, atol=1e-5)


def test_inertial_params_roundtrip(model):
    phi = model.inertial_params()
    q = np.zeros(model.nq)
    before = model.gravity_torque(q)
    model.set_inertial_params(phi)
    np.testing.assert_allclose(model.gravity_torque(q), before, atol=1e-9)
