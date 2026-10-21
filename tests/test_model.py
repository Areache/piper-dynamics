"""Model layer: kinematics, dynamics, and the sign convention.

Runs against the toy URDF in ``tests/fixtures`` by default. Set ``PIPER_URDF``
(and ``PIPER_EE_FRAME`` if needed) to run the identical suite against a real
Piper description -- worth doing once when you first set up an arm.
"""

import warnings

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


def _overrides(links, phi, order):
    # Scale every link differently so a positional mix-up cannot go unnoticed.
    return {links[i]: (phi[i] * (1.0 + 0.1 * i)).tolist() for i in order}


def test_moving_links_are_one_per_joint(model):
    links = model.moving_links()
    assert len(links) == model.nv
    assert len(set(links)) == len(links)


def test_inertial_overrides_are_matched_by_name_not_order(model):
    links = model.moving_links()
    phi = model.inertial_params().reshape(len(links), 10)
    q = np.full(model.nq, 0.3)
    model.set_inertial_overrides(_overrides(links, phi, range(len(links))))
    in_order = model.gravity_torque(q)
    model.set_inertial_overrides(_overrides(links, phi, reversed(range(len(links)))))
    np.testing.assert_allclose(model.gravity_torque(q), in_order, atol=1e-12)


def test_inertial_overrides_reject_missing_unknown_and_lumped_links(model):
    links = model.moving_links()
    good = _overrides(links, model.inertial_params().reshape(len(links), 10), range(len(links)))
    missing = dict(list(good.items())[1:])
    with pytest.raises(ValueError, match="missing"):
        model.set_inertial_overrides(missing)
    with pytest.raises(ValueError, match="not links of this URDF"):
        model.set_inertial_overrides({**good, "no_such_link": [0.0] * 10})
    from piper_dynamics.model import _frame_parents

    root = next(f.name for f in model.model.frames  # a link fixed to the world
                if f.type == pin.FrameType.BODY and _frame_parents(f)[0] == 0)
    with pytest.raises(ValueError, match="without their own parameters"):
        model.set_inertial_overrides({**good, root: [0.0] * 10})
    first = next(iter(good))
    with pytest.raises(ValueError, match="10 values"):
        model.set_inertial_overrides({**good, first: [1.0] * 4})


def test_gravity_regressor_reproduces_gravity(model):
    q = np.random.default_rng(5).uniform(-0.5, 0.5, model.nq)
    priors = np.concatenate([s.prior for s in model.link_specs()])
    np.testing.assert_allclose(model.gravity_regressor(q) @ priors, model.gravity_torque(q),
                               atol=1e-9)


def test_link_specs_accept_the_urdf_itself(model, urdf_path):
    from piper_dynamics.calibration.feasibility import BOX_SOURCES, ParamSet, Verdict, audit

    specs = model.link_specs()
    assert [s.name for s in specs] == model.moving_links()
    assert all(s.box_source in BOX_SOURCES for s in specs)
    assert specs[-1].gripper is not None and all(s.gripper is None for s in specs[:-1])
    urdf = ParamSet(tuple(s.name for s in specs), np.array([s.prior for s in specs]), units="SI")
    assert audit(urdf, specs).verdict is Verdict.FEASIBLE
    if urdf_path.endswith("toy6.urdf"):
        assert [s.box_source for s in specs[:3]] == ["kinematic-fallback", "mesh", "mesh"]


def test_estimator_warns_on_unphysical_overrides_but_still_runs(model):
    from test_params import make_params

    from piper_dynamics import TauExtEstimator

    links = model.moving_links()
    phi = model.inertial_params().reshape(len(links), 10)
    good = {name: phi[i].tolist() for i, name in enumerate(links)}
    bad = dict(good)
    bad[links[1]] = [phi[1, 0], phi[1, 0] * 1.0, 0.0, 0.0, *phi[1, 4:]]   # com 1 m out
    params = make_params(model.nv)

    params.inertial_overrides = bad
    with pytest.warns(UserWarning, match="POINTWISE_VIOLATION"):
        est = TauExtEstimator(model, params, quasi_static=True)
    z = np.zeros(model.nv)
    assert np.isfinite(est(z, z, z)).all()

    model.set_inertial_params(phi.reshape(-1))     # back to the URDF priors
    params.inertial_overrides = good
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        TauExtEstimator(model, params, quasi_static=True)
