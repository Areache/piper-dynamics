"""Physical-feasibility gate for fitted link mass parameters.

No Pinocchio: links are stub `LinkSpec`s and regressors are synthetic, with a
nullspace planted where a test needs one. The data-driven verdicts need scipy
(the ``[fit]`` extra).
"""

import numpy as np
import pytest
from test_params import make_params

from piper_dynamics.calibration import feasibility as F
from piper_dynamics.calibration.feasibility import (
    GripperAllowance,
    LinkSpec,
    ParamSet,
    Severity,
    Verdict,
    audit,
    project,
)

NAMES = ("link1", "link2", "link3", "link4", "link5", "link6")
MASS = np.array([0.7, 1.2, 0.7, 0.4, 0.35, 0.05])
COM_X = np.array([0.0, 0.2, 0.2, 0.05, 0.0, 0.0])


def make_links(source="mesh"):
    links = []
    for i, name in enumerate(NAMES):
        com = np.array([COM_X[i], 0.0, 0.01])
        lo, hi = np.minimum(com, 0) - 0.05, np.maximum(com, 0) + 0.05
        links.append(LinkSpec(name, np.r_[MASS[i], MASS[i] * com], lo, hi, source, units="SI",
                              gripper=GripperAllowance() if i == len(NAMES) - 1 else None))
    return links


def prior_values():
    return np.array([s.prior for s in make_links()])


def params(values, names=NAMES):
    return ParamSet(tuple(names), np.asarray(values, dtype=float), units="SI")


def failed(report, kind):
    return [c for c in report.checks if c.kind == kind and not c.passed]


# -- the three bad parameter sets, and a good one ------------------------------
def negative_mass():
    v = prior_values()
    v[2, 0] = -0.3
    return v


def com_outside_link():
    v = prior_values()
    v[1, 1:] = v[1, 0] * np.array([0.6, 0.0, 0.0])   # link2 is 0.3 m long
    return v


def absurd_total_mass():
    return prior_values() * 3.0                      # every com unchanged


def test_good_parameters_pass():
    report = audit(params(prior_values()), make_links())
    assert report.verdict is Verdict.FEASIBLE
    assert all(c.passed for c in report.checks)


def test_negative_mass_is_a_fail():
    report = audit(params(negative_mass()), make_links())
    assert report.verdict is Verdict.POINTWISE_VIOLATION
    (check,) = failed(report, "mass")
    assert check.link == "link3" and check.severity is Severity.FAIL


def test_com_outside_mesh_box_is_a_fail():
    report = audit(params(com_outside_link()), make_links())
    (check,) = failed(report, "com")
    assert check.link == "link2" and check.severity is Severity.FAIL
    assert "outside the mesh box" in check.detail


def test_absurd_total_mass_is_a_warn_only():
    report = audit(params(absurd_total_mass()), make_links())
    assert report.verdict is Verdict.POINTWISE_VIOLATION
    assert [c.kind for c in report.warnings] == ["total_mass"]
    assert report.failures == []


def test_kinematic_fallback_box_never_hard_fails():
    report = audit(params(com_outside_link()), make_links(source="kinematic-fallback"))
    (check,) = failed(report, "com")
    assert check.severity is Severity.WARN
    assert report.failures == []


def test_gripper_sized_end_link_passes():
    """The URDF end link may lack the gripper; a fit lumps it in. Not an error."""
    v = prior_values()
    v[5] = 0.5 * np.array([1.0, 0.0, 0.0, 0.08])
    assert audit(params(v), make_links()).verdict is Verdict.FEASIBLE


def test_end_link_beyond_gripper_allowance_is_a_warn():
    v = prior_values()
    v[5] = 2.0 * np.array([1.0, 0.0, 0.0, 0.3])
    report = audit(params(v), make_links())
    assert {c.kind for c in report.warnings} >= {"gripper_mass", "com"}
    assert report.failures == []


def test_pointwise_audit_never_says_infeasible():
    for v in (negative_mass(), com_outside_link(), absurd_total_mass(), prior_values()):
        report = audit(params(v), make_links())
        assert report.verdict in (Verdict.FEASIBLE, Verdict.POINTWISE_VIOLATION)
        assert "Y, tau not provided" in report.summary()


def test_parameters_are_matched_to_links_by_name():
    v = negative_mass()
    reordered = audit(params(v[::-1], NAMES[::-1]), make_links())
    assert {c.link for c in reordered.failures} == {"link3"}


def test_bare_arrays_are_refused():
    with pytest.raises(TypeError, match="ParamSet"):
        audit(prior_values(), make_links())


@pytest.mark.parametrize("units", [None, "A", "current"])
def test_units_must_be_si(units):
    kw = {} if units is None else {"units": units}
    with pytest.raises(ValueError, match="multiply by kt"):
        ParamSet(NAMES, prior_values(), **kw)
    with pytest.raises(ValueError, match="multiply by kt"):
        LinkSpec("link1", [1, 0, 0, 0], [-1, -1, -1], [1, 1, 1], "mesh", **kw)


def test_prior_outside_its_own_box_is_refused():
    with pytest.raises(ValueError, match="bounds are wrong"):
        LinkSpec("link1", [1.0, 0.5, 0, 0], [-0.1] * 3, [0.1] * 3, "mesh", units="SI")


def test_thresholds_are_pinned():
    assert F.M_MIN_KG == 1e-4
    assert F.RESIDUAL_SLACK == 0.05
    assert F.STAGE2_RMS_FLOOR_NM == 1e-4
    assert F.NULLSPACE_DRMS_ABS_NM == 0.02
    assert F.NULLSPACE_DRMS_REL == 0.05
    assert F.TOTAL_MASS_RANGE == (0.8, 2.5)
    assert F.GRIPPER_ALLOWANCE_MASS_KG == 1.0
    assert F.GRIPPER_ALLOWANCE_COM_RADIUS_M == 0.15
    assert F.KINEMATIC_BOX_PAD_ABS_M == 0.05
    assert F.KINEMATIC_BOX_PAD_REL == 0.5
    assert F.RANK_RTOL == 1e-6
    assert F.CHECK_TOL == 1e-9
    assert F.UNITS == "SI"
    assert GripperAllowance() == GripperAllowance(1.0, 0.15)


def test_kinematic_box_pads_the_joint_hull():
    lo, hi = F.kinematic_box(np.array([[0.3, 0.0, 0.0]]), com=np.array([0.15, -0.02, 0.0]))
    np.testing.assert_allclose(lo, [-0.15, -0.17, -0.15])
    np.testing.assert_allclose(hi, [0.45, 0.15, 0.15])


# -- verdicts with data ----------------------------------------------------------
def regressor(null_dirs=(), rows=600, seed=0):
    """Random gravity-like regressor whose nullspace contains ``null_dirs``."""
    Y = np.random.default_rng(seed).normal(scale=2.0, size=(rows, 4 * len(NAMES)))
    for d in null_dirs:
        d = d / np.linalg.norm(d)
        Y = Y - np.outer(Y @ d, d)
    return Y


def planted_direction():
    """Mostly link3's mass: static data cannot see it in this regressor."""
    d = np.zeros(4 * len(NAMES))
    d[8], d[11] = 1.0, 0.02
    return d


def artifact_case():
    d = planted_direction()
    Y = regressor([d])
    truth = prior_values().reshape(-1)
    tau = Y @ truth + np.random.default_rng(1).normal(scale=1e-3, size=len(Y))
    given = truth - 2 * MASS[2] * d          # link3 mass -0.7, same predictions
    return params(given.reshape(-1, 4)), Y, tau


def infeasible_case():
    Y = regressor()
    bad = negative_mass()                     # identifiable: Y has full column rank
    return params(bad), Y, Y @ bad.reshape(-1)


def test_feasible_parameters_stay_feasible_with_data():
    pytest.importorskip("scipy.optimize")
    Y = regressor()
    truth = prior_values()
    report = audit(params(truth), make_links(), Y, Y @ truth.reshape(-1))
    assert report.verdict is Verdict.FEASIBLE


def test_nullspace_violation_is_an_artifact_not_infeasible():
    pytest.importorskip("scipy.optimize")
    given, Y, tau = artifact_case()
    assert audit(given, make_links()).verdict is Verdict.POINTWISE_VIOLATION
    report = audit(given, make_links(), Y, tau)
    assert report.verdict is Verdict.NULLSPACE_ARTIFACT
    assert report.delta_rms_nm <= report.delta_rms_threshold_nm
    ident = report.identifiability
    assert ident.rank == ident.n_params - 1
    assert ident.null_fraction[2, 0] > 0.9          # link3 mass: invisible
    assert ident.null_fraction[1, 0] < 0.01         # link2 mass: fully determined
    assert "NULLSPACE_ARTIFACT" in report.summary()


def test_identifiable_violation_is_infeasible():
    pytest.importorskip("scipy.optimize")
    given, Y, tau = infeasible_case()
    report = audit(given, make_links(), Y, tau)
    assert report.verdict is Verdict.INFEASIBLE
    assert report.delta_rms_nm > report.delta_rms_threshold_nm
    assert report.binding


@pytest.mark.parametrize("case", [artifact_case, infeasible_case])
def test_projection_is_feasible_and_reports_the_residual_cost(case):
    pytest.importorskip("scipy.optimize")
    given, Y, tau = case()
    out, report = project(given, make_links(), Y, tau)
    assert audit(out, make_links()).verdict is Verdict.FEASIBLE
    assert audit(out, make_links(), Y, tau).verdict is Verdict.FEASIBLE
    cost = report.projection.rms_after_nm - report.rms_free_nm
    if report.verdict is Verdict.NULLSPACE_ARTIFACT:
        assert cost <= report.delta_rms_threshold_nm
        # the prior picks the representative: link3 goes back to its URDF mass
        assert abs(out.values[2, 0] - MASS[2]) < 0.05
    else:
        assert report.verdict is Verdict.INFEASIBLE
        assert cost > report.delta_rms_threshold_nm


def test_projection_leaves_feasible_parameters_alone():
    pytest.importorskip("scipy.optimize")
    Y = regressor()
    truth = params(prior_values())
    out, report = project(truth, make_links(), Y, Y @ prior_values().reshape(-1))
    assert out is truth
    assert report.projection.delta_rms_nm == 0


def test_projection_needs_data():
    with pytest.raises(ValueError, match="needs Y and tau"):
        project(params(negative_mass()), make_links(), None, None)


def test_full_tier_is_interface_only():
    with pytest.raises(NotImplementedError):
        F.audit_full(np.zeros(60), NAMES)
    with pytest.raises(NotImplementedError):
        F.project_full(np.zeros(60), NAMES, np.zeros((1, 60)), np.zeros(1))


# -- wiring --------------------------------------------------------------------
def test_static_identification_report_says_feasibility_does_not_apply():
    from piper_dynamics.calibration.identify import IdentificationReport

    z = np.zeros(6)
    report = IdentificationReport(make_params(), np.zeros((3, 6)), z, z, 1.0, "static")
    assert "physical feasibility: not applicable" in report.summary()
    report.feasibility = audit(params(negative_mass()), make_links())
    assert "POINTWISE_VIOLATION" in report.summary()


def test_diagnose_reports_absent_overrides(tmp_path, capsys):
    from piper_dynamics.cli import diagnose_main

    path = tmp_path / "arm.yaml"
    make_params().save(path)
    diagnose_main(["--params", str(path)])
    assert "nothing to audit" in capsys.readouterr().out
