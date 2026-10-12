"""Parameter loading, validation and round-tripping."""

import pytest

from piper_dynamics.params import ArmParams, JointParams, ParamError, Provenance


def make_params(nq=6):
    return ArmParams(
        joints=[JointParams(kt=0.5 + 0.1 * i, direction=1 if i % 2 == 0 else -1)
                for i in range(nq)],
        provenance=Provenance(arm_id="test-arm", calibrated_at="2026-10-12"),
    )


def test_roundtrip(tmp_path):
    p = make_params()
    path = tmp_path / "arm.yaml"
    p.save(path)
    q = ArmParams.load(path)
    assert q.nq == p.nq
    assert q.kt.tolist() == pytest.approx(p.kt.tolist())
    assert q.direction.tolist() == p.direction.tolist()
    assert q.provenance.arm_id == "test-arm"


def test_direction_must_be_unit():
    with pytest.raises(ParamError):
        JointParams(kt=1.0, direction=0)


def test_kt_must_be_positive():
    """Sign belongs in `direction`, so the two stay separable."""
    with pytest.raises(ParamError):
        JointParams(kt=-1.0)


def test_schema_version_mismatch_is_refused():
    with pytest.raises(ParamError, match="schema"):
        ArmParams.from_dict({"schema_version": 999, "joints": []})


def test_missing_file_message_points_at_calibration(tmp_path):
    with pytest.raises(ParamError, match="piper-calibrate"):
        ArmParams.load(tmp_path / "nope.yaml")


def test_warnings_flag_missing_provenance():
    p = make_params()
    p.provenance.arm_id = "unknown"
    assert any("provenance" in w for w in p.warnings())


def test_warnings_flag_high_residual():
    p = make_params()
    p.provenance.fit_rms_nm = [0.01] * 5 + [2.0]
    assert any("residual" in w for w in p.warnings())
