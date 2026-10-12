"""Shared fixtures.

Model tests run against a toy 6-DOF URDF authored in this repository, so the
model layer is exercised in CI without depending on a vendor description that
is not ours to redistribute. Point ``PIPER_URDF`` at a real Piper URDF to run
the same tests against actual hardware geometry.
"""

import os
from pathlib import Path

import pytest

FIXTURE_URDF = Path(__file__).parent / "fixtures" / "toy6.urdf"


@pytest.fixture(scope="session")
def urdf_path() -> str:
    return os.environ.get("PIPER_URDF", str(FIXTURE_URDF))


@pytest.fixture(scope="session")
def ee_frame() -> str:
    default = "link6" if "PIPER_URDF" not in os.environ else "link6"
    return os.environ.get("PIPER_EE_FRAME", default)


@pytest.fixture
def model(urdf_path, ee_frame):
    pytest.importorskip("pinocchio", reason="pinocchio not installed")
    from piper_dynamics import PiperModel

    return PiperModel(urdf_path, ee_frame=ee_frame)
