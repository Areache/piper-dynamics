"""Hardware adapter and HDF5 reader, without hardware.

The CAN adapter is exercised against a fake SDK that mimics the piper_sdk
0.6.2 message shapes, so unit scaling and the motion guards are checked on
every commit even though nobody here has an arm in CI.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from piper_dynamics.io.can import UNIT_SCALE, PiperSdkArm


class FakeSdk:
    """Raw-integer telemetry; JointCtrl teleports the arm to the target."""

    def __init__(self, joint_raw, speed_raw=0, current_raw=500, enabled=True, track=True):
        self.joint_raw = list(joint_raw)
        self.speed_raw = speed_raw
        self.current_raw = current_raw
        self.enabled = enabled
        self.track = track
        self.calls = []

    def GetArmJointMsgs(self):
        js = SimpleNamespace(**{f"joint_{i + 1}": v for i, v in enumerate(self.joint_raw)})
        return SimpleNamespace(joint_state=js)

    def GetArmHighSpdInfoMsgs(self):
        motor = SimpleNamespace(motor_speed=self.speed_raw, current=self.current_raw)
        return SimpleNamespace(**{f"motor_{i + 1}": motor for i in range(6)})

    def GetArmEnableStatus(self):
        return [self.enabled] * 6

    def MotionCtrl_2(self, *args):
        self.calls.append(("MotionCtrl_2", args))

    def JointCtrl(self, *args):
        self.calls.append(("JointCtrl", args))
        if self.track:
            self.joint_raw = list(args)


def test_read_state_scaling():
    sdk = FakeSdk([90_000, 0, 0, 0, 0, 0], speed_raw=1000, current_raw=1500)
    q, v, c = PiperSdkArm(sdk=sdk).read_state()
    assert q[0] == pytest.approx(np.pi / 2)          # 90 000 x 0.001 deg
    assert v == pytest.approx(np.ones(6))             # 1000 x 0.001 rad/s
    assert c == pytest.approx(np.full(6, 1.5))        # 1500 x 0.001 A
    assert UNIT_SCALE["velocity"] == 0.001


def test_move_j_refused_without_allow_motion():
    sdk = FakeSdk([0] * 6)
    with pytest.raises(RuntimeError, match="motion is disabled"):
        PiperSdkArm(sdk=sdk).move_j(np.zeros(6))
    assert sdk.calls == []


def test_move_j_refused_when_not_enabled():
    sdk = FakeSdk([0] * 6, enabled=False)
    with pytest.raises(RuntimeError, match="not enabled"):
        PiperSdkArm(sdk=sdk, allow_motion=True).move_j(np.zeros(6))
    assert sdk.calls == []


def test_speed_is_capped():
    with pytest.raises(ValueError):
        PiperSdkArm(sdk=FakeSdk([0] * 6), speed_pct=50)


def test_move_j_sends_move_j_in_raw_units():
    sdk = FakeSdk([0] * 6)
    target = np.deg2rad([10, -20, 30, 0, 5, -5])
    PiperSdkArm(sdk=sdk, allow_motion=True, speed_pct=10).move_j(target)
    assert sdk.calls[0] == ("MotionCtrl_2", (0x01, 0x01, 10, 0x00))
    assert sdk.calls[1] == ("JointCtrl", (10_000, -20_000, 30_000, 0, 5_000, -5_000))


def test_move_j_times_out_instead_of_pretending():
    sdk = FakeSdk([0] * 6, track=False)
    arm = PiperSdkArm(sdk=sdk, allow_motion=True)
    with pytest.raises(TimeoutError):
        arm.move_j(np.deg2rad([10, 0, 0, 0, 0, 0]), timeout_s=0.1, resend_hz=100)


# -- HDF5 -----------------------------------------------------------------
@pytest.fixture
def bimanual_episode(tmp_path):
    h5py = pytest.importorskip("h5py")
    T = 5
    qpos = np.arange(T * 14, dtype=float).reshape(T, 14)
    current = 100 + np.arange(T * 12, dtype=float).reshape(T, 12)
    effort = np.zeros((T, 14))
    effort[:, [6, 13]] = 1.0          # only the grippers populated, as piper-aio records it
    path = tmp_path / "episode_0.hdf5"
    with h5py.File(path, "w") as f:
        f["observations/qpos"] = qpos
        f["observations/qvel"] = qpos
        f["observations/motor_current"] = current
        f["observations/effort"] = effort
    return path, qpos, current


def test_read_episode_selects_each_arm(bimanual_episode):
    from piper_dynamics.io.hdf5 import read_episode

    path, qpos, current = bimanual_episode
    q, _, c = read_episode(path, arm="right")
    np.testing.assert_array_equal(q, qpos[:, 7:13])
    np.testing.assert_array_equal(c, current[:, 6:12])
    q, _, c = read_episode(path, arm="left")
    np.testing.assert_array_equal(q, qpos[:, 0:6])
    np.testing.assert_array_equal(c, current[:, 0:6])


def test_read_episode_rejects_zero_effort(bimanual_episode):
    from piper_dynamics.io.hdf5 import read_episode

    path, _, _ = bimanual_episode
    with pytest.raises(ValueError, match="all zeros"):
        read_episode(path, current_key="observations/effort", arm="left")
