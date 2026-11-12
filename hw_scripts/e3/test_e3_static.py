"""Offline tests of e3_static.py with a fake arm: command sequence, no motion while waiting for the operator,
abort paths. No SDK import or hardware connection."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e3_static as E  # noqa: E402


class FakeArm:
    """Reaches every move_j target at once; records every command with the fake time."""

    def __init__(self, q0):
        self.q, self.t, self.log, self.interrupt_at = np.asarray(q0, float), 0.0, [], None

    def get_joint_angles(self):
        return SimpleNamespace(msg=self.q.copy())

    def get_motor_states(self, i):
        return SimpleNamespace(msg=SimpleNamespace(position=self.q[i - 1], velocity=0.0, current=0.1 * i))

    def move_j(self, q):
        self.log.append(("move_j", self.t, list(q)))
        self.q = np.asarray(q, float)

    def set_speed_percent(self, pct):
        self.log.append(("speed", self.t, pct))

    def electronic_emergency_stop(self):
        self.log.append(("estop", self.t, None))

    def clock(self):
        return self.t

    def sleep(self, dt):
        self.t += dt
        if self.interrupt_at is not None and self.t >= self.interrupt_at:
            self.interrupt_at = None
            raise KeyboardInterrupt


@pytest.fixture
def poses(tmp_path):
    P = {"can_interface": "can_right", "poses": [{"q_rad": [0.1 * k, 1.0, -1.0, 0.2, -0.3, 0.1], "note": f"n{k}"}
                                                  for k in range(3)]}
    f = tmp_path / "poses.json"
    f.write_text(json.dumps(P))
    return f


def answers(*xs):
    it = iter(xs)
    return lambda prompt: next(it)


def test_poses_mode_visits_in_order(tmp_path, poses):
    arm = FakeArm([0, 0, 0, 0, 0, 0])
    out = E.main(["poses", "--poses", str(poses), "--label", "w190", "--operator-confirmed", "--out",
                  str(tmp_path / "o"), "--settle-s", "1", "--hold-s", "0.5"], robot=arm, clock=arm.clock,
                 sleep=arm.sleep, ask=answers("YES"))
    S = json.loads((out / "summary.json").read_text())
    assert [r["pose"] for r in S["records"]] == [0, 1, 2] and S["label"] == "w190" and "finished" in S
    targets = [tuple(np.round(c[2], 6)) for c in arm.log if c[0] == "move_j"]
    seen = list(dict.fromkeys(targets))
    assert seen == [tuple(np.round([0.1 * k, 1.0, -1.0, 0.2, -0.3, 0.1], 6)) for k in range(3)]
    assert arm.log[0] == ("speed", arm.log[0][1], 15)
    r = np.load(out / "rec_01.npz")
    assert r["current"].shape[1] == 6 and abs(r["current"][:, 2].mean() - 0.3) < 1e-9 and float(r["t_mono0"]) > 0


def test_weigh_mode_sends_nothing_while_waiting(tmp_path, poses):
    arm = FakeArm([0, 1.0, -1.0, 0.2, -0.3, 0.1])
    marks = []

    def ask(prompt):
        marks.append((prompt[:4], len(arm.log)))
        return "YES" if prompt.startswith("type") else ""

    out = E.main(["weigh", "--poses", str(poses), "--pose", "2", "--steps", "0,190,0", "--operator-confirmed",
                  "--out", str(tmp_path / "o"), "--settle-s", "1", "--hold-s", "0.5"], robot=arm, clock=arm.clock,
                 sleep=arm.sleep, ask=ask)
    S = json.loads((out / "summary.json").read_text())
    assert [r["step"] for r in S["records"]] == ["0", "190", "0"]
    n_after_settle = marks[1][1]                       # commands sent before the first weight prompt
    assert len(arm.log) == n_after_settle             # nothing sent after the arm reached the pose


def test_no_YES_sends_nothing(tmp_path, poses):
    arm = FakeArm([0] * 6)
    with pytest.raises(SystemExit):
        E.main(["poses", "--poses", str(poses), "--operator-confirmed", "--out", str(tmp_path / "o")], robot=arm,
               clock=arm.clock, sleep=arm.sleep, ask=answers("no"))
    assert arm.log == []


def test_interrupt_holds_measured_pose(tmp_path, poses):
    arm = FakeArm([0] * 6)
    arm.interrupt_at = 2.0
    with pytest.raises(KeyboardInterrupt):
        E.main(["poses", "--poses", str(poses), "--operator-confirmed", "--out", str(tmp_path / "o"),
                "--settle-s", "3"], robot=arm, clock=arm.clock, sleep=arm.sleep, ask=answers("YES"))
    S = json.loads((tmp_path / "o" / "summary.json").read_text())
    assert S["abort"]["status"] == "KeyboardInterrupt"
    assert arm.log[-1][0] == "move_j" and np.allclose(arm.log[-1][2], arm.q)


def test_limits(tmp_path, poses):
    with pytest.raises(SystemExit):
        E.main(["poses", "--poses", str(poses), "--operator-confirmed", "--speed-pct", "30"], robot=FakeArm([0] * 6))
    with pytest.raises(SystemExit):
        E.main(["weigh", "--poses", str(poses), "--operator-confirmed"], robot=FakeArm([0] * 6))
