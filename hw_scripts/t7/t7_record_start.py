#!/usr/bin/env python3
"""Record T7 start poses (robot side). READ-ONLY: only reads joint angles, sends no command.

The operator puts the arm in drag (teach) mode with the physical button, closes the gripper, and holds the
tip about 8 cm above the rigid test plate with the gripper pointing straight down. Each Enter records the
mean joint angles over 1 s and asks for the ruler-measured tip-to-plate gap in mm.
Output: the teach_poses.py JSON schema (+ gap_mm per pose), which t7_plan.py reads.

usage: python t7_record_start.py can_right --out start_can_right.json
"""
import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from guarded_move import (PYAGX_DEFAULT, connect, read_state, check_air_feedback,
                          wait_for_feedback, FeedbackNotReady, OffPlanFeedback)  # noqa: E402


def sample_pose(robot, clock=time.monotonic, sleep=time.sleep):
    """Read only; reject missing/stale feedback and movement rather than averaging them away."""
    wait_for_feedback(robot, clock, sleep)
    Q = []
    t0 = clock()
    while clock() - t0 < 1.0:
        extra = {}
        state = read_state(robot, extra)
        check_air_feedback(robot, state, extra)
        Q.append(state[0].copy())
        sleep(.01)
    Q = np.asarray(Q)
    if len(Q) < 20:
        raise ValueError('not enough joint samples')
    std = np.degrees(np.std(Q, axis=0))
    if std.max() > .2:
        raise ValueError('arm moved during recording; hold stationary and record again')
    return Q, std


def positive_gap(raw):
    gap = float(raw)
    if not np.isfinite(gap) or gap <= 0:
        raise ValueError('gap must be finite and positive')
    return gap


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("can_interface", choices=["can_left", "can_right"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--pyagx", default=PYAGX_DEFAULT)
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"{out} exists; pick a new name")
    out.parent.mkdir(parents=True, exist_ok=True)
    robot = connect(a.can_interface, a.pyagx)
    poses = []
    while True:
        cmd = input(f"[{len(poses)} poses] Enter = record, q = quit: ").strip().lower()
        if cmd == "q":
            break
        try:
            Q, std = sample_pose(robot)
        except (FeedbackNotReady, OffPlanFeedback, ValueError) as exc:
            print(f'  recording rejected: {exc}; no commands sent')
            continue
        print("  q [deg]:", np.round(np.degrees(Q.mean(0)), 2).tolist(), " max std", round(float(std.max()), 3))
        gap = None
        while gap is None:
            raw = input("  ruler: tip-to-plate gap in mm: ").strip()
            try:
                gap = positive_gap(raw)
            except ValueError:
                print("  (enter a number, e.g. 95)")
        note = input("  note (where on the plate, wrist tilt): ").strip()
        poses.append({"q_rad": Q.mean(0).tolist(), "q_deg": np.round(np.degrees(Q.mean(0)), 3).tolist(),
                      "q_std_deg": np.round(std, 3).tolist(), "gap_mm": gap, "note": note,
                      "joint_samples_rad": Q.tolist(), "recorded_at": datetime.now().isoformat(timespec="seconds")})
        # Persist each accepted point so a later interruption does not lose earlier points.
        out.write_text(json.dumps({"can_interface": a.can_interface,
                                   "purpose": "T7 start poses", "poses": poses}, indent=1, allow_nan=False))
    out.write_text(json.dumps({"can_interface": a.can_interface, "recorded_at": datetime.now().isoformat(timespec="seconds"),
                               "purpose": "T7 start poses", "poses": poses}, indent=1))
    print(f"saved {len(poses)} poses -> {out}")


if __name__ == "__main__":
    main()
