#!/usr/bin/env python3
"""Replay reviewed poses and record static telemetry. THIS MOVES THE ARM.

Prerequisites, all checked or asked for here:
  - an operator at the arm with the e-stop in hand (--operator-confirmed)
  - the arm already enabled in CAN mode (this script never enables it)
  - a pose file written by teach_poses.py and reviewed on the task board

For each selected pose: MOVE J at --speed-pct (<= 20), wait --settle-s, then
record --hold-s of joint angle, speed and current at --rate-hz. Writes

  <out>/raw.npz       every sample, per pose
  <out>/samples.npz   per-pose means, loadable by `piper-calibrate --samples`

Any error or Ctrl-C stops sending commands immediately; the arm holds its last
target. It does not return to the start pose on its own.

    python hw_scripts/collect_poses.py can_right hw_results/T3/poses_can_right.json \
        --indices 0,3 --confirm-each --operator-confirmed
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from piper_dynamics.calibration.collect import CalibrationSamples
from piper_dynamics.io.can import MAX_SPEED_PCT, PiperSdkArm

DRIFT_DEG = 0.2   # joint motion during the hold above this marks the pose as not static


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("can_interface")
    p.add_argument("poses", type=Path)
    p.add_argument("--indices", help="comma-separated pose indices; default all")
    p.add_argument("--speed-pct", type=int, default=10)
    p.add_argument("--settle-s", type=float, default=2.0)
    p.add_argument("--hold-s", type=float, default=3.0)
    p.add_argument("--rate-hz", type=float, default=100.0)
    p.add_argument("--confirm-each", action="store_true", help="ask before every move")
    p.add_argument("--operator-confirmed", action="store_true")
    p.add_argument("--out", type=Path)
    args = p.parse_args()
    if not args.operator_confirmed:
        p.error("operator confirmation required: --operator-confirmed")
    if not 1 <= args.speed_pct <= MAX_SPEED_PCT:
        p.error(f"--speed-pct must be 1..{MAX_SPEED_PCT}")

    spec = json.loads(args.poses.read_text())
    if spec["can_interface"] != args.can_interface:
        p.error(f"pose file was taught on {spec['can_interface']}, not {args.can_interface}")
    all_poses = np.array([pose["q_rad"] for pose in spec["poses"]])
    if args.indices:
        idx = [int(i) for i in args.indices.split(",")]
    else:
        idx = list(range(len(all_poses)))
    targets = all_poses[idx]

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = args.out or Path.home() / "piper_hw_runs" / f"poses_{args.can_interface}_{stamp}"
    out.mkdir(parents=True, exist_ok=False)

    arm = PiperSdkArm(args.can_interface, allow_motion=True, speed_pct=args.speed_pct)
    q_start, _, _ = arm.read_state()
    print("start pose [deg]:", np.round(np.rad2deg(q_start), 1).tolist())
    print(f"plan: {len(targets)} poses at speed {args.speed_pct}%, "
          f"settle {args.settle_s}s, hold {args.hold_s}s")
    for k, (i, q) in enumerate(zip(idx, targets, strict=True)):
        note = spec["poses"][i].get("note", "")
        print(f"  {k}: pose {i} [deg]", np.round(np.rad2deg(q), 1).tolist(), note)
    if input("type YES to start: ").strip() != "YES":
        print("aborted, nothing sent")
        return

    raw = {"t": [], "q": [], "v": [], "current": [], "pose_index": []}
    means_q, means_v, means_c, static, done = [], [], [], [], []
    try:
        for i, q in zip(idx, targets, strict=True):
            if args.confirm_each and input(f"move to pose {i}? [y/N] ").strip().lower() != "y":
                print("stopped by operator")
                break
            arm.move_j(q)
            time.sleep(args.settle_s)
            qs, vs, cs, ts = [], [], [], []
            for _ in range(int(args.hold_s * args.rate_hz)):
                qn, vn, cn = arm.read_state()
                qs.append(qn)
                vs.append(vn)
                cs.append(cn)
                ts.append(time.monotonic())
                time.sleep(1.0 / args.rate_hz)
            qs, vs, cs = np.array(qs), np.array(vs), np.array(cs)
            drift = np.rad2deg(np.ptp(qs, axis=0)).max()
            for key, val in (("t", ts), ("q", qs), ("v", vs), ("current", cs), ("pose_index", i)):
                raw[key].append(val)
            means_q.append(qs.mean(0))
            means_v.append(vs.mean(0))
            means_c.append(cs.mean(0))
            static.append(drift < DRIFT_DEG)
            done.append(i)
            print(f"  pose {i}: drift {drift:.3f} deg, current [A]",
                  np.round(cs.mean(0), 3).tolist())
    finally:
        if done:
            np.savez_compressed(out / "raw.npz", **{k: np.array(v) for k, v in raw.items()})
            CalibrationSamples(
                q=np.array(means_q), v=np.array(means_v), current=np.array(means_c),
                static=np.array(static), dt=1.0 / args.rate_hz,
                meta={"procedure": "static", "can_interface": args.can_interface,
                      "poses_file": str(args.poses), "pose_indices": done,
                      "speed_pct": args.speed_pct, "settle_s": args.settle_s,
                      "hold_s": args.hold_s},
            ).save(out / "samples.npz")
            print(f"saved {len(done)} poses to {out}")


if __name__ == "__main__":
    main()
