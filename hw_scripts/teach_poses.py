#!/usr/bin/env python3
"""Record operator-taught poses. Passive: sends no command and no CAN frame.

The operator puts the arm in teach (drag) mode with the arm's own button,
moves it by hand, and presses Enter here at each pose. Each pose is the mean
of ``--avg-s`` seconds of joint feedback; a pose where the arm was still
moving is flagged and kept only if the operator says so.

    python hw_scripts/teach_poses.py can_right --operator-confirmed
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from piper_dynamics.io.can import PiperSdkArm

STILL_DEG = 0.2   # joint std above this during averaging means the arm was moving


def sample(arm: PiperSdkArm, sdk, seconds: float, rate_hz: float = 100.0):
    qs, cs, teach = [], [], []
    for _ in range(max(int(seconds * rate_hz), 1)):
        q, _, c = arm.read_state()
        qs.append(q)
        cs.append(c)
        teach.append(int(sdk.GetArmStatus().arm_status.teach_status))
        time.sleep(1.0 / rate_hz)
    return np.array(qs), np.array(cs), teach


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("can_interface")
    p.add_argument("--operator-confirmed", action="store_true")
    p.add_argument("--avg-s", type=float, default=1.0)
    p.add_argument("--out", type=Path)
    args = p.parse_args()
    if not args.operator_confirmed:
        p.error("operator confirmation required: --operator-confirmed")

    from piper_sdk import C_PiperInterface_V2

    out = args.out or Path(f"hw_results/T3/poses_{args.can_interface}.json")
    if out.exists():
        p.error(f"{out} exists; pick a new --out rather than overwriting taught poses")
    sdk = C_PiperInterface_V2(args.can_interface)
    sdk.ConnectPort(piper_init=False)          # passive: no init query frames
    arm = PiperSdkArm(sdk=sdk)                  # allow_motion=False: move_j would refuse

    poses: list[dict] = []
    try:
        while True:
            cmd = input(f"[{len(poses)} poses] Enter = record, q = quit: ").strip().lower()
            if cmd == "q":
                break
            q, c, teach = sample(arm, sdk, args.avg_s)
            std_deg = np.rad2deg(q.std(axis=0))
            q_mean = q.mean(axis=0)
            print("  q [deg]:", np.round(np.rad2deg(q_mean), 1).tolist(),
                  " current [A]:", np.round(c.mean(axis=0), 3).tolist())
            if std_deg.max() > STILL_DEG:
                keep = input(f"  arm moved (max std {std_deg.max():.2f} deg). keep anyway? [y/N] ")
                if keep.strip().lower() != "y":
                    continue
            note = input("  note (optional, e.g. 'arm high, wrist down'): ").strip()
            poses.append({
                "q_rad": q_mean.tolist(),
                "q_deg": np.rad2deg(q_mean).round(3).tolist(),
                "q_std_deg": std_deg.round(3).tolist(),
                "current_a_in_teach_mode": c.mean(axis=0).round(4).tolist(),
                "teach_status": sorted(set(teach)),
                "note": note,
            })
    finally:
        sdk.DisconnectPort()

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "can_interface": args.can_interface,
        "recorded_at": datetime.now().isoformat(timespec="seconds"),
        "avg_s": args.avg_s,
        "poses": poses,
    }, indent=1))
    print(f"saved {len(poses)} poses to {out}")


if __name__ == "__main__":
    main()
