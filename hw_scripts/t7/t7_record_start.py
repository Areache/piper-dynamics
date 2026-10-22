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
from guarded_move import PYAGX_DEFAULT, connect, read_state  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("can_interface", choices=["can_left", "can_right"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--pyagx", default=PYAGX_DEFAULT)
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"{out} exists; pick a new name")
    robot = connect(a.can_interface, a.pyagx)
    poses = []
    while True:
        cmd = input(f"[{len(poses)} poses] Enter = record, q = quit: ").strip().lower()
        if cmd == "q":
            break
        Q = []
        t0 = time.monotonic()
        while time.monotonic() - t0 < 1.0:
            q, _, _, _ = read_state(robot)
            Q.append(q)
            time.sleep(0.01)
        Q = np.array(Q)
        std = np.degrees(np.nanstd(Q, 0))
        print("  q [deg]:", np.round(np.degrees(np.nanmean(Q, 0)), 2).tolist(), " max std", round(float(std.max()), 3))
        if std.max() > 0.2 and input("  arm was moving; keep? [y/N] ").strip().lower() != "y":
            continue
        gap = float(input("  ruler: tip-to-plate gap in mm: ").strip())
        note = input("  note (where on the plate, wrist tilt): ").strip()
        poses.append({"q_rad": np.nanmean(Q, 0).tolist(), "q_deg": np.round(np.degrees(np.nanmean(Q, 0)), 3).tolist(),
                      "q_std_deg": np.round(std, 3).tolist(), "gap_mm": gap, "note": note})
    out.write_text(json.dumps({"can_interface": a.can_interface, "recorded_at": datetime.now().isoformat(timespec="seconds"),
                               "purpose": "T7 start poses", "poses": poses}, indent=1))
    print(f"saved {len(poses)} poses -> {out}")


if __name__ == "__main__":
    main()
