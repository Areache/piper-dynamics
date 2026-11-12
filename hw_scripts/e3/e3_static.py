#!/usr/bin/env python3
"""E3 static recording on the right arm (robot side). THIS MOVES THE ARM.

Two modes, both built from the guarded_move.py helpers (same connect / feedback check / settle / 100 Hz static
record as T7 and E1), so no new motion code:

  poses  Known-weight pass (T5): visit every pose of a T4/T5 pose file in file order, settle, wait --settle-s,
         record --hold-s. Run once per weight with the SAME file and order, so every pose is approached from the
         same previous pose with every weight (static friction then enters the weight difference only through
         the load change).
  weigh  Same-pose differential weighing (KR6): go to one pose and stay there. For each step of --steps the
         operator hangs or removes a weight while the arm holds still (no command is sent while waiting), presses
         Enter, then --settle-s later --hold-s is recorded.

Each recording keeps t (s from its start), q, motor position, velocity, current (A), plus time.monotonic() and
time.time() at its start, so lowspd_log.py (temperatures, firmware flags) can be aligned on the same clock.
Outputs to --out: rec_XX.npz per recording, summary.json (updated after every recording).

Safety: --operator-confirmed, an enabled arm (never enabled here), typed YES; --confirm-each asks y before
each move; speed percent <= 20. Ctrl-C or any error holds the measured pose and stops; the arm never returns
on its own. Weight changes only happen at a prompt, with the arm still.

usage:
  python e3/e3_static.py poses --poses e3/poses_can_right.json --label w0 --speed-pct 15 --operator-confirmed \
      --out ~/piper_hw_runs/E3_<date>/B_w0
  python e3/e3_static.py weigh --poses e3/poses_can_right.json --pose 5 --steps 0,190,0,482,0 \
      --operator-confirmed --out ~/piper_hw_runs/E3_<date>/C_p5
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "t7"))
import guarded_move as gm  # noqa: E402

MAX_PCT = 20


def save(out, k, rec, meta, summary):
    np.savez_compressed(out / f"rec_{k:02d}.npz", **rec)
    meta = {**meta, "rec": k, "n": int(len(rec["t"])),
            "drift_deg": float(np.degrees(np.ptp(rec["q"], axis=0)).max()),
            "v_max": float(np.nanmax(np.abs(rec["v"]))),
            "current_mean_A": np.round(np.nanmean(rec["current"], 0), 4).tolist()}
    summary["records"].append(meta)
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(f"  rec {k:02d} {meta.get('pose_name', '')} {meta.get('step', '')}: drift {meta['drift_deg']:.3f} deg, "
          f"current {meta['current_mean_A']}")


def record(robot, a, clock, sleep):
    sleep(a.settle_s)
    t_mono, t_wall = clock(), time.time()
    rec = gm.record_static(robot, a.hold_s, clock, sleep)
    rec["t_mono0"], rec["t_wall0"] = np.array(t_mono), np.array(t_wall)
    return rec, {"t_mono0": t_mono, "t_wall0": t_wall, "wall_time": datetime.now().isoformat(timespec="seconds")}


def main(argv=None, robot=None, clock=time.monotonic, sleep=time.sleep, ask=input):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("mode", choices=("poses", "weigh"))
    p.add_argument("--poses", type=Path, required=True)
    p.add_argument("--channel", default="can_right")
    p.add_argument("--indices", help="poses mode: comma-separated pose indices (default all, file order)")
    p.add_argument("--pose", type=int, help="weigh mode: pose index to hold")
    p.add_argument("--steps", default="0,190,0,277.6,0,482,0", help="weigh mode: grams on the arm at each step")
    p.add_argument("--label", default="", help="free text stored in the summary, e.g. the weight on the arm")
    p.add_argument("--speed-pct", type=int, default=15)
    p.add_argument("--settle-s", type=float, default=3.0)
    p.add_argument("--hold-s", type=float, default=3.0)
    p.add_argument("--confirm-each", action="store_true")
    p.add_argument("--operator-confirmed", action="store_true")
    p.add_argument("--pyagx", default=gm.PYAGX_DEFAULT)
    p.add_argument("--out", type=Path)
    a = p.parse_args(argv)
    if not a.operator_confirmed:
        p.error("operator confirmation required: --operator-confirmed")
    if not 1 <= a.speed_pct <= MAX_PCT:
        p.error(f"--speed-pct must be 1..{MAX_PCT}")
    spec = json.loads(a.poses.read_text())
    if spec.get("can_interface", a.channel) != a.channel:
        p.error(f"pose file is for {spec['can_interface']}, not {a.channel}")
    allq = [np.asarray(x["q_rad"], float) for x in spec["poses"]]
    if a.mode == "poses":
        idx = [int(i) for i in a.indices.split(",")] if a.indices else list(range(len(allq)))
    else:
        if a.pose is None:
            p.error("weigh mode needs --pose")
        idx = [a.pose]
    steps = [s.strip() for s in a.steps.split(",")] if a.mode == "weigh" else []

    out = Path(a.out or Path.home() / "piper_hw_runs" / f"E3_{a.mode}_{datetime.now():%Y%m%d_%H%M%S}")
    out.mkdir(parents=True, exist_ok=False)
    summary = {"script": "e3_static.py", "mode": a.mode, "label": a.label, "poses_file": a.poses.name,
               "pose_indices": idx, "steps_g": steps, "speed_pct": a.speed_pct, "settle_s": a.settle_s,
               "hold_s": a.hold_s, "started": datetime.now().isoformat(timespec="seconds"), "records": []}

    if robot is None:
        robot = gm.connect(a.channel, a.pyagx)
    print("Waiting for complete, stable feedback ...")
    q_now = gm.wait_for_feedback(robot, clock, sleep)[0]
    print(f"[{a.channel}] now [deg]: {np.round(np.degrees(q_now), 2).tolist()}")
    for i in idx:
        print(f"  pose {i} [deg]: {np.round(np.degrees(allq[i]), 1).tolist()}  {spec['poses'][i].get('note', '')[:40]}")
    print(f"plan: {a.mode}, {len(idx)} pose(s) at {a.speed_pct}%, settle {a.settle_s} s, hold {a.hold_s} s"
          + (f", steps {steps} g" if steps else f", label '{a.label}'"))
    print("check: weight hanging <= 9 cm below the gripper, path clear, e-stop in hand.")
    if ask("type YES to start: ").strip() != "YES":
        raise SystemExit("aborted, nothing sent")
    try:
        gm.wait_for_feedback(robot, clock, sleep)
        robot.set_speed_percent(a.speed_pct)
        k = 0
        for i in idx:
            if a.confirm_each and ask(f"move to pose {i}? y / anything else stops: ").strip().lower() != "y":
                print("stopped by operator")
                break
            gm.settle(robot, allq[i], clock, sleep)
            if a.mode == "poses":
                rec, meta = record(robot, a, clock, sleep)
                save(out, k, rec, {**meta, "pose": i, "pose_name": f"p{i}", "label": a.label}, summary)
                k += 1
                continue
            for s in steps:
                if ask(f"step {s} g on the arm: hang / remove now, hands off, swing stopped -> Enter "
                       "(q stops): ").strip().lower() == "q":
                    print("stopped by operator")
                    break
                rec, meta = record(robot, a, clock, sleep)
                save(out, k, rec, {**meta, "pose": i, "pose_name": f"p{i}", "step": s}, summary)
                k += 1
    except (KeyboardInterrupt, TimeoutError, gm.FeedbackNotReady) as exc:
        summary["abort"] = {"status": type(exc).__name__, "detail": str(exc),
                            "hold_q_rad": gm.hold_measured_position(robot)}
        (out / "summary.json").write_text(json.dumps(summary, indent=1))
        raise
    summary["finished"] = datetime.now().isoformat(timespec="seconds")
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(f"done, {len(summary['records'])} recordings in {out}; the arm holds the last pose")
    return out


if __name__ == "__main__":
    main()
