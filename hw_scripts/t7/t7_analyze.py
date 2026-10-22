#!/usr/bin/env python3
"""T7 analysis (server side). Reads one run folder written by guarded_move.py and the plan it used.

Per trial:
  - stop point: the deepest measured position (k_meas max) = where the plate held the tip; its tip height z
  - detection point: k_meas when contact was declared, and the time from reaching the stop point to the
    detection (the plate stops the motion first, the current threshold fires after)
  - current step at contact on j2/j3 (A, relative to the free-descent mean), and the static / free spread
  - command lead: how far the command was ahead of the measured position at detection (bounds the push)
Across trials: repeatability (sd) of the stop height and of the detection height; how many trials stopped on
contact vs hit max depth. Writes <run>/analysis.json and prints a table.

usage: venv/bin/python hw_scripts/t7/t7_analyze.py hw_scripts/t7/runs/<run> --plan hw_scripts/t7/plans/<plan>.json
"""
import argparse
import json
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run"); ap.add_argument("--plan", required=True)
    a = ap.parse_args()
    run = Path(a.run); plan = json.loads(Path(a.plan).read_text())
    Z = np.asarray(plan["tip_z_m"]); step = plan["step_mm"]
    summ = json.loads((run / "summary.json").read_text())
    rows = []
    for tr in summ["trials"]:
        d = np.load(run / f"trial_{tr['trial']:02d}.npz")
        t, kc, km, hit = d["t"], d["k_cmd"].astype(int), d["k_meas"].astype(int), d["hit"]
        kmax = int(km.max()); i_stop = int(np.argmax(km >= kmax - 1))       # first sample at (within a step of) the stop point
        r = d["resid"] - d["mu_free"]
        row = {"trial": tr["trial"], "status": tr["status"], "stop_depth_mm": kmax * step, "stop_tip_z_mm": 1000 * Z[kmax]}
        Fa = d["tip_force_mean_N"] if "tip_force_mean_N" in d.files else None
        if tr["k_det"] is not None and Fa is not None and (np.nan_to_num(Fa, nan=-1.0) > 2.0).any():
            ko = int(np.argmax(np.nan_to_num(Fa, nan=-1.0) > 2.0))
            while ko > 0 and not (Fa[ko] < 0.5):
                ko -= 1
            row["onset_depth_mm"] = int(km[ko]) * step
            row["onset_tip_z_mm"] = 1000 * Z[int(km[ko])]
        if tr["k_det"] is not None:
            i_det = int(np.argmax(hit > 0)) if (hit > 0).any() else int(np.argmax(km))
            row.update({"det_depth_mm": tr["k_det"] * step, "det_tip_z_mm": 1000 * Z[tr["k_det"]],
                        "stop_to_detect_ms": 1000 * (t[i_det] - t[i_stop]),
                        "cmd_lead_at_det_mm": (kc[i_det] - km[i_det]) * step,
                        "tip_force_at_det_N": tr.get("tip_force_mean_at_det_N", tr.get("tip_force_at_det_N")), "threshold_N": tr.get("threshold_N"),
                        "free_tip_force_sd_N": tr.get("free_tip_force_sd_N"),
                        "step_j2_A": float(r[i_det, 1]), "step_j3_A": float(r[i_det, 2]),
                        "sd_free_j2_A": float(d["sd_free"][1]), "sd_free_j3_A": float(d["sd_free"][2]),
                        "sd_static_j2_A": float(d["sd_static"][1]), "sd_static_j3_A": float(d["sd_static"][2])})
        rows.append(row)
    c = [r for r in rows if r["status"] in ("contact", "contact_late", "stall")]
    res = {"run": run.name, "plan": Path(a.plan).name, "speed_mm_s": summ["speed_mm_s"], "n": len(rows), "n_contact": len(c),
           "gap_measured_mm": plan["gap_mm"], "trials": rows}
    if c:
        sz = np.array([r["stop_tip_z_mm"] for r in c]); dz = np.array([r["det_tip_z_mm"] for r in c])
        res.update({"stop_z_sd_mm": float(sz.std(ddof=1)) if len(c) > 1 else None, "det_z_sd_mm": float(dz.std(ddof=1)) if len(c) > 1 else None,
                    "stop_depth_mean_mm": float(np.mean([r["stop_depth_mm"] for r in c])),
                    "stop_to_detect_ms_median": float(np.median([r["stop_to_detect_ms"] for r in c])),
                    "cmd_lead_mm_max": float(max(r["cmd_lead_at_det_mm"] for r in c))})
        on = [r["onset_depth_mm"] for r in c if "onset_depth_mm" in r]
        if len(on) > 1:
            res.update({"onset_depth_mean_mm": float(np.mean(on)), "onset_depth_sd_mm": float(np.std(on, ddof=1))})
    (run / "analysis.json").write_text(json.dumps(res, indent=1))
    print(f"{run.name}: {len(c)}/{len(rows)} stopped on contact; measured gap {plan['gap_mm']} mm")
    for r in rows:
        if r["status"] in ("contact", "contact_late", "stall"):
            print(f"  #{r['trial']:2d} stop {r['stop_depth_mm']:6.2f} mm (z {r['stop_tip_z_mm']:.2f}) | detect {r['det_depth_mm']:6.2f} mm, "
                  f"{r['stop_to_detect_ms']:5.0f} ms after the stop, cmd lead {r['cmd_lead_at_det_mm']:.2f} mm | step j2 {r['step_j2_A']:+.3f} j3 {r['step_j3_A']:+.3f} A "
                  f"(free sd {r['sd_free_j2_A']:.3f}/{r['sd_free_j3_A']:.3f}) | F {r['tip_force_at_det_N'] or float('nan'):.1f} N vs thr {r['threshold_N'] or float('nan'):.1f}")
        else:
            print(f"  #{r['trial']:2d} {r['status']} (deepest {r['stop_depth_mm']:.2f} mm)")
    if "onset_depth_sd_mm" in res:
        print(f"  contact onset (0.2 s mean force starts rising): mean {res['onset_depth_mean_mm']:.2f} mm below start, "
              f"sd {res['onset_depth_sd_mm']:.2f} mm over {len(c)} trials")
    if len(c) > 1:
        print(f"  repeatability: stop height sd {res['stop_z_sd_mm']:.2f} mm, detection height sd {res['det_z_sd_mm']:.2f} mm; "
              f"stop->detect median {res['stop_to_detect_ms_median']:.0f} ms; max command lead {res['cmd_lead_mm_max']:.2f} mm")


if __name__ == "__main__":
    main()
