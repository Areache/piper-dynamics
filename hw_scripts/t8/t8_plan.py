#!/usr/bin/env python3
"""T8 planner (server side, needs pinocchio + the Piper X gripper URDF). Sends nothing to any arm.

Builds every target of the T8 session for t8_run.py from the reviewed T5 pose file and a few site parameters:
  excite  start pose = the T5 pose whose single-joint excursions (+-0.19 rad, j1..j5) keep the most clearance
  sweeps  per joint j2..j5, two base configurations taken from T5 poses (the joint is swept, the others stay at
          that pose), chosen for coverage of the target range under the clearance rule and for the largest
          difference in gravity torque on the swept joint; ranges clipped to the clear part
  static  the 10 T5 poses, plus a pose 15 mm above and 15 mm below each (tool orientation held) for the approach
  lift    container on the table at --site-xy: pre-grasp (fingertips 3 cm above the container top) -> grasp line,
          then a 5 cm vertical lift line, tool pointing down
  reach   container held, 5 cm horizontal line away from the base at --reach-height-m above the table
  hold / comply  3 / 2 T5 poses spanning the payload torque on j2 + j3
Clearance rule (base frame, table top at --table-z-m): the container bottom and top, the fingertip, and every
joint origin from j2 on stay >= the margins below, on every target, along every streamed line, and along the
straight joint-space path between consecutive targets (move_j interpolates in joint space). Joint limits: >= 5 deg.

usage (server, repo root): venv/bin/python hw_scripts/t8/t8_plan.py --out hw_scripts/t8/plans/t8_right.json
"""
import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pinocchio as pin

HERE = Path(__file__).resolve().parent
import sys  # noqa: E402
sys.path.insert(0, str(HERE.parent / "t7"))
from t7_plan import PLATE_Z_MM, T4, URDF, build_model, fk, ik_step_line  # noqa: E402

# Joint max speed at 100 %, from the firmware (read-only SearchMotorMaxAngleSpdAccLimit query, right arm, 2026-11-12):
# 3.00 rad/s = 171.9 deg/s on every joint, max acceleration 5.00 rad/s^2. Was the spec sheet (180/195/180/225/225/225).
# j5 answered twice with 3.00 and 2.00 rad/s while the master arm shared the bus: re-queried at the start of T8 with the
# master unplugged; if 2.00 holds, j5's triangle percent becomes 15 (see TASKS T8). The percent -> speed mapping itself is
# measured by the `speedcal` segment.
VMAX_DEG_S = [171.9] * 6
AMAX_RAD_S2 = 5.0
# Joint limits for planning stay the URDF ones (tighter than the firmware's where they differ: j6 +-120 vs +-170 deg).
TARGET_DEG = {2: (60.0, 140.0), 3: (-135.0, -50.0), 4: (-90.0, 90.0), 5: (-80.0, 15.0)}
MUST_COVER = {2: (89.3, 96.3), 3: (-72.0, -56.6)}             # the E3 ripple windows (RESULTS_ripple_v1)
SWEEP_PCT = {2: 1, 3: 1, 4: 2, 5: 2}


class Geo:
    def __init__(self, tip_m, table_z, cont_h, grip_h):
        self.m, self.fid = build_model(tip_m)
        self.d = self.m.createData()
        self.tz, self.H, self.gh = table_z, cont_h, grip_h
        self.jids = [self.m.getJointId(f"joint{i}") for i in range(3, 7)]     # j2's origin is the fixed shoulder
        self.box = None                                                          # (lo xyz, hi xyz) workspace box

    def set_box(self, Qs, margin):
        """Two boxes: one for the tool points (tip, container) from the reached fingertips, one for the joint origins
        from the reached joint origins; low z is left to the table rule."""
        self.box = {}
        for kind in ("tool", "link"):
            pts = np.array([p for q in Qs for n, p in self.points(q, False).items() if (n.startswith("j")) == (kind == "link")])
            lo, hi = pts.min(0) - margin, pts.max(0) + margin
            lo[2] = -np.inf
            self.box[kind] = (lo, hi)
        return self.box

    def points(self, q, container):
        """Lowest-relevant points: fingertip, container bottom / top (along the tool axis), joint origins j2..j6."""
        T = fk(self.m, self.d, self.fid, q)
        z = T.rotation[:, 2]
        pts = {"tip": T.translation.copy()}
        if container:
            pts["cont_bottom"] = T.translation + z * self.gh
            pts["cont_top"] = T.translation - z * (self.H - self.gh)
        for k, jid in zip(range(3, 7), self.jids):
            pts[f"j{k}"] = self.d.oMi[jid].translation.copy()
        return pts

    def clearance(self, q, container, m_tool, m_link):
        """Smallest margin (m) above the required height; negative = violation. Also returns the binding point."""
        best = (np.inf, "")
        for name, p in self.points(q, container).items():
            need = m_link if name.startswith("j") else m_tool
            c = p[2] - self.tz - need
            if c < best[0]:
                best = (float(c), name)
            if self.box is not None:
                lo, hi = self.box["link" if name.startswith("j") else "tool"]
                cb = float(np.min(np.concatenate([p - lo, hi - p])))
                if cb < best[0]:
                    best = (cb, name + " box")
        return best

    def limit_margin(self, q):
        lo, hi = self.m.lowerPositionLimit, self.m.upperPositionLimit
        return float(np.min(np.minimum(q - lo, hi - q)))

    def gravity(self, q, payload_kg=0.0):
        g = pin.computeGeneralizedGravity(self.m, self.d, q).copy()
        if payload_kg:
            pin.computeJointJacobians(self.m, self.d, q); pin.updateFramePlacements(self.m, self.d)
            J = pin.getFrameJacobian(self.m, self.d, self.fid, pin.LOCAL_WORLD_ALIGNED)[:3]
            g += J.T @ np.array([0, 0, 9.81 * payload_kg])
        return g

    def line(self, q0, delta, step_mm):
        """Straight tip line from q0 by delta (m, base frame), orientation held; joint waypoints."""
        T0 = fk(self.m, self.d, self.fid, q0)
        L = float(np.linalg.norm(delta)) * 1000.0
        n = int(np.ceil(L / step_mm)) + 1
        Q, q = [np.array(q0, float)], np.array(q0, float)
        for k in range(1, n):
            tgt = pin.SE3(T0.rotation, T0.translation + delta * (k / (n - 1)))
            q, ep, eo = ik_step_line(self.m, self.d, self.fid, q, tgt)
            if ep > 2e-4 or eo > np.radians(0.2):
                raise SystemExit(f"IK failed on a line at waypoint {k}: {ep*1000:.3f} mm / {np.degrees(eo):.3f} deg")
            if np.max(np.abs(q - Q[-1])) > 0.01:
                raise SystemExit("joint step > 0.01 rad on a line: near a singularity")
            Q.append(q.copy())
        return np.array(Q)


def check_path(geo, qa, qb, container, mt, ml, what, n=60):
    # 5 deg from every joint limit, except where an end point is a pose the arm already reached on hardware closer
    # than that (T5 pose 3 has j3 at -2.4 deg, limit 0): then the path may not come closer than that end point
    need = min(np.radians(5), geo.limit_margin(qa), geo.limit_margin(qb)) - 1e-9
    worst = (np.inf, "")
    for s in np.linspace(0, 1, n):
        q = qa + s * (qb - qa)
        c = geo.clearance(q, container, mt, ml)
        if c[0] < worst[0]:
            worst = c
        if geo.limit_margin(q) < need:
            raise SystemExit(f"{what}: closer to a joint limit than {np.degrees(need):.1f} deg")
    if worst[0] < 0:
        raise SystemExit(f"{what}: clearance violated by {-worst[0]*1000:.0f} mm at {worst[1]}")
    return worst


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", default="can_right")
    ap.add_argument("--poses", default=str(HERE.parent.parent / ".hw/results/T5/poses_can_right.json"))
    ap.add_argument("--tip-m", type=float, default=0.15)
    ap.add_argument("--table-z-m", type=float, default=PLATE_Z_MM / 1000.0, help="table top, base frame (T7 plate)")
    ap.add_argument("--container-h-m", type=float, default=0.10, help="container height")
    ap.add_argument("--grip-h-m", type=float, default=0.06, help="height above the container bottom where the fingertips close")
    ap.add_argument("--seed-deg", default="10.8,76.0,-58.2,73.0,0.4,21.7",
                    help="near-vertical pose the arm has reached on hardware (T7 p2 start, tool 4.2 deg from down); its "
                         "orientation is kept for lift and reach, its x, y is the default container site")
    ap.add_argument("--site-xy", help="container position on the table (base frame, m); default: the seed's x, y")
    ap.add_argument("--reach-height-m", type=float, default=0.13, help="fingertip height above the table for reach")
    ap.add_argument("--margin-tool-m", type=float, default=0.05, help="min height of tip / container above the table in free motion")
    ap.add_argument("--sweep-margin-tool-m", type=float, default=0.08, help="tool margin for the long slow sweeps")
    ap.add_argument("--margin-link-m", type=float, default=0.04, help="min height of joint origins j2..j6 above the table")
    ap.add_argument("--t4-dir", help="T4 static-pose npz directory (can_right_pose_*.npz): its fingertip and joint "
                                     "positions span the workspace box; default: the T5 poses only")
    ap.add_argument("--box-margin-m", type=float, default=0.08, help="workspace box = reached positions + this margin")
    ap.add_argument("--step-mm", type=float, default=0.25)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    geo = Geo(a.tip_m, a.table_z_m, a.container_h_m, a.grip_h_m)
    mt, ml = a.margin_tool_m, a.margin_link_m
    spec = json.loads(Path(a.poses).read_text())
    if spec["can_interface"] != a.arm:
        raise SystemExit("pose file is for another arm")
    P = [np.array(p["q_rad"], float) for p in spec["poses"]]
    report = {"clearance_mm": {}}
    Qbox = list(P)
    if a.t4_dir:
        import glob
        for fn in sorted(glob.glob(str(Path(a.t4_dir) / f"{a.arm}_pose_*.npz"))):
            qq = np.asarray(np.load(fn)["q"], float)
            Qbox.append(qq.mean(0) if qq.ndim == 2 else qq)
    boxes = geo.set_box(Qbox, a.box_margin_m)
    report["workspace_box_m"] = {"from_poses": len(Qbox), **{k: [np.round(v[0], 3).tolist(), np.round(v[1], 3).tolist()]
                                                              for k, v in boxes.items()}}
    print(f"workspace boxes from {len(Qbox)} reached poses + {a.box_margin_m} m: {report['workspace_box_m']}")

    # ---------------- excite start: max-min clearance over the single-joint excursions (empty, gripper open)
    rows = []
    for i, q in enumerate(P):
        worst, lim = np.inf, np.inf
        for j in range(5):
            for d in np.linspace(-0.19, 0.19, 21):
                qq = q.copy(); qq[j] += d
                worst = min(worst, geo.clearance(qq, False, mt, ml)[0]); lim = min(lim, geo.limit_margin(qq))
        rows.append((i, worst, lim))
    ok = [r for r in rows if r[2] >= np.radians(5) and r[1] >= 0]
    if not ok:
        raise SystemExit("no T5 pose keeps clearance and limits for the excitation")
    ex_i = max(ok, key=lambda r: r[1])[0]
    q_ex = P[ex_i]
    report["excite_candidates"] = [{"pose": i, "min_clear_mm": round(c * 1000, 1), "limit_margin_deg": round(float(np.degrees(m)), 1)}
                                   for i, c, m in rows]
    tri_pct = [int(round(np.degrees(0.3) / (VMAX_DEG_S[j] / 100.0))) for j in range(6)]
    speedcal = {"q0": q_ex.tolist(), "joints": [1, 2, 3, 4, 5], "pcts": [1, 2, 5, 10, 20], "amp_deg": 10.0, "settle_s": 1.0,
                "steps": {"joints": [2, 3], "sizes_deg": [0.5, 2.0], "reps": 3, "pct": 10, "hold_s": 2.0},
                "note": "single-joint move_j from the excite start pose (its +-0.19 rad excursions are clearance-checked)"}
    excite = {"start_index": ex_i, "q0": q_ex.tolist(), "joints": [1, 2, 3, 4, 5],
              "sines": [{"f_hz": f, "amp_rad": A, "dur_s": 10.0} for f, A in ((0.5, 0.19), (1.0, 0.10), (2.0, 0.05), (3.0, 0.03))],
              "tri": {"v_rad_s": 0.3, "amp_rad": 0.15, "reversals": 20, "pct": tri_pct, "vmax_rad_s": 0.45}}

    # ---------------- sweeps
    sweeps, cover = [], {}
    for j, (lo_t, hi_t) in TARGET_DEG.items():
        k = j - 1
        grid = np.arange(lo_t, hi_t + 1e-9, 0.5)
        cands = []
        for i, base in enumerate(P):
            ok_mask = []
            for th in grid:
                qq = base.copy(); qq[k] = np.radians(th)
                ok_mask.append(geo.clearance(qq, False, a.sweep_margin_tool_m, ml)[0] >= 0 and geo.limit_margin(qq) >= np.radians(5))
            ok_mask = np.array(ok_mask)
            # largest contiguous clear interval
            best, cur = (0, 0, -1), None
            for n_, okk in enumerate(ok_mask):
                if okk and cur is None:
                    cur = n_
                if (not okk or n_ == len(ok_mask) - 1) and cur is not None:
                    end = n_ if okk else n_ - 1
                    if end - cur > best[2] - best[1]:
                        best = (0, cur, end)
                    cur = None
            if best[2] < best[1] + 4:
                continue
            lo, hi = grid[best[1]], grid[best[2]]
            # the base joint value itself must lie inside so the arm can get there by sweeping from the base pose
            qq = base.copy(); qq[k] = np.radians((lo + hi) / 2)
            tau_mid = geo.gravity(qq)[k]
            cands.append({"base": i, "lo": float(lo), "hi": float(hi), "span": float(hi - lo), "tau_mid": float(tau_mid)})
        if len(cands) < 2:
            raise SystemExit(f"j{j}: fewer than two T5 base poses give a clear sweep")
        cands.sort(key=lambda c: -c["span"])
        A = cands[0]
        B = max(cands[1:], key=lambda c: (min(c["span"], A["span"]) >= 0.6 * A["span"], abs(c["tau_mid"] - A["tau_mid"])))
        for name, c in (("A", A), ("B", B)):
            base = P[c["base"]]
            qlo = base.copy(); qlo[k] = np.radians(c["lo"])
            qhi = base.copy(); qhi[k] = np.radians(c["hi"])
            check_path(geo, qlo, qhi, False, mt, ml, f"sweep j{j}{name}")
            pct = SWEEP_PCT[j]
            sweeps.append({"joint": j, "config": f"{name}(T5 pose {c['base']})", "base_index": c["base"],
                           "range_deg": [c["lo"], c["hi"]], "q_lo": qlo.tolist(), "q_hi": qhi.tolist(), "pct": pct,
                           "deg_s_nominal": VMAX_DEG_S[k] * pct / 100.0, "deg_s_max": 6.0, "passes": 2,
                           "tau_grav_mid_Nm": round(c["tau_mid"], 3),
                           "tip_xyz_lo_hi_mm": [np.round(fk(geo.m, geo.d, geo.fid, qq).translation * 1000).tolist() for qq in (qlo, qhi)]})
        cov = []
        for lo_m, hi_m in [MUST_COVER.get(j, (None, None))]:
            if lo_m is not None:
                cov = [n for n, c in (("A", A), ("B", B)) if c["lo"] <= lo_m and c["hi"] >= hi_m]
                if not cov:
                    print(f"WARNING j{j}: no config covers the E3 window {lo_m}..{hi_m} deg")
        cover[f"j{j}"] = {"target": [lo_t, hi_t], "A": [A["lo"], A["hi"]], "B": [B["lo"], B["hi"]], "covers_E3_window": cov}
    report["sweep_coverage"] = cover

    # ---------------- static: T5 poses with +-15 mm approach points (container held)
    def with_approach(i):
        """+-15 mm vertical approach points; if one would cross a joint limit (or clearance), try 10 / 5 mm, else
        that approach is dropped for this pose (None) and reported."""
        q = P[i]
        c0 = geo.clearance(q, True, mt, ml)
        if c0[0] < 0:
            raise SystemExit(f"T5 pose {i}: container clearance violated by {-c0[0]*1000:.0f} mm at {c0[1]}")
        out = {"index": i, "q": q.tolist()}
        for name, sgn in (("above", 1), ("below", -1)):
            out[f"q_{name}"] = None
            for off in (0.015, 0.010, 0.005):
                try:
                    qq = geo.line(q, np.array([0, 0, sgn * off]), a.step_mm)[-1]
                except SystemExit:
                    continue
                if (geo.clearance(qq, True, mt, ml)[0] >= 0 and
                        geo.limit_margin(qq) >= min(np.radians(1.0), geo.limit_margin(q) - np.radians(0.5))):
                    out[f"q_{name}"] = qq.tolist(); out[f"{name}_mm"] = off * 1000
                    out[f"dq_{name}_deg"] = np.round(np.degrees(qq - q), 3).tolist()
                    break
            if out[f"q_{name}"] is None:
                print(f"NOTE T5 pose {i}: no feasible {name} approach point; that visit is skipped")
        return out

    static = {"poses": [with_approach(i) for i in range(len(P))], "settle_s": 3.0, "hold_s": 2.0, "offset_mm": 15}
    seq = []
    for p in static["poses"]:
        seq += [x for x in (p["q_above"], p["q"], p["q_below"], p["q"]) if x is not None]
    worst = (np.inf, "")
    for qa, qb in zip(seq[:-1], seq[1:]):
        w = check_path(geo, np.array(qa), np.array(qb), True, mt, ml, "static transition")
        worst = min(worst, w)
    report["clearance_mm"]["static_transitions"] = [round(worst[0] * 1000, 1), worst[1]]

    # ---------------- lift site (orientation of the hardware-reached seed pose; vertical lines in the base frame)
    q_seed = np.radians([float(v) for v in a.seed_deg.split(",")])
    T_seed = fk(geo.m, geo.d, geo.fid, q_seed)
    R = T_seed.rotation
    x, y = (float(v) for v in a.site_xy.split(",")) if a.site_xy else (float(T_seed.translation[0]), float(T_seed.translation[1]))
    seed = "seed_deg"
    pre_z = a.table_z_m + a.container_h_m + 0.03
    grasp_z = a.table_z_m + a.grip_h_m
    q_pre, ep, eo = q_seed.copy(), 1, 1
    for _ in range(5):
        q_pre, ep, eo = ik_step_line(geo.m, geo.d, geo.fid, q_pre, pin.SE3(R, np.array([x, y, pre_z])), iters=200)
    if ep > 2e-4 or eo > np.radians(0.2):
        raise SystemExit(f"IK for the pre-grasp pose failed ({ep*1000:.2f} mm)")
    Q_down = geo.line(q_pre, np.array([0, 0, grasp_z - pre_z]), a.step_mm)
    Q_up = geo.line(Q_down[-1], np.array([0, 0, 0.05]), a.step_mm)
    for Q, w in ((Q_down, "lift approach"), (Q_up, "lift")):
        for q in Q:
            c = geo.clearance(q, False, 0.0, ml)          # the tip goes down to the container on purpose: links only
            if c[0] < 0:
                raise SystemExit(f"{w}: link clearance violated at {c[1]}")
            if geo.limit_margin(q) < np.radians(5):
                raise SystemExit(f"{w}: near a joint limit")
    lift = {"site_xy": [x, y], "seed_pose": seed, "q_pre": q_pre.tolist(), "Q_down": np.round(Q_down, 6).tolist(),
            "Q_up": np.round(Q_up, 6).tolist(), "step_mm": a.step_mm, "approach_mm_s": 20.0, "lift_mm": 50.0,
            "lift_mm_s": 10.0, "put_down_mm": 48.0, "still_s": 2.0}

    # ---------------- reach
    u = np.array([x, y, 0]) / np.hypot(x, y)
    q_r0 = q_pre.copy()
    q_r0, ep, eo = ik_step_line(geo.m, geo.d, geo.fid, q_r0, pin.SE3(R, np.array([x, y, a.table_z_m + a.reach_height_m])), iters=200)
    Q_out = geo.line(q_r0, 0.05 * u, a.step_mm)
    for q in Q_out:
        c = geo.clearance(q, True, mt, ml)
        if c[0] < 0 or geo.limit_margin(q) < np.radians(5):
            raise SystemExit(f"reach line: clearance {c[0]*1000:.0f} mm at {c[1]}, limit margin "
                             f"{np.degrees(geo.limit_margin(q)):.1f} deg, q {np.round(np.degrees(q), 1).tolist()}")
    lever = [float(fk(geo.m, geo.d, geo.fid, q).translation @ u) for q in (Q_out[0], Q_out[-1])]
    reach = {"Q_out": np.round(Q_out, 6).tolist(), "step_mm": a.step_mm, "mm_s": 10.0, "cycles": 3, "pause_s": 1.0,
             "radial_m": lever}

    # ---------------- hold / comply poses: span the payload torque on j2 + j3
    load = [abs(geo.gravity(q, 1.0)[1] - geo.gravity(q)[1]) + abs(geo.gravity(q, 1.0)[2] - geo.gravity(q)[2]) for q in P]
    order = list(np.argsort(load))
    full = [i for i in order if static["poses"][i]["q_above"] is not None and static["poses"][i]["q_below"] is not None]
    hold_idx = [int(full[-1]), int(full[len(full) // 2]), int(full[0])]
    hold = {"poses": [p for p in static["poses"] if p["index"] in hold_idx], "approaches": ["above", "below", "above"],
            "settle_s": 3.0, "hold_s": 10.0, "payload_Nm_per_kg_j2j3": {int(i): round(load[i], 2) for i in hold_idx}}
    comply = {"poses": [p for p in static["poses"] if p["index"] in (hold_idx[0], hold_idx[2])], "settle_s": 3.0, "hold_s": 3.0}

    anchor = {"q": q_ex.tolist(), "note": "excite start pose, empty gripper"}
    # ---------------- transitions between segment entry points (joint-space move_j at 10 %)
    # sweeps: enter each via its T5 base pose (then along the swept joint only, inside the clear range); if the
    # straight joint path from where the arm is to that base pose is not clear, go via the excite pose first
    trans = []
    prev = q_ex
    for s_ in sweeps:
        base = P[s_["base_index"]]
        via = []
        try:
            check_path(geo, prev, base, False, mt, ml, "probe")
        except SystemExit:
            check_path(geo, prev, q_ex, False, mt, ml, f"sweep j{s_['joint']} via excite pose (1)")
            check_path(geo, q_ex, base, False, mt, ml, f"sweep j{s_['joint']} via excite pose (2)")
            via.append(q_ex.tolist())
        via.append(base.tolist())
        s_["via"] = via
        w = check_path(geo, base, np.array(s_["q_lo"]), False, mt, ml, f"sweep j{s_['joint']} base -> start")
        trans.append([f"sweep {s_['joint']}{s_['config'][0]} ({len(via)} via)", round(w[0] * 1000, 1)])
        prev = np.array(s_["q_lo"])                     # every sweep ends at its low end (last pass goes down)
    for name, q in (("anchor", q_ex), ("lift_pre", q_pre)):
        trans.append([name, round(check_path(geo, prev, q, False, mt, ml, f"transition to {name}")[0] * 1000, 1)])
        prev = q
    q_top = Q_up[-1]
    A_ = lambda v: np.array(v, float)  # noqa: E731
    pairs = [("lift top -> static first", q_top, A_(static["poses"][0]["q_above"])),
             ("static last -> reach start", A_(static["poses"][-1]["q"]), Q_out[0]),
             ("reach start -> lift top", Q_out[0], q_top),
             ("lift top -> hold first", q_top, A_(hold["poses"][0]["q_above"]))]
    hseq = []
    for p in hold["poses"]:
        hseq += [A_(p["q_above"]), A_(p["q"]), A_(p["q_below"]), A_(p["q"]), A_(p["q_above"]), A_(p["q"])]
    pairs += [(f"hold sequence {n}", u_, v_) for n, (u_, v_) in enumerate(zip(hseq[:-1], hseq[1:]))]
    pairs += [("hold last -> comply first", hseq[-1], A_(comply["poses"][0]["q_above"]))]
    cseq = []
    for p in comply["poses"]:
        cseq += [A_(p["q_above"]), A_(p["q"])]
    pairs += [(f"comply sequence {n}", u_, v_) for n, (u_, v_) in enumerate(zip(cseq[:-1], cseq[1:]))]
    pairs += [("comply last -> lift top", cseq[-1], q_top)]
    for name, u_, v_ in pairs:
        trans.append([name + " (container)", round(check_path(geo, u_, v_, True, mt, ml, name)[0] * 1000, 1)])
    report["transitions_min_clear_mm"] = trans

    plan = {"kind": "t8_plan", "version": 1, "arm": a.arm, "created": datetime.now().isoformat(timespec="seconds"),
            "urdf_sha256": hashlib.sha256(Path(URDF).read_bytes()).hexdigest(), "poses_file": Path(a.poses).name,
            "tip_m": a.tip_m, "table_z_m": a.table_z_m, "container": {"h_m": a.container_h_m, "grip_h_m": a.grip_h_m},
            "margins_m": {"tool": mt, "link": ml}, "kt_T4": T4[a.arm]["kt"], "dir_T4": T4[a.arm]["dir"],
            "anchor": anchor, "speedcal": speedcal, "excite": excite, "sweeps": sweeps, "static": static, "lift": lift, "reach": reach,
            "hold": hold, "comply": comply,
            "named": {"excite_q0": q_ex.tolist(), "lift_pre": q_pre.tolist(), "lift_top": Q_up[-1].tolist(),
                      "reach_start": Q_out[0].tolist()},
            "report": report}
    body = json.dumps(plan, sort_keys=True)
    plan["sha"] = hashlib.sha256(body.encode()).hexdigest()[:12]
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(plan))
    print(f"plan {plan['sha']} -> {out}")
    print(f"excite start: T5 pose {ex_i}; triangle pct {tri_pct[:5]}")
    for c in report["excite_candidates"]:
        print("   ", c)
    for s in sweeps:
        dur = abs(s["range_deg"][1] - s["range_deg"][0]) / s["deg_s_nominal"]
        print(f"sweep j{s['joint']} {s['config']}: {s['range_deg'][0]:.1f}..{s['range_deg'][1]:.1f} deg, {s['pct']} % "
              f"~{s['deg_s_nominal']:.2f} deg/s, {dur:.0f} s per pass-direction, tau_g mid {s['tau_grav_mid_Nm']} N m, "
              f"tip mm at ends {s['tip_xyz_lo_hi_mm']}")
    print("coverage:", json.dumps(cover))
    print(f"lift site {x:.3f},{y:.3f}: seed pose {seed}, pre-grasp z {pre_z*1000:.0f} mm, grasp z {grasp_z*1000:.0f} mm, "
          f"{len(Q_down)} + {len(Q_up)} waypoints")
    print(f"reach: radial {lever[0]:.3f} -> {lever[1]:.3f} m at tip height {a.reach_height_m*1000:.0f} mm above the table")
    print(f"hold poses {hold_idx} (payload N m/kg j2+j3 {hold['payload_Nm_per_kg_j2j3']}); comply {[p['index'] for p in comply['poses']]}")
    print("static transitions min clearance", report["clearance_mm"]["static_transitions"])
    print("transitions min clearance (mm)", trans)


if __name__ == "__main__":
    main()
