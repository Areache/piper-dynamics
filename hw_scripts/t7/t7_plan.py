#!/usr/bin/env python3
"""T7 planner (server side, needs pinocchio + the Piper X gripper URDF). Sends nothing to any arm.

Turns one start pose into a guarded-descent plan for hw_scripts/t7/guarded_move.py:
  - a straight vertical line for the gripper tip, from the start pose down by --depth-mm, orientation held,
    solved as joint waypoints every --step-mm (damped least squares, warm-started);
  - the expected motor current along the line from the URDF gravity torque and the T4 kt / sign per joint,
    so the robot side can subtract the gravity change while it descends;
  - the safety limits the robot side enforces (max depth below the start, arming distance, speed cap).
Checks: IK residual, joint limits, joint step per waypoint, and that no waypoint lifts the tip.

usage (from the repo root on the server):
  venv/bin/python hw_scripts/t7/t7_plan.py --arm can_right --start-json <teach_poses json> --pose-index 0 \
      --gap-mm 80 --out hw_scripts/t7/plans/right_p0.json
  # or --start-q j1,j2,j3,j4,j5,j6 (rad) instead of --start-json/--pose-index
"""
import argparse
import hashlib
import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pinocchio as pin

HERE = Path(__file__).resolve().parent
# the Piper X URDF with the gripper is not distributed with the repo (see assets/urdf/README.md)
URDF = Path(os.environ.get("PIPER_GRIPPER_URDF", HERE.parent.parent / "piper_x_6dof_with_gripper.urdf"))
# T7 (2026-10-22): plate top at the nominal 0.15 m tip, from the contact onsets of the near-vertical poses p0-p4
PLATE_Z_MM = -9.1
# T4 static calibration (2026-10-20, gripper URDF): current [A] = dir * tau / kt + offset. j1 and j6 are not
# identifiable statically; their gravity torque is ~0 so they do not enter the expectation.
T4 = {
    "can_left": {"kt": [4.725, 3.7489, 2.6865, 0.982, 0.6598, 0.779], "dir": [-1, 1, 1, 1, 1, 1]},
    "can_right": {"kt": [4.725, 3.7786, 3.3293, 1.0499, 0.6691, 0.779], "dir": [-1, 1, 1, 1, 1, 1]},
}


def build_model(tip_m):
    full = pin.buildModelFromUrdf(str(URDF))
    lock = [full.getJointId(n) for n in full.names[1:] if n not in [f"joint{i}" for i in range(1, 7)]]
    m = pin.buildReducedModel(full, lock, pin.neutral(full)) if lock else full
    fid = m.getFrameId("gripper_base")
    m.addFrame(pin.Frame("tip", m.frames[fid].parentJoint, fid,
                         m.frames[fid].placement * pin.SE3(np.eye(3), np.array([0.0, 0.0, tip_m])), pin.FrameType.OP_FRAME))
    return m, m.getFrameId("tip")


def fk(m, d, fid, q):
    pin.framesForwardKinematics(m, d, q)
    return d.oMf[fid].copy()


def ik_step_line(m, d, fid, q, target, iters=50, lam=1e-3):
    """Solve tip pose = target (position + orientation) from warm start q."""
    for _ in range(iters):
        cur = fk(m, d, fid, q)
        err = pin.log6(cur.actInv(target)).vector          # local-frame twist error
        if np.linalg.norm(err[:3]) < 1e-5 and np.linalg.norm(err[3:]) < 1e-4:
            break
        J = pin.computeFrameJacobian(m, d, q, fid, pin.LOCAL)
        dq = J.T @ np.linalg.solve(J @ J.T + lam * np.eye(6), err)
        q = q + dq
    cur = fk(m, d, fid, q)
    err = pin.log6(cur.actInv(target)).vector
    return q, float(np.linalg.norm(err[:3])), float(np.linalg.norm(err[3:]))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", required=True, choices=["can_left", "can_right"])
    ap.add_argument("--start-json"); ap.add_argument("--pose-index", type=int, default=0)
    ap.add_argument("--start-q", help="6 joint angles in rad, comma-separated")
    ap.add_argument("--gap-mm", type=float, help="tip-to-surface gap at the start pose (or use --object-height-mm)")
    ap.add_argument("--object-height-mm", type=float,
                    help="height of the touched surface above the T7 plate (caliper); the gap then comes from FK: "
                         "tip z at the start - (plate z from the T7 touches + this height)")
    ap.add_argument("--plate-z-mm", type=float, default=PLATE_Z_MM, help="T7 plate top, nominal-tip FK frame (mm)")
    ap.add_argument("--overtravel-mm", type=float, default=5.0, help="how far below the expected plane the robot may go")
    ap.add_argument("--step-mm", type=float, default=0.25)
    ap.add_argument("--tip-m", type=float, default=0.15, help="nominal tip offset along gripper_base z (T7 solves the real one)")
    ap.add_argument("--speed-mm-s", type=float, default=10.0)
    ap.add_argument("--arm-after-mm", type=float, default=20.0, help="detection armed only after this much free descent")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if (a.gap_mm is None) == (a.object_height_mm is None):
        raise SystemExit("give exactly one of --gap-mm, --object-height-mm")
    if a.object_height_mm is not None:
        if abs(a.tip_m - 0.15) > 1e-9:
            raise SystemExit("--object-height-mm uses the plate height measured with the nominal 0.15 m tip")
        q_s = (np.array([float(x) for x in a.start_q.split(",")]) if a.start_q else
               np.array(json.loads(Path(a.start_json).read_text())["poses"][a.pose_index]["q_rad"], dtype=float))
        m0, f0 = build_model(a.tip_m); z_tip = fk(m0, m0.createData(), f0, q_s).translation[2] * 1000.0
        a.gap_mm = round(z_tip - (a.plate_z_mm + a.object_height_mm), 1)
        print(f"gap from FK: tip z {z_tip:.1f} mm - (plate {a.plate_z_mm:.1f} + object {a.object_height_mm:.1f}) = {a.gap_mm:.1f} mm")
    if a.gap_mm < a.arm_after_mm + 15:
        raise SystemExit(f"gap {a.gap_mm} mm too small: need >= arm_after ({a.arm_after_mm}) + 15 mm of armed free descent")
    if a.speed_mm_s > 20:
        raise SystemExit("speed capped at 20 mm/s")

    if a.start_q:
        q0 = np.array([float(x) for x in a.start_q.split(",")])
        src = "cli"
    else:
        spec = json.loads(Path(a.start_json).read_text())
        if spec.get("can_interface") not in (None, a.arm):
            raise SystemExit(f"pose file is for {spec['can_interface']}, not {a.arm}")
        q0 = np.array(spec["poses"][a.pose_index]["q_rad"], dtype=float)
        src = f"{Path(a.start_json).name}#{a.pose_index}"
    assert q0.shape == (6,)

    m, fid = build_model(a.tip_m)
    d = m.createData()
    T0 = fk(m, d, fid, q0)
    zaxis = T0.rotation[:, 2]
    tilt_deg = float(np.degrees(np.arccos(np.clip(-zaxis[2], -1, 1))))    # angle between tool axis and straight down
    depth = (a.gap_mm + a.overtravel_mm) / 1000.0
    n = int(np.ceil(depth / (a.step_mm / 1000.0))) + 1
    lo, hi = m.lowerPositionLimit, m.upperPositionLimit
    Q, Z, worst = [q0.copy()], [float(T0.translation[2])], (0.0, 0.0)
    q = q0.copy()
    for k in range(1, n):
        tgt = pin.SE3(T0.rotation, T0.translation - np.array([0, 0, k * a.step_mm / 1000.0]))
        q_new, ep, eo = ik_step_line(m, d, fid, q, tgt)
        if ep > 2e-4 or eo > np.radians(0.2):
            raise SystemExit(f"IK failed at waypoint {k} ({k * a.step_mm:.1f} mm): pos err {ep*1000:.3f} mm, rot err {np.degrees(eo):.3f} deg")
        if np.any(q_new < lo + np.radians(5)) or np.any(q_new > hi - np.radians(5)):
            raise SystemExit(f"waypoint {k} within 5 deg of a joint limit: {np.round(np.degrees(q_new),1).tolist()}")
        if np.max(np.abs(q_new - q)) > 0.01:
            raise SystemExit(f"joint step {np.max(np.abs(q_new-q)):.4f} rad at waypoint {k}: near a singularity, pick another start")
        worst = (max(worst[0], ep), max(worst[1], eo))
        q = q_new; Q.append(q.copy()); Z.append(float(fk(m, d, fid, q).translation[2]))
    Q = np.array(Q); Z = np.array(Z)
    if np.any(np.diff(Z) > 1e-6):
        raise SystemExit("tip height is not monotonically decreasing")

    kt, sg = np.array(T4[a.arm]["kt"]), np.array(T4[a.arm]["dir"])
    tau_g = np.array([pin.computeGeneralizedGravity(m, d, qq) for qq in Q])
    i_exp = sg * tau_g / kt
    i_exp[:, [0, 5]] = 0.0                                                    # not identifiable / no gravity
    # contact signature: an upward force F on the tip lowers the motor torque by J_z^T F, so the current changes
    # by di = -dir * J_z / kt per newton. The robot side projects its residual onto this to get a tip force.
    di_up = []
    for qq in Q:
        pin.computeJointJacobians(m, d, qq); pin.updateFramePlacements(m, d)
        di_up.append(-sg * pin.getFrameJacobian(m, d, fid, pin.LOCAL_WORLD_ALIGNED)[2] / kt)
    di_up = np.array(di_up); di_up[:, [0, 5]] = 0.0
    plan = {
        "kind": "t7_guarded_descent", "version": 2, "arm": a.arm, "created": datetime.now().isoformat(timespec="seconds"),
        "urdf_sha256": hashlib.sha256(URDF.read_bytes()).hexdigest(), "start_source": src,
        "tip_offset_m": a.tip_m, "tool_tilt_from_down_deg": round(tilt_deg, 2),
        "gap_mm": a.gap_mm, "object_height_mm": a.object_height_mm, "max_depth_mm": a.gap_mm + a.overtravel_mm, "step_mm": a.step_mm,
        "speed_mm_s": a.speed_mm_s, "arm_after_mm": a.arm_after_mm,
        # v2 (2026-10-21, after the first hardware descent): single-joint thresholds false-triggered on j2
        # stick-slip; detect a tip force instead (projection over j2..j5, 30 ms low-pass, 60 ms hold).
        "detect": {"mode": "tip_force", "joints": [1, 2, 3, 4], "lp_n": 3, "thr_sigma": 4.0, "floor_N": 3.0,
                   "hold_s": 0.06, "min_explained": 0.5},
        "retract_mm": 5.0, "start_tolerance_rad": 0.02,
        "q": np.round(Q, 6).tolist(), "tip_z_m": np.round(Z, 6).tolist(), "i_expected_A": np.round(i_exp, 5).tolist(),
        "di_per_N_up": np.round(di_up, 6).tolist(),
        "ik_worst": {"pos_mm": round(worst[0] * 1000, 4), "rot_deg": round(float(np.degrees(worst[1])), 4)},
    }
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(plan))
    di = i_exp[-1] - i_exp[0]
    print(f"plan -> {out}: {len(Q)} waypoints, {Z[0]-Z[-1]:.4f} m descent, tool tilt {tilt_deg:.1f} deg from straight down")
    print(f"  joint change over the line [deg]: {np.round(np.degrees(Q[-1]-Q[0]),2).tolist()}")
    print(f"  expected current change over the line [A] (gravity): {np.round(di,3).tolist()}")
    print(f"  current change per N of upward tip force at the start [A/N]: {np.round(di_up[0],4).tolist()}")
    print(f"  IK worst: {plan['ik_worst']}; at {a.speed_mm_s} mm/s the descent takes {(Z[0]-Z[-1])*1000/a.speed_mm_s:.1f} s")
    if tilt_deg > 10:
        print(f"  WARNING: tool axis is {tilt_deg:.1f} deg from vertical; the tip will not be the lowest point")


if __name__ == "__main__":
    main()
