#!/usr/bin/env python3
"""T8 session on the right arm (robot side). THIS MOVES THE ARM.

One script, one segment per call. Every segment replays a plan made on the server by t8_plan.py (the robot PC has
no kinematics library) and records 100 Hz rows of joint angle, motor position, velocity, current and gripper
width / force, each with time.monotonic() and time.time(). Temperatures come from hw_scripts/lowspd_log.py and
every CAN frame from `candump -L` (both listen-only, started by the operator before the first segment).

Segments (task board T8, in session order):
  anchor   empty gripper, plan anchor pose, settle, record 3 s (drift anchor; run at start, after sweeps,
           after holds, at the end)
  tap      close the gripper, operator taps it 3 times ~2 s apart, open again (start and end of each excite)
  excite   friction excitation, one joint at a time from the plan start pose: sines 0.5/1/2/3 Hz (streamed
           move_j at 50 Hz), then a triangle wave (one move_j per leg, 20 reversals)
  sweep    joint ripple sweeps: per joint and config, one slow move_j from one end of the range to the other,
           both directions, two passes
  static   with the container: the 10 T5 poses, each approached once from 15 mm above and once from 15 mm below
  lift     table lift-off: grasp the container on the table, lift 5 cm at 1 cm/s, put it back; modes "pass"
           (no stop) and "stop" (2 s still before and after), interleaved
  reach    with the container: 5 cm horizontal out and back at 1 cm/s, 3 cycles, wrist orientation fixed
  hold     long holds: 3 poses x 3 visits (above / below / above), 10 s each
  comply   arm holds one pose, the operator changes the load and types an external gauge reading per step
  pick     grasp the container at the lift site and lift it 5 cm (before static / reach / hold / comply)
  place    put the container back at the lift site and open (before the operator changes its content)
  goto     move to a named plan pose at 10 % and stop ("excite_q0", "lift_pre", "lift_top", "reach_start")

Safety (same rules as T7 / E3): --operator-confirmed, an arm that is already enabled (never enabled here),
YES typed after the plan is printed, speed percent <= 20, one joint at a time in excite / sweep. Every loop checks
the feedback; speed / excursion guards stop the segment. Ctrl-C, a guard or a timeout sends ONE move_j to the
measured pose (guarded_move.hold_measured_position; damped e-stop if the feedback is invalid), saves what was
recorded and exits. The arm never returns anywhere on its own.

usage (robot PC, PYAGX set, from the t8 directory):
  python t8_run.py excite --plan plans/t8_right.json --out ~/piper_hw_runs/T8_<date>/excite_cold --operator-confirmed
  python t8_run.py lift --plan plans/t8_right.json --label m250 --grams 251.3 --out ~/piper_hw_runs/T8_<date>/lift_m250 \
      --operator-confirmed --confirm-first
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "t7"))
import guarded_move as gm  # noqa: E402

VERSION = "t8_run 1.0 (2026-11-11)"
MAX_PCT = 20
REC_HZ = 100.0
STREAM_HZ = 50.0
GOTO_PCT = 10
GOTO_TOL = 0.004          # rad, final approach tolerance (0.23 deg)
VTOL = 0.01               # rad/s
GRIP_OPEN_M = 0.07
COLS = (["t_mono", "t_wall", "phase", "cmd"] + [f"q{j}" for j in range(1, 7)] + [f"mpos{j}" for j in range(1, 7)]
        + [f"v{j}" for j in range(1, 7)] + [f"i{j}" for j in range(1, 7)] + ["grip_m", "grip_N"])
PHASES = {"move": 0, "settle": 1, "record": 2, "sine": 3, "tri": 4, "sweep": 5, "line": 6, "grip": 7, "wait": 8,
          "tap": 9, "operator": 10}
CMDS = {"none": 0, "move_j": 1, "move_j_stream": 2, "hold": 3}


class Guard(RuntimeError):
    """A safety check stopped the segment."""


# ----------------------------------------------------------------------------- connection
def connect(channel, pyagx):
    """Same as guarded_move.connect, plus the gripper effector (it has to be created before connect())."""
    if not pyagx:
        raise SystemExit("set PYAGX to the PyAgxArm source directory, or pass --pyagx")
    if pyagx not in sys.path:
        sys.path.insert(0, pyagx)
    from pyAgxArm import AgxArmFactory, ArmModel, PiperFW, create_agx_arm_config
    cfg = create_agx_arm_config(robot=ArmModel.PIPER_X, firmeware_version=PiperFW.DEFAULT,
                                interface="socketcan", channel=channel)
    robot = AgxArmFactory.create_arm(cfg)
    grip = robot.init_effector("agx_gripper")
    robot.connect()
    return robot, grip


# ----------------------------------------------------------------------------- recorder
class Rec:
    """Rows at <= REC_HZ; every motion loop calls tick(). Marks carry the same two clocks."""

    def __init__(self, robot, grip, clock, sleep):
        self.robot, self.grip, self.clock, self.sleep = robot, grip, clock, sleep
        self.rows, self.marks, self.cmds = [], [], []
        self.last_q = None
        self.hist = []            # (t, q) over the last 0.2 s, for a finite-difference joint speed

    def qdot(self, span=0.15):
        """Joint speed (rad/s) from the joint angles, not from the motor velocity field."""
        h = self.hist
        if len(h) < 4 or h[-1][0] - h[0][0] < span * 0.9:
            return np.zeros(6)
        return (h[-1][1] - h[0][1]) / (h[-1][0] - h[0][0])

    def tick(self, phase, cmd="none"):
        q, mp, v, c = gm.read_state(self.robot)
        if q.shape != (6,) or not np.isfinite(q).all():
            raise Guard("invalid joint feedback")
        gw = gf = math.nan
        if self.grip is not None:
            gs = self.grip.get_gripper_status()
            if gs is not None:
                gw, gf = float(gs.msg.value), float(gs.msg.force)
        now = self.clock()
        self.rows.append([now, time.time(), PHASES[phase], CMDS[cmd], *q, *mp, *v, *c, gw, gf])
        self.last_q = q
        self.hist.append((now, q))
        while now - self.hist[0][0] > 0.2:
            self.hist.pop(0)
        return q, v, gw

    def mark(self, text):
        self.marks.append([self.clock(), time.time(), text])

    def send(self, target, kind="move_j"):
        self.robot.move_j([float(x) for x in target])
        self.cmds.append([self.clock(), CMDS[kind], *[float(x) for x in target]])

    def wait(self, seconds, phase="wait"):
        t0 = self.clock()
        while self.clock() - t0 < seconds:
            self.tick(phase)
            self.sleep(1.0 / REC_HZ)

    def save(self, out, name, meta):
        out.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out / f"{name}.npz", rows=np.array(self.rows, float), cols=np.array(COLS),
                            cmds=np.array(self.cmds, float).reshape(-1, 8),
                            marks_t=np.array([m[:2] for m in self.marks], float).reshape(-1, 2),
                            marks_text=np.array([m[2] for m in self.marks], dtype=str))
        meta = {**meta, "n_rows": len(self.rows), "n_cmds": len(self.cmds), "marks": self.marks,
                "phases": PHASES, "cmd_kinds": CMDS, "columns": COLS}
        (out / f"{name}.json").write_text(json.dumps(meta, indent=1, default=float))


# ----------------------------------------------------------------------------- motion primitives
STEP_RAD = 0.17          # long moves are cut into joint-space steps of <= 10 deg (see goto)


def goto(rec, target, pct=GOTO_PCT, timeout=40.0, resend_hz=20.0, tol=GOTO_TOL):
    """move_j re-sent at resend_hz until within tol and still (the T7 / E3 settle, recorded).
    The planner checks clearance along the STRAIGHT joint-space path. Whether the firmware's move_j keeps the joints
    synchronised is not known, so a move longer than STEP_RAD on any joint goes through intermediate points on that
    straight path (passed within 0.02 rad, not settled); only the final target is settled."""
    target = np.asarray(target, float)
    q0 = rec.last_q if rec.last_q is not None else rec.tick("move")[0]
    n = int(np.ceil(np.max(np.abs(target - q0)) / STEP_RAD))
    for k in range(1, n):
        _goto_one(rec, q0 + (target - q0) * k / n, pct, timeout, resend_hz, tol=0.02, still=False)
    return _goto_one(rec, target, pct, timeout, resend_hz, tol, still=True)


def _goto_one(rec, target, pct, timeout, resend_hz, tol, still):
    rec.robot.set_speed_percent(int(pct))
    t0 = rec.clock(); last = -1e9
    while True:
        if rec.clock() - last >= 1.0 / resend_hz:
            rec.send(target); last = rec.clock()
        q, v, _ = rec.tick("move", "move_j")
        if np.max(np.abs(q - target)) < tol and (not still or np.nanmax(np.abs(v)) < VTOL):
            return q
        if rec.clock() - t0 > timeout:
            raise TimeoutError(f"not at target after {timeout:.0f} s: max err {np.degrees(np.max(np.abs(q-target))):.2f} deg")
        rec.sleep(1.0 / REC_HZ)


def settle_record(rec, settle_s, hold_s):
    rec.wait(settle_s, "settle")
    i0 = len(rec.rows)
    rec.mark("record start")
    rec.wait(hold_s, "record")
    rec.mark("record end")
    r = np.array(rec.rows[i0:])
    drift = float(np.degrees(np.ptp(r[:, 4:10], axis=0)).max())
    return {"row0": i0, "row1": len(rec.rows), "drift_deg": drift, "i_mean_A": np.round(r[:, 22:28].mean(0), 4).tolist()}


def single_move(rec, target, pct, joint, vmax, expect_s, resend_hz=0.0, tol=GOTO_TOL, phase="sweep"):
    """ONE move_j (re-sent only if resend_hz > 0) and let the firmware interpolate. Guard: the moving joint's speed
    (finite difference over 0.2 s) must stay below vmax (rad/s); the other joints must not move more than 0.03 rad
    from where they started."""
    target = np.asarray(target, float)
    rec.robot.set_speed_percent(int(pct))
    rec.send(target)
    t0 = last = rec.clock()
    q_start, _, _ = rec.tick(phase, "move_j")
    others = [k for k in range(6) if k != joint]
    prog = []
    while True:
        q, v, _ = rec.tick(phase)
        vj = rec.qdot()[joint]
        if abs(vj) > vmax:
            raise Guard(f"j{joint+1} speed {abs(vj):.3f} rad/s > {vmax:.3f}")
        if np.max(np.abs(q[others] - q_start[others])) > 0.03:
            raise Guard("another joint moved > 0.03 rad during a single-joint move")
        prog.append((rec.clock(), q[joint]))
        while prog[-1][0] - prog[0][0] > 2.0:
            prog.pop(0)
        if (rec.clock() - t0 > 3.0 and prog[-1][0] - prog[0][0] > 1.9 and abs(prog[-1][1] - prog[0][1]) < 0.002
                and abs(q[joint] - target[joint]) > 0.01):
            raise Guard(f"j{joint+1} made no progress for 2 s ({np.degrees(abs(q[joint]-target[joint])):.1f} deg to go)")
        if np.max(np.abs(q - target)) < tol and np.nanmax(np.abs(v)) < VTOL:
            return rec.clock() - t0
        if rec.clock() - t0 > 2.0 * expect_s + 10.0:
            raise TimeoutError(f"single move not done after {rec.clock()-t0:.0f} s (expected {expect_s:.0f} s)")
        if resend_hz > 0 and rec.clock() - last >= 1.0 / resend_hz:
            rec.send(target); last = rec.clock()
        rec.sleep(1.0 / REC_HZ)


def stream_line(rec, Q, step_mm, speed_mm_s, lead_mm=2.0, phase="line", off_rad=0.05):
    """Streamed move_j along a joint-space waypoint list (the T7 way): the commanded index advances at speed_mm_s
    but never more than lead_mm ahead of the measured waypoint. Ends with a settle at the last waypoint."""
    Q = np.asarray(Q, float)
    n = len(Q)
    lead = max(1, int(round(lead_mm / step_mm)))
    t0 = rec.clock(); last = -1e9; k_meas = 0; k_sent = -1; k_best, t_best = 0, t0
    rec.robot.set_speed_percent(MAX_PCT)          # the T7 / E1 streaming setting
    while True:
        q, v, _ = rec.tick(phase, "none")
        lo = max(0, k_meas - 4 * lead); hi = min(n, k_meas + 4 * lead + 1)
        err = np.max(np.abs(Q[lo:hi] - q), axis=1)
        k = int(np.argmin(err))
        if err[k] > off_rad:
            raise Guard(f"joint feedback {err[k]:.3f} rad off the planned line")
        if lo + k > k_best:
            k_best, t_best = lo + k, rec.clock()
        elif k_best < n - 1 and rec.clock() - t_best > 1.0:
            raise Guard(f"no progress along the line for 1 s (waypoint {k_best} of {n})")
        k_meas = lo + k
        k_time = int((rec.clock() - t0) * speed_mm_s / step_mm)
        k_cmd = min(n - 1, k_time, k_meas + lead)
        if k_cmd != k_sent and rec.clock() - last >= 1.0 / STREAM_HZ:
            rec.send(Q[k_cmd], "move_j_stream"); last = rec.clock(); k_sent = k_cmd
        if k_cmd == n - 1 and np.max(np.abs(q - Q[-1])) < GOTO_TOL and np.nanmax(np.abs(v)) < VTOL:
            return rec.clock() - t0
        if rec.clock() - t0 > (n * step_mm / speed_mm_s) * 2.0 + 10.0:
            raise TimeoutError("line not finished in time")
        rec.sleep(1.0 / REC_HZ)


def sine(rec, q0, joint, f, amp, dur, vmax=0.8):
    rec.robot.set_speed_percent(MAX_PCT)
    t0 = rec.clock(); last = -1e9
    others = [k for k in range(6) if k != joint]
    while rec.clock() - t0 < dur:
        t = rec.clock() - t0
        if rec.clock() - last >= 1.0 / STREAM_HZ:
            tgt = np.array(q0, float); tgt[joint] += amp * math.sin(2 * math.pi * f * t)
            rec.send(tgt, "move_j_stream"); last = rec.clock()
        q, v, _ = rec.tick("sine")
        if abs(q[joint] - q0[joint]) > amp + 0.03:
            raise Guard(f"j{joint+1} excursion {abs(q[joint]-q0[joint]):.3f} rad > amplitude + 0.03")
        if abs(rec.qdot()[joint]) > vmax:
            raise Guard(f"j{joint+1} speed {abs(rec.qdot()[joint]):.2f} rad/s > {vmax}")
        if np.max(np.abs(q[others] - np.asarray(q0)[others])) > 0.05:
            raise Guard("another joint moved > 0.05 rad during a sine")
        rec.sleep(1.0 / REC_HZ)


# ----------------------------------------------------------------------------- gripper
def grip_cmd(rec, width_m, force_n):
    if rec.grip is None:
        raise Guard("no gripper effector")
    gs = rec.grip.get_gripper_status()
    if gs is None:
        raise Guard("no gripper feedback; command not sent")
    if gs.msg.mode != "width":
        raise Guard(f"gripper is in {gs.msg.mode!r} mode, T8 needs width mode")
    rec.grip.move_gripper_m(value=float(width_m), force=float(force_n))
    rec.mark(f"grip {width_m:.3f} m {force_n:.2f} N")


def grip_wait_stable(rec, still_s=0.3, tol_m=0.0005, timeout=4.0):
    t0 = rec.clock(); win = []
    while rec.clock() - t0 < timeout:
        _, _, gw = rec.tick("grip")
        win.append((rec.clock(), gw))
        while win and win[-1][0] - win[0][0] > still_s:
            win.pop(0)
        vals = [w for _, w in win if np.isfinite(w)]
        if win[-1][0] - win[0][0] >= still_s * 0.95 and vals and np.ptp(vals) < tol_m and rec.clock() - t0 > 0.5:
            rec.mark(f"grip stable {vals[-1]:.4f} m")
            return vals[-1]
        rec.sleep(1.0 / REC_HZ)
    raise Guard("gripper width not stable")


# ----------------------------------------------------------------------------- segments
def seg_anchor(rec, plan, a, ask):
    goto(rec, plan["anchor"]["q"])
    r = settle_record(rec, 3.0, 3.0)
    return {"anchor": r}


def seg_tap(rec, plan, a, ask):
    grip_cmd(rec, 0.0, a.grip_force_n); grip_wait_stable(rec)
    rec.mark("tap window start")
    ask("operator: tap the CLOSED gripper with the hard object 3 times, ~2 s apart, then press Enter: ")
    rec.mark("tap window end")
    rec.wait(1.0, "tap")
    if not a.keep_closed:
        grip_cmd(rec, GRIP_OPEN_M, a.grip_force_n); grip_wait_stable(rec)
    return {}


def seg_excite(rec, plan, a, ask):
    ex = plan["excite"]; q0 = np.asarray(ex["q0"], float)
    joints = [int(j) for j in a.joints.split(",")] if a.joints else ex["joints"]
    goto(rec, q0); rec.wait(1.0)
    info = []
    for j in joints:
        k = j - 1
        for s in ex["sines"]:
            rec.mark(f"sine j{j} f{s['f_hz']} A{s['amp_rad']}")
            sine(rec, q0, k, s["f_hz"], s["amp_rad"], s["dur_s"])
            goto(rec, q0, pct=GOTO_PCT); rec.wait(1.0)
        tri = ex["tri"]; pct = int(tri["pct"][k])
        hi = q0.copy(); hi[k] += tri["amp_rad"]; lo = q0.copy(); lo[k] -= tri["amp_rad"]
        rec.mark(f"tri j{j} pct{pct}")
        goto(rec, lo, pct=GOTO_PCT); rec.wait(0.5)
        legs = []
        for r in range(tri["reversals"] + 1):
            tgt = hi if r % 2 == 0 else lo
            legs.append(single_move(rec, tgt, pct, k, vmax=tri["vmax_rad_s"], expect_s=2 * tri["amp_rad"] / tri["v_rad_s"],
                                    resend_hz=a.resend_hz, phase="tri"))
        goto(rec, q0); rec.wait(1.0)
        info.append({"joint": j, "tri_leg_s": np.round(legs, 2).tolist()})
    return {"joints": info}


def seg_sweep(rec, plan, a, ask):
    sel = [int(j) for j in a.joints.split(",")] if a.joints else None
    done = []
    for sw in plan["sweeps"]:
        if sel and sw["joint"] not in sel:
            continue
        k = sw["joint"] - 1
        ends = [np.asarray(sw["q_lo"], float), np.asarray(sw["q_hi"], float)]
        expect = abs(sw["range_deg"][1] - sw["range_deg"][0]) / sw["deg_s_nominal"]
        for v in sw.get("via", []):                     # planner-checked entry: T5 base pose (via the excite pose if needed)
            goto(rec, v)
        goto(rec, ends[0]); rec.wait(1.0)
        for p in range(sw["passes"]):
            for d, tgt in (("up", ends[1]), ("down", ends[0])):
                rec.mark(f"sweep j{sw['joint']} {sw['config']} pass{p} {d} start")
                dt = single_move(rec, tgt, sw["pct"], k, vmax=math.radians(sw["deg_s_max"]), expect_s=expect,
                                 resend_hz=a.resend_hz)
                rec.mark(f"sweep j{sw['joint']} {sw['config']} pass{p} {d} end")
                done.append({"joint": sw["joint"], "config": sw["config"], "pass": p, "dir": d, "dur_s": round(dt, 2),
                             "deg_s": round(expect * sw["deg_s_nominal"] / max(dt, 1e-6), 3)})
                rec.wait(1.0)
    return {"sweeps": done}


def visit(rec, p, approach, pct, settle_s, hold_s, label):
    via = p["q_above"] if approach == "above" else p["q_below"]
    if via is None:                     # the planner found no feasible approach point on this side
        return {"pose": p["index"], "approach": approach, "skipped": True}
    goto(rec, via, pct=pct); rec.wait(0.3)
    goto(rec, p["q"], pct=pct)
    rec.mark(f"pose {p['index']} from {approach} {label}")
    r = settle_record(rec, settle_s, hold_s)
    return {"pose": p["index"], "approach": approach, **r}


def seg_static(rec, plan, a, ask):
    st = plan["static"]; out = []
    for n, p in enumerate(st["poses"]):
        if a.confirm_first and n == 0 or a.confirm_each:
            if ask(f"move to pose {p['index']} (from above, then below)? y / anything else stops: ").strip().lower() != "y":
                break
        for ap in ("above", "below"):
            out.append(visit(rec, p, ap, a.speed_pct, st["settle_s"], st["hold_s"], a.label))
    return {"visits": out}


def seg_hold(rec, plan, a, ask):
    h = plan["hold"]; out = []
    for n, p in enumerate(h["poses"]):
        if a.confirm_first and n == 0 or a.confirm_each:
            if ask(f"long holds at pose {p['index']}? y / anything else stops: ").strip().lower() != "y":
                break
        for v, ap in enumerate(h["approaches"]):
            out.append({**visit(rec, p, ap, a.speed_pct, h["settle_s"], h["hold_s"], a.label), "visit": v})
    return {"visits": out}


def seg_lift(rec, plan, a, ask):
    L = plan["lift"]; step = L["step_mm"]
    Qd, Qu = np.asarray(L["Q_down"]), np.asarray(L["Q_up"])
    Q_put = Qu[::-1][: int(round(L["put_down_mm"] / step)) + 1]          # stops (lift - put_down) mm above the grasp
    modes = a.modes.split(",")
    trials = []
    goto(rec, L["q_pre"]); grip_cmd(rec, GRIP_OPEN_M, a.grip_force_n); grip_wait_stable(rec)
    for n, mode in enumerate(modes):
        first = n == 0 and (a.confirm_first or a.confirm_each)
        tr = {"trial": n, "mode": mode}
        rec.mark(f"lift trial {n} {mode} start")
        stream_line(rec, Qd, step, L["approach_mm_s"])                       # pre-grasp -> grasp, gripper open
        if (first or a.confirm_each) and ask("fingers around the container, nothing touching? y closes, anything else stops: ").strip().lower() != "y":
            raise KeyboardInterrupt
        tr["t_close"] = rec.clock(); grip_cmd(rec, 0.0, a.grip_force_n)
        tr["grip_width_m"] = grip_wait_stable(rec)
        if mode == "stop":
            rec.mark("pre static"); tr["pre"] = settle_record(rec, 0.0, L["still_s"])
        tr["t_lift_cmd"] = rec.clock(); rec.mark("lift cmd")
        tr["lift_s"] = stream_line(rec, Qu, step, L["lift_mm_s"])
        if mode == "stop":
            rec.mark("post static"); tr["post"] = settle_record(rec, 0.0, L["still_s"])
        else:
            rec.wait(0.5)
        if first and ask("container held and lifted? y puts it back down, anything else stops: ").strip().lower() != "y":
            raise KeyboardInterrupt
        rec.mark("put down"); stream_line(rec, Q_put, step, L["lift_mm_s"])
        grip_cmd(rec, GRIP_OPEN_M, a.grip_force_n); grip_wait_stable(rec)
        stream_line(rec, Qd[::-1], step, L["approach_mm_s"])                 # back up to pre-grasp
        rec.mark(f"lift trial {n} end")
        trials.append(tr)
    return {"trials": trials}


def seg_pick(rec, plan, a, ask):
    """Grasp the container at the lift site and lift it 5 cm (ends at lift_top holding it)."""
    L = plan["lift"]; step = L["step_mm"]
    goto(rec, L["q_pre"]); grip_cmd(rec, GRIP_OPEN_M, a.grip_force_n); grip_wait_stable(rec)
    stream_line(rec, np.asarray(L["Q_down"]), step, L["approach_mm_s"])
    if ask("fingers around the container, nothing touching? y closes, anything else stops: ").strip().lower() != "y":
        raise KeyboardInterrupt
    grip_cmd(rec, 0.0, a.grip_force_n); w = grip_wait_stable(rec)
    stream_line(rec, np.asarray(L["Q_up"]), step, L["lift_mm_s"])
    rec.wait(1.0)
    return {"grip_width_m": w}


def seg_place(rec, plan, a, ask):
    """From anywhere with the container: move_j to lift_top, put it down, open, back up to pre-grasp."""
    L = plan["lift"]; step = L["step_mm"]; Qu = np.asarray(L["Q_up"])
    goto(rec, Qu[-1], pct=a.speed_pct); rec.wait(1.0)
    stream_line(rec, Qu[::-1][: int(round(L["put_down_mm"] / step)) + 1], step, L["lift_mm_s"])
    grip_cmd(rec, GRIP_OPEN_M, a.grip_force_n); grip_wait_stable(rec)
    stream_line(rec, np.asarray(L["Q_down"])[::-1], step, L["approach_mm_s"])
    return {}


def seg_reach(rec, plan, a, ask):
    R = plan["reach"]; Q = np.asarray(R["Q_out"]); step = R["step_mm"]
    goto(rec, Q[0], pct=a.speed_pct); rec.wait(2.0)
    cyc = []
    for c in range(R["cycles"]):
        rec.mark(f"reach {c} out"); t_out = stream_line(rec, Q, step, R["mm_s"]); rec.wait(R["pause_s"])
        rec.mark(f"reach {c} in"); t_in = stream_line(rec, Q[::-1], step, R["mm_s"]); rec.wait(R["pause_s"])
        cyc.append({"cycle": c, "out_s": round(t_out, 2), "in_s": round(t_in, 2)})
    return {"cycles": cyc}


def seg_comply(rec, plan, a, ask):
    C = plan["comply"]; out = []
    idx = [int(i) for i in a.indices.split(",")] if a.indices else [p["index"] for p in C["poses"]]
    for p in [p for p in C["poses"] if p["index"] in idx]:
        goto(rec, p["q_above"], pct=a.speed_pct); goto(rec, p["q"], pct=a.speed_pct)
        rec.wait(2.0)
        for s in a.steps.split(","):
            if ask(f"pose {p['index']}: load {s} g in the container, hands off -> Enter (q stops): ").strip().lower() == "q":
                raise KeyboardInterrupt
            r = settle_record(rec, C["settle_s"], C["hold_s"])
            reading = ask(f"  gauge reading now ({a.gauge}), number in mm: ").strip()
            out.append({"pose": p["index"], "grams": s, "gauge_mm": reading, **r})
            rec.mark(f"pose {p['index']} load {s} gauge {reading}")
    return {"steps": out, "gauge": a.gauge}


def seg_goto(rec, plan, a, ask):
    q = plan["named"][a.name]
    goto(rec, q, pct=a.speed_pct)
    return {"name": a.name}


SEGMENTS = {"anchor": seg_anchor, "tap": seg_tap, "excite": seg_excite, "sweep": seg_sweep, "static": seg_static,
            "lift": seg_lift, "pick": seg_pick, "place": seg_place, "reach": seg_reach, "hold": seg_hold, "comply": seg_comply, "goto": seg_goto}
NEEDS_CONTAINER = {"static", "reach", "hold", "comply", "pick", "place"}


def describe(plan, seg, a):
    """One line per target so the operator can read the plan before YES."""
    deg = lambda q: np.round(np.degrees(q), 1).tolist()  # noqa: E731
    if seg == "excite":
        ex = plan["excite"]
        return [f"start (T5 pose {ex['start_index']}) {deg(ex['q0'])}",
                f"joints {a.joints or ex['joints']}: sines " + ", ".join(f"{s['f_hz']} Hz x {s['amp_rad']} rad x {s['dur_s']} s" for s in ex['sines'])
                + f"; triangle +-{ex['tri']['amp_rad']} rad, {ex['tri']['reversals']} reversals, pct {ex['tri']['pct']}"]
    if seg == "sweep":
        return [f"j{s['joint']} {s['config']}: {s['range_deg'][0]:.1f}..{s['range_deg'][1]:.1f} deg at {s['pct']} % "
                f"(~{s['deg_s_nominal']:.1f} deg/s), {s['passes']} passes x 2 directions; others {deg(s['q_lo'])}"
                for s in plan["sweeps"] if not a.joints or s["joint"] in [int(j) for j in a.joints.split(",")]]
    if seg in ("static", "hold", "comply"):
        return [f"pose {p['index']} {deg(p['q'])}" for p in plan[seg]["poses"]]
    if seg == "lift":
        L = plan["lift"]
        return [f"pre-grasp {deg(L['q_pre'])}", f"grasp {deg(L['Q_down'][-1])}",
                f"modes {a.modes}; lift {L['lift_mm']} mm at {L['lift_mm_s']} mm/s, put down {L['put_down_mm']} mm"]
    if seg == "reach":
        R = plan["reach"]
        return [f"start {deg(R['Q_out'][0])}", f"end {deg(R['Q_out'][-1])}", f"{R['cycles']} cycles at {R['mm_s']} mm/s"]
    if seg == "anchor":
        return [f"anchor {deg(plan['anchor']['q'])}"]
    if seg in ("pick", "place"):
        L = plan["lift"]
        return [f"pre-grasp {deg(L['q_pre'])}", f"grasp {deg(L['Q_down'][-1])}", f"lift top {deg(L['Q_up'][-1])}"]
    if seg == "goto":
        return [f"{a.name} {deg(plan['named'][a.name])}"]
    return []


def main(argv=None, robot=None, grip=None, clock=time.monotonic, sleep=time.sleep, ask=input):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("segment", choices=sorted(SEGMENTS))
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--channel", default="can_right")
    p.add_argument("--label", default="", help="free text, e.g. m0 / m100 / cold / hot")
    p.add_argument("--grams", type=float, help="weighed content of the container (g); required with the container")
    p.add_argument("--container-g", type=float, help="weighed empty container (g); required with the container")
    p.add_argument("--speed-pct", type=int, default=GOTO_PCT)
    p.add_argument("--joints", help="excite / sweep: comma-separated joints (1-based); default all in the plan")
    p.add_argument("--modes", default=",".join(["pass", "pass", "stop"] * 5), help="lift: trial modes in order (predictions_v2: 10 pass + 5 stop)")
    p.add_argument("--steps", default="0,500,0,500,0", help="comply: grams in the container at each step")
    p.add_argument("--indices", help="comply: pose indices (default all in the plan)")
    p.add_argument("--gauge", default="dial", help="comply: external method (dial / ruler / camera)")
    p.add_argument("--name", help="goto: plan pose name")
    p.add_argument("--grip-force-n", type=float, default=1.0)
    p.add_argument("--keep-closed", action="store_true", help="tap: leave the gripper closed")
    p.add_argument("--resend-hz", type=float, default=0.0, help="single moves: re-send the target at this rate (0 = once)")
    p.add_argument("--confirm-first", action="store_true", help="ask before the first pose / trial")
    p.add_argument("--confirm-each", action="store_true", help="ask before every pose / trial")
    p.add_argument("--operator-confirmed", action="store_true")
    p.add_argument("--pyagx", default=gm.PYAGX_DEFAULT)
    a = p.parse_args(argv)
    if not a.operator_confirmed:
        p.error("operator confirmation required: --operator-confirmed")
    if not 1 <= a.speed_pct <= MAX_PCT:
        p.error(f"--speed-pct must be 1..{MAX_PCT}")
    if a.segment == "goto" and not a.name:
        p.error("goto needs --name")
    plan = json.loads(a.plan.read_text())
    if plan.get("arm") != a.channel:
        p.error(f"plan is for {plan.get('arm')}, not {a.channel}")
    if a.segment in NEEDS_CONTAINER | {"lift"} and (a.grams is None or a.container_g is None):
        p.error(f"{a.segment} uses the container: give --grams (content) and --container-g (empty container)")
    for s in plan.get("sweeps", []):
        if s["pct"] > MAX_PCT:
            p.error("plan sweep pct above the cap")
    a.out.mkdir(parents=True, exist_ok=False)
    meta = {"version": VERSION, "segment": a.segment, "label": a.label, "grams": a.grams, "container_g": a.container_g,
            "plan": a.plan.name, "plan_sha": plan.get("sha"), "args": {k: str(v) for k, v in vars(a).items()},
            "started": datetime.now().isoformat(timespec="seconds")}

    if robot is None:
        robot, grip = connect(a.channel, a.pyagx)
    print("Waiting for complete, stable feedback ...")
    q_now = gm.wait_for_feedback(robot, clock, sleep)[0]
    print(f"[{a.channel}] now [deg]: {np.round(np.degrees(q_now), 2).tolist()}")
    for line in describe(plan, a.segment, a):
        print("  " + line)
    if a.segment in NEEDS_CONTAINER | {"lift"}:
        print(f"  container {a.container_g} g + content {a.grams} g")
    print("check: e-stop in hand, path clear, left arm parked, master arm off the right bus, lowspd_log + candump running.")
    if ask("type YES to start: ").strip() != "YES":
        raise SystemExit("aborted, nothing sent")
    rec = Rec(robot, grip, clock, sleep)
    rec.mark(f"segment {a.segment} {a.label} start")
    try:
        gm.wait_for_feedback(robot, clock, sleep)
        info = SEGMENTS[a.segment](rec, plan, a, ask)
        meta.update(status="done", info=info)
    except (KeyboardInterrupt, TimeoutError, Guard, gm.FeedbackNotReady) as exc:
        hold = None
        try:
            hold = gm.hold_measured_position(robot)
        finally:
            rec.cmds.append([clock(), CMDS["hold"], *(hold or [math.nan] * 6)])
            meta.update(status=type(exc).__name__, detail=str(exc), hold_q_rad=hold)
            meta["finished"] = datetime.now().isoformat(timespec="seconds")
            rec.save(a.out, "rec", meta)
        print(f"STOPPED ({type(exc).__name__}: {exc}); holding the measured pose; data in {a.out}")
        raise SystemExit(2)
    meta["finished"] = datetime.now().isoformat(timespec="seconds")
    rec.mark(f"segment {a.segment} end")
    rec.save(a.out, "rec", meta)
    print(f"done: {a.segment} {a.label}; {len(rec.rows)} rows in {a.out}; the arm holds the last pose")
    return a.out, meta


if __name__ == "__main__":
    main()
