#!/usr/bin/env python3
"""T7 guarded descent on the real arm (robot side). THIS MOVES THE ARM.

Replays a plan made on the server by t7_plan.py: the gripper tip goes straight down from the start pose,
slowly, while the motor currents are watched; the first contact stops the descent and the arm backs off.

How it moves: move_j (firmware-planned, the same call the T4 scripts used) is re-sent with the next waypoint
of the plan, a fraction of a millimetre ahead, ~40 times a second. The commanded target is never more than a
couple of waypoints ahead, so when sending stops the arm stops within (servo lag x speed), about 1.3 mm at
10 mm/s. move_js (no planning) is never used.

How contact is detected (plan version 2, "tip_force"):
  1. 1 s static baseline at the start pose: i_base.
  2. residual r = (i - i_base) - (i_expected(at the measured position) - i_expected(start)); the gravity change
     along the line comes from the plan. r is low-passed over lp_n samples (30 ms).
  3. the first arm_after mm (plan) are free descent: the mean and spread of r over its last 10 mm absorb the
     kinetic friction of moving down; detection is armed only after that.
  4. tip force F = weighted least-squares projection of (r - mean) on j2..j5 onto the plan's contact signature
     di_per_N_up (current change per newton of upward force on the tip), weights 1 / spread^2.
     "explained" = share of the weighted deviation that a pure tip force accounts for.
  5. contact = F > max(thr_sigma * spread of F in free descent, floor_N) AND explained >= min_explained,
     held for hold_s (60 ms). A single joint slipping (j2 stick-slip at low speed) moves F little and is
     poorly explained by a tip force, so it does not trigger; a real touch moves j2 and j3 together.
  6. kinetic friction (the free-descent mean) is scaled by phi = measured tip speed / commanded speed over the
     last 100 ms, so it is removed when the arm stops (v2.1).
  7. v2.2 (after the first real contacts): on hardware the contact force arrives as 30-70 ms pulses every ~0.2 s,
     so a "held above threshold" rule fired late (~10 N) or not at all. Contact is now declared when the MEAN tip
     force over the last avg_window_s (0.2 s) exceeds max(thr_sigma x its free-descent spread, floor_N = 2 N).
     Steps 4-5 are still computed and logged (tip_force_N, explained) but no longer gate the stop.
     If max depth is reached while that mean is above 1 N, the arm is pressing: status "contact_late", back off
     at once (no hold at the bottom).
  8. v2.3: the tip force uses j2 and j3 only (--detect-joints, default 2,3). On hardware the wrist joints (j4, j5)
     answered with the opposite sign at every pose, because the first point to touch is a finger edge, not the
     nominal tip, so their response depends on where exactly the gripper meets the plate. Replaying all 24
     trials: no false alarm with j2+j3 (free-descent mean force <= 1.37 N), earlier detection, and p3 detected
     at the same plate height as p0/p2.
  9. v2.4: --motion line sends ONE move_l from the start flange pose straight down to max depth and lets the
     firmware interpolate at constant speed (speed set by --speed-pct, mm/s measured during the free descent),
     instead of re-sending move_j every 25 ms. Offline analysis of T7 showed the streamed motion is a ~5 Hz
     stop-and-go (j2 acceleration line at 4.7 Hz), which is where the pulsed contact force and part of the friction
     noise come from. Detection is unchanged. The stall backup becomes progress-based (measured speed < 20 % of
     the free-descent speed for 150 ms). --test-stop-mm X declares a fake contact at X mm in free air: use it first,
     to measure how quickly a running move_l is interrupted by the back-off move_j (summary: overshoot_mm).
  10. Every trial also writes the tip force along the WHOLE descent, F(z): npz keys fz_* and fz_XX.csv
     (t, phase 0 free / 1 armed / 2 backing off, depth, tip z, commanded depth, F, 0.2 s mean F, phi). The
     friction baseline from arming is applied back to the first 20 mm, so the free-descent part is included.
Backup stop (v2.1): if the command runs ahead of the measured position by more than max(3 mm, 3 x the
free-descent lead) for 80 ms, the arm is being held by something: status "stall", back off. This is a
position (a - q) criterion, kept only as a safety net and logged so current and position can be compared.
Hard stops: tip more than max_depth below the start (plan), timeout, Ctrl-C (holds the measured pose).

Safety: needs --operator-confirmed, an enabled arm, the gripper already closed, and typed YES; each trial asks
for y unless --no-confirm-each. Speed is capped at 40 mm/s (above 20 only on the rigid plate) and the SDK speed percent at 20; in line mode
the trial stops if the measured free-descent speed exceeds --max-line-speed (default 45 mm/s).

usage:
  python guarded_move.py --plan right_p0.json --hover-only --operator-confirmed          # move to start, record 3 s, stop
  python guarded_move.py --plan right_p0.json --trials 10 --operator-confirmed           # 10 descents, y before each
  python guarded_move.py --plan right_p0.json --motion line --speed-pct 5 --test-stop-mm 40 --operator-confirmed
Needs PyAgxArm: set PYAGX=<path to the PyAgxArm source> or pass --pyagx.
Outputs to --out (default ~/piper_hw_runs/T7_<stamp>/): trial_XX.npz per descent, summary.json, hover.npz.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

PYAGX_DEFAULT = os.environ.get("PYAGX", "")
MAX_SPEED_MM_S = 40.0
MAX_SPEED_PCT = 20


# ----------------------------------------------------------------------------- robot I/O
def connect(channel, pyagx=PYAGX_DEFAULT):
    if not pyagx:
        raise SystemExit("set PYAGX to the PyAgxArm source directory, or pass --pyagx")
    if pyagx not in sys.path:
        sys.path.insert(0, pyagx)
    from pyAgxArm import AgxArmFactory, ArmModel, PiperFW, create_agx_arm_config
    cfg = create_agx_arm_config(robot=ArmModel.PIPER_X, firmeware_version=PiperFW.DEFAULT,
                                interface="socketcan", channel=channel)
    robot = AgxArmFactory.create_arm(cfg)
    robot.connect()
    return robot


def read_state(robot):
    """q (joint feedback, rad), motor position (rad, same frame as current), velocity (rad/s), current (A)."""
    ja = robot.get_joint_angles()
    q = np.asarray(ja.msg, float) if ja is not None else np.full(6, np.nan)
    mp, v, c = np.full(6, np.nan), np.full(6, np.nan), np.full(6, np.nan)
    for i in range(6):
        ms = robot.get_motor_states(i + 1)
        if ms is not None:
            mp[i], v[i], c[i] = ms.msg.position, ms.msg.velocity, ms.msg.current
    return q, mp, v, c


def settle(robot, target, clock, sleep, tol=0.01, vtol=0.01, timeout=25.0, resend_hz=20.0):
    t0 = clock()
    while True:
        robot.move_j([float(x) for x in target])
        q, _, v, _ = read_state(robot)
        if np.all(np.isfinite(q)) and np.max(np.abs(q - target)) < tol and np.nanmax(np.abs(v)) < vtol:
            return q
        if clock() - t0 > timeout:
            raise TimeoutError(f"did not settle at target: max err {np.degrees(np.nanmax(np.abs(q-target))):.2f} deg")
        sleep(1.0 / resend_hz)


def record_static(robot, seconds, clock, sleep, hz=100.0):
    rows = []
    t0 = clock()
    while clock() - t0 < seconds:
        q, mp, v, c = read_state(robot)
        rows.append(np.concatenate([[clock() - t0], q, mp, v, c]))
        sleep(1.0 / hz)
    a = np.array(rows)
    return {"t": a[:, 0], "q": a[:, 1:7], "mpos": a[:, 7:13], "v": a[:, 13:19], "current": a[:, 19:25]}


# ----------------------------------------------------------------------------- one descent
def tip_force(rl, mu, sd, a, J):
    """Weighted LS tip force (N, + = upward on the tip) and the share of the deviation it explains."""
    dev = (rl - mu)[J]; aj = a[J]; w = 1.0 / np.maximum(sd[J], 1e-4) ** 2
    F = float((w * aj * dev).sum() / max((w * aj * aj).sum(), 1e-12))
    tot = float((w * dev * dev).sum()); fit = float((w * (dev - aj * F) ** 2).sum())
    return F, (1.0 - fit / tot) if tot > 1e-12 else 0.0


def full_descent_fz(L, aux, plan, mu, sd, A, J, lp_n, avg_window_s, n_post):
    """Tip force along the whole descent, F(z), from the logged residuals. The online log has the force only after
    arming (the free-descent friction mu is known only then); here mu, sd from arming are applied to every row,
    including the first 20 mm, with the same low-pass, friction scaling (phi) and 0.2 s mean as the online rule.
    Phases: 0 free descent before arming, 1 armed, 2 after the stop (backing off; phi = 0, approximate)."""
    n = len(L)
    step = plan["step_mm"]; zt = np.asarray(plan["tip_z_m"])
    phi, k = aux[:, 0], np.clip(aux[:, 1].astype(int), 0, len(zt) - 1)
    t, R = L[:, 0], L[:, 32:38]
    phase = np.where(L[:, 3] > 0, 1, 0); phase[n - n_post:] = 2
    F = np.full(n, np.nan)
    if mu is not None:
        for i in range(n):
            rl = np.nanmean(R[max(0, i - lp_n + 1): i + 1], 0)
            F[i] = tip_force(rl, mu * phi[i], sd, A[k[i]], J)[0]
    Fm = np.full(n, np.nan)
    j0 = 0
    for i in range(n):
        while t[i] - t[j0] > avg_window_s:
            j0 += 1
        if t[i] - t[j0] >= 0.8 * avg_window_s:
            Fm[i] = np.nanmean(F[j0: i + 1])
    return {"fz_depth_mm": k * step, "fz_tip_z_mm": zt[k] * 1000.0, "fz_cmd_depth_mm": L[:, 1] * step,
            "fz_F_N": F, "fz_F_mean_N": Fm, "fz_phi": phi, "fz_phase": phase}


def descend(robot, plan, speed_mm_s, clock, sleep, loop_hz=100.0, send_hz=50.0, log=print, hold_bottom_s=0.0,
            avg_window_s=0.2, floor_N=2.0, detect_joints=(1, 2), motion="stream", test_stop_mm=None,
            max_line_speed=45.0, line_pct=None, restore_pct=None):
    if plan.get("version", 1) < 2 or plan["detect"].get("mode") != "tip_force":
        raise SystemExit("plan is version 1 (single-joint thresholds, false-triggers on j2 stick-slip): regenerate it with t7_plan.py")
    Q = np.asarray(plan["q"]); IE = np.asarray(plan["i_expected_A"]); A = np.asarray(plan["di_per_N_up"]); step = plan["step_mm"]
    det = plan["detect"]; J = list(detect_joints) if detect_joints else det["joints"]; lp_n = det["lp_n"]
    arm_after, max_depth = plan["arm_after_mm"], plan["max_depth_mm"]
    k_max = min(len(Q) - 1, int(max_depth / step))
    retract_k = int(round(plan["retract_mm"] / step))

    base = record_static(robot, 1.0, clock, sleep)
    i_base = np.nanmean(base["current"], 0)
    sd_static = np.nanstd(base["current"], 0)

    rows, free, recent = [], [], []
    aux = []                  # per logged row: (phi, measured waypoint) -> full-descent F(z) after the trial
    mu = sd = None
    thr_F = sd_F = None
    t_over = None
    hist = []                 # (t, k_meas) for the tip speed
    fwin = []                 # (t, F) for the window mean
    thr_avg = sd_avg = None
    F_avg = np.nan
    leads = []                # command lead in free descent (mm)
    lead_thr = None; t_lead = None; t_stall = None; t_cur = None
    k_cmd, k_meas, last_send = 0, 0, -1.0
    status, k_det, t_det, F_det = "no_contact", None, None, None
    speed_ref = speed_mm_s; speeds = []; t5 = None; t_slow = None; k_stop = None; q_hold = None
    line = motion == "line"
    if line:
        fp = robot.get_flange_pose()
        if fp is None:
            raise SystemExit("no flange pose feedback: cannot plan the move_l")
        target = [float(x) for x in fp.msg]
        target[2] -= max_depth / 1000.0
        line_target = target
    t0 = clock()
    timeout = 60.0 if line else max_depth / speed_mm_s + 10.0
    if line:
        if line_pct:
            robot.set_speed_percent(line_pct)          # only the move_l runs at this percent
        robot.move_l(line_target)
    try:
        while True:
            t = clock() - t0
            k_want = min(k_max, int(speed_mm_s * t / step))
            if not line and k_want != k_cmd and t - last_send >= 1.0 / send_hz:
                k_cmd = k_want
                robot.move_j([float(x) for x in Q[k_cmd]])
                last_send = t
            q, mp, v, c = read_state(robot)
            if np.all(np.isfinite(q)):
                lo, hi = max(0, k_meas - 20), min(len(Q), k_meas + 40)
                k_meas = lo + int(np.argmin(np.abs(Q[lo:hi] - q).max(1)))
            s_meas = k_meas * step
            win = max(0.1, 3 * step / speed_ref)      # speed window covers >= 3 waypoints (k_meas is quantised)
            hist.append((t, k_meas)); hist = [h for h in hist if t - h[0] <= win]
            phi, v_tip = 0.0, 0.0
            if t - hist[0][0] > 0.5 * win:
                v_tip = (k_meas - hist[0][1]) * step / (t - hist[0][0])
                phi = float(np.clip(v_tip / speed_ref, 0.0, 1.0))
            if line:
                k_cmd = k_meas
            lead = (k_cmd - k_meas) * step
            r = (c - i_base) - (IE[k_meas] - IE[0])
            recent.append(r); recent = recent[-lp_n:]
            rl = np.nanmean(np.array(recent), 0)
            armed = mu is not None
            Fk, ex = np.nan, np.nan
            if not armed:
                if t5 is None and s_meas >= 5.0:
                    t5 = t
                if s_meas >= arm_after - 10.0:
                    free.append((t, k_meas, rl.copy()))
                    leads.append(lead)
                if s_meas >= arm_after and len(free) >= 20:
                    R = np.array([x[2] for x in free])
                    mu, sd = R.mean(0), R.std(0)
                    tf = np.array([x[0] for x in free])
                    Ff = np.array([tip_force(x[2], mu, sd, A[x[1]], J)[0] for x in free])
                    sd_F = float(Ff.std())
                    thr_F = max(det["thr_sigma"] * sd_F, det["floor_N"])
                    Fm = np.array([Ff[(tf > tk - avg_window_s) & (tf <= tk)].mean() for tk in tf if tk - tf[0] >= avg_window_s])
                    sd_avg = float(Fm.std()) if len(Fm) > 5 else sd_F
                    thr_avg = max(det["thr_sigma"] * sd_avg, floor_N)
                    lead_thr = max(3.0, 3.0 * float(np.median(leads)))
                    if t5 is not None and t > t5:
                        speeds = [(s_meas - 5.0) / (t - t5)]
                    if line and speeds:
                        speed_ref = speeds[0]
                        log(f"    move_l speed in free descent: {speed_ref:.1f} mm/s")
                    if line and speed_ref > max_line_speed:
                        status, k_det, t_det = "too_fast", k_meas, t
                        k_stop = max(0, k_meas - retract_k)
                        robot.move_j([float(x) for x in Q[k_stop]])
                        log(f"    move_l at {speed_ref:.1f} mm/s > {max_line_speed:.0f} mm/s: stopping, lower --speed-pct")
                        break
                    log(f"    armed at {s_meas:.1f} mm: free-descent spread j2 {sd[1]:.3f} A, j3 {sd[2]:.3f} A; "
                        f"{avg_window_s:.1f} s mean tip-force spread {sd_avg:.2f} N -> threshold {thr_avg:.2f} N")
            hit = False
            stall = False
            if armed:
                Fk, ex = tip_force(rl, mu * phi, sd, A[k_meas], J)
                fwin.append((t, Fk))
                fwin = [w for w in fwin if t - w[0] <= avg_window_s]
                F_avg = float(np.mean([w[1] for w in fwin])) if t - fwin[0][0] >= 0.8 * avg_window_s else np.nan
                hit = bool(np.isfinite(F_avg) and F_avg > thr_avg)
                if line:
                    if phi < 0.2 and s_meas < max_depth - 2.0:
                        t_slow = t if t_slow is None else t_slow
                        stall = t - t_slow >= 0.15
                    else:
                        t_slow = None
                elif lead > lead_thr:
                    t_lead = t if t_lead is None else t_lead
                    stall = t - t_lead >= 0.08
                else:
                    t_lead = None
            rows.append(np.concatenate([[t, k_cmd, k_meas, float(armed), float(hit), Fk, ex, F_avg], q, mp, v, c, r]))
            aux.append((phi, k_meas))
            if test_stop_mm is not None and s_meas >= test_stop_mm and not (hit or stall):
                status, k_det, t_det, F_det = "test_stop", k_meas, t, F_avg
                k_stop = max(0, k_meas - retract_k)
                robot.move_j([float(x) for x in Q[k_stop]])
                break
            if hit or stall:
                status = "contact" if hit else "stall"
                k_det, t_det, F_det = k_meas, t, F_avg
                t_cur, t_stall = (t if hit else None), (t if stall else None)
                k_stop = max(0, k_meas - retract_k)
                robot.move_j([float(x) for x in Q[k_stop]])
                break
            if (k_cmd >= k_max or line) and s_meas >= max_depth - step:
                if np.isfinite(F_avg) and F_avg > 1.0:
                    status, k_det, t_det, F_det = "contact_late", k_meas, t, F_avg
                    k_stop = max(0, k_meas - retract_k)
                    robot.move_j([float(x) for x in Q[k_stop]])
                    log(f"    max depth reached while pressing (mean tip force {F_avg:.1f} N): backing off")
                    break
                status = "max_depth"
                if hold_bottom_s > 0:
                    log(f"    reached max depth without contact: holding {hold_bottom_s:.0f} s, MEASURE the remaining tip-to-plate gap now")
                    t_h = clock()
                    while clock() - t_h < hold_bottom_s:
                        robot.move_j([float(x) for x in Q[k_meas if line else k_cmd]]); sleep(0.05)
                break
            if t > timeout:
                status = "timeout"
                if line:                      # a running move_l would keep going: hold where we are
                    k_stop = k_meas
                    robot.move_j([float(x) for x in Q[k_stop]])
                break
            sleep(1.0 / loop_hz)
    except KeyboardInterrupt:
        q, _, _, _ = read_state(robot)
        robot.move_j([float(x) for x in q])
        q_hold = [float(x) for x in q]
        status = "interrupted"
    # keep logging for 1 s after the stop (servo settling, the pressing force decaying); re-send the back-off target
    # so it also overrides a running move_l
    if k_stop is not None:
        q_hold = [float(x) for x in Q[k_stop]]
    t1 = clock(); k_after = k_max_after = k_meas; last_rs = -1.0
    n_pre_post = len(rows)
    while clock() - t1 < 1.0:
        t = clock() - t0
        if q_hold is not None and t - last_rs >= 0.1:
            robot.move_j(q_hold); last_rs = t
        q, mp, v, c = read_state(robot)
        if np.all(np.isfinite(q)):
            lo, hi = max(0, k_after - 20), min(len(Q), k_after + 40)
            k_after = lo + int(np.argmin(np.abs(Q[lo:hi] - q).max(1)))
            k_max_after = max(k_max_after, k_after)
        r = (c - i_base) - (IE[k_meas] - IE[0])
        recent.append(r); recent = recent[-lp_n:]
        Fk, ex = (tip_force(np.nanmean(np.array(recent), 0), mu, sd, A[k_meas], J) if mu is not None else (np.nan, np.nan))
        rows.append(np.concatenate([[t, k_cmd, k_meas, float(mu is not None), 0.0, Fk, ex, np.nan], q, mp, v, c, r]))
        aux.append((0.0, k_after))           # backing off: no friction model for upward motion, phi = 0
        sleep(1.0 / loop_hz)
    if line and line_pct and restore_pct:
        robot.set_speed_percent(restore_pct)          # back to the approach speed for the return to the start
    L = np.array(rows)
    fz = full_descent_fz(L, np.array(aux, float), plan, mu, sd, A, J, lp_n, avg_window_s, n_post=len(rows) - n_pre_post)
    log_d = {"t": L[:, 0], "k_cmd": L[:, 1], "k_meas": L[:, 2], "armed": L[:, 3], "hit": L[:, 4],
             "tip_force_N": L[:, 5], "explained": L[:, 6], "tip_force_mean_N": L[:, 7],
             "q": L[:, 8:14], "mpos": L[:, 14:20], "v": L[:, 20:26], "current": L[:, 26:32], "resid": L[:, 32:38],
             "i_base": i_base, "sd_static": sd_static, "mu_free": mu if mu is not None else np.full(6, np.nan),
             "sd_free": sd if sd is not None else np.full(6, np.nan),
             "thr_N": np.nan if thr_avg is None else thr_avg, "sd_F_free_N": np.nan if sd_F is None else sd_F,
             "sd_Fmean_free_N": np.nan if sd_avg is None else sd_avg, "avg_window_s": avg_window_s, **fz}
    summ = {"status": status, "k_det": k_det, "t_det_s": t_det,
            "det_depth_mm": None if k_det is None else k_det * step,
            "det_tip_z_m": None if k_det is None else plan["tip_z_m"][k_det],
            "tip_force_mean_at_det_N": F_det, "threshold_N": thr_avg, "free_tip_force_mean_sd_N": sd_avg,
            "free_tip_force_sd_N": sd_F, "detect_joints": [j + 1 for j in J],
            "t_current_contact_s": t_cur, "t_stall_s": t_stall, "lead_threshold_mm": lead_thr,
            "motion": motion, "free_speed_mm_s": None if not speeds else float(np.median(speeds)),
            "overshoot_mm": None if k_det is None else max(0.0, (k_max_after - k_det) * step),
            "sd_static_j23_A": [float(sd_static[1]), float(sd_static[2])],
            "sd_free_j23_A": None if sd is None else [float(sd[1]), float(sd[2])]}
    return log_d, summ


# ----------------------------------------------------------------------------- main
def main(argv=None, robot=None, clock=time.monotonic, sleep=time.sleep, ask=input):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plan", required=True)
    ap.add_argument("--trials", type=int, default=1)
    ap.add_argument("--speed-mm-s", type=float, help="override the plan speed (<= 40)")
    ap.add_argument("--speed-pct", type=int, default=20, help="SDK speed percent for move_j (<= 20)")
    ap.add_argument("--hover-only", action="store_true", help="move to the start pose, record 3 s, stop")
    ap.add_argument("--no-confirm-each", action="store_true")
    ap.add_argument("--hold-at-bottom-s", type=float, default=15.0, help="if no contact, hold at max depth this long so the gap can be measured")
    ap.add_argument("--avg-window-s", type=float, default=0.2, help="window of the mean tip force used for the contact rule")
    ap.add_argument("--floor-n", type=float, default=2.0, help="lowest contact threshold on the mean tip force (N)")
    ap.add_argument("--detect-joints", default="2,3", help="joints (1-based) used for the tip force; default j2,j3")
    ap.add_argument("--motion", choices=["stream", "line", "alt"], default="stream",
                    help="stream: move_j every 25 ms (v2.3); line: one move_l; alt: trial 0 stream, 1 line, 2 stream, ...")
    ap.add_argument("--line-speed-pct", type=int, help="line mode: SDK speed percent for the move_l only (the approach and "
                    "the return to the start keep --speed-pct); 1 percent is about 4 mm/s")
    ap.add_argument("--max-line-speed", type=float, default=45.0, help="line mode: stop if the measured speed is above this (mm/s)")
    ap.add_argument("--test-stop-mm", type=float, help="free-air test: declare a fake contact at this depth and back off")
    ap.add_argument("--operator-confirmed", action="store_true")
    ap.add_argument("--pyagx", default=PYAGX_DEFAULT)
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    plan = json.loads(Path(a.plan).read_text())
    if plan.get("kind") != "t7_guarded_descent":
        raise SystemExit("not a T7 plan")
    speed = a.speed_mm_s or plan["speed_mm_s"]
    if a.motion == "alt" and a.line_speed_pct is None:
        raise SystemExit("--motion alt needs --line-speed-pct (the stream trials use --speed-mm-s)")
    if a.line_speed_pct is not None and not 1 <= a.line_speed_pct <= MAX_SPEED_PCT:
        raise SystemExit(f"--line-speed-pct must be 1..{MAX_SPEED_PCT}")
    if not 0 < speed <= MAX_SPEED_MM_S or not 1 <= a.speed_pct <= MAX_SPEED_PCT:
        raise SystemExit(f"speed must be in (0, {MAX_SPEED_MM_S}] mm/s and speed-pct in [1, {MAX_SPEED_PCT}]")
    if not a.operator_confirmed:
        raise SystemExit("this moves the arm: rerun with --operator-confirmed once the operator holds the e-stop")
    out = Path(a.out or Path.home() / "piper_hw_runs" / f"T7_{datetime.now():%Y%m%d_%H%M%S}")
    out.mkdir(parents=True, exist_ok=False)

    if robot is None:
        robot = connect(plan["arm"], a.pyagx)
    q_now, _, _, _ = read_state(robot)
    q0 = np.asarray(plan["q"][0])
    print(f"[{plan['arm']}] now   [deg]: {np.round(np.degrees(q_now), 2).tolist()}")
    print(f"[{plan['arm']}] start [deg]: {np.round(np.degrees(q0), 2).tolist()}  (max diff {np.degrees(np.nanmax(np.abs(q_now-q0))):.1f} deg)")
    print(f"plan: gap {plan['gap_mm']} mm, max depth {plan['max_depth_mm']} mm below start, {speed} mm/s, "
          f"armed after {plan['arm_after_mm']} mm, tool tilt {plan['tool_tilt_from_down_deg']} deg, trials {a.trials}")
    print("check: gripper CLOSED, plane under the tip, nothing else within reach, e-stop in hand.")
    if ask("type YES to start: ").strip() != "YES":
        raise SystemExit("aborted")

    robot.set_speed_percent(a.speed_pct)
    settle(robot, q0, clock, sleep)
    hov = record_static(robot, 3.0, clock, sleep)
    np.savez_compressed(out / "hover.npz", **hov)
    print(f"  at start; static current spread j2 {np.nanstd(hov['current'][:,1]):.4f} A, j3 {np.nanstd(hov['current'][:,2]):.4f} A")
    summary = {"plan": Path(a.plan).name, "arm": plan["arm"], "speed_mm_s": speed, "speed_pct": a.speed_pct, "motion": a.motion, "line_speed_pct": a.line_speed_pct,
               "test_stop_mm": a.test_stop_mm,
               "started": datetime.now().isoformat(timespec="seconds"), "trials": []}
    if not a.hover_only:
        for n in range(a.trials):
            if not a.no_confirm_each and ask(f"trial {n}: y to descend, anything else to stop: ").strip().lower() != "y":
                break
            log_d, s = descend(robot, plan, speed, clock, sleep, hold_bottom_s=a.hold_at_bottom_s,
                               avg_window_s=a.avg_window_s, floor_N=a.floor_n,
                               detect_joints=tuple(int(x) - 1 for x in a.detect_joints.split(",")),
                               motion=("stream", "line")[n % 2] if a.motion == "alt" else a.motion,
                               test_stop_mm=a.test_stop_mm,
                               max_line_speed=a.max_line_speed, line_pct=a.line_speed_pct, restore_pct=a.speed_pct)
            np.savez_compressed(out / f"trial_{n:02d}.npz", **log_d)
            np.savetxt(out / f"fz_{n:02d}.csv", np.c_[log_d["t"], log_d["fz_phase"], log_d["fz_depth_mm"], log_d["fz_tip_z_mm"],
                                                   log_d["fz_cmd_depth_mm"], log_d["fz_F_N"], log_d["fz_F_mean_N"], log_d["fz_phi"]],
                       delimiter=",", fmt="%.4f", header="t_s,phase,depth_mm,tip_z_mm,cmd_depth_mm,F_N,F_mean_N,phi", comments="")
            s["trial"] = n
            s["wall_time"] = datetime.now().isoformat(timespec="seconds")
            summary["trials"].append(s)
            (out / "summary.json").write_text(json.dumps(summary, indent=1))
            print(f"  trial {n}: {s['status']}" + (f" at {s['det_depth_mm']:.2f} mm below start" if s["k_det"] is not None else "")
                  + (f", overshoot {s['overshoot_mm']:.2f} mm" if s.get("overshoot_mm") is not None else "")
                  + (f", free speed {s['free_speed_mm_s']:.1f} mm/s" if s.get("free_speed_mm_s") else ""))
            settle(robot, q0, clock, sleep)
            if s["status"] in ("interrupted", "timeout", "max_depth", "stall", "contact_late", "too_fast"):
                break
            if s["status"] == "test_stop" and (s.get("overshoot_mm") or 0) > 3.0:
                print(f"  back-off overshoot {s['overshoot_mm']:.1f} mm > 3 mm: move_l is not interrupted fast enough, stopping")
                break
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(f"done -> {out}\ncopy this folder to the server's hw_scripts/t7/runs/ (see README).")
    return out


if __name__ == "__main__":
    main()
