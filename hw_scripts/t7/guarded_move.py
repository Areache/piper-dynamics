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
  11. v2.5 (logging only, control identical to v2.4): every cycle also stores the feedback time, the expected
     gravity current at the current waypoint, the threshold in use and the detector state (0 blind = first
     20 mm, 1 monitoring, 2 stop decided, 3 backing off, 4 done); every command sent (move_j / move_l) goes to a
     command log with its send time; motor temperature / status fields are stored if the feedback messages
     already carry them. After the stop, logging continues until the arm is back at the back-off target and
     still for 0.5 s (at least 1 s, at most 5 s), so the loading and the unloading branch are both recorded.
     Each trial also writes trial_XX.json (metadata, field legend). Tip x, y, z are computed offline from q on
     the server (t7_rays.py): the robot PC has no kinematics library.
  13. v2.5.2: wait for complete, fresh and stable startup feedback before confirmation. Reject joint
     feedback far from the planned descent instead of treating its nearest waypoint as actual depth.
  14. v2.6.0 (opt-in only): --air-calibrate records a completely empty, clearance-checked path with the
     legacy force estimator logged but NOT used to stop. All position/feedback/timeout protections remain.
     air_baseline.py builds a candidate from three complete passes. --air-baseline + --air-validate tests
     it independently WITH force stopping enabled. Only a qualified baseline can be used with objects.
     The baseline is frozen, bound to the exact plan/speed/setup, and only the free-window offset is aligned.
  15. v2.6.1: baseline validation/object runs finish at the final command within one waypoint and 0.1 deg
     per joint. Calibration still requires full measured coverage; an old timeout cannot be qualified.
  17. v2.6.3: --max-depth-mm caps a qualified object run at a shorter prefix without changing
     its plan/profile binding. It never expands coverage and disables the bottom measurement dwell.
  18. v2.6.4: share the endpoint rule with offline qualification: final command sent,
     <=0.1 deg error on every joint and <=0.5 mm nearest-waypoint index lag. The index is
     not a Cartesian depth measurement. New validation logs must prove the final command;
     training coverage and commanded limits remain unchanged. Historical timeouts stay failed.
  16. v2.6.2: if successful calibration stops one waypoint short, build only the common measured prefix
     and cap every later target at that prefix. No baseline samples or commands are extrapolated.
  12. v2.5.1: interruption, timeout and protection stops never return to the start or begin another trial.
     Timeout holds the measured pose in both motion modes. Interrupts/timeouts during approach, post-stop
     logging or return are also caught and held; invalid joint feedback uses the damped emergency stop.
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
from air_baseline import (BaselineError, binding, endpoint, expected_residual, load_profile,
                          baseline_endpoint_reached, ENDPOINT_POLICY, ENDPOINT_LAG_MM,
                          ENDPOINT_JOINT_DEG)

PYAGX_DEFAULT = os.environ.get("PYAGX", "")
TRIAL_LEGEND = {
    "state": "0 blind (first 20 mm, friction only), 1 monitoring, 2 stop decided on this row, 3 backing off, 4 done (still)",
    "cmd_kind": "1 stream move_j, 2 back-off move_j, 3 back-off re-send, 4 hold at bottom, 5 hold after interruption/timeout, 6 move_l",
    "t": "cycle start (s, monotonic, from the start of the descent)", "t_fb": "feedback received (s, same clock)",
    "cmd_idx": "index into cmd_* of the last command sent before this row (-1 none)",
    "k_meas": "waypoint nearest the measured q (frozen at the stop on rows after it, as in v2.4)",
    "k_now": "waypoint nearest the measured q on every row, including after the stop",
    "i_grav": "expected gravity current at k_now (A)", "thr_row_N": "contact threshold in use (N, NaN before arming)",
    "fz_*": "tip force along the whole descent (see full_descent_fz)",
    "tip_xyz": "not stored: computed from q on the server (t7_rays.py)"}
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


EXTRA_KEYS = ("temp", "status", "err", "collision", "stall")   # motor fields kept if the messages carry them


def _extras(obj, prefix, out):
    for name in dir(obj):
        if name.startswith("_") or not any(k in name.lower() for k in EXTRA_KEYS):
            continue
        try:
            val = getattr(obj, name)
        except Exception:
            continue
        if isinstance(val, (bool, int, float, np.integer, np.floating)):
            out[prefix + name] = float(val)


def read_state(robot, extra=None):
    """q (joint feedback, rad), motor position (rad, same frame as current), velocity (rad/s), current (A).
    extra (dict): filled with numeric temperature / status fields found on the messages already read, and the
    message timestamp if it has one; no additional SDK calls."""
    ja = robot.get_joint_angles()
    q = np.asarray(ja.msg, float) if ja is not None else np.full(6, np.nan)
    mp, v, c = np.full(6, np.nan), np.full(6, np.nan), np.full(6, np.nan)
    for i in range(6):
        ms = robot.get_motor_states(i + 1)
        if ms is not None:
            mp[i], v[i], c[i] = ms.msg.position, ms.msg.velocity, ms.msg.current
            if extra is not None:
                _extras(ms.msg, f"m{i + 1}_", extra)
                stamp = getattr(ms, 'timestamp', None)
                if isinstance(stamp, (int, float)):
                    extra[f'm{i + 1}_stamp'] = stamp
    if extra is not None and ja is not None:
        for name in ("timestamp", "time_stamp", "stamp"):
            val = getattr(ja, name, None)
            if isinstance(val, (int, float)):
                extra["fb_stamp"] = float(val)
    return q, mp, v, c


class FeedbackNotReady(RuntimeError):
    """No motion has been authorized with usable startup feedback."""


class OffPlanFeedback(RuntimeError):
    """The measured joints cannot establish a position on this descent."""


def check_air_feedback(robot, state, extra):
    """New air modes require finite live joint/motor feedback throughout the descent."""
    if not all(x.shape == (6,) and np.isfinite(x).all() for x in state):
        raise OffPlanFeedback('invalid joint/motor state during air-baseline operation')
    parser = getattr(robot, '_parser', None)
    if parser is not None:
        pairs = [getattr(parser, name, None) for name in ('joint_12', 'joint_34', 'joint_56')]
        stamps = [getattr(pair, 'timestamp', float('nan')) for pair in pairs]
        motor_stamps = [extra.get(f'm{i}_stamp', float('nan')) for i in range(1, 7)]
        now = time.time()
        if (not all(np.isfinite(t) and 0 <= now-t <= .2 for t in stamps + motor_stamps)
                or max(stamps)-min(stamps) > .03):
            raise OffPlanFeedback('stale/incomplete joint or motor feedback during air-baseline operation')


def wait_for_feedback(robot, clock, sleep, timeout=3.0):
    """Read only: require 0.25 s of complete stable feedback, allowing genuine zero angles."""
    t0 = clock()
    good_since = None
    anchor = None
    while clock() - t0 < timeout:
        state = read_state(robot)
        valid = all(x.shape == (6,) and np.isfinite(x).all() for x in state)
        parser = getattr(robot, "_parser", None)
        if parser is not None:
            # PyAgxArm's aggregate getter can return zero-filled missing pairs.
            pairs = [getattr(parser, name, None) for name in ("joint_12", "joint_34", "joint_56")]
            stamps = [getattr(pair, "timestamp", 0.0) for pair in pairs]
            now = time.time()  # python-can receive timestamps use wall time
            valid = valid and all(pair is not None for pair in pairs)
            valid = valid and all(np.isfinite(t) and 0 <= now - t <= 0.2 for t in stamps)
            valid = valid and max(stamps) - min(stamps) <= 0.03
        q = state[0]
        if not valid:
            good_since = anchor = None
        elif anchor is None or np.max(np.abs(q - anchor)) > np.radians(0.5):
            good_since, anchor = clock(), q.copy()
        elif clock() - good_since >= 0.25:
            return state
        sleep(0.01)
    raise FeedbackNotReady("joint/motor feedback incomplete, stale or unstable for 3 s; no motion sent")


def locate_on_plan(Q, q, lo, hi):
    if q.shape != (6,) or not np.isfinite(q).all():
        raise OffPlanFeedback("invalid joint feedback during descent")
    errors = np.max(np.abs(Q[lo:hi] - q), axis=1)
    k = int(np.argmin(errors))
    if errors[k] > 0.05:
        raise OffPlanFeedback(f"joint feedback is {errors[k]:.3f} rad off the descent; depth is unknown")
    return lo + k


def hold_measured_position(robot, send=None):
    """Hold a valid measured joint pose; never send NaN or a cached plan target."""
    try:
        ja = robot.get_joint_angles()
        q = np.asarray(ja.msg, float) if ja is not None else np.full(6, np.nan)
    except Exception:
        robot.electronic_emergency_stop()
        raise
    if q.shape != (6,) or not np.all(np.isfinite(q)):
        robot.electronic_emergency_stop()
        raise RuntimeError("invalid joint feedback: damped emergency stop sent")
    target = [float(x) for x in q]
    (robot.move_j if send is None else send)(target)
    return target


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


def full_descent_fz(L, aux, plan, mu, sd, A, J, lp_n, avg_window_s, n_post, air_profile=None):
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
            if air_profile is not None:
                if i >= n - n_post:
                    continue  # downward air profile is invalid during the upward back-off
                lo = max(0, i-lp_n+1)
                rl = np.mean([R[j]-expected_residual(air_profile, int(k[j])) for j in range(lo, i+1)], axis=0)
                F[i] = tip_force(rl, mu, sd, A[k[i]], J)[0]
            else:
                rl = np.nanmean(R[max(0, i - lp_n + 1): i + 1], 0)
                F[i] = tip_force(rl, mu * phi[i], sd, A[k[i]], J)[0]
    Fm = np.full(n, np.nan)
    j0 = 0
    for i in range(n):
        while t[i] - t[j0] > avg_window_s:
            j0 += 1
        if t[i] - t[j0] >= 0.8 * avg_window_s:
            Fm[i] = np.nanmean(F[j0: i + 1]) if np.isfinite(F[j0:i+1]).any() else np.nan
    if air_profile is not None and n_post:
        Fm[-n_post:] = np.nan
    return {"fz_depth_mm": k * step, "fz_tip_z_mm": zt[k] * 1000.0, "fz_cmd_depth_mm": L[:, 1] * step,
            "fz_F_N": F, "fz_F_mean_N": Fm, "fz_phi": phi, "fz_phase": phase}


def runtime_endpoint(plan, profile, max_depth_mm=None):
    """Keep the full plan/profile binding; optionally stop at a shorter prefix."""
    k = endpoint(plan, profile)
    if max_depth_mm is None:
        return k
    if (profile is None or not np.isfinite(max_depth_mm) or
            not plan['arm_after_mm'] + 3 < max_depth_mm <= k * plan['step_mm']):
        raise BaselineError('max-depth must exceed arming + 3 mm and stay within qualified plan coverage')
    capped = int(np.floor(max_depth_mm / plan['step_mm']))
    if capped * plan['step_mm'] <= plan['arm_after_mm'] + 3:
        raise BaselineError('rounded max-depth leaves no detection interval')
    return capped


def descend(robot, plan, speed_mm_s, clock, sleep, loop_hz=100.0, send_hz=50.0, log=print, hold_bottom_s=0.0,
            avg_window_s=0.2, floor_N=2.0, detect_joints=(1, 2), motion="stream", test_stop_mm=None,
            max_line_speed=45.0, line_pct=None, restore_pct=None,
            air_calibration=False, air_validation=False, air_profile=None, max_depth_mm=None):
    if plan.get("version", 1) < 2 or plan["detect"].get("mode") != "tip_force":
        raise SystemExit("plan is version 1 (single-joint thresholds, false-triggers on j2 stick-slip): regenerate it with t7_plan.py")
    Q = np.asarray(plan["q"]); IE = np.asarray(plan["i_expected_A"]); A = np.asarray(plan["di_per_N_up"]); step = plan["step_mm"]
    det = plan["detect"]; J = list(detect_joints) if detect_joints else det["joints"]; lp_n = det["lp_n"]
    arm_after, max_depth = plan["arm_after_mm"], plan["max_depth_mm"]
    if max_depth_mm is not None and (air_calibration or air_validation or motion != 'stream'):
        raise BaselineError('runtime max-depth is only for qualified stream object runs')
    k_max = runtime_endpoint(plan, air_profile, max_depth_mm)
    if max_depth_mm is not None:
        hold_bottom_s = 0.0  # no stationary measurement dwell at a capped object-test endpoint
    if air_profile is not None:
        max_depth = k_max*step  # never command beyond the common measured calibration prefix
    retract_k = int(round(plan["retract_mm"] / step))

    base = record_static(robot, 1.0, clock, sleep)
    i_base = np.nanmean(base["current"], 0)
    sd_static = np.nanstd(base["current"], 0)
    if (air_calibration or air_validation or air_profile is not None) and not np.isfinite(base['current']).all():
        raise OffPlanFeedback('invalid static currents before air-baseline descent')

    rows, free, recent = [], [], []
    aux = []                  # per logged row: (phi, measured waypoint) -> full-descent F(z) after the trial
    ext = []                  # per logged row: t_fb, last command index, state, waypoint now, threshold, i_grav(6)
    mex = []                  # per logged row: dict of motor extras (may stay empty)
    cmds = []                 # every command: (t_send, kind, 6 values); kind 1 stream, 2 back-off, 3 re-send,
                              # 4 hold at bottom, 5 hold after interruption/timeout, 6 move_l (flange pose)
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
    line_target = None
    speed_ref = speed_mm_s; speeds = []; t5 = None; t_slow = None; k_stop = None; q_hold = None
    line = motion == "line"
    air_mode = air_calibration or air_validation or air_profile is not None
    if air_mode and line:
        raise BaselineError('air baseline supports stream motion only')
    t_fast = None
    if line:
        fp = robot.get_flange_pose()
        if fp is None:
            raise SystemExit("no flange pose feedback: cannot plan the move_l")
        target = [float(x) for x in fp.msg]
        target[2] -= max_depth / 1000.0
        line_target = target
    t0 = clock()

    def send_j(qv, kind):
        robot.move_j(qv)
        cmds.append((clock() - t0, kind, [float(x) for x in qv]))

    timeout = 60.0 if line else max_depth / speed_mm_s + 10.0
    if line:
        if line_pct:
            robot.set_speed_percent(line_pct)          # only the move_l runs at this percent
        robot.move_l(line_target)
        cmds.append((clock() - t0, 6, [float(x) for x in line_target]))
    try:
        while True:
            t = clock() - t0
            k_want = min(k_max, int(speed_mm_s * t / step))
            if not line and k_want != k_cmd and t - last_send >= 1.0 / send_hz:
                k_cmd = k_want
                send_j([float(x) for x in Q[k_cmd]], 1)
                last_send = t
            mx = {}
            q, mp, v, c = read_state(robot, mx)
            if air_mode:
                check_air_feedback(robot, (q, mp, v, c), mx)
            t_fb = clock() - t0
            lo, hi = max(0, k_meas - 20), min(len(Q), k_meas + 40)
            k_meas = locate_on_plan(Q, q, lo, hi)
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
            corrected = r - expected_residual(air_profile, k_meas) if air_profile is not None else r
            recent.append(corrected); recent = recent[-lp_n:]
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
                        send_j([float(x) for x in Q[k_stop]], 2)
                        log(f"    move_l at {speed_ref:.1f} mm/s > {max_line_speed:.0f} mm/s: stopping, lower --speed-pct")
                        break
                    log(f"    armed at {s_meas:.1f} mm: free-descent spread j2 {sd[1]:.3f} A, j3 {sd[2]:.3f} A; "
                        f"{avg_window_s:.1f} s mean tip-force spread {sd_avg:.2f} N -> threshold {thr_avg:.2f} N")
            hit = False
            stall = False
            if armed:
                Fk, ex = tip_force(rl, mu if air_profile is not None else mu * phi, sd, A[k_meas], J)
                fwin.append((t, Fk))
                fwin = [w for w in fwin if t - w[0] <= avg_window_s]
                F_avg = float(np.mean([w[1] for w in fwin])) if t - fwin[0][0] >= 0.8 * avg_window_s else np.nan
                # Only the separately confirmed EMPTY calibration disables this unreliable
                # contact estimator. Validation and object operation retain contact stopping.
                hit = bool(not air_calibration and np.isfinite(F_avg) and F_avg > thr_avg)
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
            ext.append([t_fb, len(cmds) - 1, 1 if armed else 0, k_meas, np.nan if thr_avg is None else thr_avg, *IE[k_meas]])
            mex.append(mx)
            if air_mode:
                fast = v_tip > max(1.5 * speed_mm_s, speed_mm_s + 5)
                t_fast = (t if t_fast is None else t_fast) if fast else None
                if t_fast is not None and t-t_fast >= .15:
                    status, k_det, t_det = 'too_fast', k_meas, t
                    k_stop = max(0, k_meas-retract_k)
                    send_j([float(x) for x in Q[k_stop]], 2)
                    break
            if test_stop_mm is not None and s_meas >= test_stop_mm and not (hit or stall):
                status, k_det, t_det, F_det = "test_stop", k_meas, t, F_avg
                k_stop = max(0, k_meas - retract_k)
                send_j([float(x) for x in Q[k_stop]], 2)
                break
            if hit or stall:
                status = "contact" if hit else "stall"
                k_det, t_det, F_det = k_meas, t, F_avg
                t_cur, t_stall = (t if hit else None), (t if stall else None)
                k_stop = max(0, k_meas - retract_k)
                send_j([float(x) for x in Q[k_stop]], 2)
                break
            at_endpoint = (baseline_endpoint_reached(k_cmd, k_meas, k_max, q, Q, step)
                           if air_profile is not None else
                           ((k_cmd >= k_max or line) and s_meas >= max_depth - step))
            if at_endpoint:
                if air_calibration or air_validation:
                    status = 'air_complete' if air_calibration else 'air_validated'
                    k_det, t_det = k_meas, t
                    k_stop = max(0, k_meas-retract_k)
                    send_j([float(x) for x in Q[k_stop]], 2)
                    break
                if np.isfinite(F_avg) and F_avg > 1.0:
                    status, k_det, t_det, F_det = "contact_late", k_meas, t, F_avg
                    k_stop = max(0, k_meas - retract_k)
                    send_j([float(x) for x in Q[k_stop]], 2)
                    log(f"    max depth reached while pressing (mean tip force {F_avg:.1f} N): backing off")
                    break
                status = "max_depth"
                if hold_bottom_s > 0:
                    log(f"    reached max depth without contact: holding {hold_bottom_s:.0f} s, MEASURE the remaining tip-to-plate gap now")
                    t_h = clock()
                    while clock() - t_h < hold_bottom_s:
                        send_j([float(x) for x in Q[k_meas if line else k_cmd]], 4); sleep(0.05)
                break
            if t > timeout:
                status = "timeout"
                k_stop = None
                q_hold = hold_measured_position(robot, lambda target: send_j(target, 5))
                break
            sleep(1.0 / loop_hz)
    except KeyboardInterrupt:
        k_stop = None                  # an earlier back-off must not overwrite this hold
        q_hold = hold_measured_position(robot, lambda target: send_j(target, 5))
        status = "interrupted"
    # keep logging for 1 s after the stop (servo settling, the pressing force decaying); re-send the back-off target
    # so it also overrides a running move_l
    if k_stop is not None:
        q_hold = [float(x) for x in Q[k_stop]]
    t1 = clock(); k_after = k_max_after = k_meas; last_rs = -1.0
    n_pre_post = len(rows)
    if ext and status in ("contact", "stall", "test_stop", "contact_late", "too_fast", "air_complete", "air_validated"):
        ext[-1][2] = 2                                 # the row on which the stop was decided
    qwin = []                                          # (t, q) over the last 0.5 s, for "still"
    post_end = "cap"
    while True:
        el = clock() - t1
        if el >= 5.0:
            break
        t = clock() - t0
        if q_hold is not None and t - last_rs >= 0.1:
            robot.move_j(q_hold); last_rs = t
            cmds.append((t, 3, list(q_hold)))
        mx = {}
        q, mp, v, c = read_state(robot, mx)
        if air_mode:
            check_air_feedback(robot, (q, mp, v, c), mx)
        t_fb = clock() - t0
        if np.all(np.isfinite(q)):
            lo, hi = max(0, k_after - 20), min(len(Q), k_after + 40)
            k_after = locate_on_plan(Q, q, lo, hi)
            if el < 1.0:                               # overshoot over the first second, as in v2.4
                k_max_after = max(k_max_after, k_after)
            qwin.append((t, q.copy())); qwin = [w for w in qwin if t - w[0] <= 0.5]
        r = (c - i_base) - (IE[k_meas] - IE[0])
        recent.append(r); recent = recent[-lp_n:]
        Fk, ex = (tip_force(np.nanmean(np.array(recent), 0), mu, sd, A[k_meas], J)
                  if mu is not None and air_profile is None else (np.nan, np.nan))
        rows.append(np.concatenate([[t, k_cmd, k_meas, float(mu is not None), 0.0, Fk, ex, np.nan], q, mp, v, c, r]))
        aux.append((0.0, k_after))           # backing off: no friction model for upward motion, phi = 0
        ext.append([t_fb, len(cmds) - 1, 3, k_after, np.nan if thr_avg is None else thr_avg, *IE[k_after]])
        mex.append(mx)
        reached = k_stop is None or k_after <= k_stop + 1
        still = (len(qwin) > 5 and t - qwin[0][0] >= 0.45
                 and float(np.max(np.abs(np.array([w[1] for w in qwin]) - qwin[0][1]))) < 5e-4)
        if el >= 1.0 and reached and still:
            post_end = "still"
            break
        sleep(1.0 / loop_hz)
    E = np.array(ext, float)
    if post_end == "still":
        E[(E[:, 2] == 3) & (E[:, 0] >= E[-1, 0] - 0.5), 2] = 4
    if line and line_pct and restore_pct:
        robot.set_speed_percent(restore_pct)          # back to the approach speed for the return to the start
    L = np.array(rows)
    fz = full_descent_fz(L, np.array(aux, float), plan, mu, sd, A, J, lp_n, avg_window_s,
                         n_post=len(rows) - n_pre_post, air_profile=air_profile)
    log_d = {"t": L[:, 0], "k_cmd": L[:, 1], "k_meas": L[:, 2], "armed": L[:, 3], "hit": L[:, 4],
             "tip_force_N": L[:, 5], "explained": L[:, 6], "tip_force_mean_N": L[:, 7],
             "q": L[:, 8:14], "mpos": L[:, 14:20], "v": L[:, 20:26], "current": L[:, 26:32], "resid": L[:, 32:38],
             "i_base": i_base, "sd_static": sd_static, "mu_free": mu if mu is not None else np.full(6, np.nan),
             "sd_free": sd if sd is not None else np.full(6, np.nan),
             "thr_N": np.nan if thr_avg is None else thr_avg, "sd_F_free_N": np.nan if sd_F is None else sd_F,
             "sd_Fmean_free_N": np.nan if sd_avg is None else sd_avg, "avg_window_s": avg_window_s, **fz,
             "t_fb": E[:, 0], "cmd_idx": E[:, 1].astype(int), "state": E[:, 2].astype(int), "k_now": E[:, 3].astype(int),
             "thr_row_N": E[:, 4], "i_grav": E[:, 5:11],
             "cmd_t": np.array([x[0] for x in cmds]), "cmd_kind": np.array([x[1] for x in cmds], int),
             "cmd_val": np.array([x[2] for x in cmds]).reshape(-1, 6)}
    mkeys = sorted({k for d in mex for k in d})
    for kname in mkeys:
        log_d["mx_" + kname] = np.array([d.get(kname, np.nan) for d in mex])
    log_d['air_baseline_active'] = np.array(air_profile is not None)
    if air_profile is not None:
        baseline_rows = np.full((len(L), 6), np.nan)
        for i in range(n_pre_post):
            baseline_rows[i] = expected_residual(air_profile, int(E[i, 3]))
        log_d['air_baseline_expected_A'] = baseline_rows
        log_d['air_baseline_offset_A'] = mu if mu is not None else np.full(6, np.nan)
    summ = {"status": status, "k_det": k_det, "t_det_s": t_det,
            "det_depth_mm": None if k_det is None else k_det * step,
            "det_tip_z_m": None if k_det is None else plan["tip_z_m"][k_det],
            "tip_force_mean_at_det_N": F_det, "threshold_N": thr_avg, "free_tip_force_mean_sd_N": sd_avg,
            "free_tip_force_sd_N": sd_F, "detect_joints": [j + 1 for j in J],
            "t_current_contact_s": t_cur, "t_stall_s": t_stall, "lead_threshold_mm": lead_thr,
            "motion": motion, "free_speed_mm_s": None if not speeds else float(np.median(speeds)),
            "overshoot_mm": None if k_det is None else max(0.0, (k_max_after - k_det) * step),
            "sd_static_j23_A": [float(sd_static[1]), float(sd_static[2])],
            "sd_free_j23_A": None if sd is None else [float(sd[1]), float(sd[2])],
            "version": "2.6.4", "post_end": post_end,
            "post_s": float(E[-1, 0] - E[n_pre_post, 0]) if len(E) > n_pre_post else 0.0,
            "n_rows": int(len(E)), "n_cmds": len(cmds), "motor_extra_fields": mkeys,
            "line_target_flange": line_target,
            "force_model": 'same_path_air_baseline' if air_profile is not None else 'legacy_free_window',
            "air_mode": 'calibration' if air_calibration else ('validation' if air_validation else None),
            "baseline_profile_sha256": air_profile['profile_sha256'] if air_profile is not None else None}
    if air_profile is not None:
        summ['endpoint_policy'] = ENDPOINT_POLICY
        summ['endpoint_tolerance_mm'] = ENDPOINT_LAG_MM
        summ['endpoint_joint_tolerance_deg'] = ENDPOINT_JOINT_DEG
        summ['commanded_endpoint_mm'] = k_max*step
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
    ap.add_argument('--air-calibrate', action='store_true', help='EMPTY path only: record full path; force estimator is diagnostic, not a contact stop')
    ap.add_argument('--air-baseline', help='same-path baseline JSON (must be qualified unless --air-validate)')
    ap.add_argument('--max-depth-mm', type=float, help='qualified stream object runs only: cap commands at a shorter plan prefix, rounded DOWN to a waypoint; never extends baseline coverage; no bottom dwell')
    ap.add_argument('--air-validate', action='store_true', help='one independent EMPTY run with baseline and force stops enabled')
    ap.add_argument('--air-clearance-mm', type=float, help='operator-confirmed lower bound on clearance from start tip to nearest surface anywhere along the EMPTY path')
    ap.add_argument('--baseline-context', help='unique token for this uninterrupted powered session and unchanged unloaded closed gripper/setup')
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
    a.air_profile = None
    advanced = a.air_calibrate or a.air_validate or a.air_baseline is not None
    if advanced:
        if a.air_calibrate and (a.air_baseline or a.air_validate):
            raise SystemExit('--air-calibrate cannot use a baseline or validation mode')
        if a.air_validate and not a.air_baseline:
            raise SystemExit('--air-validate requires --air-baseline')
        if (a.motion != 'stream' or a.hover_only or a.test_stop_mm is not None or
                a.detect_joints != '2,3' or a.avg_window_s != .2 or a.floor_n != 2 or
                not 0 < speed <= 10 or not 1 <= a.trials <= 3):
            raise SystemExit('air baseline requires stream <=10 mm/s, joints 2,3, window .2 s, floor 2 N, 1..3 trials; no hover/test-stop')
        if a.no_confirm_each:
            raise SystemExit('air-baseline operations require confirmation before each trial')
        if a.air_validate and a.trials != 1:
            raise SystemExit('independent validation uses exactly one trial')
        if a.air_calibrate or a.air_validate:
            if (a.air_clearance_mm is None or not np.isfinite(a.air_clearance_mm) or
                    a.air_clearance_mm < plan['max_depth_mm'] + 30):
                raise SystemExit('EMPTY path needs --air-clearance-mm at least max depth + 30 mm; remove all objects first')
        try:
            expected = binding(plan, speed, a.speed_pct, a.baseline_context)
            if a.air_baseline:
                a.air_profile = load_profile(a.air_baseline, expected, allow_candidate=a.air_validate)
                endpoint(plan, a.air_profile)  # validate common measured prefix before any movement
        except (BaselineError, ValueError, KeyError) as exc:
            raise SystemExit(str(exc))
    elif a.air_clearance_mm is not None or a.baseline_context is not None:
        raise SystemExit('air-clearance/context options require an air-baseline mode')
    if a.max_depth_mm is not None:
        if a.air_profile is None or a.air_calibrate or a.air_validate or a.motion != 'stream':
            raise SystemExit('--max-depth-mm requires a qualified baseline in ordinary stream object mode')
        try:
            runtime_endpoint(plan, a.air_profile, a.max_depth_mm)
        except BaselineError as exc:
            raise SystemExit(str(exc))
    out = Path(a.out or Path.home() / "piper_hw_runs" / f"T7_{datetime.now():%Y%m%d_%H%M%S}")
    out.mkdir(parents=True, exist_ok=False)
    if advanced:
        (out/'plan_snapshot.json').write_text(json.dumps(plan, indent=1))
        if a.air_profile is not None:
            (out/'baseline_snapshot.json').write_text(json.dumps(a.air_profile, indent=1))

    if robot is None:
        robot = connect(plan["arm"], a.pyagx)
    # Startup readiness failure must not issue a hold, mode, enable or movement command.
    print("Waiting for complete, stable feedback ...")
    initial_state = wait_for_feedback(robot, clock, sleep)
    try:
        return run_trials(a, plan, speed, out, robot, clock, sleep, ask, initial_state)
    except OffPlanFeedback as exc:
        robot.electronic_emergency_stop()
        (out / "abort.json").write_text(json.dumps({"status": "invalid_feedback", "detail": str(exc),
            "version": "2.6.3", "wall_time": datetime.now().isoformat(timespec="seconds")}, indent=1))
        raise
    except (KeyboardInterrupt, TimeoutError, BaselineError) as exc:
        # Covers approach, baseline, post-stop logging, and return as well as descent.
        status = ('interrupted' if isinstance(exc, KeyboardInterrupt) else
                  ('baseline_invalid' if isinstance(exc, BaselineError) else 'timeout'))
        abort = {"status": status, "detail": str(exc), "version": "2.6.3",
                 "wall_time": datetime.now().isoformat(timespec="seconds")}
        try:
            abort["hold_q_rad"] = hold_measured_position(robot)
        finally:
            (out / "abort.json").write_text(json.dumps(abort, indent=1))
        raise


def run_trials(a, plan, speed, out, robot, clock, sleep, ask, initial_state):
    q_now, _, _, _ = initial_state
    q0 = np.asarray(plan["q"][0])
    print(f"[{plan['arm']}] now   [deg]: {np.round(np.degrees(q_now), 2).tolist()}")
    print(f"[{plan['arm']}] start [deg]: {np.round(np.degrees(q0), 2).tolist()}  (max diff {np.degrees(np.nanmax(np.abs(q_now-q0))):.1f} deg)")
    print(f"plan: gap {plan['gap_mm']} mm, max depth {plan['max_depth_mm']} mm below start, {speed} mm/s, "
          f"armed after {plan['arm_after_mm']} mm, tool tilt {plan['tool_tilt_from_down_deg']} deg, trials {a.trials}")
    if a.air_profile is not None:
        print(f"baseline coverage: {endpoint(plan, a.air_profile)*plan['step_mm']:.2f} mm; "
              f"command limit: {runtime_endpoint(plan, a.air_profile, a.max_depth_mm)*plan['step_mm']:.2f} mm below start")
    if a.max_depth_mm is not None:
        print('Runtime depth cap active; printed plan gap is the source-plan reference, not a measurement of the current object. No bottom dwell.')
    empty = a.air_calibrate or a.air_validate
    if empty:
        print(f'EMPTY workspace: declared clearance {a.air_clearance_mm} mm; remove cup/blocks and keep hands clear, e-stop in hand.')
        print('Force stopping DISABLED for calibration only; position, speed, feedback and timeout guards remain.'
              if a.air_calibrate else 'Independent air validation: force stopping ENABLED; do not place any object.')
    else:
        print("check: gripper CLOSED, plane under the tip, nothing else within reach, e-stop in hand.")
    if a.baseline_context:
        print(f'Setup {a.baseline_context}: confirm same uninterrupted power, unloaded CLOSED gripper and unchanged hardware/trajectory conditions.')
    confirmation = 'AIR CLEAR' if empty else 'YES'
    if ask(f"type {confirmation} to start: ").strip() != confirmation:
        raise SystemExit("aborted")

    wait_for_feedback(robot, clock, sleep)  # user may have waited at the prompt
    robot.set_speed_percent(a.speed_pct)
    settle(robot, q0, clock, sleep)
    hov = record_static(robot, 3.0, clock, sleep)
    np.savez_compressed(out / "hover.npz", **hov)
    print(f"  at start; static current spread j2 {np.nanstd(hov['current'][:,1]):.4f} A, j3 {np.nanstd(hov['current'][:,2]):.4f} A")
    summary = {"plan": Path(a.plan).name, "arm": plan["arm"], "speed_mm_s": speed, "speed_pct": a.speed_pct, "motion": a.motion, "line_speed_pct": a.line_speed_pct,
               "test_stop_mm": a.test_stop_mm,
               "requested_max_depth_mm": a.max_depth_mm,
               "started": datetime.now().isoformat(timespec="seconds"), "trials": [],
               "air_mode": 'calibration' if a.air_calibrate else ('validation' if a.air_validate else None),
               "baseline_context": a.baseline_context, "air_clearance_mm": a.air_clearance_mm,
               "empty_workspace_confirmed": bool(empty),
               "baseline_profile_sha256": a.air_profile['profile_sha256'] if a.air_profile is not None else None}
    if not a.hover_only:
        for n in range(a.trials):
            if not a.no_confirm_each and ask(f"trial {n}: y to descend, anything else to stop: ").strip().lower() != "y":
                break
            log_d, s = descend(robot, plan, speed, clock, sleep, hold_bottom_s=a.hold_at_bottom_s,
                               avg_window_s=a.avg_window_s, floor_N=a.floor_n,
                               detect_joints=tuple(int(x) - 1 for x in a.detect_joints.split(",")),
                               motion=("stream", "line")[n % 2] if a.motion == "alt" else a.motion,
                               test_stop_mm=a.test_stop_mm,
                               max_line_speed=a.max_line_speed, line_pct=a.line_speed_pct, restore_pct=a.speed_pct,
                               air_calibration=a.air_calibrate, air_validation=a.air_validate, air_profile=a.air_profile,
                               max_depth_mm=a.max_depth_mm)
            np.savez_compressed(out / f"trial_{n:02d}.npz", **log_d)
            np.savetxt(out / f"fz_{n:02d}.csv", np.c_[log_d["t"], log_d["fz_phase"], log_d["fz_depth_mm"], log_d["fz_tip_z_mm"],
                                                   log_d["fz_cmd_depth_mm"], log_d["fz_F_N"], log_d["fz_F_mean_N"], log_d["fz_phi"]],
                       delimiter=",", fmt="%.4f", header="t_s,phase,depth_mm,tip_z_mm,cmd_depth_mm,F_N,F_mean_N,phi", comments="")
            s["trial"] = n
            s["wall_time"] = datetime.now().isoformat(timespec="seconds")
            (out / f"trial_{n:02d}.json").write_text(json.dumps({**s, "plan": Path(a.plan).name, "legend": TRIAL_LEGEND}, indent=1))
            summary["trials"].append(s)
            (out / "summary.json").write_text(json.dumps(summary, indent=1))
            print(f"  trial {n}: {s['status']}" + (f" at {s['det_depth_mm']:.2f} mm below start" if s["k_det"] is not None else "")
                  + (f", overshoot {s['overshoot_mm']:.2f} mm" if s.get("overshoot_mm") is not None else "")
                  + (f", free speed {s['free_speed_mm_s']:.1f} mm/s" if s.get("free_speed_mm_s") else ""))
            # Only a successful trial may initiate a new motion back to q0.
            # Protection stops have already backed off; interrupts/timeouts hold.
            if s["status"] not in ("contact", "test_stop", "air_complete", "air_validated"):
                print(f"  stopping after {s['status']}; no automatic return to start")
                break
            if (a.air_calibrate or a.air_validate or a.air_profile is not None) and s.get('post_end') != 'still':
                print('  back-off did not settle; stopping without return or another trial')
                break
            if (s.get("overshoot_mm") or 0) > 3.0:
                print(f"  back-off overshoot {s['overshoot_mm']:.1f} mm > 3 mm: stopping without return to start")
                break
            settle(robot, q0, clock, sleep)
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(f"done -> {out}\ncopy this folder to the server's hw_scripts/t7/runs/ (see README).")
    return out


if __name__ == "__main__":
    main()
