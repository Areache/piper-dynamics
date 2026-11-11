#!/usr/bin/env python3
"""Passive CAN logger for the slow Piper feedback: motor / driver temperature and the firmware status flags.
READ-ONLY: opens the CAN interface as a listener and never sends a frame, so it can run next to any control
process (guarded_move.py, collect_poses.py, the ROS leader-follower node) without taking control.

What it records (Piper X, frame layout from PyAgxArm `ArmMsgFeedbackLowSpd` / `ArmMsgFeedbackStatus`, big-endian):
  0x261..0x266  per joint, ~10 Hz: bus voltage (0.1 V), driver temperature (int16, degC), motor temperature
                (int8, degC), status byte, bus current (0.001 A). Status bits: 0 voltage low, 1 motor over-temp,
                2 driver over-current, 3 driver over-temp, 4 collision protection, 5 driver error, 6 enabled,
                7 stall protection.
  0x2A1         arm status: control mode, arm status, mode feedback, teach status, motion status, trajectory
                index, 16-bit error code.
Only frames whose content changes are written for 0x2A1; the 0x26x frames are written at most --hz per joint.
Every row carries time.monotonic() (same clock as the motion scripts on this PC) and time.time().

Outputs (--out, default ~/piper_hw_runs/lowspd_<channel>_<stamp>/):
  lowspd.csv   t_mono, t_wall, joint, volt_V, driver_C, motor_C, status, voltage_low, motor_hot, overcurrent,
               driver_hot, collision, driver_err, enabled, stall, bus_A
  status.csv   t_mono, t_wall, ctrl_mode, arm_status, mode_feedback, teach_status, motion_status, traj_num, err_code
  marks.csv    t_mono, t_wall, text   (type a line + Enter while it runs to mark an event, e.g. "power cycle")
  meta.json    channel, start/stop times, frame counts, rate per ID
On exit (Ctrl-C) it prints the temperature range per joint and every time a collision / stall / error bit was set.

usage:
  python hw_scripts/lowspd_log.py can_right --out ~/piper_hw_runs/E3_<date>/lowspd
Needs python-can (installed with PyAgxArm).
"""
from __future__ import annotations

import argparse
import csv
import json
import select
import sys
import time
from datetime import datetime
from pathlib import Path

LOW_IDS = {0x261 + j: j + 1 for j in range(6)}
STATUS_ID = 0x2A1
BITS = ("voltage_low", "motor_hot", "overcurrent", "driver_hot", "collision", "driver_err", "enabled", "stall")
ALARM_BITS = ("collision", "stall", "driver_err", "overcurrent", "motor_hot", "driver_hot", "voltage_low")


def _s(v, bits):
    return v - (1 << bits) if v >= 1 << (bits - 1) else v


def decode_low(data: bytes) -> dict:
    d = bytes(data)
    st = d[5]
    out = {"volt_V": int.from_bytes(d[0:2], "big") * 0.1,
           "driver_C": _s(int.from_bytes(d[2:4], "big"), 16),
           "motor_C": _s(d[4], 8),
           "status": st,
           "bus_A": int.from_bytes(d[6:8], "big") * 1e-3}
    for k, name in enumerate(BITS):
        out[name] = (st >> k) & 1
    return out


def decode_status(data: bytes) -> tuple:
    d = bytes(data)
    return (d[0], d[1], d[2], d[3], d[4], d[5], int.from_bytes(d[6:8], "big"))


def main(argv=None, bus=None, clock=time.monotonic, wall=time.time):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("channel")
    p.add_argument("--out", type=Path)
    p.add_argument("--hz", type=float, default=10.0, help="max rows per joint per second for 0x261..0x266")
    p.add_argument("--duration-s", type=float, default=0.0, help="stop after this many seconds (0 = until Ctrl-C)")
    a = p.parse_args(argv)
    out = a.out or Path.home() / "piper_hw_runs" / f"lowspd_{a.channel}_{datetime.now():%Y%m%d_%H%M%S}"
    out.mkdir(parents=True, exist_ok=False)
    if bus is None:
        import can
        bus = can.Bus(interface="socketcan", channel=a.channel, receive_own_messages=False)
    f_low, f_st, f_mk = (open(out / n, "w", newline="") for n in ("lowspd.csv", "status.csv", "marks.csv"))
    w_low, w_st, w_mk = csv.writer(f_low), csv.writer(f_st), csv.writer(f_mk)
    w_low.writerow(["t_mono", "t_wall", "joint", "volt_V", "driver_C", "motor_C", "status", *BITS, "bus_A"])
    w_st.writerow(["t_mono", "t_wall", "ctrl_mode", "arm_status", "mode_feedback", "teach_status", "motion_status",
                   "traj_num", "err_code"])
    w_mk.writerow(["t_mono", "t_wall", "text"])
    t0m, t0w = clock(), wall()
    last_row = {j: -1e9 for j in range(1, 7)}
    last_bits = {j: 0 for j in range(1, 7)}
    last_status = None
    counts, temps, alarms = {}, {j: [None, None, None, None] for j in range(1, 7)}, []
    print(f"logging {a.channel} -> {out} (listen only; Ctrl-C to stop; type a note + Enter to mark an event)")
    try:
        while True:
            now = clock()
            if a.duration_s and now - t0m > a.duration_s:
                break
            if sys.stdin is not None and sys.stdin.isatty():
                r, _, _ = select.select([sys.stdin], [], [], 0)
                if r:
                    line = sys.stdin.readline().strip()
                    if line:
                        w_mk.writerow([f"{now:.4f}", f"{wall():.4f}", line]); f_mk.flush()
                        print(f"  mark @ {now - t0m:.1f} s: {line}")
            msg = bus.recv(timeout=0.05)
            if msg is None:
                continue
            now, nw = clock(), wall()
            aid = msg.arbitration_id
            counts[aid] = counts.get(aid, 0) + 1
            if aid in LOW_IDS and len(msg.data) >= 8:
                j = LOW_IDS[aid]
                v = decode_low(msg.data)
                alarm = sum(v[b] << k for k, b in enumerate(BITS) if b in ALARM_BITS)
                if alarm != last_bits[j]:
                    set_now = [b for b in ALARM_BITS if v[b]]
                    alarms.append((round(now - t0m, 3), j, set_now))
                    print(f"  j{j} status bits {set_now or 'cleared'} @ {now - t0m:.2f} s")
                    last_bits[j] = alarm
                elif now - last_row[j] < 1.0 / a.hz:
                    continue
                last_row[j] = now
                tr = temps[j]
                for k, key in ((0, "motor_C"), (2, "driver_C")):
                    tr[k] = v[key] if tr[k] is None else min(tr[k], v[key])
                    tr[k + 1] = v[key] if tr[k + 1] is None else max(tr[k + 1], v[key])
                w_low.writerow([f"{now:.4f}", f"{nw:.4f}", j, f"{v['volt_V']:.1f}", v["driver_C"], v["motor_C"],
                                v["status"], *(v[b] for b in BITS), f"{v['bus_A']:.3f}"])
            elif aid == STATUS_ID and len(msg.data) >= 8:
                st = decode_status(msg.data)
                if st != last_status:
                    w_st.writerow([f"{now:.4f}", f"{nw:.4f}", *st]); f_st.flush()
                    last_status = st
    except KeyboardInterrupt:
        pass
    finally:
        t1m, t1w = clock(), wall()
        for f in (f_low, f_st, f_mk):
            f.close()
        dur = max(t1m - t0m, 1e-9)
        meta = {"channel": a.channel, "listen_only": True, "start_mono": t0m, "start_wall": t0w,
                "stop_mono": t1m, "stop_wall": t1w, "duration_s": round(dur, 2),
                "frames": {hex(k): n for k, n in sorted(counts.items())},
                "rate_hz": {hex(k): round(n / dur, 2) for k, n in sorted(counts.items()) if k in LOW_IDS or k == STATUS_ID},
                "temp_range_C": {f"j{j}": {"motor": tr[:2], "driver": tr[2:]} for j, tr in temps.items()},
                "alarm_changes": alarms}
        (out / "meta.json").write_text(json.dumps(meta, indent=1))
        print(f"stopped after {dur:.0f} s")
        for j, tr in temps.items():
            print(f"  j{j}: motor {tr[0]}..{tr[1]} C, driver {tr[2]}..{tr[3]} C")
        print(f"  status-bit changes: {len(alarms)}" + ("" if not alarms else f" (first: {alarms[0]})"))
        if hasattr(bus, "shutdown"):
            bus.shutdown()
    return out


if __name__ == "__main__":
    main()
