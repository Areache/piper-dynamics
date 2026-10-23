#!/usr/bin/env python3
"""Offline test of guarded_move.py against a simulated arm (server side, no hardware).

The simulated arm moves along the plan's line: its position follows the last move_j target with a first-order
lag (tau = 133 ms, the servo lag measured on the recorded data). A rigid plane sits --true-gap-mm below the
start; past it the tip cannot move and a contact force builds as servo stiffness x (lagged target - plane).
Motor current = gravity (the plan's expectation x --gravity-scale, i.e. deliberate model error)
+ kinetic friction (sign of motion, 0.2 s ramp) + J^T F through kt + white noise. Time is simulated.

usage: venv/bin/python hw_scripts/t7/t7_sim.py --plan hw_scripts/t7/plans/x.json --true-gap-mm 78 --trials 5
"""
import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pinocchio as pin

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import guarded_move as G  # noqa: E402
from t7_plan import T4, build_model  # noqa: E402


class SimArm:
    def __init__(self, plan, true_gap_mm, gravity_scale=1.05, k_servo_n_per_mm=2.0, fric_A=(0, 0.15, 0.12, 0.03, 0.02, 0.0),
                 noise_A=0.02, tau=0.133, seed=0, stickslip_A=0.0, pulsed=False, line_speed=10.0, line_cancel_s=0.05):
        self.Q = np.asarray(plan["q"]); self.IE = np.asarray(plan["i_expected_A"]); self.step = plan["step_mm"]
        self.gap = true_gap_mm; self.gs = gravity_scale; self.ks = k_servo_n_per_mm; self.tau = tau
        self.fric = np.asarray(fric_A); self.noise = noise_A; self.rng = np.random.default_rng(seed)
        self.t = 0.0; self.s_target = 0.0; self.s_lag = 0.0; self.s = 0.0; self.v = 0.0; self.move_t = 0.0
        kt, sg = np.array(T4[plan["arm"]]["kt"]), np.array(T4[plan["arm"]]["dir"])
        m, fid = build_model(plan["tip_offset_m"]); d = m.createData()
        Jz = []
        for q in self.Q[:: max(1, len(self.Q) // 200)]:
            pin.computeJointJacobians(m, d, q); pin.updateFramePlacements(m, d)
            Jz.append(pin.getFrameJacobian(m, d, fid, pin.LOCAL_WORLD_ALIGNED)[2])
        self.Jz = np.array(Jz); self.Jz_every = max(1, len(self.Q) // 200)
        self.kt, self.sg = kt, sg
        self.line_target = None; self.line_speed = line_speed; self.line_cancel_s = line_cancel_s; self.cancel_at = None
        self.pulsed = pulsed      # contact force delivered as 50 ms pulses every 200 ms (same mean), as seen on hardware
        self.ss_A = stickslip_A; self.ss_on = False      # j2 random telegraph between 0 and +ss_A while moving
        self.contacts = []      # absolute sim time when the tip reached the plane
        self.retracts = []      # (time, contact force) when a move_j moved the target back up
        self.in_contact = False

    # --- clock
    def clock(self):
        return self.t

    def sleep(self, dt):
        n = max(1, int(round(dt / 0.001)))
        for _ in range(n):
            if self.line_target is not None:
                if self.cancel_at is not None and self.t >= self.cancel_at:
                    self.line_target = None; self.s_target = self.pending_target
                else:
                    self.s_target = min(self.line_target, self.s_target + self.line_speed * 0.001)
            self.s_lag += (self.s_target - self.s_lag) * (0.001 / self.tau)
            s_new = min(self.s_lag, self.gap)
            self.v = (s_new - self.s) / 0.001
            if abs(self.v) > 0.5:
                self.move_t = min(self.move_t + 0.001, 0.2)
            else:
                self.move_t = max(self.move_t - 0.002, 0.0)
            self.s = s_new
            self.t += 0.001
            if self.ss_A and abs(self.v) > 0.5 and self.rng.random() < 0.001 / 0.15:
                self.ss_on = not self.ss_on
            if not self.in_contact and self.s >= self.gap - 1e-9:
                self.in_contact = True; self.contacts.append(self.t)
            elif self.in_contact and self.s < self.gap - 0.5:
                self.in_contact = False

    # --- robot API used by guarded_move
    def _k(self, s):
        return int(np.clip(round(s / self.step), 0, len(self.Q) - 1))

    def set_speed_percent(self, p):
        pass

    def get_flange_pose(self):
        return SimpleNamespace(msg=[0.3, 0.0, 0.2 - self.s / 1000.0, 0.0, 0.0, 0.0])

    def move_l(self, pose):
        self.line_target = (0.2 - pose[2]) * 1000.0; self.cancel_at = None

    def move_j(self, q):
        q = np.asarray(q)
        k = int(np.argmin(np.abs(self.Q - q).max(1)))
        if self.line_target is not None:      # a move_j interrupts a running move_l after line_cancel_s
            if self.cancel_at is None:
                self.cancel_at = self.t + self.line_cancel_s
                if self.in_contact:
                    self.retracts.append((self.t, self.ks * max(0.0, self.s_lag - self.gap)))
            self.pending_target = k * self.step
            return
        if k * self.step < self.s_target - 1e-9 and self.in_contact:
            self.retracts.append((self.t, self.ks * max(0.0, self.s_lag - self.gap)))
        self.s_target = k * self.step

    def get_joint_angles(self):
        return SimpleNamespace(msg=list(self.Q[self._k(self.s)] + self.rng.normal(0, 2e-5, 6)))

    def _current(self):
        k = self._k(self.s)
        force = self.ks * max(0.0, self.s_lag - self.gap)                    # N, pushing the tip up
        if self.pulsed:
            force = force * 4.0 if (self.t % 0.2) < 0.05 else 0.0
        tau_c = -self.Jz[min(k // self.Jz_every, len(self.Jz) - 1)] * force   # joint torque from an upward force
        fr = self.fric * np.sign(self.v) * (self.move_t / 0.2)               # the 0.2 s friction ramp seen on hardware
        i = self.IE[k] * self.gs + fr + self.sg * tau_c / self.kt + self.rng.normal(0, self.noise, 6)
        if self.ss_on:
            i[1] += self.ss_A
        return i, force

    def get_motor_states(self, idx):
        i, force = self._current()
        k = self._k(self.s)
        return SimpleNamespace(msg=SimpleNamespace(position=float(self.Q[k][idx - 1]), velocity=float(self.v / 1000 * 0.01),
                                                   current=float(i[idx - 1]), torque=0.0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", required=True); ap.add_argument("--true-gap-mm", type=float, required=True)
    ap.add_argument("--trials", type=int, default=5); ap.add_argument("--speed-mm-s", type=float)
    ap.add_argument("--gravity-scale", type=float, default=1.05); ap.add_argument("--k-servo", type=float, default=2.0)
    ap.add_argument("--noise", type=float, default=0.02); ap.add_argument("--out", required=True)
    ap.add_argument("--stickslip", type=float, default=0.0, help="j2 random-telegraph amplitude (A), as seen on hardware at 5 mm/s")
    ap.add_argument("--pulsed", action="store_true", help="contact force as 50 ms pulses every 200 ms")
    ap.add_argument("--gm-args", default="", help="extra arguments passed to guarded_move.py")
    ap.add_argument("--line-speed", type=float, default=10.0, help="simulated move_l speed (mm/s)")
    ap.add_argument("--line-cancel-s", type=float, default=0.05, help="simulated delay before a move_j interrupts a move_l")
    a = ap.parse_args()
    plan = json.loads(Path(a.plan).read_text())
    sim = SimArm(plan, a.true_gap_mm, a.gravity_scale, a.k_servo, noise_A=a.noise, stickslip_A=a.stickslip, pulsed=a.pulsed,
                 line_speed=a.line_speed, line_cancel_s=a.line_cancel_s)
    argv = ["--plan", a.plan, "--trials", str(a.trials), "--operator-confirmed", "--no-confirm-each", "--out", a.out]
    if a.speed_mm_s:
        argv += ["--speed-mm-s", str(a.speed_mm_s)]
    argv += a.gm_args.split()
    out = G.main(argv, robot=sim, clock=sim.clock, sleep=sim.sleep, ask=lambda *_: "YES")
    s = json.loads((Path(out) / "summary.json").read_text())
    det = [t["det_depth_mm"] for t in s["trials"] if t["det_depth_mm"] is not None]
    print(f"\nSIM: true gap {a.true_gap_mm} mm, gravity model error {100*(a.gravity_scale-1):+.0f}%, servo stiffness {a.k_servo} N/mm, noise {a.noise} A")
    print(f"  statuses: {[t['status'] for t in s['trials']]}")
    lat = []
    for tc in sim.contacts:
        nxt = [r for r in sim.retracts if r[0] >= tc]
        if nxt:
            lat.append((nxt[0][0] - tc, nxt[0][1]))
    if lat:
        L = np.array(lat)
        print(f"  contact -> back-off latency {np.round(L[:,0]*1000).astype(int).tolist()} ms; contact force at back-off {np.round(L[:,1],1).tolist()} N")
    if det:
        det = np.array(det)
        print(f"  detected at {np.round(det,2).tolist()} mm -> beyond the true plane by {np.round(det - a.true_gap_mm,2).tolist()} mm "
              f"(mean {np.mean(det - a.true_gap_mm):.2f}, sd {np.std(det):.2f})")


if __name__ == "__main__":
    main()
