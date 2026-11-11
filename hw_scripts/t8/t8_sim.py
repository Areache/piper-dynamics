#!/usr/bin/env python3
"""T8 dry run (server side, no hardware): every t8_run.py segment against a simulated arm, in session order, plus
stop-path fault injection. Needs pinocchio + the gripper URDF (same as t8_plan.py).

Simulated arm (joint space, time simulated in 2 ms steps):
  - move_j: after a 50 ms command delay each joint runs toward the latest target, speed <= spec x percent,
    acceleration <= 4 rad/s^2, trapezoidal stop; streamed targets simply replace the target
  - current = dir * tau / kt_true; tau = URDF gravity + payload (container, only while it is off the table)
    + Coulomb / viscous friction (temperature-dependent) + joint-angle ripple + a per-stop constant + noise.
    kt_true, ripple, friction and the per-stop constant are the "truths" t8_analyze.py must recover from the dry-run
    data (sim_truth.json)
  - gripper: width moves at 50 mm/s; the container (50 mm wide) is attached when the fingers close within 10 mm of
    its grip point; released on open; a release more than 10 mm above the table counts as a drop
  - every 20 ms: table / workspace-box clearance of the fingertip, container and joint origins (t8_plan.Geo)
Checks written to dryrun.json / dryrun.md: per segment sim duration, operator prompts, min clearance + binding
point, max joint speed, guard trips, drops; stop-path tests: Ctrl-C at a random time in each moving segment, a
too-fast sweep (speed mapping x4), an off-path line (joint offset) -> each must end with exactly one move_j to the
measured pose and the arm still within 0.5 s.

usage (server): venv/bin/python hw_scripts/t8/t8_sim.py --plan hw_scripts/t8/plans/t8_right.json --out /tmp/t8_dry
"""
import argparse
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pinocchio as pin

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import t8_plan as TP  # noqa: E402
import t8_run as R  # noqa: E402

KBC = np.array([4.725, 4.134, 4.725, 1.232, 0.779, 0.779])
TRUTH = {
    "kt_ratio": [1.0, 0.91, 0.80, 0.85, 0.86, 1.0],                  # kt_true / nominal k*b*c
    "ripple": {"2": [[0.15, 0.156, 0.3]], "3": [[0.15, 1.00, 1.1], [0.06, 0.52, 2.0]], "4": [[0.03, 1.0, 0.0]],
               "5": [[0.02, 1.0, 0.5]]},                               # joint: [amp N m, period deg, phase rad]
    "fc_cold": [0.45, 1.10, 0.50, 0.10, 0.05, 0.03], "fv": [0.5, 1.0, 0.6, 0.1, 0.05, 0.03],
    "fc_temp_coeff": -0.01,                                            # per degC above 35
    "visit_sd": [0.0, 0.10, 0.10, 0.03, 0.02, 0.0], "stiction": [0.0, 0.15, 0.20, 0.03, 0.02, 0.0],
    "noise_A": 0.02, "delay_s": 0.05, "sink_mm_per_N": 0.5,
}
DIR = np.array([-1, 1, 1, 1, 1, 1])


class SimGrip:
    def __init__(self, arm):
        self.arm = arm

    def get_gripper_status(self):
        a = self.arm
        return SimpleNamespace(msg=SimpleNamespace(value=a.gw, force=a.gforce if a.attached else 0.0, mode="width"))

    def move_gripper_m(self, value=0.0, force=1.0):
        self.arm.g_target, self.arm.g_force_cmd = float(value), float(force)


class SimArm:
    def __init__(self, plan, geo, seed=0, speed_scale=1.0):
        self.p, self.geo = plan, geo
        self.rng = np.random.default_rng(seed)
        self.t = 0.0
        self.q = np.array(plan["excite"]["q0"], float); self.qd = np.zeros(6)
        self.target = self.q.copy(); self.pending = []           # (t_effective, target, pct)
        self.pct = 10; self.speed_scale = speed_scale
        self.kt = KBC * np.array(TRUTH["kt_ratio"])
        self.gw, self.g_target, self.g_force_cmd, self.gforce = 0.07, 0.07, 1.0, 0.0
        self.attached = False; self.content_g = 0.0; self.container_g = 120.0
        L = plan["lift"]
        self.site_grip = TP.fk(geo.m, geo.d, geo.fid, np.array(L["Q_down"][-1])).translation.copy()
        self.cont_pos = self.site_grip.copy()                     # grip point of the container
        self.visit = np.zeros(6); self.still_t = 0.0; self.moved = True; self.last_dir = np.zeros(6)
        self.estop = False; self.offset = np.zeros(6); self.block = None
        self.log = {"min_clear": (np.inf, ""), "drops": 0, "max_qd": np.zeros(6), "viol": []}
        self.next_clear = 0.0; self.cmds = 0
        self._cur = np.zeros(6); self._update_current()

    # ---- clock
    def clock(self):
        return self.t

    def sleep(self, dt):
        n = max(1, int(round(dt / 0.002)))
        for _ in range(n):
            self._step(0.002)
        if getattr(self, "interrupt_at", None) is not None and self.t >= self.interrupt_at:
            self.interrupt_at = None
            raise KeyboardInterrupt

    def temp(self):
        return 35.0 + 15.0 * (1 - math.exp(-self.t / 1800.0))

    def _step(self, h):
        while self.pending and self.pending[0][0] <= self.t:
            _, self.target, self.pct = self.pending.pop(0)
        if self.estop:
            self.qd[:] = 0
        else:
            vlim = np.radians(TP.VMAX_DEG_S) * self.pct / 100.0 * self.speed_scale
            err = self.target - self.q
            vdes = np.sign(err) * np.minimum(vlim, np.sqrt(2 * 4.0 * np.abs(err)))
            self.qd += np.clip(vdes - self.qd, -4.0 * h, 4.0 * h)
            self.q = self.q + self.qd * h
            snap = (np.abs(self.target - self.q) < 2e-4) & (np.abs(self.qd) < 0.02)     # servo settles on target
            self.q[snap], self.qd[snap] = self.target[snap], 0.0
        self.log["max_qd"] = np.maximum(self.log["max_qd"], np.abs(self.qd))
        moving = np.abs(self.qd) > 1e-3
        self.last_dir = np.where(moving, np.sign(self.qd), self.last_dir)
        if moving.any():
            self.moved, self.still_t = True, 0.0
        else:
            self.still_t += h
            if self.moved and self.still_t > 0.1:                       # came to rest: redraw the per-stop constant
                self.visit = self.rng.normal(0, TRUTH["visit_sd"]) + np.array(TRUTH["stiction"]) * self.last_dir
                self.moved = False
        # gripper
        tip = None
        if self.attached or (self.g_target < 0.05 and self.gw <= 0.0501):
            tip = TP.fk(self.geo.m, self.geo.d, self.geo.fid, self.q).translation.copy()
        if not self.attached:
            self.gw += float(np.clip(self.g_target - self.gw, -0.05 * h, 0.05 * h))
            if tip is not None and np.linalg.norm(tip - self.cont_pos) < 0.01:
                self.attached, self.gw, self.gforce = True, 0.05, self.g_force_cmd
        elif self.g_target > 0.05:
            self.attached = False
            if tip[2] - self.site_grip[2] > 0.01:
                self.log["drops"] += 1
            self.cont_pos = tip.copy(); self.cont_pos[2] = max(self.site_grip[2], min(self.cont_pos[2], self.site_grip[2]))
        else:
            self.cont_pos = tip
        if self.block is not None and self.t > self.block[0]:
            k = self.block[1]
            self.q[k] -= self.qd[k] * h; self.qd[k] = 0.0      # joint k blocked (an obstacle)
        self.t += h
        if self.t >= self.next_clear:
            self.next_clear = self.t + 0.02
            c = self.geo.clearance(self.q, self.attached, 0.0, 0.0)
            if self.attached and self.cont_pos[2] - self.site_grip[2] < 0.002:
                c = self.geo.clearance(self.q, False, 0.0, 0.0)       # standing on the table / being lifted off it
            if c[0] < self.log["min_clear"][0]:
                self.log["min_clear"] = c
            if c[0] < 0:
                self.log["viol"].append((round(self.t, 2), c[1], round(c[0] * 1000, 1)))
            self._update_current()

    def _update_current(self):
        g = self.geo
        tau = pin.computeGeneralizedGravity(g.m, g.d, self.q).copy()
        if self.attached and self.cont_pos[2] - self.site_grip[2] > 0.0005:
            pin.computeJointJacobians(g.m, g.d, self.q); pin.updateFramePlacements(g.m, g.d)
            J = pin.getFrameJacobian(g.m, g.d, g.fid, pin.LOCAL_WORLD_ALIGNED)[:3]
            tau += J.T @ np.array([0, 0, 9.81 * (self.content_g + self.container_g) / 1000.0])
        fc = np.array(TRUTH["fc_cold"]) * (1 + TRUTH["fc_temp_coeff"] * (self.temp() - 35.0))
        tau += fc * np.tanh(self.qd / 0.01) + np.array(TRUTH["fv"]) * self.qd
        th = np.degrees(self.q)
        for j, comps in TRUTH["ripple"].items():
            k = int(j) - 1
            for amp, per, ph in comps:
                tau[k] += amp * math.sin(2 * math.pi * th[k] / per + ph)
        if not np.any(np.abs(self.qd) > 1e-3):
            tau += self.visit
        self._cur = DIR * tau / self.kt

    # ---- robot API used by t8_run / guarded_move
    def set_speed_percent(self, p):
        self.pct = int(p)

    def move_j(self, q):
        self.cmds += 1
        self.pending.append((self.t + TRUTH["delay_s"], np.asarray(q, float).copy(), self.pct))

    def electronic_emergency_stop(self):
        self.estop = True

    def get_joint_angles(self):
        return SimpleNamespace(msg=list(np.round(self.q + self.offset, 5) + self.rng.normal(0, 2e-5, 6)))

    def get_motor_states(self, i):
        c = self._cur[i - 1] + self.rng.normal(0, TRUTH["noise_A"])
        return SimpleNamespace(msg=SimpleNamespace(position=float(self.q[i - 1]), velocity=float(self.qd[i - 1]), current=float(c)))


def make_ask(sim, log, gauge=True):
    def ask(prompt):
        log.append(prompt.strip())
        if "type YES" in prompt:
            return "YES"
        if "load " in prompt and " g in the container" in prompt:
            sim.content_g = float(prompt.split("load ")[1].split(" g")[0]); return ""
        if "gauge reading" in prompt:
            load_N = 9.81 * (sim.content_g + sim.container_g) / 1000.0
            return f"{TRUTH['sink_mm_per_N'] * load_N + sim.rng.normal(0, 0.05):.2f}"
        if "tap" in prompt:
            return ""
        return "y"
    return ask


SESSION = [  # (segment, label, extra argv, content grams or None)
    ("anchor", "a0", [], None), ("tap", "cold_start", [], None), ("excite", "cold", [], None), ("tap", "cold_end", [], None),
    ("sweep", "", [], None), ("anchor", "a1", [], None),
] + [x for m in (0, 100, 250, 500) for x in (
    ("lift", f"m{m}", ["--confirm-first"], m), ("pick", f"m{m}", [], m), ("static", f"m{m}", ["--confirm-first"], m),
    ("reach", f"m{m}", [], m), ("place", f"m{m}", [], m))] + [
    ("pick", "hold500", [], 500), ("hold", "m500", ["--confirm-first"], 500),
    ("comply", "c500", ["--steps", "500,0,500,0"], 500), ("hold", "m0", ["--confirm-first"], 0), ("place", "hold0", [], 0),
    ("anchor", "a2", [], None), ("tap", "hot_start", [], None), ("excite", "hot", [], None), ("tap", "hot_end", [], None),
    ("anchor", "a3", [], None)]


def run_seg(sim, plan_path, out, seg, label, extra, grams, prompts):
    argv = [seg, "--plan", str(plan_path), "--out", str(out / f"{seg}_{label or 'x'}"), "--operator-confirmed", "--label", label] + extra
    if grams is not None:
        argv += ["--grams", str(grams), "--container-g", str(sim.container_g)]
        sim.content_g = float(grams)
    t0, c0 = sim.t, len(sim.log["viol"])
    sim.log["min_clear"] = (np.inf, "")
    try:
        R.main(argv, robot=sim, grip=SimGrip(sim), clock=sim.clock, sleep=sim.sleep, ask=make_ask(sim, prompts))
        status = "done"
    except SystemExit as e:
        status = f"exit {e.code}"
    meta = json.loads((out / f"{seg}_{label or 'x'}" / "rec.json").read_text())
    return {"segment": seg, "label": label, "status": meta["status"] if status == "done" else status, "detail": meta.get("detail"),
            "sim_s": round(sim.t - t0, 1), "min_clear_mm": round(sim.log["min_clear"][0] * 1000, 1),
            "binding": sim.log["min_clear"][1], "violations": len(sim.log["viol"]) - c0}


def stop_test(plan, geo, plan_path, out, seg, kind, grams=None, seed=1):
    sim = SimArm(plan, geo, seed=seed, speed_scale=4.0 if kind == "too_fast" else 1.0)
    if seg in ("static", "reach", "hold", "comply"):        # start holding the container at lift top
        sim.q = np.array(plan["lift"]["Q_up"][-1]); sim.target = sim.q.copy(); sim.attached = True; sim.gw = 0.05
        sim.cont_pos = TP.fk(geo.m, geo.d, geo.fid, sim.q).translation.copy()
    if kind == "ctrl_c":
        sim.interrupt_at = 20.0 + sim.rng.uniform(0, 30.0)
    if kind == "blocked":                          # after 15 s j3 stops following (something holds the arm)
        sim.block = (15.0, 2)
    extra = {"sweep": ["--joints", "3"], "excite": ["--joints", "2"]}.get(seg, [])
    d = out / "stop_tests" / f"{seg}_{kind}"
    argv = [seg, "--plan", str(plan_path), "--out", str(d), "--operator-confirmed"] + extra
    if seg in R.NEEDS_CONTAINER | {"lift"}:
        argv += ["--grams", "250", "--container-g", "120"]
    try:
        R.main(argv, robot=sim, grip=SimGrip(sim), clock=sim.clock, sleep=sim.sleep, ask=make_ask(sim, []))
        return {"segment": seg, "fault": kind, "result": "FAIL: segment finished, fault not caught"}
    except SystemExit:
        pass
    meta = json.loads((d / "rec.json").read_text())
    z = np.load(d / "rec.npz")
    cmds = z["cmds"]; last = cmds[-1]
    n_cmd_after = sim.cmds
    q_meas = np.array(meta.get("hold_q_rad") or [np.nan] * 6)
    t_stop = sim.t
    q_before = sim.q.copy()
    for _ in range(250):
        sim._step(0.002)
    moved = float(np.degrees(np.max(np.abs(sim.q - q_before))))
    ok = (int(last[1]) == R.CMDS["hold"] and sim.cmds == n_cmd_after and moved < 0.05
          and np.allclose(sim.target, q_meas, atol=1e-3))
    return {"segment": seg, "fault": kind, "status": meta["status"], "detail": meta.get("detail"), "t_stop_s": round(t_stop, 2),
            "moved_after_stop_deg": round(moved, 3), "last_cmd_is_hold": int(last[1]) == R.CMDS["hold"],
            "still": moved < 0.05, "target_is_measured": bool(np.allclose(sim.target, q_meas, atol=1e-3)),
            "result": "PASS" if ok else "FAIL"}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plan", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", help="comma-separated segment names (session order kept)")
    ap.add_argument("--no-stop-tests", action="store_true")
    a = ap.parse_args()
    plan_path = Path(a.plan); plan = json.loads(plan_path.read_text())
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    geo = TP.Geo(plan["tip_m"], plan["table_z_m"], plan["container"]["h_m"], plan["container"]["grip_h_m"])
    geo.box = {k: (np.array(v[0], float), np.array(v[1], float)) for k, v in plan["report"]["workspace_box_m"].items() if k in ("tool", "link")}
    for k in geo.box:
        geo.box[k][0][2] = -np.inf
    sim = SimArm(plan, geo)
    rows, prompts = [], []
    for seg, label, extra, grams in SESSION:
        if a.only and seg not in a.only.split(","):
            continue
        n0 = len(prompts)
        r = run_seg(sim, plan_path, out, seg, label, extra, grams, prompts)
        r["prompts"] = len(prompts) - n0
        r["max_joint_speed_deg_s"] = np.round(np.degrees(sim.log["max_qd"]), 1).tolist(); sim.log["max_qd"][:] = 0
        rows.append(r)
        print(f"{seg:7s} {label:10s} {r['status']:6s} {r['sim_s']:7.1f} s  clear {r['min_clear_mm']:6.1f} mm ({r['binding']})"
              f"  viol {r['violations']}  prompts {r['prompts']}  vmax {r['max_joint_speed_deg_s']}")
    tests = []
    if not a.no_stop_tests:
        for seg, kind in (("excite", "ctrl_c"), ("sweep", "ctrl_c"), ("sweep", "too_fast"), ("static", "ctrl_c"),
                          ("lift", "ctrl_c"), ("reach", "ctrl_c"), ("hold", "ctrl_c"), ("lift", "blocked"), ("sweep", "blocked")):
            t = stop_test(plan, geo, plan_path, out, seg, kind)
            tests.append(t); print("stop test", t)
    total = sum(r["sim_s"] for r in rows)
    res = {"plan_sha": plan.get("sha"), "segments": rows, "total_sim_min": round(total / 60, 1),
           "operator_prompts": len(prompts), "drops": sim.log["drops"], "violations": sim.log["viol"][:20],
           "stop_tests": tests, "truth": TRUTH, "sim_temp_end_C": round(sim.temp(), 1)}
    (out / "dryrun.json").write_text(json.dumps(res, indent=1, default=float))
    (out / "sim_truth.json").write_text(json.dumps(TRUTH, indent=1))
    print(f"total {total/60:.1f} min simulated, {len(prompts)} prompts, drops {sim.log['drops']}, violations {len(sim.log['viol'])}")


if __name__ == "__main__":
    main()
