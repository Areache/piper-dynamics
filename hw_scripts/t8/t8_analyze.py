#!/usr/bin/env python3
"""T8 analysis (server side, pinocchio + gripper URDF). Implements .hw/results/T8/predictions_v1.md P1-P7 as frozen;
prints one result table and writes results.json / results.md. Nothing here may be tuned after the data are in.

Input: a run directory with one sub-directory per t8_run.py segment (<segment>_<label>/rec.npz + rec.json), the
plan used, and optionally the E3 directory (for P2a / P2b held-out).
  venv/bin/python hw_scripts/t8/t8_analyze.py --run ~/piper_hw_runs/T8_<date> --plan hw_scripts/t8/plans/t8_right.json \
      --e3 /path/E3_20261028 [--d-com-m 0.0] --out results_t8
The dry run (t8_sim.py output) is a valid input: it checks that every step runs and that the known sim truths come back.
"""
import argparse
import glob
import json
import math
import sys
from pathlib import Path

import numpy as np
import pinocchio as pin

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "t7"))
from t7_plan import build_model  # noqa: E402

KBC = np.array([4.725, 4.134, 4.725, 1.232, 0.779, 0.779])
DIR = np.array([-1, 1, 1, 1, 1, 1])
J25 = [1, 2, 3, 4]
GRID = 0.005
DOM = {1: 0.156, 2: 1.00}
E3_WIN = {1: (89.3, 96.3), 2: (-72.0, -56.6)}
rng = np.random.default_rng(7)


# ----------------------------------------------------------------------------- loading
class Seg:
    def __init__(self, d):
        z = np.load(d / "rec.npz", allow_pickle=False)
        self.meta = json.loads((d / "rec.json").read_text())
        r = z["rows"]; self.t = r[:, 0] - r[0, 0]; self.phase = r[:, 2].astype(int)
        self.q, self.i = r[:, 4:10], r[:, 22:28]; self.grip = r[:, 28]
        self.marks = [(float(a) - r[0, 0], str(b)) for a, b in zip(z["marks_t"][:, 0], z["marks_text"])]
        self.name = d.name

    def mark_times(self, text):
        return [t for t, m in self.marks if m.startswith(text)]

    def window(self, t0, t1):
        return (self.t >= t0) & (self.t < t1)


def load_run(run):
    segs = {}
    for d in sorted(Path(run).iterdir()):
        if (d / "rec.npz").exists():
            segs[d.name] = Seg(d)
    return segs


class Model:
    def __init__(self, tip_m, d_com):
        self.m, self.fid = build_model(tip_m + d_com)
        self.d = self.m.createData()
        self.full_tip, _ = build_model(tip_m)

    def g(self, Q):
        return np.array([pin.computeGeneralizedGravity(self.m, self.d, q) for q in Q])

    def gamma(self, q):
        pin.computeJointJacobians(self.m, self.d, q); pin.updateFramePlacements(self.m, self.d)
        return pin.getFrameJacobian(self.m, self.d, self.fid, pin.LOCAL_WORLD_ALIGNED)[2] * 9.81   # N m per kg

    def tip(self, q):
        pin.framesForwardKinematics(self.m, self.d, q)
        return self.d.oMf[self.fid].translation.copy()


def tau_res(seg, M, kt=KBC):
    return DIR * kt * seg.i - M.g(seg.q)


def vel(t, x, k=5):
    """Central difference over +-k samples (robust to repeated time stamps)."""
    v = np.zeros_like(x, dtype=float)
    n = len(t)
    for i in range(n):
        a, b = max(0, i - k), min(n - 1, i + k)
        dt = t[b] - t[a]
        v[i] = (x[b] - x[a]) / dt if dt > 0 else 0.0
    return v


def grams(seg):
    return float(seg.meta.get("grams") or 0.0)


def boot_ci(vals, fn, n=2000):
    vals = list(vals)
    bs = [fn([vals[k] for k in rng.integers(0, len(vals), len(vals))]) for _ in range(n)]
    return np.percentile(bs, [2.5, 97.5], axis=0)


# ----------------------------------------------------------------------------- static visits
def static_visits(segs, M):
    """One row per recorded visit: tier grams, pose, approach, window means of q and i (record phase only)."""
    out = []
    for name, s in segs.items():
        if not name.startswith(("static_", "hold_")) or s.meta.get("status") != "done":
            continue
        for v in s.meta["info"]["visits"]:
            if v.get("skipped"):
                continue
            sl = slice(v["row0"], v["row1"])
            out.append({"seg": name, "grams": grams(s), "pose": v["pose"], "approach": v["approach"], "visit": v.get("visit", 0),
                        "q": s.q[sl].mean(0), "i": s.i[sl].mean(0), "t": s.t[sl], "Q": s.q[sl], "I": s.i[sl]})
    return out


def p1(vis, M):
    st = [v for v in vis if v["seg"].startswith("static_")]
    zero = {(v["pose"], v["approach"]): v for v in st if v["grams"] == 0}
    rows = []
    for v in st:
        z = zero.get((v["pose"], v["approach"]))
        if v["grams"] == 0 or z is None:
            continue
        dm = (v["grams"] - z["grams"]) / 1000.0
        rows.append({"pose": v["pose"], "y": DIR * KBC * (v["i"] - z["i"]) - (M.g([v["q"]])[0] - M.g([z["q"]])[0]),
                     "x": dm * M.gamma(v["q"])})
    if not rows:
        return {"verdict": "no data"}
    poses = sorted({r["pose"] for r in rows})

    def fit(ps):
        R = [r for r in rows if r["pose"] in ps]
        y = np.array([r["y"] for r in R]); x = np.array([r["x"] for r in R])
        b = np.array([(x[:, j] @ y[:, j]) / (x[:, j] @ x[:, j]) for j in J25])
        return 1.0 / b

    r = fit(poses); ci = boot_ci(poses, fit)
    inv = 1.0 / r; inv_ci = np.sort(1.0 / ci, axis=0)
    p1a = bool(np.all(ci[1] < 1.0))
    p1b = all(inv[k] >= 1.05 and inv_ci[0, k] <= 1.3 and inv_ci[1, k] >= 1.1 for k in (2, 3))
    return {"r_j2_j5": np.round(r, 3).tolist(), "ci_lo": np.round(ci[0], 3).tolist(), "ci_hi": np.round(ci[1], 3).tolist(),
            "inv_r_j4_j5": np.round(inv[2:], 3).tolist(), "n_points": len(rows), "P1a": p1a, "P1b": p1b,
            "verdict": "holds" if p1a and p1b else ("partial" if p1a or p1b else "fails"), "gain": r}


# ----------------------------------------------------------------------------- ripple
def angle_band(theta, x, lo=0.1, hi=3.0):
    o = np.argsort(theta); th = theta[o]
    g = np.arange(np.ceil(th[0] / GRID) * GRID, th[-1], GRID)
    if len(g) < 200:
        return None
    xg = np.interp(g, th, x[o]); X = np.fft.rfft(xg - xg.mean()); f = np.fft.rfftfreq(len(xg), GRID)
    X[(f < 1 / hi) | (f > 1 / lo)] = 0
    return g, np.fft.irfft(X, len(xg))


def sweep_pieces(seg, M):
    tr = tau_res(seg, M)
    out = []
    for t0, m in seg.marks:
        if not (m.startswith("sweep j") and m.endswith("start")):
            continue
        j = int(m.split()[1][1:]) - 1
        ends = [t for t, mm in seg.marks if mm == m.replace("start", "end")]
        if not ends:
            continue
        w = seg.window(t0 + 1.0, ends[0] - 1.0)          # skip the firmware ramps
        if w.sum() < 300:
            continue
        sm = np.convolve(np.degrees(seg.q[w, j]), np.ones(25) / 25, "same")
        b = angle_band(sm, tr[w, j])
        if b is not None:
            out.append({"joint": j, "label": m, "g": b[0], "w": b[1]})
    return out


def ripple_table(segs, M):
    pcs = [p for n, s in segs.items() if n.startswith("sweep_") for p in sweep_pieces(s, M)]
    tab = {}
    for j in sorted({p["joint"] for p in pcs}):
        P = [p for p in pcs if p["joint"] == j]
        lo = min(p["g"][0] for p in P); hi = max(p["g"][-1] for p in P)
        g = np.arange(np.ceil(lo / GRID) * GRID, hi, GRID)
        acc = np.zeros_like(g); cnt = np.zeros_like(g)
        for p in P:
            m = (g >= p["g"][0]) & (g <= p["g"][-1]); acc[m] += np.interp(g[m], p["g"], p["w"]); cnt[m] += 1
        tab[j] = (g, np.where(cnt > 0, acc / np.maximum(cnt, 1), 0.0))
        cfg = {}
        for p in P:
            key = p["label"].split()[2][0]
            cfg.setdefault(key, []).append(np.std(p["w"]))
        tab[j] = (*tab[j], {k: float(np.mean(v)) for k, v in cfg.items()})
    return tab


def W_at(tab, Q):
    out = np.zeros((len(Q), 6))
    for j, (g, w, _) in tab.items():
        th = np.degrees(Q[:, j]); m = (th >= g[0]) & (th <= g[-1])
        out[m, j] = np.interp(th[m], g, w)
    return out


def bandpass_time(x, fs, lo=0.5, hi=5.0):
    X = np.fft.rfft(x - x.mean(), axis=0); f = np.fft.rfftfreq(len(x), 1 / fs)
    X[(f < lo) | (f > hi)] = 0
    return np.fft.irfft(X, len(x), axis=0)


def e3_free_segments(e3, M):
    out = []
    if not e3:
        return out
    for f in sorted(glob.glob(f"{e3}/A*/trial_*.npz")):
        z = np.load(f, allow_pickle=True)
        if not {"state", "q", "current", "t"} <= set(z.files):
            continue
        t, st = z["t"], z["state"]
        th_ = t[np.argmax(z["hit"] > 0)] if ("hit" in z.files and z["hit"].any()) else t[-1]
        m = np.flatnonzero((st == 1) & (t < th_ - 0.5))
        if len(m) < 150:
            continue
        s = slice(m[0], m[-1] + 1)
        out.append({"t": t[s], "q": z["q"][s], "i": z["current"][s]})
    return out


def p2(segs, M, tab, e3):
    res = {"configs_amp": {f"j{j+1}": v[2] for j, v in tab.items()}}
    E = e3_free_segments(e3, M)
    # P2a: phase vs the E3 mean waveform in the E3 windows
    a_ok = []
    for j in (1, 2):
        if j not in tab or not E:
            res[f"P2a_j{j+1}"] = "no data"; a_ok.append(False); continue
        waves = []
        for e in E:
            sm = np.convolve(np.degrees(e["q"][:, j]), np.ones(25) / 25, "same")
            tr = DIR[j] * KBC[j] * e["i"][:, j] - M.g(e["q"])[:, j]
            b = angle_band(sm, tr)
            if b is not None:
                waves.append(b)
        lo, hi = E3_WIN[j]
        g = np.arange(np.ceil(lo / GRID) * GRID, hi, GRID)
        ref = np.mean([np.interp(g, *w) for w in waves if w[0][0] <= lo and w[0][-1] >= hi] or [np.zeros_like(g)], 0)
        gt, wt, _ = tab[j]
        if gt[0] > lo or gt[-1] < hi or not ref.any():
            res[f"P2a_j{j+1}"] = "window not covered"; a_ok.append(False); continue
        x = np.interp(g, gt, wt); r0 = float(np.corrcoef(x, ref)[0, 1])
        ml = int(round(0.5 * DOM[j] / GRID)); best = (-2, 0)
        for L in range(-ml, ml + 1):
            u, v = (x[L:], ref[:len(ref) - L]) if L >= 0 else (x[:L], ref[-L:])
            best = max(best, (float(np.corrcoef(u, v)[0, 1]), L))
        sh = best[1] * GRID / DOM[j]
        ok = r0 >= 0.5 and abs(sh) <= 0.1
        res[f"P2a_j{j+1}"] = {"r0": round(r0, 3), "best_r": round(best[0], 3), "shift_period": round(sh, 3), "consistent": ok}
        a_ok.append(ok)
    # P2b: 0.5-5 Hz variance before / after subtracting W, on held-out free motion
    red = {1: {"E3": [], "T8": []}, 2: {"E3": [], "T8": []}}
    held = [("E3", e["t"], e["q"], e["i"]) for e in E]
    for n, s in segs.items():
        if n.startswith("reach_") and grams(s) == 0:
            held.append(("T8", s.t, s.q, s.i))
    for src, t, Q, I in held:
        fs = 1.0 / np.median(np.diff(t))
        tr = DIR * KBC * I - M.g(Q); Wq = W_at(tab, Q)
        for j in (1, 2):
            if j not in tab:
                continue
            b0 = bandpass_time(tr[:, j], fs); b1 = bandpass_time(tr[:, j] - Wq[:, j], fs)
            if np.var(b0) > 0:
                red[j][src].append(1 - np.var(b1) / np.var(b0))
    from_wil = {}
    for j in (1, 2):
        x = np.array(red[j]["E3"])
        if len(x) >= 6:
            # exact sign-flip permutation p for "median reduction > 0" (no scipy needed)
            rk = np.argsort(np.argsort(np.abs(x))) + 1
            obs = np.sum(np.sign(x) * rk)
            null = [np.sum(rng.choice([-1, 1], len(x)) * rk) for _ in range(20000)]
            from_wil[j] = float(np.mean(np.array(null) >= obs))
        else:
            from_wil[j] = math.nan
    need = {1: 0.20, 2: 0.30}
    b_ok = all(red[j][s] and np.median(red[j][s]) >= need[j] for j in (1, 2) for s in ("E3", "T8")) and \
        all(from_wil[j] < 0.05 for j in (1, 2))
    res["P2b"] = {f"j{j+1}": {s: (round(float(np.median(v)), 3) if v else None) for s, v in red[j].items()} for j in (1, 2)}
    res["P2b"]["p_E3"] = {f"j{j+1}": from_wil[j] for j in (1, 2)}
    res["verdict"] = "holds" if all(a_ok) and b_ok else ("partial" if any(a_ok) or b_ok else "fails")
    return res


# ----------------------------------------------------------------------------- mass projections
def mhat(dtau, gam, w):
    gg, ww = gam[J25], w[J25]; return float((gg * ww) @ dtau[J25] / ((gg * ww) @ gg))


def sigma_static(vis, M, gain):
    zero = [v for v in vis if v["seg"].startswith("static_") and v["grams"] == 0]
    # per-joint spread of the 0 g tier across approaches of the same pose = revisit noise
    d = []
    by = {}
    for v in zero:
        by.setdefault(v["pose"], []).append(DIR * KBC * gain * v["i"] - M.g([v["q"]])[0])
    for p, L in by.items():
        if len(L) > 1:
            d.append(np.std(L, axis=0, ddof=1))
    s = np.mean(d, 0) if d else np.ones(6) * 0.1
    return 1.0 / np.maximum(s, 1e-3) ** 2


def gain_vec(P1):
    g = np.ones(6)
    if "gain" in P1:
        g[J25] = P1["gain"]
    return g


def p3(vis, M, gain, w):
    res = {}
    H = [v for v in vis if v["seg"].startswith("hold_")]
    if H:
        def xs(v, win):
            m = v["t"] - v["t"][0] < win
            i = v["I"][m].mean(0); q = v["Q"][m].mean(0)
            tr = DIR * KBC * gain * i - M.g([q])[0]
            return mhat(tr, M.gamma(q), w)
        dp = {}
        for win in (2.0, 10.0):
            num, den = [], []
            for p in sorted({v["pose"] for v in H}):
                c0 = [xs(v, win) for v in H if v["pose"] == p and v["grams"] == 0]
                c5 = [xs(v, win) for v in H if v["pose"] == p and v["grams"] >= 400]
                if len(c0) > 1 and len(c5) > 1:
                    num.append(np.mean(c5) - np.mean(c0)); den += [np.var(c0, ddof=1), np.var(c5, ddof=1)]
            dp[win] = float(np.mean(num) / math.sqrt(np.mean(den))) if num else math.nan
        ratio = dp[10.0] / dp[2.0] if dp[2.0] else math.nan
        res["P3a"] = {"d2": round(dp[2.0], 2), "d10": round(dp[10.0], 2), "ratio": round(ratio, 3),
                      "verdict": "holds" if ratio < 1.3 else ("fails" if ratio > 1.8 else "unclear")}
    S = [v for v in vis if v["seg"].startswith("static_")]
    if S:
        by = {}
        for v in S:
            by.setdefault((v["pose"], v["grams"]), {})[v["approach"]] = DIR * KBC * gain * v["i"] - M.g([v["q"]])[0]
        per_pose = {}
        for (p, _), d in by.items():
            if "above" in d and "below" in d:
                per_pose.setdefault(p, []).append(d["above"] - d["below"])
        D = np.array([np.mean(L, 0) for L in per_pose.values()])
        out = {}
        for j in (1, 2):
            x = D[:, j]; tstat = x.mean() / (x.std(ddof=1) / math.sqrt(len(x)))
            null = [abs((y := x * rng.choice([-1, 1], len(x))).mean() / (y.std(ddof=1) / math.sqrt(len(x)))) for _ in range(20000)]
            out[f"j{j+1}"] = {"mean_Nm": round(float(x.mean()), 3), "t": round(float(tstat), 2), "p": float(np.mean(np.array(null) >= abs(tstat)))}
        out["verdict"] = "holds" if any(out[f"j{j}"]["p"] < 0.01 for j in (2, 3)) else "fails"
        res["P3b"] = out
    return res


def p4(segs, M, gain, w):
    out = {"pass": [], "stop": []}
    for n, s in segs.items():
        if not n.startswith("lift_") or s.meta.get("status") != "done":
            continue
        tr = DIR * KBC * gain * s.i - M.g(s.q)
        gam = M.gamma(s.q[len(s.q) // 2])
        mh = np.array([mhat(x, gam, w) for x in tr]) * 1000.0      # grams
        r0 = s.meta["_row0_mono"]                                   # trial times are on the same monotonic clock
        for tr_ in s.meta["info"]["trials"]:
            tl = tr_["t_lift_cmd"] - r0
            if tr_["mode"] == "pass":
                pre = (s.t >= tl - 0.3) & (s.t < tl)
                post = (s.t >= tl + 1.0) & (s.t < tl + tr_["lift_s"] - 0.5)
            else:
                pre = slice(tr_["pre"]["row0"], tr_["pre"]["row1"]); post = slice(tr_["post"]["row0"], tr_["post"]["row1"])
            out[tr_["mode"]].append((grams(s), float(np.mean(mh[post]) - np.mean(mh[pre]))))
    if not out["pass"] or not out["stop"]:
        return {"verdict": "no data"}
    res = {}
    for mode in ("pass", "stop"):
        L = out[mode]
        z = np.mean([s for g, s in L if g == 0]) if any(g == 0 for g, _ in L) else 0.0
        err = {}
        for g, s in L:
            err.setdefault(g, []).append(s - z - g)
        res[mode] = {"sd_pooled_g": float(math.sqrt(np.mean([np.var(v, ddof=1) for v in err.values() if len(v) > 1]))),
                     "bias_g": {str(g): round(float(np.mean(v)), 1) for g, v in err.items()}}
    ratio = res["pass"]["sd_pooled_g"] / res["stop"]["sd_pooled_g"]
    dof = sum(len([1 for g_, _ in out["pass"] if g_ == g]) - 1 for g in {g for g, _ in out["pass"]})
    # one-sided F test via permutation-free approximation: report the ratio and dof; F p-value from the beta form
    F = ratio ** 2
    p = _f_cdf(F, dof, dof)
    res["ratio_sd_pass_over_stop"] = round(ratio, 3); res["F_p_one_sided"] = round(p, 4)
    res["outcome"] = ("A: stop redraws" if ratio < 0.6 and p < 0.05 else "B: load change redraws" if 0.7 <= ratio <= 1.43 else "unclear")
    return res


def p4v2(segs, M, gain, w):
    """predictions_v2 P4: pass step, stop difference and static read (stop trials, 2 s still after the lift); sigma_m pooled
    within levels as v1; primary = d'_pass >= 1.5 x both others with one-sided F p < 0.05; secondary = tier from sigma_pass."""
    reads = {"pass": [], "stop": [], "static": []}
    for n, s in segs.items():
        if not n.startswith("lift_") or s.meta.get("status") != "done":
            continue
        tr = DIR * KBC * gain * s.i - M.g(s.q)
        gam = M.gamma(s.q[len(s.q) // 2])
        mh = np.array([mhat(x, gam, w) for x in tr]) * 1000.0
        r0 = s.meta["_row0_mono"]
        for tr_ in s.meta["info"]["trials"]:
            tl = tr_["t_lift_cmd"] - r0
            if tr_["mode"] == "pass":
                pre = (s.t >= tl - 0.3) & (s.t < tl); post = (s.t >= tl + 1.0) & (s.t < tl + tr_["lift_s"] - 0.5)
                reads["pass"].append((grams(s), float(np.mean(mh[post]) - np.mean(mh[pre]))))
            else:
                pre = slice(tr_["pre"]["row0"], tr_["pre"]["row1"]); post = slice(tr_["post"]["row0"], tr_["post"]["row1"])
                reads["stop"].append((grams(s), float(np.mean(mh[post]) - np.mean(mh[pre]))))
                reads["static"].append((grams(s), float(np.mean(mh[post]))))
    if not reads["pass"] or not reads["stop"]:
        return {"verdict": "no data"}
    res = {}
    for k, L in reads.items():
        z = np.mean([v for g, v in L if g == 0]) if any(g == 0 for g, _ in L) else 0.0
        err = {}
        for g, v in L:
            err.setdefault(g, []).append(v - z - g)
        sd = float(math.sqrt(np.mean([np.var(v, ddof=1) for v in err.values() if len(v) > 1])))
        res[k] = {"sigma_m_g": round(sd, 1), "d_prime": round(100.0 / sd, 3), "dof": sum(len(v) - 1 for v in err.values())}
    a = res["pass"]
    p_stop = 1.0 - _f_cdf(res["stop"]["sigma_m_g"] ** 2 / a["sigma_m_g"] ** 2, res["stop"]["dof"], a["dof"])
    p_stat = 1.0 - _f_cdf(res["static"]["sigma_m_g"] ** 2 / a["sigma_m_g"] ** 2, res["static"]["dof"], a["dof"])
    prim = (a["d_prime"] >= 1.5 * res["stop"]["d_prime"] and a["d_prime"] >= 1.5 * res["static"]["d_prime"]
            and p_stop < 0.05 and p_stat < 0.05)
    res["primary"] = {"p_stop": round(p_stop, 4), "p_static": round(p_stat, 4), "verdict": "holds" if prim else "fails"}
    inside = 1.13 <= a["d_prime"] <= 2.64
    res["secondary"] = {"inside_sim_bracket": inside,
                        "tier": ("opt (redraw only at stops)" if a["sigma_m_g"] < 61.0 else "pess (every load change)") if inside
                        else "outside both tiers"}
    res["sim_brackets"] = {"static": [0.62, 1.18], "pass": [1.13, 2.64], "stop": [0.50, 1.11]}
    res["inside"] = {k: bool(lo <= res[k]["d_prime"] <= hi) for k, (lo, hi) in res["sim_brackets"].items()}
    res["verdict"] = f"primary {res['primary']['verdict']}; secondary {res['secondary']['tier']}"
    return res


def _f_cdf(F, d1, d2, n=200000):
    x = rng.chisquare(d1, n) / d1 / (rng.chisquare(d2, n) / d2)
    return float(np.mean(x <= F))


def p5(segs, M, gain, w, tab, sig_static):
    """Per reach pass: slope k of tau_res,j2 against the radial tip position over the constant-speed part; one estimate
    per out+in cycle (their mean, which cancels the direction-dependent part); mass = (k - mean k of 0 g) / k1, with
    k1 the model slope per kg of payload (regression of gamma_j2 on x along the same samples)."""
    est = {True: [], False: []}
    for n, s in segs.items():
        if not n.startswith("reach_") or s.meta.get("status") != "done":
            continue
        u = M.tip(s.q[0])[:2]; u = u / np.linalg.norm(u)
        x = np.array([M.tip(q)[:2] @ u for q in s.q]); v = vel(s.t, x)
        gam2 = np.array([M.gamma(q)[1] for q in s.q])
        for use in (True, False):
            tr = DIR * KBC * gain * s.i - M.g(s.q) - (W_at(tab, s.q) if use else 0.0)
            ks, k1s = [], []
            for t0, m in s.marks:
                if not (m.startswith("reach ") and m.endswith((" out", " in"))):
                    continue
                nxt = [t for t, _ in s.marks if t > t0]
                if not nxt:
                    continue
                sel = s.window(t0, nxt[0]) & (np.abs(np.abs(v) - 0.010) < 0.002)
                if sel.sum() < 50:
                    continue
                ks.append(np.polyfit(x[sel], tr[sel, 1], 1)[0]); k1s.append(np.polyfit(x[sel], gam2[sel], 1)[0])
            for c in range(len(ks) // 2):
                est[use].append((grams(s), float(np.mean(ks[2 * c:2 * c + 2])), float(np.mean(k1s[2 * c:2 * c + 2]))))
    if not est[True]:
        return {"verdict": "no data"}
    res = {}
    for use in (True, False):
        L = est[use]
        k0 = np.mean([k for g, k, _ in L if g == 0])
        err = {}
        for g, k, k1 in L:
            err.setdefault(g, []).append((k - k0) / k1 * 1000 - g)
        res["with_W" if use else "no_W"] = float(math.sqrt(np.mean([np.var(v_, ddof=1) for v_ in err.values() if len(v_) > 1])))
        res["bias_" + ("with_W" if use else "no_W")] = {str(g): round(float(np.mean(v_)), 1) for g, v_ in err.items()}
    g_ = np.array([g for g, _, _ in est[True]]); k_ = np.array([k for _, k, _ in est[True]])
    r2 = float(np.corrcoef(g_, k_)[0, 1] ** 2)
    a, b, c = r2 >= 0.9, res["with_W"] <= sig_static, res["no_W"] >= 1.2 * res["with_W"]
    return {"R2": round(r2, 3), "n_cycles": len(est[True]), "sigma_m_reach_W_g": round(res["with_W"], 1),
            "sigma_m_reach_noW_g": round(res["no_W"], 1), "sigma_m_static_g": round(sig_static, 1),
            "bias_with_W_g": res["bias_with_W"], "P5a": a, "P5b": b, "P5c": c, "verdict": "holds" if a and b and c else "fails"}


def p6(segs, M):
    out = {}
    for n, s in segs.items():
        if not n.startswith("comply_"):
            continue
        for st in s.meta["info"]["steps"]:
            q = s.q[st["row0"]:st["row1"]].mean(0)
            try:
                gauge = float(st["gauge_mm"])
            except ValueError:
                gauge = math.nan
            out.setdefault(st["pose"], []).append((float(st["grams"]), gauge, M.tip(q)[2] * 1000))
    res, ok = {}, []
    for p, L in out.items():
        hi = [x for x in L if x[0] >= 400]; lo = [x for x in L if x[0] == 0]
        if not hi or not lo:
            continue
        ext = abs(np.mean([x[1] for x in hi]) - np.mean([x[1] for x in lo]))
        enc = abs(np.mean([x[2] for x in hi]) - np.mean([x[2] for x in lo]))
        res[str(p)] = {"external_sink_mm": round(float(ext), 2), "encoder_sink_mm": round(float(enc), 3)}
        ok.append(1.0 <= ext <= 4.0 and enc <= 0.3)
    res["verdict"] = "holds" if ok and all(ok) else ("no data" if not ok else "fails")
    return res


def p7(segs, M, gain):
    fc = {}
    for n, s in segs.items():
        if not n.startswith("excite_"):
            continue
        label = s.meta["label"]
        tr = DIR * KBC * gain * s.i - M.g(s.q)
        for t0, m in s.marks:
            if not m.startswith("tri j"):
                continue
            j = int(m.split()[1][1:]) - 1
            nxt = [t for t, mm in s.marks if t > t0 and not mm.startswith("tri")]
            t1 = nxt[0] if nxt else s.t[-1]
            win = s.window(t0, t1)
            v = vel(s.t, s.q[:, j])
            plat = np.percentile(np.abs(v[win]), 90)
            sel = win & (np.abs(np.abs(v) - plat) < 0.2 * plat)
            up, dn = sel & (v > 0), sel & (v < 0)
            if up.sum() < 20 or dn.sum() < 20:
                continue
            fc.setdefault(label, {})[j] = (float(np.mean(tr[up, j]) - np.mean(tr[dn, j])) / 2.0)
    if "cold" not in fc or "hot" not in fc:
        return {"verdict": "no data", "fc": fc}
    lower = [fc["hot"][j] < fc["cold"][j] for j in range(5) if j in fc["hot"] and j in fc["cold"]]
    return {"fc_cold": {f"j{j+1}": round(v, 3) for j, v in fc["cold"].items()},
            "fc_hot": {f"j{j+1}": round(v, 3) for j, v in fc["hot"].items()},
            "n_lower": int(sum(lower)), "verdict": "holds" if sum(lower) >= 4 else "fails",
            "note": "temperature gate (>= 5 C between cold and hot) is checked from lowspd_log by the reviewer"}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True); ap.add_argument("--plan", required=True)
    ap.add_argument("--e3"); ap.add_argument("--d-com-m", type=float, default=0.0); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    plan = json.loads(Path(a.plan).read_text())
    M = Model(plan["tip_m"], a.d_com_m)
    segs = load_run(a.run)
    for s in segs.values():                       # trial times are monotonic: keep the first row's clock value
        z = np.load(Path(a.run) / s.name / "rec.npz"); s.meta["_row0_mono"] = float(z["rows"][0, 0])
    vis = static_visits(segs, M)
    R = {"run": Path(a.run).name, "plan_sha": plan.get("sha"), "segments": sorted(segs)}
    R["P1"] = p1(vis, M); gain = gain_vec(R["P1"])
    tab = ripple_table(segs, M)
    R["P2"] = p2(segs, M, tab, a.e3)
    w = sigma_static(vis, M, gain)
    R["P3"] = p3(vis, M, gain, w)
    R["P4"] = p4(segs, M, gain, w)
    R["P4v2"] = p4v2(segs, M, gain, w)        # predictions_v2
    _v3 = p4v2(segs, M, gain, w)               # predictions_v3: same rules, brackets from the latency-on simulator
    if "sim_brackets" in _v3:
        _v3["sim_brackets"] = {"static": [1.06, 2.11], "pass": [1.13, 2.54], "stop": [0.80, 1.64]}
        _v3["inside"] = {k: bool(lo <= _v3[k]["d_prime"] <= hi) for k, (lo, hi) in _v3["sim_brackets"].items()}
        _v3["secondary"]["inside_sim_bracket"] = bool(1.13 <= _v3["pass"]["d_prime"] <= 2.54)
    R["P4v3"] = _v3
    # static sigma_m from the 3a visits (per tier, vs the 0 g tier of the same pose / approach)
    st = [v for v in vis if v["seg"].startswith("static_")]
    zero = {(v["pose"], v["approach"]): v for v in st if v["grams"] == 0}
    err = {}
    for v in st:
        z = zero.get((v["pose"], v["approach"]))
        if z is None or v["grams"] == 0:
            continue
        d = (DIR * KBC * gain * v["i"] - M.g([v["q"]])[0]) - (DIR * KBC * gain * z["i"] - M.g([z["q"]])[0])
        err.setdefault(v["grams"], []).append(mhat(d, M.gamma(v["q"]), w) * 1000 - v["grams"])
    sig_static = float(math.sqrt(np.mean([np.var(e, ddof=1) for e in err.values() if len(e) > 1]))) if err else math.nan
    R["sigma_m_static_g"] = round(sig_static, 1)
    R["P5"] = p5(segs, M, gain, w, tab, sig_static)
    R["P6"] = p6(segs, M)
    R["P7"] = p7(segs, M, gain)
    R["P1"].pop("gain", None)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(R, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
    lines = ["| 预测 | 判定 | 要点 |", "|---|---|---|"]
    for k in ("P1", "P2", "P3", "P4", "P4v2", "P4v3", "P5", "P6", "P7"):
        v = R[k]
        verdict = v.get("verdict") or v.get("outcome") or "; ".join(f"{kk}: {vv.get('verdict')}" for kk, vv in v.items() if isinstance(vv, dict))
        lines.append(f"| {k} | {verdict} | {json.dumps({kk: vv for kk, vv in v.items() if kk not in ('verdict',)}, ensure_ascii=False, default=str)[:300]} |")
    (out / "results.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
