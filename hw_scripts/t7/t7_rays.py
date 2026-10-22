#!/usr/bin/env python3
"""One guarded descent = one ray. Turns each recorded descent into free-space evidence and contact stiffness.
Server side (pinocchio + the gripper URDF), offline, reads v2.2-v2.5 logs (older logs: fields they lack -> "缺").

Per descent:
  x, y          tip position at detection (FK of the measured q, nominal tip 0.15 m along gripper_base z)
  z_start       tip z at the first logged row
  z_mon         tip z where monitoring starts (arming). The first 20 mm are friction-only and NOT covered by
                the certificate: [z_mon, z_start] is blind and is written out as such.
  z_det         tip z at the row where the stop was decided
  z_onset       last row before the stop where the centred 0.2 s mean force is inside the noise band
  z_fc          first contact: Theil-Sen line of the instantaneous F(z) on the loading branch (onset -> lowest
                point), evaluated at the top of the noise band
  overshoot     z_fc - z_det, and z_onset - z_det (mm, > 0 = detected below the first contact)
  certificate   max causal 0.2 s mean tip force over the monitored rows above max(z_onset, z_fc), and its
                margin to the threshold: "[z_onset, z_mon] is free at F_thr"
  k_load        loading stiffness -dF/dz (N/mm); k_unload and hysteresis (N*mm = mJ, > 0 = energy lost) only if
                >= 5 back-off rows are above the band, else "分辨不出"
  F_rest_after  median force over the last 0.2 s (arm still after backing off): the friction preload left by
                the last direction of motion shows up here
Tip force: recomputed for every row from the logged residuals with the v2.3 rule (j2+j3 weighted LS onto the
plan's tip-force signature, free-descent friction mu x phi, 30 ms low-pass) so that all versions are comparable;
after the stop phi = 0 (no friction model for upward motion). Noise band top = 3 x the free-descent spread of the
0.2 s mean force (logged sd_Fmean_free_N). Theil-Sen (median pairwise slope) because streamed move_j delivers the
contact force as ~5 Hz pulses. On a rigid plate the 4-5 mm back-off takes ~0.1 s, so the unloading branch is
often not resolvable at 100 Hz. Tip z comes from the motor-side joint angles, so after contact it includes the
arm's own bending: k_load is the series stiffness (arm + servo + object) seen by the robot, not the object's.

usage: venv/bin/python hw_scripts/t7/t7_rays.py [run folders ...] [--out rays.csv]
       (no folders: the 33 T7 contact descents used in t7_final.py)
"""
import argparse, csv, json, sys
from pathlib import Path
import numpy as np
import pinocchio as pin

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from t7_plan import build_model  # noqa: E402

T7_RUNS = ["T7_right_p0_v10_rep", "T7_right_p2_v10_rep"] + ["T7_right_p%d_v23" % i for i in (0, 1, 3, 4, 5, 6)]
M, FID = build_model(0.15); D = M.createData()
J = [1, 2]


def tip_xyz(Qrows):
    out = np.empty((len(Qrows), 3))
    for i, q in enumerate(Qrows):
        pin.framesForwardKinematics(M, D, np.asarray(q, float)); out[i] = D.oMf[FID].translation
    return out * 1000.0


def tip_force(rl, mu, sd, a):
    dev = (rl - mu)[J]; aj = a[J]; w = 1.0 / np.maximum(sd[J], 1e-4) ** 2
    return float((w * aj * dev).sum() / max((w * aj * aj).sum(), 1e-12))


def recompute_F(d, plan, speed, kd):
    """Instantaneous tip force on every row (v2.3 rule). kd = row where the stop was decided (rows after it are
    the back-off; their waypoint comes from q, phi = 0)."""
    Q = np.asarray(plan["q"]); A = np.asarray(plan["di_per_N_up"]); step = plan["step_mm"]
    mu, sd, R, t = d["mu_free"], d["sd_free"], d["resid"], d["t"]
    n = len(t)
    if not np.all(np.isfinite(mu[J])):
        return np.full(n, np.nan), None
    if "k_now" in d:
        k = d["k_now"].astype(int)
    else:
        k = d["k_meas"].astype(int).copy()
        for i in range(kd + 1, n):
            lo, hi = max(0, k[i - 1] - 20), min(len(Q), k[i - 1] + 40)
            k[i] = lo + int(np.argmin(np.abs(Q[lo:hi] - d["q"][i]).max(1)))
    # phi as online: tip speed over a window covering >= 3 waypoints, divided by the reference speed
    phi = np.zeros(n)
    if "fz_phi" in d:
        phi = d["fz_phi"]
    else:
        win = max(0.1, 3 * step / speed); j0 = 0
        km = d["k_meas"]
        for i in range(min(kd + 1, n)):
            while t[i] - t[j0] > win:
                j0 += 1
            if t[i] - t[j0] > 0.5 * win:
                phi[i] = np.clip((km[i] - km[j0]) * step / (t[i] - t[j0]) / speed, 0, 1)
    F = np.full(n, np.nan)
    for i in range(n):
        rl = np.nanmean(R[max(0, i - 2): i + 1], 0)
        F[i] = tip_force(rl, mu * phi[i], sd, A[min(k[i], len(A) - 1)])
    return F, k


def window_mean(t, F, w=0.2, centred=False):
    out = np.full(len(t), np.nan)
    c = np.r_[0.0, np.cumsum(np.nan_to_num(F))]; cn = np.r_[0, np.cumsum(np.isfinite(F))]
    for i in range(len(t)):
        lo_t, hi_t = (t[i] - w / 2, t[i] + w / 2) if centred else (t[i] - w, t[i])
        a = np.searchsorted(t, lo_t, side="right" if not centred else "left"); b = np.searchsorted(t, hi_t, side="right")
        if (t[min(b, len(t)) - 1] - t[a]) >= 0.8 * w and cn[b] - cn[a] > 0:
            out[i] = (c[b] - c[a]) / (cn[b] - cn[a])
    return out


def theil_sen(x, y):
    """Median pairwise slope and the matching intercept (robust to the pulsed contact force)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    i, j = np.triu_indices(len(x), 1)
    dx = x[j] - x[i]; ok = np.abs(dx) > 1e-6
    if ok.sum() < 3:
        return np.nan, np.nan
    b = float(np.median((y[j] - y[i])[ok] / dx[ok]))
    return b, float(np.median(y - b * x))


def one_ray(run_dir, tr, plan, S):
    d = dict(np.load(run_dir / f"trial_{tr['trial']:02d}.npz"))
    t = d["t"]; n = len(t)
    speed = S.get("speed_mm_s") or plan["speed_mm_s"]
    if tr.get("motion") == "line" and tr.get("free_speed_mm_s"):
        speed = tr["free_speed_mm_s"]
    if "state" in d and (d["state"] == 2).any():
        kd = int(np.flatnonzero(d["state"] == 2)[0])
    else:
        kd = int(np.argmax(d["hit"] > 0)) if (d["hit"] > 0).any() else int(np.argmax(d["k_meas"]))
    F, k = recompute_F(d, plan, speed, kd)
    P = tip_xyz(d["q"]); z = P[:, 2]
    Fm = window_mean(t, F, 0.2, centred=False)            # what the detector saw
    Fc = window_mean(t, F, 0.2, centred=True)             # offline, no lag
    armed = d["armed"] > 0
    i_mon = int(np.argmax(armed)) if armed.any() else None
    thr = float(d["thr_N"]) if np.isfinite(d["thr_N"]) else np.nan
    sdm = float(d["sd_Fmean_free_N"]) if np.isfinite(d["sd_Fmean_free_N"]) else np.nan
    band = 3 * sdm                                         # top of the noise band of the 0.2 s mean
    i_low = kd + int(np.argmin(z[kd:]))                    # turn point (lowest tip) after the stop decision
    rec = {"run": run_dir.name, "trial": tr["trial"], "status": tr["status"], "speed_mm_s": round(float(speed), 1),
           "version": tr.get("version", "2.5" if "state" in d else ("2.4" if "fz_F_N" in d else "<=2.3")),
           "x_mm": P[kd, 0], "y_mm": P[kd, 1], "z_start_mm": z[0], "z_mon_mm": z[i_mon] if i_mon is not None else np.nan,
           "z_det_mm": z[kd], "z_low_mm": z[i_low], "thr_N": thr, "band_N": band,
           "z_onset_mm": np.nan, "z_fc_mm": np.nan, "overshoot_mm": np.nan, "overshoot_onset_mm": np.nan,
           "k_load_N_per_mm": np.nan, "n_load": 0, "k_unload_N_per_mm": np.nan, "n_unload": 0, "hysteresis_mJ": np.nan,
           "F_peak_N": float(np.nanmax(F[kd:i_low + 3])) if np.isfinite(F[kd:i_low + 3]).any() else np.nan,
           "F_rest_after_N": float(np.nanmedian(F[-20:])), "backoff_s": np.nan, "unload": "缺",
           "cert_max_N": np.nan, "cert_margin_N": np.nan, "cert_interval_mm": "", "blind_interval_mm": "", "note": ""}
    # onset: last row before the stop where the centred 0.2 s mean is inside the noise band
    inside = np.flatnonzero((Fc[:kd + 1] <= band) & np.isfinite(Fc[:kd + 1]))
    i_on = int(inside[-1]) if len(inside) else None
    if i_on is not None:
        rec["z_onset_mm"] = z[i_on]; rec["overshoot_onset_mm"] = z[i_on] - z[kd]
        seg = np.arange(i_on, i_low + 1); seg = seg[np.isfinite(F[seg])]
        rec["n_load"] = len(seg)
        if len(seg) >= 4 and np.ptp(z[seg]) > 0.3:
            b, a = theil_sen(z[seg], F[seg])               # loading branch of the instantaneous force vs tip z
            if np.isfinite(b) and b < 0:
                rec["k_load_N_per_mm"] = -b
                rec["z_fc_mm"] = (band - a) / b
                rec["overshoot_mm"] = rec["z_fc_mm"] - z[kd]
            else:
                rec["note"] += "loading slope not negative; "
        else:
            rec["note"] += f"loading branch {len(seg)} rows / {np.ptp(z[seg]) if len(seg) else 0:.1f} mm; "
    # back-off time: from the turn until the tip z stops changing
    z_end = float(np.median(z[-10:]))
    near = np.flatnonzero(np.abs(z[i_low:] - z_end) < 0.3)
    rec["backoff_s"] = float(t[i_low + near[0]] - t[i_low]) if len(near) else np.nan
    # unloading: after the turn, rows until the centred mean is back inside the band
    after = np.arange(i_low, n)
    back = after[np.isfinite(Fc[after]) & (Fc[after] <= band)]
    i_out = int(back[0]) if len(back) else None
    if i_out is not None:
        seg = np.arange(i_low, i_out + 1); seg = seg[np.isfinite(F[seg]) & (F[seg] > band)]
        rec["n_unload"] = len(seg)
        if len(seg) >= 5 and np.ptp(z[seg]) > 0.3:
            b, a = theil_sen(z[seg], F[seg]); rec["k_unload_N_per_mm"] = -b; rec["unload"] = "ok"
            if i_on is not None:
                loop = np.arange(i_on, i_out + 1); loop = loop[np.isfinite(F[loop])]
                rec["hysteresis_mJ"] = float(-np.trapezoid(F[loop] - band, z[loop]))
        else:
            rec["unload"] = f"分辨不出（{len(seg)} 行在噪声带以上）"
    # certificate: monitored rows above the onset (the conservative end of the contact-free interval)
    z_ref = rec["z_onset_mm"] if np.isfinite(rec["z_onset_mm"]) else z[kd]
    if np.isfinite(rec["z_fc_mm"]):
        z_ref = max(z_ref, rec["z_fc_mm"])
    if i_mon is not None:
        mon = np.arange(i_mon, kd + 1); mon = mon[(z[mon] > z_ref) & np.isfinite(Fm[mon])]
        if len(mon):
            rec["cert_max_N"] = float(np.max(Fm[mon])); rec["cert_margin_N"] = thr - rec["cert_max_N"]
            rec["cert_interval_mm"] = f"[{z_ref:.1f}, {z[i_mon]:.1f}]"; rec["blind_interval_mm"] = f"[{z[i_mon]:.1f}, {z[0]:.1f}]"
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="*"); ap.add_argument("--out", default=str(HERE / "runs" / "t7_rays.csv"))
    ap.add_argument("--status", default="contact", help="statuses to include (comma-separated)")
    a = ap.parse_args()
    runs = [Path(r) for r in a.runs] or [HERE / "runs" / r for r in T7_RUNS]
    keep = set(a.status.split(","))
    recs = []
    for rd in runs:
        S = json.loads((rd / "summary.json").read_text())
        plan = json.loads((HERE / "plans" / S["plan"]).read_text())
        for tr in S["trials"]:
            if tr["status"] in keep and (rd / f"trial_{tr['trial']:02d}.npz").exists():
                recs.append(one_ray(rd, tr, plan, S))
    keys = list(dict.fromkeys(k for r in recs for k in r))
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, keys); w.writeheader()
        for r in recs:
            w.writerow({k: (f"{v:.3f}" if isinstance(v, float) else v) for k, v in r.items()})
    print(f"{len(recs)} rays -> {a.out}")
    R = recs
    def arr(k):
        return np.array([r[k] for r in R], float)
    cm = arr("cert_margin_N"); cx = arr("cert_max_N"); th = arr("thr_N")
    print("\n(1) certificate: max causal 0.2 s mean force above the first contact, and margin to the threshold")
    print(f"    n {np.isfinite(cm).sum()}/{len(R)}; max force median {np.nanmedian(cx):.2f} N (worst {np.nanmax(cx):.2f}); "
          f"margin median {np.nanmedian(cm):.2f} N, min {np.nanmin(cm):.2f} N, quartiles {np.nanpercentile(cm, 25):.2f}-{np.nanpercentile(cm, 75):.2f}; "
          f"max force / threshold median {np.nanmedian(cx / th):.2f}")
    print("\n(2) per group: loading stiffness, overshoot (detection below first contact), peak force, rest force after")
    print("    group           n   k_load N/mm         overshoot fit mm    overshoot onset mm  F_peak N            F_rest_after N      unload ok")
    groups = sorted({r["run"].split("_v")[0] if r["run"].startswith("T7") else r["run"] for r in R})
    for p in groups + ["all"]:
        sub = [r for r in R if p == "all" or (r["run"].split("_v")[0] if r["run"].startswith("T7") else r["run"]) == p]
        def ms(k):
            x = np.array([r[k] for r in sub], float); x = x[np.isfinite(x)]
            return f"{x.mean():6.2f} ± {x.std():4.2f} ({len(x):2d})" if len(x) else "  缺               "
        nu = sum(1 for r in sub if r["unload"] == "ok")
        print(f"    {p[-12:]:14s} {len(sub):3d}  {ms('k_load_N_per_mm')}  {ms('overshoot_mm')}  {ms('overshoot_onset_mm')}  "
              f"{ms('F_peak_N')}  {ms('F_rest_after_N')}  {nu}/{len(sub)}")
    print()
    for key, lab in (("overshoot_mm", "fit (z_fc - z_det)"), ("overshoot_onset_mm", "onset (z_onset - z_det)")):
        for sp in sorted({r["speed_mm_s"] for r in R}):
            ov = np.array([r[key] for r in R if r["speed_mm_s"] == sp], float); ov = ov[np.isfinite(ov)]
            if len(ov):
                print(f"(3) overshoot, {lab}, {sp} mm/s: mean {ov.mean():.2f} mm, sd {ov.std():.2f} mm, "
                      f"range {ov.min():.2f}-{ov.max():.2f} (n {len(ov)})")
    ku = arr("k_unload_N_per_mm"); hy = arr("hysteresis_mJ")
    print(f"    unloading resolvable in {np.isfinite(ku).sum()}/{len(R)}; back-off time median {np.nanmedian(arr('backoff_s')):.2f} s"
          + (f"; k_unload median {np.nanmedian(ku):.2f} N/mm, hysteresis median {np.nanmedian(hy):.2f} mJ" if np.isfinite(ku).any() else ""))
    notes = [f"{r['run']}#{r['trial']}: {r['note']}" for r in R if r["note"]]
    if notes:
        print("\nnotes:\n  " + "\n  ".join(notes))


if __name__ == "__main__":
    main()
