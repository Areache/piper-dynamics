# Validation

Without a force sensor, the only way to know whether `tau_ext` is real is to
apply a force you already know. That is what this procedure does, and it is
the thing that decides whether anyone else can trust the output.

> **Status: procedure written, results not yet collected.** The plots below
> are described, not measured. If you run this, please open an issue with
> yours.

---

## 0. First: check your units (2 minutes)

Before anything else. Most "the estimate is 1000x off" reports are a unit
scale, not a physics problem.

Hold the arm still, gripper empty, in two very different poses — arm
outstretched horizontally, and arm folded up vertically.

```python
r1 = est.decompose(q_outstretched, v_zero, current_1)
r2 = est.decompose(q_folded,       v_zero, current_2)
print(r1.tau_meas, r1.tau_model)
print(r2.tau_meas, r2.tau_model)
```

What you should see:

- `tau_meas` and `tau_model` the same **order of magnitude** and the same
  **sign** in both poses. If they differ by ~1000x, fix `UNIT_SCALE` in
  `io/can.py`. If they have opposite signs, fix `direction` in your params.
- `tau_meas` **changing a lot** between the two poses. If it barely moves,
  your firmware is already compensating gravity somewhere and this package's
  model of the signal is wrong — stop and open an issue.
- `tau_ext` close to zero in both. This is the real test of calibration: an
  empty gripper at rest means no external load.

The last one is the single most useful number in this package. Whatever
`|tau_ext|` is here with nothing in the gripper, that is your noise floor, and
nothing below it is a measurement.

## 1. Known-weight validation (30 minutes)

**What you need:** a set of weights spanning your payload range. Calibration
weights if you have them; water bottles on a kitchen scale work fine — write
down the actual masses.

**Procedure**

1. Pick ~10 poses spread across the workspace where the gripper can hold a
   weight safely. Avoid near-singular configurations; `wrench_from_tau_ext`
   will flag them, and they are where the estimate is worst.
2. Empty gripper: record ~2 s at rest in each pose. This is your zero.
3. For each weight (e.g. 100 g, 250 g, 500 g, 1 kg): grip it, visit the same
   poses, record ~2 s at rest in each.
4. Estimate:

```python
from piper_dynamics.estimation import estimate_payload_mass

for mass_true, (q, v, c) in recordings.items():
    tau_ext = est.batch(q, v, c)
    out = estimate_payload_mass(model, q, tau_ext)
    print(f"{mass_true:5.2f} kg  ->  {out.mass_kg:5.2f} ± {out.stderr_kg:.3f} kg"
          f"   (observability {out.observability:.1f})")
```

**What to report**

- A scatter of estimated vs true mass, with a y=x line and error bars. This
  goes at the top of the README.
- Residual vs pose. If the error is systematic with pose rather than random,
  the remaining error is in the gravity model, not in the noise — and that is
  fixable by better identification.
- The empty-gripper number from step 0, as your floor.

**What counts as good:** that depends on your task. For distinguishing "holding
something" from "not holding something", a floor well under the lightest object
you care about is enough. For reading mass as a continuous physical property,
you want the residual scatter small relative to the differences you intend to
resolve. State which one you measured; do not quote a single accuracy number
without saying what for.

## 2. Free-space consistency (continuous check)

A check that needs no weights and can run over data you already have: during
free-space motion with an empty gripper, `tau_ext` should be ≈ 0 at all times.

```python
tau_ext = est.batch(q, v, c)
print(np.abs(tau_ext).mean(axis=0))      # per joint
```

Any structure here is model error. Useful diagnostics:

- **Grows with velocity** → friction is unidentified or wrong (expected: static
  calibration does not identify friction).
- **Grows with acceleration** → inertial parameters are off, or your
  acceleration estimate is noisy. Try `quasi_static=True` and see if it drops.
- **Constant offset in one joint** → `current_offset`, or something attached to
  the arm that is not in the URDF (a camera, a different gripper).
- **Varies with pose at rest** → the gravity model is wrong. This is the one
  that matters most for payload estimation, because it is indistinguishable
  from a payload.

## 3. What this cannot tell you

Known-weight validation measures the quasi-static, gravity-dominated regime.
It says nothing about:

- fast contact transients (bandwidth is limited by the telemetry rate and by
  the smoothing in your acceleration estimate);
- forces in directions the Jacobian barely projects onto the joints — at some
  configurations an entire force direction is carried by the structure and is
  invisible to the motors;
- torques about the end-effector axis, which on a 6-DOF arm with a parallel
  gripper are poorly observed.

None of these are fixed by better calibration. They are properties of
estimating force from joint torque, and they are the honest reason a wrist F/T
sensor is still worth its price for some tasks.
