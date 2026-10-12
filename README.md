# piper-dynamics

**Identified dynamics, external torque and payload estimation for the AgileX Piper arm.**

> Status: early. The estimation core is implemented and unit-tested; the
> reference calibration has not been measured on hardware yet. See
> [Status](#status) before depending on it.

---

## The problem

Your recorded episodes have an `effort` field. It looks like force data. It is
not useful as force data.

What the motors report is total torque, and almost all of it is the arm
holding itself up:

```
tau_meas  =  M(q)q̈ + C(q,q̇)q̇ + g(q)  +  tau_friction  +  tau_ext
             └────────── the arm carrying itself ──────┘     └─ what you want
```

The gravity term swings by several N·m across the workspace. A 200 g object in
the gripper moves the reading by a fraction of one. Feed `effort` straight into
a policy and it learns *what pose the arm is in* — which the cameras already
told it — not *what the arm is touching*.

Arms with built-in joint torque sensors hand you the residual directly: the
subtraction happens in firmware, which is why work like
[FACTR](https://arxiv.org/abs/2502.17432) can treat external joint torque as a
field you simply read. Piper gives you the signal from *before* that
subtraction.

**This package does the subtraction in software.**

```python
from piper_dynamics import PiperModel, ArmParams, TauExtEstimator

model  = PiperModel("piper_description.urdf")
params = ArmParams.load("params/my_arm.yaml")
est    = TauExtEstimator(model, params, quasi_static=True)

tau_ext = est.batch(q, v, current)     # (T, 6) N·m — external torque only
```

Works offline on recorded HDF5. No arm required, no extra hardware required.

---

## Install

```bash
pip install piper-dynamics            # core: numpy, pinocchio, pyyaml
pip install piper-dynamics[hdf5]      # + read/annotate ALOHA-style episodes
pip install piper-dynamics[can]       # + live arm access, for calibration
```

The URDF is not bundled — see [`assets/urdf/README.md`](assets/urdf/README.md).

---

## Three commands

```bash
# 1. Calibrate your own arm (~10 min, no extra hardware)
piper-calibrate --urdf piper.urdf --arm-id lab-piper-01 --by you

# 2. Add tau_ext to episodes you already recorded
piper-annotate --urdf piper.urdf --params params/my_arm.yaml \
               --quasi-static episodes/*.hdf5

# 3. Check a parameter file before trusting it
piper-diagnose --params params/my_arm.yaml
```

Step 1 is not optional and the numbers are not transferable. Unit-to-unit
variation on low-cost arms is larger than the signal being measured —
[`params/README.md`](params/README.md) explains how much larger.

---

## What you get

| | |
|---|---|
| `TauExtEstimator` | external joint torque, the main output |
| `wrench_from_tau_ext` | end-effector wrench, with a conditioning flag so near-singular solves announce themselves |
| `estimate_payload_mass` | how heavy is the thing in the gripper — analytic least squares, no learning |
| `PiperModel` | FK, Jacobians, RNEA, inertial regressor (thin Pinocchio wrapper) |
| `piper-calibrate` | identification + a residual report you can read |

Read [`conventions.py`](src/piper_dynamics/conventions.py) before interpreting
any sign. It is short, and every sign error in this kind of code comes from an
unstated assumption.

---

## Status

Honest accounting of what is and is not done:

**Works and is tested** (30 tests, no hardware needed)
- `tau_ext` estimation, quasi-static and dynamic paths
- payload mass estimation — end-to-end test recovers a known mass through the
  full chain (current → torque → subtract dynamics → mass) to 1e-6
- wrench recovery, verified against a known applied force
- static (gravity-only) identification
- HDF5 annotation

The model layer runs in CI against a toy 6-DOF URDF authored in this repo, so
the sign conventions are checked on every commit. Point `PIPER_URDF` at a real
Piper description to run the same suite against actual geometry:

```bash
PIPER_URDF=/path/to/piper.urdf pytest
```

**Not done yet**
- the reference calibration in `params/piper_reference.yaml` is a placeholder,
  not a measurement
- `io/can.py` SDK field names are untested against hardware
- dynamic identification of inertial parameters (`identify_full`) — static
  identification plus `quasi_static=True` is the supported path
- `docs/validation.md` describes the known-weight validation; the measured
  results are not in yet

**Not planned**
- real-time control. This estimates; it does not command. If you want
  compliance control, `tau_ext` from here is an input to your controller, not
  a replacement for one.

---

## How accurate is it?

The honest answer is "measure it on your arm", and
[`docs/validation.md`](docs/validation.md) gives the procedure: hang known
weights, plot estimated mass against true mass. That plot is the only thing
that tells you whether the output is good enough for what you want to do with
it. Any residual model error shows up as apparent mass, so the plot *is* the
error bar.

If you run it, please open an issue with your residual plot — a scatter across
several arms is far more informative than one from mine.

---

## Related

- [`piper_control`](https://pypi.org/project/piper-control/) — SDK wrapper,
  includes gravity *compensation* (feed-forward torque to hold the arm up).
  Complementary: that package makes the arm hold itself; this one reads what
  is left over.
- [`piper-planner`](https://pypi.org/project/piper-planner/) — Pinocchio-based
  trajectory planning.
- [FACTR](https://arxiv.org/abs/2502.17432) (RSS 2025) — what external joint
  torque buys you in a policy, on arms that provide it in hardware.
- [ForceVLA](https://arxiv.org/abs/2505.22159) (NeurIPS 2025) — force as a
  first-class VLA modality, using a wrist F/T sensor.

## Contributing

Calibration results from other arms are the most valuable contribution: they
turn an unverified default into a distribution. Issues with a residual plot
attached get looked at first.

## License

MIT. See [LICENSE](LICENSE).
