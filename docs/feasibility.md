# Physical feasibility of fitted link parameters

A held-out R² says the torque predictions are good. It does not say the
parameters behind them are physical. Low-cost arms make the gap wide: on a
CRANE-X7 identified without CAD (arXiv 2605.15949), the median relative
parameter difference was 4.1 %, the 90th percentile 72.8 % and the worst
689 %. Piper has harmonic drives, current feedback and no joint torque
sensors, so a free fit readily returns negative masses and centres of mass
outside the link while predicting torque well.

`piper_dynamics.calibration.feasibility` is the gate fitted link parameters
pass before anyone trusts them. `audit()` only checks. `project()` changes
parameters, and only when it is called explicitly.

## Two tiers, and why the gravity tier has no LMI

| | gravity tier (`audit`, `project`) | full tier (`audit_full`, `project_full`) |
|---|---|---|
| data | static poses | dynamic excitation (T6 onwards) |
| parameters per link | `m, m*cx, m*cy, m*cz` | those plus the 6 rotational inertia entries |
| constraints | `m > 0`, centre of mass in a box, total mass in a range | pseudo-inertia `J(pi) >= 0` (Wensing, Kim, Slotine 2017) |
| solver | QP (scipy SLSQP) | SDP |
| status | implemented | interface only, raises `NotImplementedError` |

At rest the dynamics reduce to `g(q)`, and gravity acts on a link's mass and
centre of mass only. The rotational inertia never enters `g(q)`, so its six
regressor columns are identically zero on static data.

The pseudo-inertia LMI constrains all ten parameters at once. Run it on
static data and those six entries are unconstrained by measurement. The
solver then fills them with whatever satisfies the LMI, typically the prior
or a point on the boundary. The output looks like an identified inertia
tensor and contains no information. Worse, it looks physically consistent,
which is exactly what the gate is meant to certify.

So the gravity tier checks only what static data determines. The full tier
waits until dynamic data has made the inertia columns non-zero.

Even the four gravity parameters per link are not all identifiable. Static
data determines only some linear combinations of them. On a recorded Piper X
dataset the gravity regressor had rank 10 of 24. That is why the gate judges
violations against the data, not parameter by parameter.

## Reading the verdict

| verdict | meaning | what to do |
|---|---|---|
| `FEASIBLE` | The parameters as given satisfy every constraint. | Nothing. |
| `NULLSPACE_ARTIFACT` | They violate a constraint, but a feasible parameter set fits the data as well (ΔRMS ≤ max(0.02 N·m, 5 % of the free-fit RMS)). | **Not a failure.** The model is fine; static data just cannot pin these parameters down. Use `project()` for a feasible representative. |
| `INFEASIBLE` | The best feasible set fits the data clearly worse than the free fit. | Look at the binding constraints. Suspect the URDF kinematics, the units or kt, an unmodelled payload, or the bound itself (see the gripper allowance below). |
| `POINTWISE_VIOLATION` | A constraint is violated and no `Y, tau` was given. | Run `audit` again with the fit's data. Without it, nullspace drift and real infeasibility look identical, so `INFEASIBLE` is never returned. |

ΔRMS is the RMS of the best feasible fit (stage 1 below) minus the RMS of
the unconstrained least-squares fit.

Two parts of the report explain the call:

- **Regressor rank** and **`null_fraction[link, parameter]`**. This is the
  share of each parameter the data cannot see: 1 means invisible, 0 means
  fully determined, in SI coordinates. A `NULLSPACE_ARTIFACT` whose violating
  parameters have null fractions near 1 is the expected case.
- **Binding constraints** at the best feasible point. For `INFEASIBLE`, these
  are the constraints the data is pushing against.

Example, on recorded static frames with the Piper X URDF. The minimum-norm
free fit gives a total arm mass of 0.3 kg and a link3 centre of mass 5.7 m
from its joint, which is wildly unphysical pointwise. Yet the verdict is
`NULLSPACE_ARTIFACT` with ΔRMS +0.0000 N·m. Every violation lives in
directions the data cannot see, and `project()` returns masses of
0.71/1.16/0.57/0.28/0.28/0.03 kg with the same residual.

## Severity

| severity | checks |
|---|---|
| FAIL | `m <= 1e-4 kg`; centre of mass outside a box taken from the URDF collision geometry (`mesh`) |
| WARN | centre of mass outside a `kinematic-fallback` box; end link beyond the gripper allowance; total mass outside 0.8×–2.5× the URDF total |

A WARN comes from a conservative estimate and never rejects a parameter set
on its own.

## Box sources

`PiperModel.link_specs()` tries the URDF collision geometry first, resolving
`package://` meshes through `package_dirs`. Any link whose geometry cannot be
loaded falls back to a **kinematic box**. That is the hull of the link's
joint origin, the origins of the joints and frames it carries, and its URDF
centre of mass, padded by `max(5 cm, 0.5 × link length)`. AgileX
descriptions reference meshes by `package://` path, so without the
description package installed every link uses the fallback. The report names
the source of each link's box.

## The end link and the gripper allowance: the most likely false alarm

The Piper X URDF describes the bare arm: link6 weighs 7 g. Both arms carry an
AGX gripper. A fit lumps the gripper's mass and centre of mass into link6, so
checking link6 against the URDF's own geometry would fail every correct fit.
The end link therefore gets a `GripperAllowance`:

- **mass**: URDF mass + `GRIPPER_ALLOWANCE_MASS_KG` = 1.0 kg. The stock Piper
  description, gripper included, puts 0.51 kg in link6; doubling it leaves
  room for a camera mount and cabling.
- **centre of mass**: within a cube of half-width
  `GRIPPER_ALLOWANCE_COM_RADIUS_M` = 0.15 m around the flange. The stock
  gripper's fingers sit 0.136 m out, so the gripper's centre of mass cannot
  be further.

Both are estimates, so violations on the end link are WARN. A heavier tool,
or a wrist camera on a long bracket, can exceed them. **This is the check
most likely to misjudge.** When an `INFEASIBLE` verdict lists only the end
link as binding, check the hardware before the fit.

## Units

Parameters must be SI (kg, kg·m) and declared with `units="SI"` on every
`ParamSet` and `LinkSpec`; anything else raises. A fit in motor-current
units absorbs the torque constant kt into its parameters. Multiply by kt
before calling the gate, or the masses are off by a factor of kt.

## Calling it from outside this package

The module imports neither Pinocchio nor this package's model, so a script
with its own URDF parsing can fill the dataclasses directly:

```python
from piper_dynamics.calibration.feasibility import LinkSpec, ParamSet, audit, project

links = [LinkSpec(name, prior4, com_lo, com_hi, "kinematic-fallback", units="SI"), ...]
fitted = ParamSet(tuple(names), phi.reshape(-1, 4), units="SI")
report = audit(fitted, links, Y, tau)   # Y: (rows, 4*n_links), link-major, N·m per SI unit
print(report.summary())
```

Inside the package, `PiperModel.link_specs()` and
`PiperModel.gravity_regressor(q)` produce the same inputs.

## How `project` decides

There are two stages, with no weight trading data against the prior:

1. `min ||Y φ − τ||²` subject to the constraints. Its optimum `r*` is the best
   any feasible parameter set can do.
2. `min ||φ − φ_urdf||²` subject to the constraints and
   `||Y φ − τ||² ≤ r*(1 + 0.05)`. The prior only picks a representative among
   equally good fits.

Both stages are convex: a QP, then a strictly convex objective over linear
constraints plus one convex quadratic constraint. The solution found is
therefore the global optimum, and a verdict cannot come from a local minimum.
The rise in residual from the given parameters to the projected ones is
reported, and is itself the measure of how unphysical the input was.

## Thresholds

All are module constants, pinned by `tests/test_feasibility.py`.

| constant | value |
|---|---|
| `M_MIN_KG` | 1e-4 kg |
| `RESIDUAL_SLACK` (ε) | 0.05 |
| `NULLSPACE_DRMS_ABS_NM`, `NULLSPACE_DRMS_REL` | 0.02 N·m, 5 % |
| `TOTAL_MASS_RANGE` | 0.8×–2.5× URDF total |
| `GRIPPER_ALLOWANCE_MASS_KG`, `GRIPPER_ALLOWANCE_COM_RADIUS_M` | 1.0 kg, 0.15 m |
| `KINEMATIC_BOX_PAD_ABS_M`, `KINEMATIC_BOX_PAD_REL` | 5 cm, 0.5 × link length |
| `RANK_RTOL` | 1e-6 |
| `STAGE2_RMS_FLOOR_NM` | 1e-4 N·m |
