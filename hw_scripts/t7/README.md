# T7 / E1: current-based contact probing (guarded vertical descent)

The closed gripper descends straight down from a taught start pose. The motor currents alone tell when the
fingertips touch a surface: the arm then stops and backs off 5 mm. No force sensor is used. Results and
history are in `.hw/TASKS.md` (T7).

| file | runs on | moves the arm |
|---|---|---|
| `t7_record_start.py` | robot PC | no (reads joint angles while the operator drags the arm) |
| `t7_plan.py` | any machine with pinocchio + the gripper URDF | no |
| `guarded_move.py` | robot PC (numpy + PyAgxArm) | **yes** |
| `t7_analyze.py` | any | no |
| `t7_sim.py` | any (pinocchio) | no: a toy simulated arm to check the control flow, not the physics |

## Pipeline

1. `t7_record_start.py can_right --out start.json`: the operator drags the closed gripper (fingers down) to a
   start pose about 6-10 cm above the surface. Read-only.
2. `t7_plan.py --arm can_right --start-json start.json --pose-index 0 --gap-mm G --out plan.json`: the planner
   solves IK for a vertical tip line every 0.25 mm down to the gap plus 5 mm. It also writes the expected
   gravity current along the line (URDF + static calibration T4) and the tip-force signature: the change in
   current per newton of upward tip force.
   - `--object-height-mm H` replaces the ruler gap. The gap comes from FK: the tip height at the start minus
     (the plate height measured by earlier touches + H).
3. `guarded_move.py --plan plan.json --trials N --operator-confirmed`: descends, detects, backs off and logs
   every sample (`trial_XX.npz`, `summary.json`).

The URDF is not in this repository (see `assets/urdf/README.md`). Put it at the repository root as
`piper_x_6dof_with_gripper.urdf`, or set `PIPER_GRIPPER_URDF`. Set `PYAGX` to the PyAgxArm source directory,
or pass `--pyagx`.

## Contact rule (v2.3, unchanged in v2.4)

- **Residual** = current − the static baseline before the descent − the planned gravity change. Kinetic
  friction from the first 20 mm of free descent is also subtracted, scaled by measured / commanded speed.
- **Tip force** = residuals of j2 and j3 projected onto the tip-force signature by weighted least squares.
  - The wrist joints are left out. When the gripper is tilted, a finger edge touches first, and j4/j5 then
    give the opposite sign.
- **Contact** = the 0.2 s mean tip force exceeds max(4 × its free-descent spread, 2 N).
  - It is a mean, not an instantaneous test, because on hardware the contact force arrives as 30-70 ms
    pulses about every 0.2 s.
- **Speed**: 10 mm/s is the default. At 5 mm/s, j2 turns at about 0.008 rad/s and sits in stick-slip, and
  single-joint thresholds false-trigger.
- **Backup stops**:
  - `contact_late`: still pressing at the maximum depth.
  - `stall`: the arm is not advancing.
  - `max_depth`: reached the maximum depth with no contact; holds 15 s so the remaining gap can be measured.

## v2.4 (not yet run on hardware)

- `--motion line`: one `move_l` from the start flange pose straight down to the maximum depth, instead of
  streaming `move_j` every 25 ms. The streamed motion is a ~5 Hz stop-and-go, with a line at 4.7 Hz in the j2
  acceleration.
  - The speed is set by `--speed-pct` and measured during the free descent. The trial stops (`too_fast`) above
    `--max-line-speed`.
  - After a contact, a `move_j` back-off interrupts the `move_l` and is re-sent for 1 s.
- `--test-stop-mm X`: a fake contact at X mm in free air. It measures how far the arm keeps going after a
  stop is decided (`overshoot_mm`). Use it before any line-mode contact. Trials stop if the overshoot exceeds
  3 mm.
- The speed cap is raised to 40 mm/s, for a 10/20/40 mm/s comparison on a rigid plate.

## Safety

- Operator present with the e-stop.
- `--operator-confirmed` and a typed `YES`, plus `y` before each trial unless `--no-confirm-each`.
- SDK speed percent ≤ 20. `move_js` is never used.
- Above 10 mm/s, only on a rigid plate.
