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
| `t7_rays.py` | any (pinocchio) | no: one record per descent (free-space certificate, first contact, stiffness) |

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

## v2.4

- `--motion line`: one `move_l` from the start flange pose straight down to the maximum depth, instead of
  streaming `move_j` every 25 ms. The motivation was a ~5 Hz stop-and-go seen offline in streamed T7 descents
  (a line at 4.7 Hz in the j2 acceleration); the E1 in-air runs did not show it.
  - The speed is set by `--speed-pct` and measured during the free descent. The trial stops (`too_fast`) above
    `--max-line-speed`.
  - After a contact, a `move_j` back-off interrupts the `move_l` and is re-sent for 1 s.
  - `--line-speed-pct` sets the percent for the `move_l` only. The approach and the return to the start keep
    `--speed-pct`. On the right arm, 1 percent is about 4 mm/s.
  - Measured in free air: a stop is decided and the arm is held within 30-40 ms in both modes. Line mode
    overshoots 1.25 mm at 40 mm/s; stream overshoots 0.75 mm at 20 mm/s and 1.5 mm at 40 mm/s.
- `--motion alt`: trials alternate stream / line within one run, so the two modes see the same drift. Each
  trial records its `wall_time`.
- `--test-stop-mm X`: a fake contact at X mm in free air. It measures how far the arm keeps going after a
  stop is decided (`overshoot_mm`). Use it before any line-mode contact. Trials stop if the overshoot exceeds
  3 mm.
- The speed cap is raised to 40 mm/s, for a 10/20/40 mm/s comparison on a rigid plate.
- Every trial writes the tip force along the whole descent, F(z), as `fz_XX.csv` and `fz_*` npz keys: time,
  phase (0 free / 1 armed / 2 backing off), depth, tip z, commanded depth, F, 0.2 s mean F, phi. The
  free-descent friction baseline from arming is applied back to the first 20 mm.

Hardware results so far (right arm, rigid plate):
- In free air, `move_j` interrupts `move_l` within 30-40 ms.
- From 10 to 40 mm/s, the contact onset does not move. Detection depth and peak force grow with speed, and
  line mode is no better than stream. Stream at 10 mm/s stays the default.

## v2.5: every descent is a full ray (logging only)

Control is identical to v2.4.
- **Simulator**: every `t7_sim.py` scenario × 3 seeds. Decisions, the command sequence and the shared log rows
  are bit-identical.
- **Replay**: 79 recorded descents give the same detection, stop and back-off.
- **Compute**: +0.09 ms per cycle.

Per trial, `trial_XX.npz` adds these keys:

| key | content |
|---|---|
| `t_fb` | feedback received (s, same clock as `t`) |
| `state` | 0 blind (first 20 mm, friction only), 1 monitoring, 2 stop decided on this row, 3 backing off, 4 done (still) |
| `k_now` | waypoint nearest the measured q on every row (`k_meas` stays frozen after the stop, as before) |
| `i_grav` | expected gravity current at `k_now` (A, 6 joints) |
| `thr_row_N` | contact threshold in use (NaN before arming) |
| `cmd_idx` | index of the last command sent before this row |
| `cmd_t`, `cmd_kind`, `cmd_val` | every command: send time; kind (1 stream `move_j`, 2 back-off, 3 back-off re-send, 4 hold at bottom, 5 hold after Ctrl-C, 6 `move_l`); 6 joint angles or the flange pose |
| `mx_*` | motor temperature / status fields, only if the feedback messages already carry them (no extra SDK calls) |
| `fz_*` | tip force along the whole descent (see above) |

- **Logging after the stop** continues until the arm is back at the back-off target and still for 0.5 s (at
  least 1 s, at most 5 s), so both the loading and the unloading branch are recorded.
- **Metadata**: `trial_XX.json` holds the trial summary and a field legend.
- **Tip x, y, z** are not computed on the robot PC (no kinematics library). `t7_rays.py` computes them from q
  on the server.

`t7_rays.py` (server, pinocchio) turns each descent into one record:
- **Points**:
  - probe x, y;
  - start height;
  - monitoring start height (the first 20 mm are friction-only and not covered: written out as the blind
    interval);
  - detection height;
  - contact onset and first contact (Theil-Sen line of the loading branch at the top of the noise band);
  - overshoot.
- **Certificate**: the maximum 0.2 s mean force over the monitored rows above the first contact, and its
  margin to the threshold. In words: this interval is free at F_thr.
- **Stiffness**: loading and unloading stiffness, and hysteresis.
  - Tip z comes from the motor-side joints, so the stiffness is the series stiffness of arm, servo and object.

On the 33 T7 contacts:
- **Certificate**: holds 33/33, margin median 0.75 N.
- **Overshoot at 10 mm/s**: 1.53 ± 0.27 mm within one session (a near-constant bias that cancels in
  differences).
- **Loading stiffness**: about 1-2 N/mm.
- **Unloading branch**: resolvable in only 8/33. On a rigid plate the 4-5 mm back-off takes about 0.09 s.

## Safety

- Operator present with the e-stop.
- `--operator-confirmed` and a typed `YES`, plus `y` before each trial unless `--no-confirm-each`.
- SDK speed percent ≤ 20. `move_js` is never used.
- Above 10 mm/s, only on a rigid plate.
