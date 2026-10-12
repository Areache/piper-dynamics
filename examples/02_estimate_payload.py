"""Estimate the mass of a held object from tau_ext.

Expects a recording where the arm picks something up partway through and
holds it, quasi-statically.

    python examples/02_estimate_payload.py --urdf piper.urdf \
        --params params/my_arm.yaml --episode episodes/lift_0.hdf5 \
        --hold-start 400 --hold-end 900
"""

# %%
import argparse

import numpy as np

from piper_dynamics import ArmParams, PiperModel, TauExtEstimator
from piper_dynamics.estimation import estimate_payload_mass
from piper_dynamics.io.hdf5 import read_episode

p = argparse.ArgumentParser()
p.add_argument("--urdf", required=True)
p.add_argument("--params", required=True)
p.add_argument("--episode", required=True)
p.add_argument("--hold-start", type=int, required=True, help="first sample while holding")
p.add_argument("--hold-end", type=int, required=True, help="last sample while holding")
p.add_argument("--free-start", type=int, default=0, help="first sample with empty gripper")
p.add_argument("--free-end", type=int, default=100)
args = p.parse_args()

model = PiperModel(args.urdf)
params = ArmParams.load(args.params)
est = TauExtEstimator(model, params, quasi_static=True)

q, v, current = read_episode(args.episode, n_joints=6)
tau_ext = est.batch(q, v, current)

# %% Empty gripper first -- this is your noise floor.
free = slice(args.free_start, args.free_end)
baseline = estimate_payload_mass(model, q[free], tau_ext[free])
print(f"empty gripper: {baseline.mass_kg:+.3f} kg  (this is the floor, want ~0)")

# %% Now the held window.
hold = slice(args.hold_start, args.hold_end)
out = estimate_payload_mass(model, q[hold], tau_ext[hold])
print(f"held object:   {out.mass_kg:+.3f} ± {out.stderr_kg:.3f} kg")
flag = "" if out.observable else "  <-- too low, see below"
print(f"observability: {out.observability:.2f}{flag}")

if not out.observable:
    print(
        "\nThe poses in this window barely load the motors with the payload's\n"
        "weight, so the estimate is noise rather than measurement. Use a window\n"
        "where the arm is more extended, or a longer one spanning more poses."
    )

# %% Wrist joints are often model error rather than physics. Try without them.
mask = np.array([True, True, True, False, False, False])
out_arm_only = estimate_payload_mass(model, q[hold], tau_ext[hold], joint_mask=mask)
print(f"\nfirst 3 joints only: {out_arm_only.mass_kg:+.3f} ± {out_arm_only.stderr_kg:.3f} kg")
print("If this is much better, your wrist calibration is the thing to improve.")
