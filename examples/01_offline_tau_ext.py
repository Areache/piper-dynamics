"""Compute tau_ext for a recorded episode. No hardware required.

    python examples/01_offline_tau_ext.py --urdf piper.urdf \
        --params params/my_arm.yaml --episode episodes/ep_0.hdf5
"""

# %%
import argparse

import matplotlib.pyplot as plt
import numpy as np

from piper_dynamics import ArmParams, PiperModel, TauExtEstimator
from piper_dynamics.io.hdf5 import inspect, read_episode

p = argparse.ArgumentParser()
p.add_argument("--urdf", required=True)
p.add_argument("--params", required=True)
p.add_argument("--episode", required=True)
p.add_argument("--current-key", default="observations/motor_current")
p.add_argument("--arm", choices=("left", "right"), help="bimanual recording: which arm")
args = p.parse_args()

# %% What is actually in this file?
# Recorders disagree about field names and units. Look before you compute.
inspect(args.episode)

# %% Load
model = PiperModel(args.urdf)
params = ArmParams.load(args.params)
for w in params.warnings():
    print("warning:", w)

est = TauExtEstimator(model, params, quasi_static=True)
q, v, current = read_episode(args.episode, current_key=args.current_key, n_joints=6, arm=args.arm)
print(f"{len(q)} samples, {q.shape[1]} joints")

# %% Decompose
result = est.decompose(q[0], v[0], current[0])
print("measured  ", np.round(result.tau_meas, 3))
print("model     ", np.round(result.tau_model, 3))
print("residual  ", np.round(result.tau_ext, 3))

tau_ext = est.batch(q, v, current)

# %% The plot that makes the point
# Raw effort is dominated by pose. tau_ext is not. If the two panels look the
# same, something is wrong -- start with docs/validation.md step 0.
fig, axes = plt.subplots(2, 1, figsize=(9, 5), sharex=True)
axes[0].plot(current, lw=0.8)
axes[0].set_ylabel("raw effort")
axes[0].set_title("before: the arm holding itself up")
axes[1].plot(tau_ext, lw=0.8)
axes[1].axhline(0, color="k", lw=0.5)
axes[1].set_ylabel("tau_ext [N·m]")
axes[1].set_xlabel("sample")
axes[1].set_title("after: what the outside world did")
fig.tight_layout()
plt.show()
