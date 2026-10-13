"""Read and annotate ALOHA-style HDF5 episodes.

This is the offline path, and it is the default one on purpose: most people
who could use this package already have episodes on disk and no arm in front
of them. Give it a dataset and it adds ``/observations/tau_ext`` next to the
fields that are already there.

Layout assumed (the ALOHA/ACT convention, as produced by the common Piper
teleop recorders)::

    /observations/qpos          (T, 14)      left j0..j5, left gripper, right j0..j5, right gripper
    /observations/qvel          (T, 14)
    /observations/motor_current (T, 12)      left j0..j5, right j0..j5 -- A
    /observations/effort        (T, 14)      see below
    /action                     (T, 14)

The current field is the one that varies most between recorders, which is why
``current_key`` is explicit. On piper-aio recordings ``observations/effort`` is
all zeros for the arm joints (the ROS driver fills only the gripper slot), so
the default is ``motor_current``; ``motor_torque`` there is the same current
times a fixed SDK coefficient, not an independent measurement. Check what
yours contains before trusting the output: ``inspect(path)`` prints the tree.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

try:
    import h5py
except ImportError:  # pragma: no cover
    h5py = None


def _require_h5py() -> None:
    if h5py is None:
        raise ImportError("h5py is required: `pip install piper-dynamics[hdf5]`")


def inspect(path: str | Path) -> dict[str, Any]:
    """Print and return the dataset tree, with shapes and dtypes."""
    _require_h5py()
    out: dict[str, Any] = {}

    def visit(name, obj):
        if isinstance(obj, h5py.Dataset):
            out[name] = {"shape": obj.shape, "dtype": str(obj.dtype)}
            print(f"  {name:<40} {str(obj.shape):<16} {obj.dtype}")

    with h5py.File(path, "r") as f:
        print(f"{path}:")
        f.visititems(visit)
    return out


#: First column of each arm in a bimanual piper-aio recording: (qpos/qvel, current).
#: qpos carries a gripper slot per arm, motor_current does not.
BIMANUAL_COLUMNS = {"left": (0, 0), "right": (7, 6)}


def read_episode(
    path: str | Path,
    qpos_key: str = "observations/qpos",
    qvel_key: str = "observations/qvel",
    current_key: str = "observations/motor_current",
    n_joints: int | None = 6,
    arm: str | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(q, v, current)``, each ``(T, n_joints)``.

    ``n_joints`` slices off the gripper. For a bimanual recording pass
    ``arm="left"`` or ``"right"``: feeding both arms to a 6-DOF model silently
    produces garbage, and the two arms start at different columns in qpos and
    in motor_current (see ``BIMANUAL_COLUMNS``).

    Raises if the selected current columns are all zero, which is what an
    unpopulated ``effort`` field looks like.
    """
    _require_h5py()
    with h5py.File(path, "r") as f:
        q = np.asarray(f[qpos_key][()], dtype=float)
        v = np.asarray(f[qvel_key][()], dtype=float)
        c = np.asarray(f[current_key][()], dtype=float)
    q0, c0 = (0, 0) if arm is None else BIMANUAL_COLUMNS[arm]
    if n_joints is not None:
        q, v, c = q[:, q0:q0 + n_joints], v[:, q0:q0 + n_joints], c[:, c0:c0 + n_joints]
    elif arm is not None:
        raise ValueError("arm selection needs n_joints")
    if c.size and not c.any():
        raise ValueError(
            f"{current_key} is all zeros for the selected joints in {path}. "
            f"On piper-aio recordings use observations/motor_current."
        )
    return q, v, c


def annotate_episode(
    path: str | Path,
    estimator,
    out_path: str | Path | None = None,
    tau_ext_key: str = "observations/tau_ext",
    overwrite: bool = False,
    **read_kw,
) -> Path:
    """Compute ``tau_ext`` for an episode and write it back.

    Writes to ``out_path`` (default: alongside, with a ``.tau_ext`` suffix)
    rather than mutating the original. Your recorded episodes are expensive;
    this package is not going to be the thing that corrupts them.
    """
    _require_h5py()
    path = Path(path)
    out_path = Path(out_path) if out_path else path.with_suffix(".tau_ext" + path.suffix)

    q, v, c = read_episode(path, **read_kw)
    tau_ext = estimator.batch(q, v, c)

    # Copy only once, so annotating the second arm adds to the first arm's output.
    if out_path != path and not out_path.exists():
        import shutil
        shutil.copy2(path, out_path)

    with h5py.File(out_path, "a") as f:
        if tau_ext_key in f:
            if not overwrite:
                raise FileExistsError(
                    f"{tau_ext_key} already exists in {out_path}. Pass "
                    f"overwrite=True if you mean to replace it."
                )
            del f[tau_ext_key]
        ds = f.create_dataset(tau_ext_key, data=tau_ext)
        ds.attrs["units"] = "N·m"
        ds.attrs["convention"] = "tau_meas - tau_model - tau_fric; see piper_dynamics.conventions"
        ds.attrs["produced_by"] = "piper-dynamics"
        prov = estimator.params.provenance
        ds.attrs["params_arm_id"] = prov.arm_id
        ds.attrs["params_calibrated_at"] = prov.calibrated_at
    return out_path
