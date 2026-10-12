"""Read and annotate ALOHA-style HDF5 episodes.

This is the offline path, and it is the default one on purpose: most people
who could use this package already have episodes on disk and no arm in front
of them. Give it a dataset and it adds ``/observations/tau_ext`` next to the
fields that are already there.

Layout assumed (the ALOHA/ACT convention, as produced by the common Piper
teleop recorders)::

    /observations/qpos          (T, nq)
    /observations/qvel          (T, nq)
    /observations/effort        (T, nq)      <- or motor_current, see below
    /action                     (T, nq)

The current field is the one that varies most between recorders, which is why
``current_key`` is explicit. Check what yours contains before trusting the
output: ``inspect(path)`` prints the tree.
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


def read_episode(
    path: str | Path,
    qpos_key: str = "observations/qpos",
    qvel_key: str = "observations/qvel",
    current_key: str = "observations/effort",
    n_joints: int | None = 6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(q, v, current)``, each ``(T, n_joints)``.

    ``n_joints`` slices off the gripper (and, for a bimanual recording, you
    will want to call this twice with different slices -- a 14-dim ALOHA
    action is two arms, and feeding both to a 6-DOF model silently produces
    garbage).
    """
    _require_h5py()
    with h5py.File(path, "r") as f:
        q = np.asarray(f[qpos_key][()], dtype=float)
        v = np.asarray(f[qvel_key][()], dtype=float)
        c = np.asarray(f[current_key][()], dtype=float)
    if n_joints is not None:
        q, v, c = q[:, :n_joints], v[:, :n_joints], c[:, :n_joints]
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

    if out_path != path:
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
