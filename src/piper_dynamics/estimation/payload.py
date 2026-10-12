"""Payload mass estimation from external joint torque.

The question this answers: *how heavy is the thing the gripper is holding?*

A payload of mass ``m`` at end-effector offset ``c`` contributes, quasi-
statically,

    tau_ext = m * J_v(q)^T @ (-g)

which is linear in ``m``. So mass is a one-parameter least squares -- no
learning required, and that is the point: it gives you an analytic baseline to
check any learned estimator against.

Caveats, in order of how often they bite:

1. **Quasi-static only.** During acceleration the payload also contributes
   inertial terms. Estimate while the arm is moving slowly, or hold still.
2. **Observability depends on pose.** If ``J_v^T g`` is small -- the payload's
   weight is carried by the structure rather than the motors -- the regressor
   is weak and the estimate is noise. ``PayloadEstimate.observability`` is
   there to be checked, not ignored.
3. **Model error floors the accuracy.** Whatever bias is left in ``tau_ext``
   after calibration shows up as apparent mass. ``docs/validation.md`` has the
   known-weight procedure for measuring that floor on your own arm.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # avoid importing pinocchio to use the rest of the package
    from ..model import PiperModel

from ..conventions import GRAVITY_WORLD


@dataclass
class PayloadEstimate:
    mass_kg: float
    #: Standard error of the fit, kg. Not a guarantee -- it reports scatter,
    #: not systematic model error.
    stderr_kg: float
    #: ||b|| summed over the window. Low means the poses did not load the
    #: motors with the payload's weight; treat the estimate as unobserved.
    observability: float
    n_samples: int

    @property
    def observable(self) -> bool:
        return self.observability > 1.0


def payload_regressor(
    model: PiperModel,
    q: np.ndarray,
    com_offset: np.ndarray | None = None,
    gravity: np.ndarray | None = None,
) -> np.ndarray:
    """``b(q)`` such that ``tau_ext = m * b(q)`` for a point mass at the EE.

    ``com_offset`` shifts the payload's centre of mass in the end-effector
    frame -- a long tool held at its base is not a point mass at the flange.
    """
    g = GRAVITY_WORLD if gravity is None else np.asarray(gravity, dtype=float)
    if com_offset is None:
        Jv = model.jacobian_linear(q)
        return Jv.T @ (-g)
    # Shift the reference point: the payload hangs at `com_offset` in EE frame.
    T = model.fk(q)
    R = T[:3, :3]
    r = R @ np.asarray(com_offset, dtype=float)
    J = model.jacobian(q)                 # world-aligned, EE reference point
    Jv, Jw = J[:3, :], J[3:, :]
    # Velocity of a point offset by r: v_P = v_ee + w x r, so J_P = J_v - [r]x J_w.
    Jv_shifted = Jv - np.cross(r, Jw.T).T
    return Jv_shifted.T @ (-g)


def estimate_payload_mass(
    model: PiperModel,
    q: np.ndarray,
    tau_ext: np.ndarray,
    com_offset: np.ndarray | None = None,
    joint_mask: np.ndarray | None = None,
) -> PayloadEstimate:
    """Least-squares payload mass over a window of samples.

    Parameters
    ----------
    q, tau_ext
        ``(T, nq)`` arrays over a window during which the payload is held and
        the arm is quasi-static.
    joint_mask
        Optional boolean ``(nq,)`` selecting which joints to fit on. Wrist
        joints are often dominated by friction and their own model error;
        excluding them can improve the estimate more than any amount of
        filtering.
    """
    q = np.atleast_2d(q)
    tau_ext = np.atleast_2d(tau_ext)
    if len(q) != len(tau_ext):
        raise ValueError("q and tau_ext must have the same number of samples")

    B = np.stack([payload_regressor(model, q[t], com_offset) for t in range(len(q))])
    if joint_mask is not None:
        B = B[:, joint_mask]
        tau_ext = tau_ext[:, joint_mask]

    b = B.ravel()
    y = tau_ext.ravel()
    bb = float(b @ b)
    if bb < 1e-12:
        return PayloadEstimate(np.nan, np.nan, 0.0, len(q))

    m = float(b @ y / bb)
    resid = y - m * b
    dof = max(len(y) - 1, 1)
    stderr = float(np.sqrt((resid @ resid) / dof / bb))
    return PayloadEstimate(
        mass_kg=m,
        stderr_kg=stderr,
        observability=float(np.sqrt(bb / len(q))),
        n_samples=len(q),
    )


def estimate_payload_mass_from_lsq(b: np.ndarray, tau_ext: np.ndarray) -> PayloadEstimate:
    """Same fit, given a precomputed regressor. Used by tests and by callers
    who build ``b`` from a different model."""
    b = np.asarray(b, dtype=float).ravel()
    y = np.asarray(tau_ext, dtype=float).ravel()
    bb = float(b @ b)
    if bb < 1e-12:
        return PayloadEstimate(np.nan, np.nan, 0.0, y.size)
    m = float(b @ y / bb)
    resid = y - m * b
    stderr = float(np.sqrt((resid @ resid) / max(y.size - 1, 1) / bb))
    return PayloadEstimate(m, stderr, float(np.sqrt(bb / max(y.size, 1))), y.size)
