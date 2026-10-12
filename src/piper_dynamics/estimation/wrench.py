"""External joint torque -> end-effector wrench.

``tau_ext = -J^T F``, so recovering ``F`` means solving an underdetermined-in-
general linear system. On a 6-DOF arm ``J^T`` is square, but it is singular at
singular configurations and badly conditioned near them -- which is exactly
where a naive inverse produces enormous fictitious forces.

This module therefore always reports conditioning alongside the wrench, and
refuses to pretend a near-singular solve is trustworthy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # avoid importing pinocchio to use the rest of the package
    from ..model import PiperModel



@dataclass
class WrenchResult:
    """Wrench applied *by the environment on the robot*, in the EE frame."""

    wrench: np.ndarray          # (6,) [fx, fy, fz, mx, my, mz]
    condition_number: float
    trustworthy: bool

    @property
    def force(self) -> np.ndarray:
        return self.wrench[:3]

    @property
    def moment(self) -> np.ndarray:
        return self.wrench[3:]


def wrench_from_tau_ext(
    model: PiperModel,
    q: np.ndarray,
    tau_ext: np.ndarray,
    cond_limit: float = 1e3,
    rcond: float = 1e-6,
    local: bool = False,
) -> WrenchResult:
    """Least-squares wrench consistent with ``tau_ext`` at configuration ``q``.

    Parameters
    ----------
    cond_limit
        Above this condition number of ``J^T`` the result is flagged untrusted.
        It is still returned -- silently dropping samples hides the fact that
        your trajectory spends time near singularities.
    """
    J = model.jacobian(q, local=local)       # 6 x nv
    JT = J.T                                  # nv x 6
    wrench, *_ = np.linalg.lstsq(-JT, np.asarray(tau_ext, dtype=float), rcond=rcond)
    cond = float(np.linalg.cond(JT))
    return WrenchResult(wrench=wrench, condition_number=cond, trustworthy=cond < cond_limit)


def wrench_batch(
    model: PiperModel,
    q: np.ndarray,
    tau_ext: np.ndarray,
    **kw,
) -> tuple[np.ndarray, np.ndarray]:
    """``(T, 6)`` wrenches and a ``(T,)`` boolean trust mask."""
    q = np.atleast_2d(q)
    tau_ext = np.atleast_2d(tau_ext)
    results = [wrench_from_tau_ext(model, q[t], tau_ext[t], **kw) for t in range(len(q))]
    return (
        np.stack([r.wrench for r in results]),
        np.array([r.trustworthy for r in results]),
    )
