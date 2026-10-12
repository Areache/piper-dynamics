"""Joint friction model.

Static + Coulomb + viscous, with a smooth sign function. The smoothing is not
cosmetic: ``sign(v)`` makes tau_ext flip by ``2*fc`` every time a joint crosses
zero velocity, which puts a spike into exactly the signal you are trying to
read.
"""

from __future__ import annotations

import numpy as np

#: Velocity scale of the smooth sign, rad/s. Below this, friction is ramped
#: rather than switched. Roughly: the velocity you consider "stopped".
DEFAULT_EPS = 0.05


def smooth_sign(v: np.ndarray, eps: float = DEFAULT_EPS) -> np.ndarray:
    return np.tanh(np.asarray(v, dtype=float) / eps)


def friction_torque(
    v: np.ndarray,
    fc: np.ndarray,
    fv: np.ndarray,
    fs: np.ndarray | None = None,
    eps: float = DEFAULT_EPS,
) -> np.ndarray:
    """Friction torque opposing motion, same shape as ``v``.

    ``fs`` (stiction) decays with velocity: it acts near standstill and fades
    once the joint is clearly moving.
    """
    v = np.asarray(v, dtype=float)
    s = smooth_sign(v, eps)
    tau = np.asarray(fc) * s + np.asarray(fv) * v
    if fs is not None:
        tau = tau + np.asarray(fs) * s * np.exp(-((v / eps) ** 2))
    return tau
