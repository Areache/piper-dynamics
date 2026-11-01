"""Momentum observer: external torque without joint acceleration.

``TauExtEstimator`` needs ``a`` for the inertial term, and ``a`` comes from
differentiating the encoder velocity: the noise that amplifies lands in
``tau_ext`` during fast motion. The generalized-momentum observer avoids
``a`` altogether. With ``p = M(q) v`` and ``Mdot = C + C^T``, the dynamics give
(see ``piper_dynamics.conventions`` for the sign of ``tau_ext``)::

    pdot = tau_meas - tau_fric + C(q,v)^T v - g(q) - tau_ext

and the observer::

    r(t) = K_O * ( integral_0^t (tau_meas - tau_fric + C^T v - g - r) ds  -  (p(t) - p(0)) )

satisfies ``rdot = K_O (tau_ext - r)``: a first-order low-pass of ``tau_ext``
with time constant ``1 / K_O`` per joint, in the same convention as
``TauExtEstimator``. The integral is a forward-Euler sum, so keep
``K_O * dt`` well below 1.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ..model import PiperModel

from ..params import ArmParams
from .friction import DEFAULT_EPS, friction_torque
from .tau_ext import current_to_torque


class MomentumObserver:
    """Discrete generalized-momentum observer over a recorded episode.

    Example
    -------
    >>> obs = MomentumObserver(model, params, gain=25.0)          # doctest: +SKIP
    >>> tau_ext = obs.batch(q, v, current, dt=0.02)               # doctest: +SKIP
    """

    def __init__(
        self,
        model: PiperModel,
        params: ArmParams,
        gain: float | np.ndarray = 25.0,
        friction_eps: float = DEFAULT_EPS,
    ) -> None:
        if params.nq != model.nv:
            raise ValueError(
                f"parameter file describes {params.nq} joints but the URDF has {model.nv} DOF"
            )
        self.model = model
        self.params = params
        self.gain = np.broadcast_to(np.asarray(gain, dtype=float), (model.nv,)).copy()
        self.friction_eps = friction_eps

    def batch(
        self,
        q: np.ndarray,
        v: np.ndarray,
        current: np.ndarray,
        dt: float | np.ndarray,
    ) -> np.ndarray:
        """``(T, nq)`` in, ``(T, nq)`` out; ``dt`` scalar or per-sample ``(T,)``
        (the step from sample ``k`` to ``k + 1``). ``r[0] = 0``."""
        q, v, current = (np.atleast_2d(np.asarray(x, dtype=float)) for x in (q, v, current))
        T = len(q)
        if not (len(v) == len(current) == T):
            raise ValueError("q, v, current must have the same number of samples")
        dts = np.broadcast_to(np.asarray(dt, dtype=float), (T,))
        if np.any(self.gain * dts.max() >= 1.0):
            raise ValueError("K_O * dt >= 1: the forward-Euler observer would be unstable")
        tau_m = current_to_torque(current, self.params)
        tau_f = friction_torque(v, self.params.fc, self.params.fv, self.params.fs, self.friction_eps)
        r = np.zeros_like(q)
        integ = np.zeros(q.shape[1])
        p0 = self.model.mass_matrix(q[0]) @ v[0]
        for k in range(T):
            p = self.model.mass_matrix(q[k]) @ v[k]
            r[k] = self.gain * (integ - (p - p0))
            beta = self.model.coriolis_matrix(q[k], v[k]).T @ v[k] - self.model.gravity_torque(q[k])
            integ = integ + (tau_m[k] - tau_f[k] + beta - r[k]) * dts[k]
        return r
