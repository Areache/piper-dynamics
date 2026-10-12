"""Excitation trajectories for identification.

Identification is a least-squares problem, and its conditioning is decided
entirely by where you drive the arm. A trajectory that only sweeps one joint
at a time cannot separate coupled inertial parameters, no matter how clean the
data is.

The standard answer is a periodic Fourier-series trajectory per joint, with
coefficients chosen to minimise the condition number of the stacked regressor
subject to joint limits. This module gives you the trajectory family and a
simple random-restart search over it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class FourierTrajectory:
    """Finite Fourier series, periodic and zero-velocity at the endpoints.

    q_j(t) = q0_j + sum_k [ a_jk/(w k) sin(w k t) - b_jk/(w k) cos(w k t) ]
    """

    q0: np.ndarray           # (nq,) offset
    a: np.ndarray            # (nq, n_harmonics)
    b: np.ndarray            # (nq, n_harmonics)
    base_freq: float = 0.1   # Hz

    @property
    def period(self) -> float:
        return 1.0 / self.base_freq

    def __call__(self, t: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return ``(q, v, a)`` at times ``t``, each ``(len(t), nq)``."""
        t = np.atleast_1d(np.asarray(t, dtype=float))[:, None, None]   # (T,1,1)
        w = 2 * np.pi * self.base_freq
        k = np.arange(1, self.a.shape[1] + 1)[None, None, :]           # (1,1,K)
        wk = w * k
        A, B = self.a[None], self.b[None]                              # (1,nq,K)

        q = self.q0[None] + np.sum(A / wk * np.sin(wk * t) - B / wk * np.cos(wk * t), axis=2)
        v = np.sum(A * np.cos(wk * t) + B * np.sin(wk * t), axis=2)
        acc = np.sum(-A * wk * np.sin(wk * t) + B * wk * np.cos(wk * t), axis=2)
        return q, v, acc

    def respects_limits(
        self, q_min: np.ndarray, q_max: np.ndarray, v_max: np.ndarray, n: int = 500
    ) -> bool:
        t = np.linspace(0, self.period, n)
        q, v, _ = self(t)
        return bool(
            (q >= q_min).all() and (q <= q_max).all() and (np.abs(v) <= v_max).all()
        )


def random_trajectory(
    nq: int,
    q_min: np.ndarray,
    q_max: np.ndarray,
    v_max: np.ndarray,
    n_harmonics: int = 5,
    base_freq: float = 0.1,
    rng: np.random.Generator | None = None,
    max_tries: int = 2000,
) -> FourierTrajectory:
    """Sample a limit-respecting excitation trajectory.

    Rejection sampling, deliberately: it is short, has no dependencies, and
    identification quality is driven far more by *running several different*
    trajectories than by optimising one of them to the last decimal.
    """
    rng = np.random.default_rng() if rng is None else rng
    q_mid = (np.asarray(q_min) + np.asarray(q_max)) / 2
    scale = np.asarray(v_max)[:, None] / (2 * n_harmonics)
    for _ in range(max_tries):
        a = rng.normal(0, 1, (nq, n_harmonics)) * scale
        b = rng.normal(0, 1, (nq, n_harmonics)) * scale
        traj = FourierTrajectory(q0=q_mid.copy(), a=a, b=b, base_freq=base_freq)
        if traj.respects_limits(q_min, q_max, v_max):
            return traj
    raise RuntimeError(
        "could not sample a trajectory inside the joint limits. Lower "
        "`base_freq` or `v_max`, or widen the limits."
    )


def static_poses(
    q_min: np.ndarray, q_max: np.ndarray, n: int = 40, rng: np.random.Generator | None = None
) -> np.ndarray:
    """Random static poses, for gravity-only (quasi-static) identification.

    Start here. Static identification needs no acceleration estimate, no
    velocity filtering and no excitation design -- it identifies the gravity
    terms and the torque constants, which is most of what ``tau_ext`` needs.
    Add the dynamic trajectories only once the static fit is good.
    """
    rng = np.random.default_rng() if rng is None else rng
    return rng.uniform(np.asarray(q_min), np.asarray(q_max), size=(n, len(q_min)))
