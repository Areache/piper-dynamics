"""The core estimator: raw telemetry in, external joint torque out.

This is the one function most users will call. Everything else in the package
either feeds it (model, params, calibration) or consumes it (wrench, payload).

See ``piper_dynamics.conventions`` for the sign convention -- read it before
interpreting the output.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # avoid importing pinocchio to use the rest of the package
    from ..model import PiperModel

from ..calibration.feasibility import ParamSet, Verdict, audit
from ..params import ArmParams
from .friction import DEFAULT_EPS, friction_torque


def current_to_torque(current: np.ndarray, params: ArmParams) -> np.ndarray:
    """Motor current -> joint torque, using the identified constants."""
    current = np.asarray(current, dtype=float)
    return params.direction * params.kt * (current - params.current_offset)


@dataclass
class TauExtResult:
    """Decomposition of one sample, kept so callers can see *why*.

    A bare ``tau_ext`` number is hard to debug. Keeping the terms lets you ask
    the useful question -- is this residual real, or is it my gravity model
    being wrong in this pose?
    """

    tau_ext: np.ndarray
    tau_meas: np.ndarray
    tau_model: np.ndarray
    tau_fric: np.ndarray

    @property
    def explained_fraction(self) -> np.ndarray:
        """Per-joint ``|model| / |meas|``. Near 1 in free space, lower under load."""
        denom = np.abs(self.tau_meas)
        denom[denom < 1e-9] = np.nan
        return np.abs(self.tau_model + self.tau_fric) / denom


class TauExtEstimator:
    """Estimate external joint torque from position, velocity and current.

    Example
    -------
    >>> model = PiperModel("piper_description.urdf")             # doctest: +SKIP
    >>> params = ArmParams.load("params/my_arm.yaml")            # doctest: +SKIP
    >>> est = TauExtEstimator(model, params)                     # doctest: +SKIP
    >>> tau_ext = est(q, v, current)                             # doctest: +SKIP
    """

    def __init__(
        self,
        model: PiperModel,
        params: ArmParams,
        friction_eps: float = DEFAULT_EPS,
        quasi_static: bool = False,
    ) -> None:
        if params.nq != model.nv:
            raise ValueError(
                f"parameter file describes {params.nq} joints but the URDF has "
                f"{model.nv} DOF. Mismatched model/params produce plausible-looking "
                f"nonsense -- fix this rather than slicing."
            )
        self.model = model
        self.params = params
        self.friction_eps = friction_eps
        self.quasi_static = quasi_static
        if params.inertial_overrides:
            # Identified inertias take precedence over the URDF's CAD values.
            # They are audited against the URDF (specs taken before the
            # overwrite): a warning, never a block.
            specs = model.link_specs()
            model.set_inertial_overrides(params.inertial_overrides)
            report = audit(ParamSet.from_inertial_overrides(params.inertial_overrides,
                                                            [s.name for s in specs]), specs)
            if report.verdict is not Verdict.FEASIBLE:
                warnings.warn(report.headline() + " Run piper-diagnose --urdf for details.",
                              stacklevel=2)

    def __call__(
        self,
        q: np.ndarray,
        v: np.ndarray,
        current: np.ndarray,
        a: np.ndarray | None = None,
    ) -> np.ndarray:
        """Convenience: return just ``tau_ext``."""
        return self.decompose(q, v, current, a).tau_ext

    def decompose(
        self,
        q: np.ndarray,
        v: np.ndarray,
        current: np.ndarray,
        a: np.ndarray | None = None,
    ) -> TauExtResult:
        """Full decomposition of a single sample."""
        q = np.asarray(q, dtype=float)
        v = np.zeros_like(q) if self.quasi_static else np.asarray(v, dtype=float)
        a = np.zeros_like(q) if (a is None or self.quasi_static) else np.asarray(a, dtype=float)

        tau_meas = current_to_torque(current, self.params)
        tau_model = self.model.rnea(q, v, a)
        tau_fric = friction_torque(
            v, self.params.fc, self.params.fv, self.params.fs, self.friction_eps
        )
        return TauExtResult(
            tau_ext=tau_meas - tau_model - tau_fric,
            tau_meas=tau_meas,
            tau_model=tau_model,
            tau_fric=tau_fric,
        )

    def batch(
        self,
        q: np.ndarray,
        v: np.ndarray,
        current: np.ndarray,
        a: np.ndarray | None = None,
    ) -> np.ndarray:
        """Vectorised over time. ``(T, nq)`` in, ``(T, nq)`` out.

        Pinocchio is called per-sample; this is a loop, not magic. For a long
        episode it is still fast enough to annotate offline (order 10k
        samples/s), which is why the offline path is the default one.
        """
        q = np.atleast_2d(q)
        v = np.atleast_2d(v)
        current = np.atleast_2d(current)
        a = np.zeros_like(q) if a is None else np.atleast_2d(a)
        if not (len(q) == len(v) == len(current) == len(a)):
            raise ValueError("q, v, current, a must have the same number of samples")
        return np.stack([self(q[t], v[t], current[t], a[t]) for t in range(len(q))])


def finite_difference_accel(v: np.ndarray, dt: float, smooth: int = 5) -> np.ndarray:
    """Joint acceleration from a velocity trace, with a moving-average first.

    Differentiating a noisy velocity signal amplifies the noise, and that noise
    lands straight in ``tau_ext``. If your data is quasi-static, prefer
    ``quasi_static=True`` over feeding in a noisy ``a``.
    """
    v = np.atleast_2d(np.asarray(v, dtype=float))
    if smooth > 1:
        kernel = np.ones(smooth) / smooth
        v = np.stack([np.convolve(v[:, j], kernel, mode="same") for j in range(v.shape[1])], 1)
    a = np.gradient(v, dt, axis=0)
    return a
