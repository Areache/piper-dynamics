"""Parameter identification: least squares on the regressor.

Two procedures, in increasing order of ambition:

``identify_static``
    Gravity-only. Uses static poses to fit, per joint, the torque constant and
    the Coulomb/static friction offset against the URDF's gravity torque.
    Needs no acceleration, no velocity filtering, no excitation design.
    **Start here.** It is enough for quasi-static ``tau_ext`` and for payload
    estimation, which is what most users want.

``identify_full``
    Inertial parameters as well, from dynamic excitation trajectories. Uses
    Pinocchio's joint torque regressor. More powerful and much easier to get
    wrong -- a poorly conditioned regressor will happily return physically
    impossible inertias that fit the data.

A note on what is identifiable: current and torque enter the model only as the
product ``kt * I``, and a constant torque offset is indistinguishable from a
current offset. Do not read individual numbers as physical truths; read the
residual.
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # avoid importing pinocchio to use the rest of the package
    from ..model import PiperModel

from ..params import ArmParams, JointParams, Provenance
from .collect import CalibrationSamples


@dataclass
class IdentificationReport:
    params: ArmParams
    residual_nm: np.ndarray          # (T, nq) after fit
    rms_nm: np.ndarray               # (nq,)
    r2: np.ndarray                   # (nq,) coefficient of determination
    condition_number: float
    procedure: str

    def summary(self) -> str:
        lines = [f"identification: {self.procedure}", ""]
        lines.append(f"{'joint':>6} {'kt':>9} {'fc':>8} {'rms N·m':>9} {'R²':>7}")
        for i, j in enumerate(self.params.joints):
            lines.append(
                f"{i:>6} {j.kt:>9.4f} {j.fc:>8.3f} {self.rms_nm[i]:>9.3f} {self.r2[i]:>7.3f}"
            )
        lines.append("")
        lines.append(f"regressor condition number: {self.condition_number:.1f}")
        bad = np.where(self.rms_nm > 0.5)[0]
        if bad.size:
            lines.append(
                f"WARNING joints {bad.tolist()} have residuals above 0.5 N·m. "
                f"tau_ext on those joints will be model error, not physics."
            )
        return "\n".join(lines)


def identify_static(
    model: PiperModel,
    samples: CalibrationSamples,
    arm_id: str = "unknown",
    firmware: str = "unknown",
    calibrated_by: str = "unknown",
) -> IdentificationReport:
    """Fit ``kt`` and a constant friction offset per joint, from static poses.

    At rest the model reduces to ``tau = g(q)``, so for each joint ``j``::

        kt_j * (I_j - offset_j) = g_j(q)

    which is a two-parameter linear fit of ``g_j`` against ``I_j``. The slope
    is ``1/kt_j``... inverted: we fit ``g_j = kt_j * I_j - kt_j * offset_j``,
    i.e. a straight line through (current, gravity torque).

    The sign of the slope gives ``direction``; we keep ``kt`` positive so the
    two stay separable (see ``params.JointParams``).
    """
    mask = samples.static.astype(bool)
    q = samples.q[mask]
    current = samples.current[mask]
    if len(q) < 10:
        raise ValueError(
            f"only {len(q)} static samples; the fit needs at least ~10 well-spread "
            f"poses per joint to be meaningful. See calibration.trajectories.static_poses."
        )

    g = np.stack([model.gravity_torque(q[t]) for t in range(len(q))])   # (T, nq)

    joints: list[JointParams] = []
    resid = np.zeros_like(g)
    r2 = np.zeros(g.shape[1])
    conds = []
    for j in range(g.shape[1]):
        A = np.column_stack([current[:, j], np.ones(len(q))])
        coef, *_ = np.linalg.lstsq(A, g[:, j], rcond=None)
        slope, intercept = float(coef[0]), float(coef[1])
        conds.append(float(np.linalg.cond(A)))

        direction = 1 if slope >= 0 else -1
        kt = abs(slope)
        # intercept = -kt*direction*current_offset  ->  solve for current_offset
        current_offset = -intercept / slope if abs(slope) > 1e-12 else 0.0

        pred = A @ coef
        resid[:, j] = g[:, j] - pred
        ss_tot = float(np.sum((g[:, j] - g[:, j].mean()) ** 2))
        r2[j] = 1.0 - float(resid[:, j] @ resid[:, j]) / ss_tot if ss_tot > 1e-12 else np.nan

        joints.append(
            JointParams(
                kt=kt if kt > 1e-9 else 1e-9,
                direction=direction,
                current_offset=current_offset,
                # Static poses cannot separate Coulomb from viscous friction;
                # both stay zero until a dynamic procedure fills them in.
                fc=0.0, fv=0.0, fs=0.0,
            )
        )

    rms = np.sqrt(np.mean(resid**2, axis=0))
    params = ArmParams(
        joints=joints,
        provenance=Provenance(
            arm_id=arm_id,
            firmware=firmware,
            calibrated_by=calibrated_by,
            calibrated_at=_dt.date.today().isoformat(),
            urdf_source=str(model.urdf_path),
            notes="static (gravity-only) identification; friction not identified",
            fit_rms_nm=rms.tolist(),
            n_samples=int(len(q)),
        ),
    )
    return IdentificationReport(
        params=params,
        residual_nm=resid,
        rms_nm=rms,
        r2=r2,
        condition_number=float(np.max(conds)),
        procedure="static",
    )


def identify_full(model: PiperModel, samples: CalibrationSamples, **kw):
    """Identify inertial parameters from dynamic excitation.

    TODO: the regressor side is straightforward --
    ``Y = model.joint_torque_regressor(q, v, a)`` stacked over time, solved
    against ``kt*I - tau_fric`` with friction columns appended. The hard parts,
    which are why this is not written yet:

    * acceleration must come from a properly filtered velocity trace, or
      from the commanded trajectory rather than the measurement;
    * the regressor is rank-deficient (base parameters only are identifiable),
      so it needs a QR-based reduction, not a plain lstsq;
    * the result should be projected onto physically consistent inertias,
      otherwise you get negative masses that fit beautifully.

    Until then, `identify_static` plus `quasi_static=True` on the estimator is
    the supported path.
    """
    raise NotImplementedError(
        "full dynamic identification is not implemented. Use identify_static "
        "and run the estimator with quasi_static=True."
    )
