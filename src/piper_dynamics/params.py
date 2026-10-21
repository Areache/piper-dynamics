"""Arm parameters: load, validate, carry provenance.

Parameters are *data*, not code. Nothing in this package hard-codes a torque
constant or a friction coefficient. The values shipped in
``params/piper_reference.yaml`` are one person's arm, and are a starting point
and a sanity check -- not a substitute for calibrating your own.

Every parameter file records where it came from. If you cannot say which arm,
which firmware and which date a number came from, the number is not usable by
anyone else.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

SCHEMA_VERSION = 1


class ParamError(ValueError):
    """Raised when a parameter file is malformed, stale or implausible."""


@dataclass
class Provenance:
    """Where these numbers came from. Required -- see module docstring."""

    arm_id: str = "unknown"           # your label for this physical arm
    model: str = "piper"              # piper | piper-x | piper-l | piper-h
    firmware: str = "unknown"
    calibrated_by: str = "unknown"
    calibrated_at: str = "unknown"    # ISO date
    urdf_source: str = "unknown"
    notes: str = ""
    # Filled in by the identification step; read by `piper-diagnose`.
    fit_rms_nm: list[float] | None = None
    n_samples: int | None = None


@dataclass
class JointParams:
    """Per-joint electrical and friction parameters.

    kt
        Current-to-torque constant at the joint output, N·m/A. Folds in the
        gear ratio, so this is *joint* torque per *motor* current unit.
    direction
        +1 or -1. Sign of the motor current relative to the URDF joint axis.
        Do not fix a sign error by negating kt -- keep magnitudes positive and
        the sign here, so the two stay separable.
    current_offset
        Current reading at zero torque, in the same unit the SDK reports.
    fc, fv
        Coulomb and viscous friction, N·m and N·m·s/rad.
    fs
        Static ("stiction") friction, N·m. Applied with a smooth sign so the
        estimator does not chatter around zero velocity.
    """

    kt: float
    direction: int = 1
    current_offset: float = 0.0
    fc: float = 0.0
    fv: float = 0.0
    fs: float = 0.0

    def __post_init__(self) -> None:
        if self.direction not in (1, -1):
            raise ParamError(f"direction must be +1 or -1, got {self.direction}")
        if self.kt <= 0:
            raise ParamError(f"kt must be positive (use `direction` for sign), got {self.kt}")


@dataclass
class ArmParams:
    """Everything needed to turn raw telemetry into tau_ext."""

    joints: list[JointParams]
    provenance: Provenance = field(default_factory=Provenance)
    #: Optional per-link inertial overrides, keyed by URDF link name: every
    #: link in `PiperModel.moving_links()`, each ``[m, m*cx, m*cy, m*cz, Ixx,
    #: Ixy, Iyy, Ixz, Iyz, Izz]`` in SI about the joint frame. Nothing in this
    #: package writes it yet; absent means "trust the URDF".
    inertial_overrides: dict[str, list[float]] = field(default_factory=dict)
    #: Unit of the `current` field as reported by your SDK, for the record.
    current_unit: str = "A"
    schema_version: int = SCHEMA_VERSION

    # -- vectorised views -------------------------------------------------
    @property
    def kt(self) -> np.ndarray:
        return np.array([j.kt for j in self.joints])

    @property
    def direction(self) -> np.ndarray:
        return np.array([j.direction for j in self.joints], dtype=float)

    @property
    def current_offset(self) -> np.ndarray:
        return np.array([j.current_offset for j in self.joints])

    @property
    def fc(self) -> np.ndarray:
        return np.array([j.fc for j in self.joints])

    @property
    def fv(self) -> np.ndarray:
        return np.array([j.fv for j in self.joints])

    @property
    def fs(self) -> np.ndarray:
        return np.array([j.fs for j in self.joints])

    @property
    def nq(self) -> int:
        return len(self.joints)

    # -- io ---------------------------------------------------------------
    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ArmParams:
        version = d.get("schema_version", 0)
        if version != SCHEMA_VERSION:
            raise ParamError(
                f"parameter file is schema v{version}, this package expects "
                f"v{SCHEMA_VERSION}. Re-run `piper-calibrate` rather than "
                f"hand-editing: the meaning of the fields may have changed."
            )
        if "joints" not in d:
            raise ParamError("parameter file has no `joints` section")
        joints = [JointParams(**j) for j in d["joints"]]
        prov = Provenance(**d.get("provenance", {}))
        return cls(
            joints=joints,
            provenance=prov,
            inertial_overrides=d.get("inertial_overrides", {}),
            current_unit=d.get("current_unit", "A"),
            schema_version=version,
        )

    @classmethod
    def load(cls, path: str | Path) -> ArmParams:
        path = Path(path)
        if not path.exists():
            raise ParamError(
                f"{path} not found. Calibrate your own arm with `piper-calibrate`, "
                f"or start from params/piper_reference.yaml -- but read "
                f"params/README.md first: reference values are not expected to "
                f"transfer between units."
            )
        with path.open() as f:
            return cls.from_dict(yaml.safe_load(f))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "current_unit": self.current_unit,
            "provenance": dataclasses.asdict(self.provenance),
            "joints": [dataclasses.asdict(j) for j in self.joints],
            "inertial_overrides": self.inertial_overrides,
        }

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False, default_flow_style=False)

    # -- checks -----------------------------------------------------------
    def warnings(self) -> list[str]:
        """Plausibility checks. Not fatal -- `piper-diagnose` prints these."""
        out: list[str] = []
        p = self.provenance
        if p.arm_id == "unknown" or p.calibrated_at == "unknown":
            out.append(
                "provenance is incomplete: without arm_id and calibration date "
                "these numbers cannot be interpreted by anyone else (including "
                "you, in six months)."
            )
        for i, j in enumerate(self.joints):
            if not np.isfinite([j.kt, j.fc, j.fv, j.fs]).all():
                out.append(f"joint {i}: non-finite parameter")
            if j.fc < 0 or j.fv < 0 or j.fs < 0:
                out.append(f"joint {i}: negative friction coefficient ({j.fc=}, {j.fv=}, {j.fs=})")
        if p.fit_rms_nm is not None:
            for i, r in enumerate(p.fit_rms_nm):
                if r > 0.5:
                    out.append(
                        f"joint {i}: identification residual {r:.2f} N·m is high. "
                        f"tau_ext from this joint will be dominated by model error; "
                        f"see docs/validation.md."
                    )
        return out
