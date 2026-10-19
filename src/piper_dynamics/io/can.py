"""Live arm access.

Deliberately a thin protocol rather than an SDK dependency. Everything in
``calibration.collect`` needs is:

    move_j(q)                 -> command a joint-space target
    read_state()              -> (q, v, current)

Wrap whichever SDK version you have. The adapter below targets ``piper_sdk``
0.6.2; if your version differs, copy it and fix the field names rather than
pinning an SDK version in this package's dependencies.

Units matter and the SDK's are not SI. ``UNIT_SCALE`` is where you correct
that, and ``docs/validation.md`` has a two-minute check that you got it right.
"""

from __future__ import annotations

import time
from typing import Protocol

import numpy as np


class ArmInterface(Protocol):
    """The whole hardware contract this package relies on."""

    def move_j(self, q: np.ndarray) -> None: ...
    def read_state(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]: ...


#: Conversion from SDK units to SI. Checked against piper_sdk 0.6.2 source and
#: a passive read on two arms (2026-10-12); re-check if your SDK differs.
UNIT_SCALE = {
    "angle": np.deg2rad(0.001),   # GetArmJointMsgs joint_i: 0.001 deg -> rad
    "velocity": 0.001,            # GetArmHighSpdInfoMsgs motor_speed: 0.001 rad/s -> rad/s
    "current": 0.001,             # GetArmHighSpdInfoMsgs current: 0.001 A -> A (signed int16)
}

#: Upper bound on the SDK speed percentage this adapter will send.
MAX_SPEED_PCT = 20


class PiperSdkArm:
    """Adapter over ``piper_sdk`` (``C_PiperInterface_V2``).

    Reading is passive. Motion needs ``allow_motion=True`` and an arm that is
    already enabled: this class never enables, disables or reconfigures the
    arm, so whoever runs it has to have done that deliberately.

    ``move_j`` resends a CAN-mode MOVE J command at ``resend_hz`` until the arm
    settles inside ``tol_rad`` with near-zero speed, and raises on timeout
    instead of moving on -- a pose that was not reached must not be recorded
    as if it had been.
    """

    def __init__(
        self,
        can_port: str = "can0",
        n_joints: int = 6,
        allow_motion: bool = False,
        speed_pct: int = 10,
        sdk=None,
    ) -> None:
        if not 1 <= speed_pct <= MAX_SPEED_PCT:
            raise ValueError(f"speed_pct must be in 1..{MAX_SPEED_PCT}, got {speed_pct}")
        if sdk is None:
            try:
                from piper_sdk import C_PiperInterface_V2
            except ImportError as exc:  # pragma: no cover
                raise ImportError(
                    "piper_sdk is required for live access: "
                    "`pip install piper-dynamics[can]`"
                ) from exc
            sdk = C_PiperInterface_V2(can_port)
            sdk.ConnectPort()
        self.piper = sdk
        self.n_joints = n_joints
        self.allow_motion = allow_motion
        self.speed_pct = int(speed_pct)

    def read_state(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        joint = self.piper.GetArmJointMsgs().joint_state
        hs = self.piper.GetArmHighSpdInfoMsgs()
        q = np.array([getattr(joint, f"joint_{i + 1}") for i in range(self.n_joints)], dtype=float)
        v = np.array([getattr(hs, f"motor_{i + 1}").motor_speed for i in range(self.n_joints)],
                     dtype=float)
        c = np.array([getattr(hs, f"motor_{i + 1}").current for i in range(self.n_joints)],
                     dtype=float)
        return (
            q * UNIT_SCALE["angle"],
            v * UNIT_SCALE["velocity"],
            c * UNIT_SCALE["current"],
        )

    def move_j(
        self,
        q: np.ndarray,
        tol_rad: float = np.deg2rad(0.5),
        settle_speed: float = 0.01,
        timeout_s: float = 20.0,
        resend_hz: float = 50.0,
    ) -> None:
        if not self.allow_motion:
            raise RuntimeError(
                "motion is disabled. Construct PiperSdkArm(..., allow_motion=True) "
                "only after an operator has cleared the workspace and holds the e-stop."
            )
        q = np.asarray(q, dtype=float)
        if q.shape != (self.n_joints,) or not np.isfinite(q).all():
            raise ValueError(f"target must be {self.n_joints} finite joint angles, got {q!r}")
        enabled = self.piper.GetArmEnableStatus()
        if not all(enabled[: self.n_joints]):
            raise RuntimeError(
                f"arm is not enabled on all joints ({enabled}); enable it explicitly first"
            )

        target = [int(round(x / UNIT_SCALE["angle"])) for x in q]
        t0 = time.monotonic()
        while True:
            # ctrl_mode 0x01 = CAN command, move_mode 0x01 = MOVE J, is_mit_mode 0x00 = position.
            self.piper.MotionCtrl_2(0x01, 0x01, self.speed_pct, 0x00)
            self.piper.JointCtrl(*target)
            q_now, v_now, _ = self.read_state()
            if np.max(np.abs(q_now - q)) < tol_rad and np.max(np.abs(v_now)) < settle_speed:
                return
            if time.monotonic() - t0 > timeout_s:
                raise TimeoutError(
                    f"did not reach target within {timeout_s} s: "
                    f"max error {np.rad2deg(np.max(np.abs(q_now - q))):.2f} deg"
                )
            time.sleep(1.0 / resend_hz)
