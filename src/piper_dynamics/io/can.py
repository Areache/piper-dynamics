"""Live arm access.

Deliberately a thin protocol rather than an SDK dependency. Everything in
``calibration.collect`` needs is:

    move_j(q)                 -> command a joint-space target
    read_state()              -> (q, v, current)

Wrap whichever SDK version you have. The adapter below targets ``piper_sdk``
as of writing; if your version differs, copy it and fix the field names rather
than pinning an SDK version in this package's dependencies.

Units matter and the SDK's are not SI: joint angles have historically been
reported in 0.001 degree, and the current field's unit has changed between
firmware revisions. ``UNIT_SCALE`` is where you correct that, and
``docs/validation.md`` has a two-minute check that you got it right.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np


class ArmInterface(Protocol):
    """The whole hardware contract this package relies on."""

    def move_j(self, q: np.ndarray) -> None: ...
    def read_state(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]: ...


#: Conversion from SDK units to SI. VERIFY THESE AGAINST YOUR FIRMWARE.
UNIT_SCALE = {
    "angle": np.deg2rad(0.001),   # 0.001 deg -> rad
    "velocity": np.deg2rad(0.001),
    "current": 0.001,             # mA -> A  (check: some builds report 0.1%)
}


class PiperSdkArm:
    """Adapter over ``piper_sdk``.

    TODO(hardware): field names below are written against the SDK's
    ``GetArmJointMsgs`` / ``GetArmHighSpdInfoMsgs`` shape and have NOT been
    tested on hardware in this repository. Fix them against your SDK version,
    then run the unit check in docs/validation.md before calibrating.
    """

    def __init__(self, can_port: str = "can0", n_joints: int = 6) -> None:
        try:
            from piper_sdk import C_PiperInterface_V2
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "piper_sdk is required for live access: "
                "`pip install piper-dynamics[can]`"
            ) from exc
        self.n_joints = n_joints
        self.piper = C_PiperInterface_V2(can_port)
        self.piper.ConnectPort()

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

    def move_j(self, q: np.ndarray) -> None:
        raise NotImplementedError(
            "wire this to your SDK's joint control call, and set the motion "
            "mode and speed scale first. Left unimplemented on purpose: a "
            "wrong move_j is the one failure in this package that can break "
            "hardware."
        )
