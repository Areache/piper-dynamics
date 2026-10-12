"""Data collection from a live arm.

This is the only module that talks to hardware, and it is kept deliberately
thin: move somewhere, wait for settling, record telemetry, repeat. Everything
else in the package works offline from the ``.npz`` this produces.

Requires the ``[can]`` extra and a connected arm.

SAFETY
------
Calibration drives the arm through its workspace unattended. Before running:
clear the workspace, keep the e-stop in hand, start with
``--speed-scale 0.2``. The static procedure below never commands torque; it
only commands positions and reads telemetry.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class CalibrationSamples:
    """Raw telemetry. Deliberately a dumb container -- no derived quantities,
    so a re-identification never has to re-run the hardware."""

    q: np.ndarray            # (T, nq) rad
    v: np.ndarray            # (T, nq) rad/s
    current: np.ndarray      # (T, nq) SDK units
    static: np.ndarray       # (T,) bool: sample taken at rest
    dt: float
    meta: dict

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path, q=self.q, v=self.v, current=self.current,
            static=self.static, dt=self.dt, meta=np.array([repr(self.meta)]),
        )

    @classmethod
    def load(cls, path: str | Path) -> CalibrationSamples:
        d = np.load(path, allow_pickle=False)
        meta = eval(str(d["meta"][0])) if "meta" in d else {}   # noqa: S307 - our own repr
        return cls(
            q=d["q"], v=d["v"], current=d["current"],
            static=d["static"], dt=float(d["dt"]), meta=meta,
        )


def collect_static(
    arm,
    poses: np.ndarray,
    settle_s: float = 1.5,
    average_s: float = 0.5,
    rate_hz: float = 100.0,
) -> CalibrationSamples:
    """Drive to each pose, let it settle, average the telemetry.

    ``arm`` is any object exposing ``move_j(q)`` and ``read_state() ->
    (q, v, current)``. That protocol is small on purpose: wrap whichever SDK
    version you have rather than pinning one here. See ``io/can.py``.

    Averaging over ``average_s`` at rest is what makes static calibration
    accurate: it removes the current ripple that would otherwise dominate the
    torque constants.
    """
    n_avg = max(int(average_s * rate_hz), 1)
    qs, vs, cs = [], [], []
    for i, q_target in enumerate(poses):
        arm.move_j(q_target)
        time.sleep(settle_s)
        buf_q, buf_v, buf_c = [], [], []
        for _ in range(n_avg):
            q, v, c = arm.read_state()
            buf_q.append(q)
            buf_v.append(v)
            buf_c.append(c)
            time.sleep(1.0 / rate_hz)
        qs.append(np.mean(buf_q, axis=0))
        vs.append(np.mean(buf_v, axis=0))
        cs.append(np.mean(buf_c, axis=0))
        print(f"  pose {i + 1}/{len(poses)} done", flush=True)

    q = np.stack(qs)
    v = np.stack(vs)
    c = np.stack(cs)
    return CalibrationSamples(
        q=q, v=v, current=c, static=np.ones(len(q), dtype=bool),
        dt=1.0 / rate_hz, meta={"procedure": "static", "settle_s": settle_s},
    )


def collect_trajectory(arm, traj, duration_s: float, rate_hz: float = 100.0):
    """Track an excitation trajectory and log telemetry.

    TODO(hardware): needs a position-streaming loop at `rate_hz` and a check
    that the arm is actually tracking (log the tracking error alongside, and
    refuse to identify from data where it is large -- an untracked trajectory
    silently breaks the acceleration term).
    """
    raise NotImplementedError(
        "dynamic trajectory collection is not implemented yet. Static "
        "calibration (collect_static) identifies gravity terms and torque "
        "constants, which is enough for quasi-static tau_ext and payload "
        "estimation. See docs/validation.md for what dynamic identification "
        "would add."
    )
