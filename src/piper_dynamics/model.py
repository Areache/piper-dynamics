"""Rigid-body model of the arm: URDF in, kinematics and RNEA out.

Thin wrapper over Pinocchio. The only reason it exists is to keep the rest of
the package from caring about Pinocchio's API, and to put the end-effector
frame lookup in one place.

The URDF is *not* vendored in this repository -- see ``assets/urdf/README.md``.
Point ``PiperModel.from_urdf`` at AgileX's description package, or at your own.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .conventions import GRAVITY_WORLD

# Imported lazily, not at module scope: the pure-math half of this package
# (params, friction, the payload least squares) is useful without Pinocchio,
# and CI runs those tests on machines that do not have it.
pin = None


def _require_pin():
    global pin
    if pin is None:
        try:
            import pinocchio as _pin
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "pinocchio is required for PiperModel. `pip install pin` (the "
                "PyPI wheel) or install it from conda-forge/robotpkg. The rest "
                "of this package works without it."
            ) from exc
        pin = _pin
    return pin


class PiperModel:
    """Kinematics and dynamics of one arm.

    Parameters
    ----------
    urdf_path
        Path to the URDF. Meshes are not needed for dynamics.
    ee_frame
        Name of the end-effector frame in the URDF. Piper's description
        packages have used more than one name across releases, so this is
        explicit rather than guessed; `list_frames()` tells you what is there.
    gravity
        World-frame gravity. Override for a non-upright mount -- and only here,
        never also in the URDF.
    """

    def __init__(
        self,
        urdf_path: str | Path,
        ee_frame: str = "link6",
        gravity: np.ndarray | None = None,
    ) -> None:
        pin = _require_pin()
        urdf_path = Path(urdf_path)
        if not urdf_path.exists():
            raise FileNotFoundError(f"URDF not found: {urdf_path}")
        self.urdf_path = urdf_path
        self.model = pin.buildModelFromUrdf(str(urdf_path))
        self.data = self.model.createData()
        self.model.gravity.linear = np.asarray(
            GRAVITY_WORLD if gravity is None else gravity, dtype=float
        )
        if not self.model.existFrame(ee_frame):
            raise ValueError(
                f"frame {ee_frame!r} is not in this URDF. Available frames:\n  "
                + "\n  ".join(self.list_frames())
            )
        self.ee_frame = ee_frame
        self.ee_id = self.model.getFrameId(ee_frame)

    # -- introspection ----------------------------------------------------
    def list_frames(self) -> list[str]:
        return [f.name for f in self.model.frames]

    @property
    def nq(self) -> int:
        return self.model.nq

    @property
    def nv(self) -> int:
        return self.model.nv

    # -- kinematics -------------------------------------------------------
    def fk(self, q: np.ndarray) -> np.ndarray:
        """End-effector pose as a 4x4 homogeneous transform in the base frame."""
        q = np.asarray(q, dtype=float)
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacement(self.model, self.data, self.ee_id)
        return self.data.oMf[self.ee_id].homogeneous

    def jacobian(self, q: np.ndarray, local: bool = False) -> np.ndarray:
        """6 x nv end-effector Jacobian.

        Default is ``LOCAL_WORLD_ALIGNED``: translation and rotation axes are
        world-aligned but the reference point is the end-effector. That is the
        frame wrenches are most often wanted in. ``local=True`` gives the
        end-effector's own frame.
        """
        q = np.asarray(q, dtype=float)
        ref = pin.ReferenceFrame.LOCAL if local else pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
        pin.computeJointJacobians(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        return pin.getFrameJacobian(self.model, self.data, self.ee_id, ref)

    def jacobian_linear(self, q: np.ndarray) -> np.ndarray:
        """Translational part of the Jacobian, 3 x nv, world-aligned."""
        return self.jacobian(q)[:3, :]

    # -- dynamics ---------------------------------------------------------
    def rnea(self, q: np.ndarray, v: np.ndarray, a: np.ndarray) -> np.ndarray:
        """Inverse dynamics ``M(q)a + C(q,v)v + g(q)`` of the unloaded arm."""
        q, v, a = (np.asarray(x, dtype=float) for x in (q, v, a))
        return pin.rnea(self.model, self.data, q, v, a).copy()

    def gravity_torque(self, q: np.ndarray) -> np.ndarray:
        """``g(q)`` alone -- the quasi-static case."""
        q = np.asarray(q, dtype=float)
        return pin.computeGeneralizedGravity(self.model, self.data, q).copy()

    def joint_torque_regressor(
        self, q: np.ndarray, v: np.ndarray, a: np.ndarray
    ) -> np.ndarray:
        """``nv x (10*njoints)`` regressor, linear in the inertial parameters.

        ``tau = Y(q,v,a) @ phi`` where ``phi`` stacks each body's
        ``[m, m*c_x, m*c_y, m*c_z, I_xx, I_xy, I_yy, I_xz, I_yz, I_zz]``.
        This is what makes inertial identification a least-squares problem.
        """
        q, v, a = (np.asarray(x, dtype=float) for x in (q, v, a))
        return pin.computeJointTorqueRegressor(self.model, self.data, q, v, a).copy()

    def inertial_params(self) -> np.ndarray:
        """Current inertial parameter vector ``phi``, as in the regressor."""
        return np.concatenate([self.model.inertias[i].toDynamicParameters()
                               for i in range(1, self.model.njoints)])

    def set_inertial_params(self, phi: np.ndarray) -> None:
        """Write an identified ``phi`` back into the model."""
        phi = np.asarray(phi, dtype=float)
        expected = 10 * (self.model.njoints - 1)
        if phi.size != expected:
            raise ValueError(f"expected {expected} inertial parameters, got {phi.size}")
        for i in range(1, self.model.njoints):
            self.model.inertias[i] = pin.Inertia.FromDynamicParameters(
                phi[10 * (i - 1) : 10 * i]
            )
        self.data = self.model.createData()

    @classmethod
    def from_urdf(cls, urdf_path: str | Path, **kw) -> PiperModel:
        return cls(urdf_path, **kw)
