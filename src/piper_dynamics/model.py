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


def _frame_parents(frame) -> tuple[int, int]:
    """``(parent joint, parent frame)``; Pinocchio 3 renamed both attributes."""
    if hasattr(frame, "parentJoint"):
        return frame.parentJoint, frame.parentFrame
    return frame.parent, frame.previousFrame


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
    locked_joints
        Joints to freeze at their neutral position so the model has one DOF
        per motor. AgileX descriptions model the two gripper fingers as
        prismatic ``joint7``/``joint8``; left in, they make the model 8-DOF
        and no 6-joint parameter file matches it. The default locks those two
        when present, lumping the fingers into ``link6`` at the closed
        position. Pass ``()`` to keep every joint.
    """

    #: Gripper finger joints in AgileX Piper descriptions.
    DEFAULT_LOCKED = ("joint7", "joint8")

    def __init__(
        self,
        urdf_path: str | Path,
        ee_frame: str = "link6",
        gravity: np.ndarray | None = None,
        locked_joints: tuple[str, ...] | list[str] | None = None,
    ) -> None:
        pin = _require_pin()
        urdf_path = Path(urdf_path)
        if not urdf_path.exists():
            raise FileNotFoundError(f"URDF not found: {urdf_path}")
        self.urdf_path = urdf_path
        model = pin.buildModelFromUrdf(str(urdf_path))
        if locked_joints is None:
            locked_joints = [n for n in self.DEFAULT_LOCKED if model.existJointName(n)]
        missing = [n for n in locked_joints if not model.existJointName(n)]
        if missing:
            raise ValueError(f"cannot lock {missing}: not joints of this URDF")
        if locked_joints:
            ids = [model.getJointId(n) for n in locked_joints]
            model = pin.buildReducedModel(model, ids, pin.neutral(model))
        self.locked_joints = tuple(locked_joints)
        self.model = model
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

    def gravity_regressor(self, q: np.ndarray) -> np.ndarray:
        """``nv x (4*njoints)``: ``g(q) = Y @ phi_g`` with ``phi_g`` stacking
        each link's ``[m, m*cx, m*cy, m*cz]``, the only parameters a static
        pose sees. Column order matches `link_specs`."""
        z = np.zeros(self.nv)
        Y = self.joint_torque_regressor(q, z, z)
        return Y.reshape(self.nv, -1, 10)[:, :, :4].reshape(self.nv, -1)

    def link_specs(self, package_dirs: list[str] | None = None, gripper=None) -> list:
        """`calibration.feasibility.LinkSpec` for every moving link.

        The centre-of-mass box comes from the URDF collision geometry when it
        can be loaded (``package_dirs`` resolves ``package://`` meshes), and
        otherwise from the link's joint origins (``kinematic-fallback``). The
        end link gets a gripper allowance (default `GripperAllowance()`),
        because a fit lumps whatever the URDF leaves out into it.
        """
        from .calibration.feasibility import UNITS, GripperAllowance, LinkSpec, kinematic_box

        boxes = self._geometry_boxes(package_dirs)
        specs = []
        for i, name in enumerate(self.moving_links(), start=1):
            prior = self.model.inertias[i].toDynamicParameters()[:4]
            com = prior[1:] / prior[0] if prior[0] > 0 else None
            if i in boxes:
                lo, hi = boxes[i]
                if com is not None:
                    lo, hi = np.minimum(lo, com), np.maximum(hi, com)
                source = "mesh"
            else:
                carried = [self.model.jointPlacements[j].translation
                           for j in range(1, self.model.njoints) if self.model.parents[j] == i]
                carried += [f.placement.translation for f in self.model.frames
                            if _frame_parents(f)[0] == i]
                lo, hi = kinematic_box(np.array(carried).reshape(-1, 3), com)
                source = "kinematic-fallback"
            end = i == self.model.njoints - 1
            specs.append(LinkSpec(name, prior, lo, hi, source, units=UNITS,
                                  gripper=(gripper or GripperAllowance()) if end else None))
        return specs

    def _geometry_boxes(self, package_dirs) -> dict[int, tuple[np.ndarray, np.ndarray]]:
        """Collision-geometry bounding box per joint, in the joint frame.
        Empty when the geometry cannot be loaded (meshes not found, no
        collision library); the caller then falls back to kinematics."""
        try:
            full = pin.buildModelFromUrdf(str(self.urdf_path))
            kw = {"package_dirs": package_dirs} if package_dirs else {}
            geom = pin.buildGeomFromUrdf(full, str(self.urdf_path),
                                         pin.GeometryType.COLLISION, **kw)
            if self.locked_joints:
                ids = [full.getJointId(n) for n in self.locked_joints]
                _, geom = pin.buildReducedModel(full, geom, ids, pin.neutral(full))
            boxes: dict[int, tuple[np.ndarray, np.ndarray]] = {}
            for obj in geom.geometryObjects:
                obj.geometry.computeLocalAABB()
                lo = np.asarray(obj.geometry.aabb_local.min_)
                hi = np.asarray(obj.geometry.aabb_local.max_)
                corners = np.array([[x, y, z] for x in (lo[0], hi[0])
                                    for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
                pts = corners @ obj.placement.rotation.T + obj.placement.translation
                j = obj.parentJoint
                if j in boxes:
                    pts = np.vstack([pts, *boxes[j]])
                boxes[j] = (pts.min(axis=0), pts.max(axis=0))
            return boxes
        except Exception:  # noqa: BLE001 -- any failure means "no geometry"
            return {}

    def inertial_params(self) -> np.ndarray:
        """Current inertial parameter vector ``phi``, as in the regressor."""
        return np.concatenate([self.model.inertias[i].toDynamicParameters()
                               for i in range(1, self.model.njoints)])

    def moving_links(self) -> list[str]:
        """URDF link carried by each joint, in joint order.

        These are the keys of ``ArmParams.inertial_overrides``. A link welded
        on by a fixed joint, or a locked gripper finger, is lumped into the
        link it hangs from and has no inertial parameters of its own.
        """
        body = pin.FrameType.BODY
        names = []
        for i in range(1, self.model.njoints):
            joint_frame = self.model.getFrameId(self.model.names[i], pin.FrameType.JOINT)
            carried = [f.name for f in self.model.frames
                       if f.type == body and _frame_parents(f) == (i, joint_frame)]
            if len(carried) != 1:
                raise ValueError(
                    f"joint {self.model.names[i]!r} carries links {carried}; expected exactly one"
                )
            names.append(carried[0])
        return names

    def set_inertial_overrides(self, overrides: dict[str, list[float]]) -> None:
        """Write per-link inertial parameters, looked up by URDF link name.

        Every moving link must be present, and nothing else: a partial or
        misnamed set raises instead of being matched by position, because a
        positional match moves parameters onto the wrong link without a sound.
        """
        links = self.moving_links()
        unknown = sorted(k for k in overrides if not self.model.existBodyName(k))
        lumped = sorted(k for k in overrides if k not in links and k not in unknown)
        missing = [k for k in links if k not in overrides]
        problems = []
        if unknown:
            problems.append(f"not links of this URDF: {unknown}")
        if lumped:
            problems.append(f"links without their own parameters (fixed or locked): {lumped}")
        if missing:
            problems.append(f"missing: {missing}")
        bad = {k: np.asarray(v).shape for k, v in overrides.items()
               if k in links and np.asarray(v).shape != (10,)}
        if bad:
            problems.append(f"expected 10 values per link, got shapes {bad}")
        if problems:
            raise ValueError(
                "inertial_overrides must name every moving link exactly once "
                f"({links}). " + "; ".join(problems)
            )
        self.set_inertial_params(np.concatenate([np.asarray(overrides[k], dtype=float)
                                                 for k in links]))

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
