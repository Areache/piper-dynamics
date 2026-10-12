"""Sign and frame conventions used throughout this package.

Everything here is a convention, not a derivation. It is written down because
every sign error in this kind of code traces back to an unstated assumption.

Torque balance
--------------
We model the measured joint torque as::

    tau_meas = tau_model(q, v, a) + tau_fric(v) + tau_ext

where

``tau_meas``
    Torque reported by the motors, obtained from current via
    ``tau_meas = direction * kt * (current - current_offset)``.
``tau_model``
    Rigid-body dynamics of the *unloaded* arm, ``M(q)a + C(q,v)v + g(q)``,
    computed by RNEA from the URDF plus identified inertial parameters.
``tau_fric``
    Joint friction (static / Coulomb / viscous), identified per joint.
``tau_ext``
    Everything the model does not explain. This is the output we care about:
    the extra joint torque the motors must supply because of the outside
    world -- a grasped payload, a contact, an obstacle.

So the estimator is a subtraction::

    tau_ext = tau_meas - tau_model - tau_fric

Wrench convention
-----------------
``tau_ext`` relates to an external wrench ``F`` applied *by the environment on
the robot* at a given frame as::

    tau_ext = -J(q)^T @ F

Holding a payload of mass ``m`` at the end-effector therefore gives, in the
quasi-static case::

    F        = m * g_vec                     (gravity pulls down on the robot)
    tau_ext  = -J_v^T @ (m * g_vec)
             =  m * J_v^T @ (-g_vec)         (motors must pull up)

with ``g_vec = (0, 0, -9.81)`` in the world frame. ``payload.py`` builds its
regressor from exactly that last line.

Gravity
-------
Gravity lives in the Pinocchio model (``model.gravity.linear``). If the arm is
not mounted upright, rotate it there -- never bake a mount rotation into the
URDF and the gravity vector at the same time.
"""

from __future__ import annotations

import numpy as np

#: Standard gravity in the world frame, m/s^2.
GRAVITY_WORLD = np.array([0.0, 0.0, -9.81])

#: Number of actuated joints on a standard Piper (gripper excluded).
PIPER_NQ = 6
