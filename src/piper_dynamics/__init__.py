"""piper-dynamics -- identified dynamics and force estimation for the Piper arm.

The problem this solves: the ``effort`` field in your recorded episodes is a
raw current reading. It is dominated by the arm holding itself up, so feeding
it to a policy teaches the policy about pose, not about contact. This package
subtracts the arm's own dynamics and gives you what is left.

    from piper_dynamics import PiperModel, ArmParams, TauExtEstimator

    model  = PiperModel("piper_description.urdf")
    params = ArmParams.load("params/my_arm.yaml")
    est    = TauExtEstimator(model, params, quasi_static=True)

    tau_ext = est.batch(q, v, current)        # (T, 6) N·m

Read ``piper_dynamics.conventions`` before interpreting the sign of anything.
"""

from .conventions import GRAVITY_WORLD, PIPER_NQ
from .estimation import (
    PayloadEstimate,
    TauExtEstimator,
    TauExtResult,
    current_to_torque,
    estimate_payload_mass,
    wrench_from_tau_ext,
)
from .model import PiperModel
from .params import ArmParams, JointParams, ParamError, Provenance

__version__ = "0.0.1"

__all__ = [
    "GRAVITY_WORLD",
    "PIPER_NQ",
    "ArmParams",
    "JointParams",
    "ParamError",
    "PayloadEstimate",
    "PiperModel",
    "Provenance",
    "TauExtEstimator",
    "TauExtResult",
    "__version__",
    "current_to_torque",
    "estimate_payload_mass",
    "wrench_from_tau_ext",
]
