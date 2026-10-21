"""Physical feasibility of identified link mass parameters.

A good fit does not make good parameters. On a low-cost arm with current
feedback and no joint torque sensors, a free least-squares fit of link
parameters happily returns negative masses and centres of mass outside the
link -- and still predicts torque well, because the data cannot see the
difference. This module is the gate such parameters pass before anyone
trusts them. ``docs/feasibility.md`` explains how to read its verdicts.

Two tiers, kept apart on purpose:

**Gravity tier** (this module, usable now). Static poses determine only the
first mass moments of each link, ``[m, m*cx, m*cy, m*cz]``; the rotational
inertia never enters ``g(q)``. The checks are therefore pointwise and linear:
positive mass, centre of mass inside a box around the link, total mass in a
plausible range. Even those four numbers per link are not all identifiable --
static data pins down only some linear combinations of them -- so a violation
is judged against the data whenever the data is supplied: see `Verdict`.

**Full tier** (`audit_full`, after dynamic excitation). Only once the
rotational inertia is identified does the pseudo-inertia condition
``J(pi) >= 0`` apply. It needs all ten parameters of a link; running it on
static data would fit inertias the data never constrained. Interface only.

Conventions
-----------
Parameters are per link, ``[m, m*cx, m*cy, m*cz]`` in SI (kg, kg*m), expressed
in the frame of the joint that carries the link -- the first four entries of
Pinocchio's ``Inertia.toDynamicParameters()``. A regressor ``Y`` passed in has
one column per parameter, link-major in the order of the ``LinkSpec`` list,
and maps parameters to joint torque in N*m: ``tau = Y @ phi``.

This module does not import Pinocchio. Callers build `LinkSpec` from their own
model: `PiperModel.link_specs` does it inside this package, and a script
outside it can fill the same dataclass from its own URDF parsing.
"""

from __future__ import annotations

import enum
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np

# -- thresholds -------------------------------------------------------------
# Module constants rather than arguments: a gate whose thresholds are chosen
# per call is not a gate. tests/test_feasibility.py pins every default.

M_MIN_KG: float = 1e-4
"""Lower bound on every link mass. A mass below it is a FAIL."""

RESIDUAL_SLACK: float = 0.05
"""Stage 2 of `project` keeps the mean squared residual within
``(1 + RESIDUAL_SLACK)`` of the best feasible one (the epsilon of the
two-stage projection)."""

STAGE2_RMS_FLOOR_NM: float = 1e-4
"""Absolute slack added to the stage-2 residual bound, so that an exact fit
(noise-free data) leaves stage 2 a feasible interior. Far below the torque
resolution of 1 mA of motor current."""

NULLSPACE_DRMS_ABS_NM: float = 0.02
"""A violation is a nullspace artifact if the best feasible fit is worse than
the free fit by at most ``max(NULLSPACE_DRMS_ABS_NM, NULLSPACE_DRMS_REL *
rms_free)``; otherwise the parameters are infeasible."""

NULLSPACE_DRMS_REL: float = 0.05
"""Relative part of the nullspace-artifact criterion; see
`NULLSPACE_DRMS_ABS_NM`."""

TOTAL_MASS_RANGE: tuple[float, float] = (0.8, 2.5)
"""Allowed total moving mass, as multiples of the URDF total. Wide on the
upper side because the URDF may describe the bare arm while the real one
carries a gripper. Out of range is a WARN."""

GRIPPER_ALLOWANCE_MASS_KG: float = 1.0
"""Mass the end link may carry beyond its URDF value. Basis: the stock Piper
description puts 0.51 kg in link6 with the gripper fingers lumped in; the
allowance doubles that for camera mounts and cabling."""

GRIPPER_ALLOWANCE_COM_RADIUS_M: float = 0.15
"""Half-width of the cube around the end-link origin that its centre of mass
may occupy. Basis: the stock Piper gripper's fingers sit 0.136 m from the
flange, so the gripper's centre of mass cannot be further out."""

KINEMATIC_BOX_PAD_ABS_M: float = 0.05
"""Minimum padding of a kinematic-fallback box, in metres."""

KINEMATIC_BOX_PAD_REL: float = 0.5
"""Padding of a kinematic-fallback box as a fraction of the link length."""

RANK_RTOL: float = 1e-6
"""Singular values of the regressor below ``RANK_RTOL * s_max`` count as zero."""

CHECK_TOL: float = 1e-9
"""Numerical tolerance of every pointwise check (kg or kg*m)."""

UNITS: str = "SI"
"""The only unit system accepted."""

BOX_SOURCES: tuple[str, ...] = ("mesh", "kinematic-fallback")
"""Where a link's centre-of-mass box came from. ``mesh`` is the URDF's
collision geometry; ``kinematic-fallback`` is the hull of the link's joint
origins, padded, used when no geometry could be loaded."""

_UNSET = "<unset>"
_UNITS_HELP = (
    "Parameters must be in SI (kg, kg*m) and declared with units='SI'. A fit "
    "done in motor-current units has the torque constant kt absorbed into "
    "theta; multiply by kt before passing the parameters here."
)


class Verdict(str, enum.Enum):
    """What an audit concluded. See ``docs/feasibility.md``."""

    FEASIBLE = "FEASIBLE"
    """The parameters as given satisfy every constraint."""
    NULLSPACE_ARTIFACT = "NULLSPACE_ARTIFACT"
    """They violate a constraint, but feasible parameters fit the data as well:
    the violation lives in directions static data cannot see. Not a failure."""
    INFEASIBLE = "INFEASIBLE"
    """The best feasible parameters fit the data clearly worse."""
    POINTWISE_VIOLATION = "POINTWISE_VIOLATION"
    """A constraint is violated and no data was given to say which of the two
    above it is."""


class Severity(str, enum.Enum):
    FAIL = "FAIL"
    """Definitely unphysical: a non-positive mass, a centre of mass outside the
    link's geometry."""
    WARN = "WARN"
    """Outside a conservative estimate (fallback box, gripper allowance, total
    mass range). Never enough on its own to reject a parameter set."""


_MEANING = {
    Verdict.FEASIBLE: "every link passes as given.",
    Verdict.NULLSPACE_ARTIFACT: (
        "violated as given, but feasible parameters fit the data equally well. "
        "The model is fine; static data just does not determine these "
        "parameters uniquely. Not a failure -- project() returns a feasible "
        "representative."
    ),
    Verdict.INFEASIBLE: (
        "the best feasible parameters fit the data clearly worse. Suspect the "
        "URDF kinematics, the units/kt, an unmodelled payload, or the bounds "
        "listed as binding below."
    ),
    Verdict.POINTWISE_VIOLATION: (
        "violated as given. Y, tau not provided: cannot tell nullspace drift "
        "from genuine infeasibility. Re-run audit with the fit's data."
    ),
}


def _check_units(units: str, what: str) -> None:
    if units != UNITS:
        got = "missing" if units == _UNSET else repr(units)
        raise ValueError(f"{what}: units {got}. {_UNITS_HELP}")


# -- inputs -------------------------------------------------------------------
@dataclass(frozen=True)
class GripperAllowance:
    """How much the end link may exceed its URDF: the URDF may not model the
    gripper, and a fit will lump the gripper into the end link."""

    mass_kg: float = GRIPPER_ALLOWANCE_MASS_KG
    com_radius_m: float = GRIPPER_ALLOWANCE_COM_RADIUS_M


@dataclass(frozen=True, eq=False)
class LinkSpec:
    """What the gate needs to know about one link. Pure data.

    ``prior`` is the URDF's ``[m, m*cx, m*cy, m*cz]``; ``com_lo``/``com_hi``
    bound the centre of mass in the joint frame, in metres. ``gripper`` is set
    on the end link only. The prior must satisfy the link's own bounds.
    """

    name: str
    prior: np.ndarray
    com_lo: np.ndarray
    com_hi: np.ndarray
    box_source: str
    units: str = _UNSET
    gripper: GripperAllowance | None = None

    def __post_init__(self) -> None:
        _check_units(self.units, f"LinkSpec {self.name!r}")
        for attr, size in (("prior", 4), ("com_lo", 3), ("com_hi", 3)):
            value = np.asarray(getattr(self, attr), dtype=float).reshape(-1)
            if value.size != size or not np.isfinite(value).all():
                raise ValueError(f"LinkSpec {self.name!r}: {attr} must be {size} finite numbers")
            object.__setattr__(self, attr, value)
        if self.box_source not in BOX_SOURCES:
            raise ValueError(f"box_source must be one of {BOX_SOURCES}, got {self.box_source!r}")
        if np.any(self.com_lo > self.com_hi):
            raise ValueError(f"LinkSpec {self.name!r}: com_lo > com_hi")
        lo, hi = self.box
        m, mc = self.prior[0], self.prior[1:]
        if m < M_MIN_KG or np.any(mc < m * lo - CHECK_TOL) or np.any(mc > m * hi + CHECK_TOL):
            raise ValueError(
                f"LinkSpec {self.name!r}: the URDF prior violates the link's own bounds; "
                f"the bounds are wrong, not the prior"
            )

    @property
    def box(self) -> tuple[np.ndarray, np.ndarray]:
        """The centre-of-mass box actually enforced, gripper allowance included."""
        if self.gripper is None:
            return self.com_lo, self.com_hi
        r = self.gripper.com_radius_m
        return np.minimum(self.com_lo, -r), np.maximum(self.com_hi, r)

    @property
    def box_severity(self) -> Severity:
        exact = self.box_source == "mesh" and self.gripper is None
        return Severity.FAIL if exact else Severity.WARN

    @property
    def box_label(self) -> str:
        return self.box_source + ("+gripper" if self.gripper is not None else "")


@dataclass(frozen=True, eq=False)
class ParamSet:
    """Gravity-tier parameters of every link, ``values[i] = [m, m*cx, m*cy,
    m*cz]`` of ``links[i]``. ``units`` must be ``"SI"``."""

    links: tuple[str, ...]
    values: np.ndarray
    units: str = _UNSET

    def __post_init__(self) -> None:
        _check_units(self.units, "ParamSet")
        links = tuple(self.links)
        values = np.asarray(self.values, dtype=float)
        if values.shape != (len(links), 4):
            raise ValueError(f"values must have shape ({len(links)}, 4), got {values.shape}")
        if len(set(links)) != len(links):
            raise ValueError(f"duplicate link names in {links}")
        object.__setattr__(self, "links", links)
        object.__setattr__(self, "values", values)

    @classmethod
    def from_inertial_overrides(
        cls, overrides: Mapping[str, Sequence[float]], links: Sequence[str]
    ) -> ParamSet:
        """Gravity-tier view of ``ArmParams.inertial_overrides`` (SI by schema)."""
        if set(overrides) != set(links):
            raise ValueError(
                f"inertial_overrides keys {sorted(overrides)} do not match links {list(links)}"
            )
        values = np.array([np.asarray(overrides[k], dtype=float)[:4] for k in links])
        return cls(tuple(links), values, units=UNITS)

    def aligned(self, links: Sequence[LinkSpec]) -> np.ndarray:
        """Values reordered to ``links``, matched by name. Flat, link-major."""
        names = [s.name for s in links]
        if set(names) != set(self.links):
            raise ValueError(
                f"parameter links {list(self.links)} do not match link specs {names}"
            )
        index = {n: i for i, n in enumerate(self.links)}
        return self.values[[index[n] for n in names]].reshape(-1)


def kinematic_box(
    points: np.ndarray, com: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Fallback centre-of-mass box for a link without usable geometry.

    ``points`` are the link's joint origin and the origins of the joints and
    frames it carries, in its own frame; ``com`` is the URDF centre of mass.
    The box is their hull padded by ``max(KINEMATIC_BOX_PAD_ABS_M,
    KINEMATIC_BOX_PAD_REL * link length)`` on every side. It is a conservative
    estimate, so a violation of it is only ever a WARN.
    """
    pts = np.vstack([np.zeros(3), np.asarray(points, dtype=float).reshape(-1, 3)])
    if com is not None:
        pts = np.vstack([pts, np.asarray(com, dtype=float).reshape(1, 3)])
    length = float(np.max(np.linalg.norm(pts, axis=1)))
    pad = max(KINEMATIC_BOX_PAD_ABS_M, KINEMATIC_BOX_PAD_REL * length)
    return pts.min(axis=0) - pad, pts.max(axis=0) + pad


# -- constraints --------------------------------------------------------------
@dataclass(frozen=True)
class _Row:
    link: str
    kind: str          # "mass", "com", "gripper_mass", "total_mass"
    severity: Severity


def _constraints(links: Sequence[LinkSpec]) -> tuple[np.ndarray, np.ndarray, list[_Row]]:
    """All constraints as ``A @ phi <= b``, one labelled row each."""
    n = 4 * len(links)
    rows_a, rows_b, labels = [], [], []

    def add(coef: dict[int, float], bound: float, label: _Row) -> None:
        a = np.zeros(n)
        for k, v in coef.items():
            a[k] = v
        rows_a.append(a)
        rows_b.append(bound)
        labels.append(label)

    for i, spec in enumerate(links):
        m = 4 * i
        add({m: -1.0}, -M_MIN_KG, _Row(spec.name, "mass", Severity.FAIL))
        lo, hi = spec.box
        for ax in range(3):
            add({m: lo[ax], m + 1 + ax: -1.0}, 0.0, _Row(spec.name, "com", spec.box_severity))
            add({m: -hi[ax], m + 1 + ax: 1.0}, 0.0, _Row(spec.name, "com", spec.box_severity))
        if spec.gripper is not None:
            add({m: 1.0}, spec.prior[0] + spec.gripper.mass_kg,
                _Row(spec.name, "gripper_mass", Severity.WARN))
    ref = sum(s.prior[0] for s in links)
    every_mass = {4 * i: 1.0 for i in range(len(links))}
    add(every_mass, TOTAL_MASS_RANGE[1] * ref, _Row("(arm)", "total_mass", Severity.WARN))
    add({k: -1.0 for k in every_mass}, -TOTAL_MASS_RANGE[0] * ref,
        _Row("(arm)", "total_mass", Severity.WARN))
    return np.array(rows_a), np.array(rows_b), labels


# -- report -------------------------------------------------------------------
@dataclass
class ConstraintCheck:
    """One pointwise check. ``violation`` is in ``unit`` and zero if passed."""

    link: str
    kind: str
    severity: Severity
    passed: bool
    violation: float
    unit: str
    detail: str


@dataclass
class Identifiability:
    """How much of each parameter static data can see.

    ``null_fraction[i, k]`` is the squared norm of the nullspace component of
    the unit vector of parameter ``k`` of link ``i``, in SI coordinates: 1
    means the data says nothing about it, 0 means it is fully determined.
    """

    rank: int
    n_params: int
    null_fraction: np.ndarray


@dataclass
class Projection:
    rms_before_nm: float
    rms_after_nm: float
    params: ParamSet
    note: str = ""

    @property
    def delta_rms_nm(self) -> float:
        return self.rms_after_nm - self.rms_before_nm


@dataclass
class FeasibilityReport:
    verdict: Verdict
    links: tuple[str, ...]
    values: np.ndarray                  # (n_links, 4) as audited
    box_source: dict[str, str]
    checks: list[ConstraintCheck]
    total_mass_kg: float
    total_mass_ref_kg: float
    data_provided: bool = False
    rms_given_nm: float | None = None
    rms_free_nm: float | None = None
    rms_feasible_nm: float | None = None      # stage-1 optimum, computed on a violation
    delta_rms_threshold_nm: float | None = None
    identifiability: Identifiability | None = None
    binding: list[str] = field(default_factory=list)
    projection: Projection | None = None

    @property
    def failures(self) -> list[ConstraintCheck]:
        return [c for c in self.checks if not c.passed and c.severity is Severity.FAIL]

    @property
    def warnings(self) -> list[ConstraintCheck]:
        return [c for c in self.checks if not c.passed and c.severity is Severity.WARN]

    @property
    def delta_rms_nm(self) -> float | None:
        if self.rms_feasible_nm is None or self.rms_free_nm is None:
            return None
        return self.rms_feasible_nm - self.rms_free_nm

    def headline(self) -> str:
        return (f"physical feasibility: {self.verdict.value} "
                f"({len(self.failures)} FAIL, {len(self.warnings)} WARN) -- "
                f"{_MEANING[self.verdict]}")

    def summary(self) -> str:
        lines = [f"physical feasibility (gravity tier): {self.verdict.value}",
                 f"  {_MEANING[self.verdict]}", ""]
        ident = self.identifiability
        null_head = f" {'null fraction m/mcx/mcy/mcz':>28}" if ident is not None else ""
        lines.append(f"{'link':>10} {'m kg':>8} {'com cm (x, y, z)':>24} {'box':>26}{null_head}")
        for i, name in enumerate(self.links):
            m, mc = self.values[i, 0], self.values[i, 1:]
            com = (f"({mc[0] / m * 100:6.1f},{mc[1] / m * 100:6.1f},{mc[2] / m * 100:6.1f})"
                   if m > M_MIN_KG else "undefined (m <= 0)")
            row = f"{name:>10} {m:>8.3f} {com:>24} {self.box_source[name]:>26}"
            if ident is not None:
                row += " " + f"{' '.join(f'{v:5.2f}' for v in ident.null_fraction[i]):>28}"
            lines.append(row)
        lo, hi = (r * self.total_mass_ref_kg for r in TOTAL_MASS_RANGE)
        lines.append(f"total mass {self.total_mass_kg:.3f} kg (URDF {self.total_mass_ref_kg:.3f} "
                     f"kg; WARN outside {lo:.2f}-{hi:.2f})")
        if self.data_provided:
            data = f"data: rms as given {self.rms_given_nm:.4f}, free fit {self.rms_free_nm:.4f}"
            if self.rms_feasible_nm is not None:
                data += (f", best feasible {self.rms_feasible_nm:.4f}"
                         f" (dRMS {self.delta_rms_nm:+.4f},"
                         f" artifact if <= {self.delta_rms_threshold_nm:.4f})")
            lines.append(data + " N·m")
            lines.append(f"regressor rank {ident.rank} of {ident.n_params} parameters")
            if self.binding:
                lines.append("binding at the best feasible point: " + ", ".join(self.binding))
        else:
            lines.append("Y, tau not provided: cannot tell nullspace drift from genuine "
                         "infeasibility; pointwise checks only.")
        for c in self.failures + self.warnings:
            lines.append(f"{c.severity.value} {c.link} {c.kind}: {c.detail}")
        if self.projection is not None:
            p = self.projection
            lines.append(f"projection: rms {p.rms_before_nm:.4f} -> {p.rms_after_nm:.4f} N·m "
                         f"({p.delta_rms_nm:+.4f}). {p.note}".rstrip())
        return "\n".join(lines)


# -- core ---------------------------------------------------------------------
def _pointwise(phi: np.ndarray, links: Sequence[LinkSpec]) -> list[ConstraintCheck]:
    A, b, rows = _constraints(links)
    slack = A @ phi - b                     # > 0 means violated
    checks = []
    for i, spec in enumerate(links):
        m, mc = phi[4 * i], phi[4 * i + 1 : 4 * i + 4]
        mine = [k for k, r in enumerate(rows) if r.link == spec.name]
        for kind in ("mass", "com", "gripper_mass"):
            ks = [k for k in mine if rows[k].kind == kind]
            if not ks:
                continue
            worst = float(max(slack[ks].max(), 0.0))
            ok = worst <= CHECK_TOL
            if kind == "mass":
                detail = f"m = {m:.4g} kg (must be >= {M_MIN_KG:g})"
                checks.append(ConstraintCheck(spec.name, kind, rows[ks[0]].severity, ok,
                                              worst, "kg", detail))
            elif kind == "com":
                lo, hi = spec.box
                if m > M_MIN_KG:
                    c = mc / m
                    out = float(np.max(np.maximum(lo - c, c - hi).clip(min=0)))
                    detail = (f"com {np.round(c * 100, 1).tolist()} cm, "
                              f"{out * 100:.1f} cm outside the {spec.box_label} box")
                else:
                    detail = f"com undefined with m = {m:.4g} kg"
                checks.append(ConstraintCheck(spec.name, kind, spec.box_severity, ok,
                                              worst, "kg*m", detail))
            else:
                limit = spec.prior[0] + spec.gripper.mass_kg
                detail = f"m = {m:.3f} kg above URDF + gripper allowance {limit:.3f} kg"
                checks.append(ConstraintCheck(spec.name, kind, Severity.WARN, ok,
                                              worst, "kg", detail))
    total = float(sum(phi[0::4]))
    ref = float(sum(s.prior[0] for s in links))
    worst = float(max(slack[-2:].max(), 0.0))
    checks.append(ConstraintCheck(
        "(arm)", "total_mass", Severity.WARN, worst <= CHECK_TOL, worst, "kg",
        f"total {total:.3f} kg vs URDF {ref:.3f} kg (allowed x{TOTAL_MASS_RANGE[0]}"
        f"-x{TOTAL_MASS_RANGE[1]})"))
    return checks


def _identifiability(Y: np.ndarray) -> Identifiability:
    # SI coordinates, deliberately not column-normalised: normalising would
    # blow a nearly-invisible column back up to full visibility. Gravity
    # columns of m (lever arm x g) and of m*c (g) are of similar size anyway.
    if Y.shape[0] < Y.shape[1]:  # zero rows leave the nullspace alone; keep vt square
        Y = np.vstack([Y, np.zeros((Y.shape[1] - Y.shape[0], Y.shape[1]))])
    _, s, vt = np.linalg.svd(Y, full_matrices=False)   # never the (rows x rows) U
    rank = int(np.sum(s > RANK_RTOL * s[0])) if s[0] > 0 else 0
    frac = np.sum(vt[rank:] ** 2, axis=0)
    return Identifiability(rank, Y.shape[1], frac.reshape(-1, 4))


def _rms(Y: np.ndarray, tau: np.ndarray, phi: np.ndarray) -> float:
    return float(np.sqrt(np.mean((Y @ phi - tau) ** 2)))


def _check_data(Y, tau, n_params: int) -> tuple[np.ndarray, np.ndarray]:
    Y = np.asarray(Y, dtype=float)
    tau = np.asarray(tau, dtype=float).reshape(-1)
    if Y.ndim != 2 or Y.shape != (tau.size, n_params):
        raise ValueError(f"Y must be ({tau.size}, {n_params}) for {tau.size} torques, "
                         f"got {Y.shape}")
    return Y, tau


def _solve(fun, jac, x0, A, b, extra=()):
    try:  # [fit] extra; a pointwise audit does not need it
        from scipy.optimize import minimize
    except ImportError as exc:
        raise ImportError(
            "judging a violation against data needs scipy: pip install 'piper-dynamics[fit]'"
        ) from exc

    cons = [{"type": "ineq", "fun": lambda x: b - A @ x, "jac": lambda x: -A}, *extra]
    res = minimize(fun, x0, jac=jac, constraints=cons, method="SLSQP",
                   options={"ftol": 1e-15, "maxiter": 1000})
    return res


def _stage1(Y, tau, A, b, x0):
    """``min ||Y phi - tau||^2`` over the feasible set; returns (phi, mean square)."""
    n = len(tau)
    H, g = Y.T @ Y / n, Y.T @ tau / n
    c = float(tau @ tau) / n
    res = _solve(lambda x: x @ H @ x - 2 * g @ x + c, lambda x: 2 * (H @ x - g), x0, A, b)
    x = res.x
    if not res.success or np.max(A @ x - b) > CHECK_TOL:
        raise RuntimeError(f"stage 1 (best feasible fit) did not converge: {res.message}")
    return x, float(np.mean((Y @ x - tau) ** 2))


def _stage2(Y, tau, A, b, prior, x1, ms1):
    """``min ||phi - prior||^2`` over the feasible set within the residual bound."""
    n = len(tau)
    H, g = Y.T @ Y / n, Y.T @ tau / n
    c = float(tau @ tau) / n
    bound = ms1 * (1 + RESIDUAL_SLACK) + STAGE2_RMS_FLOOR_NM ** 2
    fit = {"type": "ineq",
           "fun": lambda x: (bound - (x @ H @ x - 2 * g @ x + c)) / bound,
           "jac": lambda x: -2 * (H @ x - g) / bound}
    res = _solve(lambda x: float((x - prior) @ (x - prior)), lambda x: 2 * (x - prior),
                 x1, A, b, extra=(fit,))
    x = res.x
    ok = (res.success and np.max(A @ x - b) <= CHECK_TOL
          and np.mean((Y @ x - tau) ** 2) <= bound * (1 + 1e-6))
    return (x, "") if ok else (x1, f"stage 2 did not converge ({res.message}); "
                                   f"returned the stage-1 point, which is feasible.")


def _analyse(params: ParamSet, links: Sequence[LinkSpec], Y, tau):
    if not isinstance(params, ParamSet):
        raise TypeError(f"params must be a ParamSet, got {type(params).__name__}. {_UNITS_HELP}")
    links = list(links)
    if not links or not all(isinstance(s, LinkSpec) for s in links):
        raise TypeError("links must be a non-empty sequence of LinkSpec")
    if (Y is None) != (tau is None):
        raise ValueError("pass both Y and tau, or neither")
    phi = params.aligned(links)
    checks = _pointwise(phi, links)
    ok = all(c.passed for c in checks)
    names = tuple(s.name for s in links)
    report = FeasibilityReport(
        verdict=Verdict.FEASIBLE if ok else Verdict.POINTWISE_VIOLATION,
        links=names,
        values=phi.reshape(-1, 4),
        box_source={s.name: s.box_label for s in links},
        checks=checks,
        total_mass_kg=float(phi[0::4].sum()),
        total_mass_ref_kg=float(sum(s.prior[0] for s in links)),
    )
    if Y is None:
        return report, phi, None
    Y, tau = _check_data(Y, tau, phi.size)
    free = np.linalg.lstsq(Y, tau, rcond=None)[0]
    report.data_provided = True
    report.identifiability = _identifiability(Y)
    report.rms_given_nm = _rms(Y, tau, phi)
    report.rms_free_nm = _rms(Y, tau, free)
    report.delta_rms_threshold_nm = max(NULLSPACE_DRMS_ABS_NM,
                                        NULLSPACE_DRMS_REL * report.rms_free_nm)
    if ok:
        return report, phi, (Y, tau, None, None)
    A, b, rows = _constraints(links)
    x1, ms1 = _stage1(Y, tau, A, b, free)
    report.rms_feasible_nm = float(np.sqrt(ms1))
    report.binding = sorted({f"{rows[k].link} {rows[k].kind} ({rows[k].severity.value})"
                             for k in np.where(A @ x1 - b > -1e-7)[0]})
    report.verdict = (Verdict.NULLSPACE_ARTIFACT
                      if report.delta_rms_nm <= report.delta_rms_threshold_nm
                      else Verdict.INFEASIBLE)
    return report, phi, (Y, tau, x1, ms1)


def audit(
    params: ParamSet,
    links: Sequence[LinkSpec],
    Y: np.ndarray | None = None,
    tau: np.ndarray | None = None,
) -> FeasibilityReport:
    """Check gravity-tier parameters; never modifies them.

    Without ``Y, tau`` only pointwise checks run and the verdict is
    ``FEASIBLE`` or ``POINTWISE_VIOLATION`` -- never ``INFEASIBLE``, because a
    violation alone cannot be told apart from nullspace drift. With them, a
    violating set is judged ``NULLSPACE_ARTIFACT`` or ``INFEASIBLE`` by how
    much the best feasible fit loses against the free fit, and the report
    carries the regressor rank and per-parameter nullspace fractions that
    explain the call. Needs scipy (``[fit]``) only when data is given and a
    constraint is violated.
    """
    return _analyse(params, links, Y, tau)[0]


def project(
    params: ParamSet,
    links: Sequence[LinkSpec],
    Y: np.ndarray,
    tau: np.ndarray,
) -> tuple[ParamSet, FeasibilityReport]:
    """Move infeasible parameters to a feasible set that fits the data best.

    Two stages, no tuning weight between data and prior:

    1. ``min ||Y phi - tau||^2`` subject to the feasibility constraints; the
       optimum ``r*`` is the best any feasible parameter set can do.
    2. ``min ||phi - phi_urdf||^2`` subject to the same constraints and
       ``||Y phi - tau||^2 <= r* (1 + RESIDUAL_SLACK)`` (plus
       ``STAGE2_RMS_FLOOR_NM``). The prior only picks a representative among
       equally good fits; it never trades against the data.

    Both stages are convex: stage 1 is a quadratic programme with linear
    inequalities (``m*lo <= m*c <= m*hi`` is linear in ``(m, m*c)``), stage 2
    a strictly convex objective over linear constraints and one convex
    quadratic one. Every KKT point of a convex problem is a global optimum, so
    the SLSQP solution -- warm-started from the free least-squares fit -- is
    the global one, and a verdict cannot be an artefact of a local minimum.
    At about 24 variables this takes milliseconds.

    The returned report describes the input (its verdict and checks) and
    carries ``projection``: residual before and after and the new parameters,
    which pass every pointwise check. The increase in residual is itself the
    measure of how unphysical the input was. Never called implicitly.
    """
    if Y is None or tau is None:
        raise ValueError("project() needs Y and tau: the nearest feasible point is "
                         "defined by the data residual, not by parameter distance")
    report, phi, (Y, tau, x1, ms1) = _analyse(params, links, Y, tau)
    if report.verdict is Verdict.FEASIBLE:
        report.projection = Projection(report.rms_given_nm, report.rms_given_nm, params,
                                       "already feasible; unchanged.")
        return params, report
    A, b, _ = _constraints(links)
    prior = np.concatenate([s.prior for s in links])
    x2, note = _stage2(Y, tau, A, b, prior, x1, ms1)
    out = ParamSet(report.links, x2.reshape(-1, 4), units=UNITS)
    if not all(c.passed for c in _pointwise(x2, links)):  # pragma: no cover - solver guard
        raise RuntimeError("projected parameters fail a pointwise check")
    report.projection = Projection(report.rms_given_nm, _rms(Y, tau, x2), out,
                                   note or "projected parameters pass every check.")
    return out, report


# -- full tier (interface only) -----------------------------------------------
def audit_full(phi_full: np.ndarray, links: Sequence[str], **kw) -> FeasibilityReport:
    """Full physical consistency of all ten inertial parameters per link.

    For use once dynamic excitation data identifies the rotational inertia
    (T6 onwards). The condition is that each link's pseudo-inertia matrix

        J = [[0.5 tr(I_c) * 1 - I_c + m c c^T, m c], [m c^T, m]]  >= 0

    is positive semidefinite (Wensing, Kim, Slotine, "Linear Matrix Inequalities
    for Physically Consistent Inertial Parameter Identification", RA-L 2017),
    which implies positive mass, a positive definite inertia and the triangle
    inequalities at once. Traversaro et al. (IROS 2016) parametrise the same
    set as a manifold instead. Projection then becomes a semidefinite
    programme. Do not call this with parameters from static data: the inertia
    entries were never constrained by it, see ``docs/feasibility.md``.
    """
    raise NotImplementedError("full-tier feasibility (pseudo-inertia LMI) is not implemented")


def project_full(phi_full: np.ndarray, links: Sequence[str], Y: np.ndarray, tau: np.ndarray,
                 **kw) -> tuple[np.ndarray, FeasibilityReport]:
    """Full-tier counterpart of `project`: the same two stages with the
    pseudo-inertia LMI as the constraint set (an SDP). Interface only."""
    raise NotImplementedError("full-tier projection (pseudo-inertia SDP) is not implemented")
