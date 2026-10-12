"""Three commands, no more.

    piper-calibrate   collect + identify + report  -> params/my_arm.yaml
    piper-annotate    add tau_ext to recorded episodes
    piper-diagnose    check a parameter file, compare against the reference

Each is a thin wrapper over the library. If a command grows options nobody
uses, that is a sign the library API is wrong, not that the CLI needs a
subcommand tree.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from .calibration import (
    collect_static,
    compare_to_reference,
    identify_static,
    static_poses,
    write_report,
)
from .calibration.collect import CalibrationSamples
from .model import PiperModel
from .params import ArmParams, ParamError

REFERENCE_PARAMS = Path(__file__).resolve().parents[2] / "params" / "piper_reference.yaml"


def _add_model_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--urdf", required=True, help="path to the Piper URDF")
    p.add_argument("--ee-frame", default="link6", help="end-effector frame name in the URDF")


# -- piper-calibrate ------------------------------------------------------
def calibrate_main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="piper-calibrate",
        description="Collect static poses from a live arm and identify its parameters.",
        epilog="SAFETY: this drives the arm. Clear the workspace, keep the e-stop in hand.",
    )
    _add_model_args(p)
    p.add_argument("--can-port", default="can0")
    p.add_argument("--n-poses", type=int, default=40)
    p.add_argument("--arm-id", required=True, help="your label for this physical arm")
    p.add_argument("--firmware", default="unknown")
    p.add_argument("--by", default="unknown", help="who ran the calibration")
    p.add_argument("--out", default="params/my_arm.yaml")
    p.add_argument("--report-dir", default="reports/calibration")
    p.add_argument("--samples", help="skip collection, re-identify from a saved .npz")
    p.add_argument("--save-samples", default="reports/calibration/samples.npz")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    model = PiperModel(args.urdf, ee_frame=args.ee_frame)

    if args.samples:
        samples = CalibrationSamples.load(args.samples)
        print(f"loaded {len(samples.q)} samples from {args.samples}")
    else:
        from .io.can import PiperSdkArm

        lo = np.array([model.model.lowerPositionLimit[i] for i in range(model.nq)])
        hi = np.array([model.model.upperPositionLimit[i] for i in range(model.nq)])
        # Stay off the hard limits: poses at the stops are where tracking and
        # friction are worst, and they contribute little to the fit.
        margin = 0.1 * (hi - lo)
        poses = static_poses(lo + margin, hi - margin, n=args.n_poses,
                             rng=np.random.default_rng(args.seed))
        print(f"collecting {len(poses)} static poses on {args.can_port} ...")
        arm = PiperSdkArm(args.can_port, n_joints=model.nq)
        samples = collect_static(arm, poses)
        samples.save(args.save_samples)
        print(f"raw samples saved to {args.save_samples}")

    report = identify_static(
        model, samples, arm_id=args.arm_id, firmware=args.firmware, calibrated_by=args.by
    )
    print()
    print(report.summary())
    write_report(report, args.report_dir)
    report.params.save(args.out)
    print(f"\nparameters written to {args.out}")
    print("Next: validate against known weights -- see docs/validation.md.")
    return 0


# -- piper-annotate -------------------------------------------------------
def annotate_main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="piper-annotate",
        description="Add an observations/tau_ext field to recorded HDF5 episodes.",
    )
    _add_model_args(p)
    p.add_argument("--params", required=True, help="your calibrated parameter file")
    p.add_argument("episodes", nargs="+", help="hdf5 episode files (globs expanded by the shell)")
    p.add_argument("--current-key", default="observations/effort")
    p.add_argument("--n-joints", type=int, default=6,
                   help="joints to slice; for a bimanual recording run once per arm")
    p.add_argument("--quasi-static", action="store_true",
                   help="ignore velocity/acceleration terms (recommended for static-only calib)")
    p.add_argument("--in-place", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args(argv)

    from .estimation import TauExtEstimator
    from .io.hdf5 import annotate_episode

    model = PiperModel(args.urdf, ee_frame=args.ee_frame)
    params = ArmParams.load(args.params)
    for w in params.warnings():
        print(f"warning: {w}", file=sys.stderr)
    est = TauExtEstimator(model, params, quasi_static=args.quasi_static)

    for ep in args.episodes:
        out = annotate_episode(
            ep, est,
            out_path=ep if args.in_place else None,
            current_key=args.current_key,
            n_joints=args.n_joints,
            overwrite=args.overwrite,
        )
        print(f"{ep} -> {out}")
    return 0


# -- piper-diagnose -------------------------------------------------------
def diagnose_main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="piper-diagnose",
        description="Sanity-check a parameter file and compare it to the reference arm.",
    )
    p.add_argument("--params", required=True)
    p.add_argument("--reference", default=str(REFERENCE_PARAMS))
    args = p.parse_args(argv)

    try:
        params = ArmParams.load(args.params)
    except ParamError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    prov = params.provenance
    print(f"arm_id         {prov.arm_id}")
    print(f"model          {prov.model}")
    print(f"firmware       {prov.firmware}")
    print(f"calibrated     {prov.calibrated_at} by {prov.calibrated_by}")
    print(f"urdf           {prov.urdf_source}")
    print(f"samples        {prov.n_samples}")
    print()

    warnings = params.warnings()
    if warnings:
        for w in warnings:
            print(f"warning: {w}")
        print()

    try:
        reference = ArmParams.load(args.reference)
        print(compare_to_reference(params, reference))
        print()
        print("Unit-to-unit variation is expected; see params/README.md for how "
              "much is normal.")
    except ParamError:
        print("(no reference parameters available for comparison)")

    return 1 if warnings else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(annotate_main())
