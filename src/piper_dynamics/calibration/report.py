"""Calibration reports.

A number without a residual plot is not a calibration. This module produces
the artifacts a user needs to judge whether their own fit worked -- and the
artifacts they should paste into an issue when it did not.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .identify import IdentificationReport


def write_report(
    report: IdentificationReport,
    out_dir: str | Path,
    reference: object | None = None,
) -> Path:
    """Write ``summary.txt`` and, if matplotlib is available, residual plots."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "summary.txt").write_text(report.summary() + "\n")
    report.params.save(out_dir / "params.yaml")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed (`pip install piper-dynamics[plot]`); "
              "skipping plots -- summary.txt still written.")
        return out_dir

    nq = report.residual_nm.shape[1]
    fig, axes = plt.subplots(nq, 1, figsize=(7, 1.6 * nq), sharex=True)
    axes = np.atleast_1d(axes)
    for j in range(nq):
        axes[j].plot(report.residual_nm[:, j], lw=0.9)
        axes[j].axhline(0, color="k", lw=0.5)
        axes[j].set_ylabel(f"J{j}")
        axes[j].text(
            0.99, 0.9, f"rms {report.rms_nm[j]:.3f} N·m",
            transform=axes[j].transAxes, ha="right", va="top", fontsize=8,
        )
    axes[-1].set_xlabel("sample")
    fig.suptitle("Identification residual (measured − model)")
    fig.tight_layout()
    fig.savefig(out_dir / "residuals.png", dpi=130)
    plt.close(fig)

    print(f"report written to {out_dir}/")
    return out_dir


def compare_to_reference(params, reference) -> str:
    """Per-joint comparison against reference values.

    Unit-to-unit variation on low-cost arms is real and expected. This is a
    smell test, not a verdict: a joint that is 3x off the reference is worth a
    second look at your collection procedure; one that is 20% off probably is
    not.
    """
    lines = [f"{'joint':>6} {'kt (yours)':>12} {'kt (ref)':>10} {'ratio':>8}"]
    for i, (j, r) in enumerate(zip(params.joints, reference.joints, strict=False)):
        ratio = j.kt / r.kt if r.kt > 1e-12 else np.inf
        flag = "  <-- check" if (ratio > 2 or ratio < 0.5) else ""
        lines.append(f"{i:>6} {j.kt:>12.4f} {r.kt:>10.4f} {ratio:>8.2f}{flag}")
    return "\n".join(lines)
