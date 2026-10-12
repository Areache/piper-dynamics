# Contributing

## The most useful contribution

**Calibration results from your arm.** The package ships one person's numbers;
what it needs is a distribution. Run:

```bash
piper-calibrate --urdf piper.urdf --arm-id <yours> --by <you>
```

and open an issue with `reports/calibration/summary.txt` and `residuals.png`.
That tells everyone how much unit-to-unit variation is real, which is the
single biggest open question about whether this approach transfers.

Validation runs (`docs/validation.md`) are the second most useful: an
estimated-vs-true mass plot from an arm that is not mine is worth more than
any amount of additional code.

## Development

```bash
pip install -e ".[dev]"
pytest
ruff check .
```

Tests run without hardware and without a Piper URDF — the model layer is
exercised against `tests/fixtures/toy6.urdf`. If you have a real arm, run the
suite against it once:

```bash
PIPER_URDF=/path/to/piper.urdf pytest
```

## If you are adding an estimator

Two things are expected of anything that produces a physical quantity:

1. **A synthetic round-trip test.** Generate data from the model with a known
   answer, then recover it. `tests/test_integration.py` is the pattern.
2. **It reports its own confidence.** Not just a number: a standard error, an
   observability measure, or a trust flag. A plausible wrong answer with no
   warning attached is worse than no answer, because the user builds on it.

## Scope

In scope: turning Piper telemetry into physical quantities, and the
calibration that makes that possible.

Out of scope: real-time control, policy learning, and generalising to arms
nobody here can test on. The internals are robot-agnostic, but the package is
scoped to one arm on purpose.
