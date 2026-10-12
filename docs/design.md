# Design notes

Why the package is shaped the way it is. Short, because the decisions are few.

## Offline first

The core library never touches CAN. Hand it arrays and it works — no arm, no
`piper_sdk`, no CAN interface, no root.

This is the decision everything else follows from. Most people who could use
this already have episodes on disk and no arm in front of them, and a library
that requires hardware excludes them. It also means the test suite runs in CI,
which a hardware-coupled design cannot.

Live access lives behind the `[can]` extra, and the whole hardware contract is
two methods (`move_j`, `read_state`) so that a changing SDK touches one file.

## Parameters are data, not code

No torque constant appears in a `.py` file. Parameters load from YAML, carry
provenance, and are gitignored by default.

The alternative — shipping "the" Piper constants — would be worse than useless:
unit-to-unit variation on these arms is larger than the payload signal, so
borrowed constants produce confident wrong answers. The shipped reference is a
format example and an order-of-magnitude check, and says so.

The corollary: **the calibration procedure is the product**, more than any
particular number. A user who runs `piper-calibrate` gets something correct for
their hardware; a user who copies someone's YAML gets a plausible-looking lie.

## Static identification before dynamic

`identify_static` is implemented; `identify_full` raises `NotImplementedError`
with an explanation.

Static (gravity-only) identification needs no acceleration estimate, no
velocity filtering and no excitation design, and it delivers what most users
need: torque constants and gravity terms, which is enough for quasi-static
`tau_ext` and payload estimation. Dynamic identification adds inertial
parameters and adds three ways to get a physically impossible answer that fits
the data beautifully. Shipping a half-correct version of it would be worse than
shipping none.

## Report the conditioning, don't hide it

`wrench_from_tau_ext` returns a wrench *and* a condition number *and* a
trust flag, and returns the untrusted result rather than dropping it.

Silently filtering bad samples hides the fact that a trajectory spends time
near singularities — which is something the user needs to know about their
data, not something the library should quietly paper over.

Same reasoning behind `TauExtResult` keeping `tau_meas`, `tau_model` and
`tau_fric` instead of only the residual: when a number looks wrong, the useful
question is *which term is wrong*, and only the decomposition answers it.

## Smooth friction

`sign(v)` makes `tau_ext` jump by `2*fc` every zero crossing — a spike in
exactly the signal being read. `tanh(v/eps)` costs nothing and removes it.

## Not in scope

**Real-time control.** This estimates; it does not command. `tau_ext` is an
input to a controller, not a controller.

**A general multi-robot abstraction.** The internals are robot-agnostic —
nothing but the defaults is Piper-specific — but the package is named, scoped
and documented for one arm. Doing one arm properly is what makes it adoptable;
generalising before that is how tools end up fitting nobody.
