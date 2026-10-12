# Parameters

Parameters are data. Nothing in this package hard-codes a torque constant or a
friction coefficient, and the file in this directory is a format example, not a
calibration you can borrow.

## Why you cannot use someone else's numbers

Low-cost arms vary unit to unit. Gear preload, motor binning, assembly
tolerance and firmware revision all move the current-to-torque relationship,
and the variation is large enough to swamp the signal you are trying to
measure. A payload of 200 g on a Piper produces a joint torque of the same
order as a 10% error in `kt`.

So: **run `piper-calibrate` on your own arm.** It takes about ten minutes and
needs no extra hardware.

```bash
piper-calibrate --urdf /path/to/piper.urdf --arm-id lab-piper-01 --by you
```

## How much deviation from the reference is normal

Once a measured reference exists here, `piper-diagnose` will compare against
it. The intended reading:

| ratio to reference | interpretation |
|---|---|
| 0.8 – 1.25 | normal unit-to-unit variation |
| 0.5 – 0.8 or 1.25 – 2 | plausible, but check your current unit scaling |
| outside 0.5 – 2 | something is wrong: unit scale, joint ordering, or sign |

A ratio near 1000 or 0.001 almost always means a unit conversion (mA vs A, or
0.001 deg vs rad), not a physically different arm. `io/can.py:UNIT_SCALE` is
where that gets fixed.

## Provenance is mandatory

Every file records `arm_id`, firmware, date, who ran it, and which URDF was
used. A parameter file without provenance is not reproducible and
`piper-diagnose` will say so. This matters more than it sounds: six months
later you will have two arms, three firmware revisions and no memory of which
file belongs to which.

## Files here are gitignored by default

`.gitignore` excludes `params/*.yaml` except the reference. Your calibration
describes your hardware, not the package -- keep it with your own project.
