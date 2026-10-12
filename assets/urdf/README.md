# URDF

**The Piper URDF is not vendored in this repository.**

It is AgileX's, distributed with their description packages under their own
terms. Rather than copy it here and inherit a licensing question, this package
takes a path:

```python
model = PiperModel("/path/to/piper_description/urdf/piper_description.urdf")
```

## Where to get one

- AgileX's official ROS/ROS2 description packages for your arm variant
  (Piper, Piper-X, Piper-L, Piper-H -- they are **not** interchangeable).
- `piper_ros` ships URDFs alongside the MoveIt configuration.
- If you have modified the arm (a different gripper, a wrist camera, a tool),
  use your modified URDF. The whole point of this package is that the model
  matches the hardware; a stock URDF on a modified arm puts the mass of your
  camera into `tau_ext` as a constant offset.

## Which frame is the end-effector

Frame naming has changed across description releases. `PiperModel` will not
guess:

```python
PiperModel(urdf, ee_frame="link6")      # raises and lists the real frames if wrong
```

If you are attaching a tool, add it to the URDF rather than compensating for
it in your own code -- then `payload_regressor` is measuring the object in the
gripper, which is what you wanted.

## A note on the inertial values

Stock URDFs typically carry CAD-exported inertias. They are good enough for
visualisation and planning, and noticeably not good enough for torque-level
work: this is why `piper_control`'s gravity compensation defaults to fitting a
polynomial rather than trusting the URDF. `piper-calibrate` identifies the
parameters that matter from measurement and writes them into
`inertial_overrides`; the URDF supplies the kinematic structure, which is the
part it gets right.
