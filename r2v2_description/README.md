# R2V2 model provenance

`source/r2v2_with_hand/` is an unmodified extraction of the user-supplied
`r2v2_with_hand.zip` received on 2026-09-07. The archive supplied no README,
hardware SDK, manufacturer/model identification, or license declaration.
Do not infer hardware specifications or redistribution permissions from it.

SHA-256:

- ZIP: `994d02337b9ec5f500a4fcc92fb09e53d7f1987b938245c1e00bb3d2e3b82c7a`
- XML: `fbc20b4e6d7c8a78c8fe88721be219a7b831817c3f372fa97cec10250738b050`
- URDF: `835d73f0dae071321e8ab469390a3a63fd51a2adf02151f05edd0db947ad5eef`

The source has 28 body hinges, 22 finger hinges, a floating base and 28 motors.
Its URDF describes ten finger mimic relationships, absent from its XML.
Head joints are fixed. The meshes and all original kinematic/inertial
parameters are retained; this is **not** the previous AMO_R2 training model.

`model.py` builds a runtime variant with 12 additional torque-limited hand
motors and ten URDF-derived equality constraints. It assigns diagnostic geom
names and rendering groups (left visual 1, right visual 2, collision 3), sets
the integration/solver settings, and explicitly excludes ten directly
adjacent palm/finger-base body pairs. This matches the moving-parent filter
in the floating model when the palms instead become fixed to the world.
Non-adjacent self-collisions are kept enabled. Original collision meshes,
joint damping and armature are unchanged.

Two variants:

- `build_model(fixture=False)`: full floating robot, `nq=57, nv=56, nu=40,
  neq=10`. Body state/action remains an explicitly named 28-channel group.
- `build_model(fixture=True)`: copies only the two wrist/hand subtrees,
  removes the two wrist joints, and mounts them on static visual fixtures.
  `nq=22, nv=22, nu=12, neq=10`. Gravity and contact physics stay enabled.

Hand trajectories and fixed-wrist physics have been validated. A narrow
[free-cylinder grasp probe](../docs/r2v2_cylinder_grasp.md) also passes at the
documented placement. Full-body standing, Reach transfer, general grasping,
and hardware use are not validated.

See [the run guide](../docs/r2v2_hand_validation.md).
