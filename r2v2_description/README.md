# Robot resource interface

Robot XML, URDF, meshes and visual textures are supplied separately under
`R2V2_RESOURCE_ROOT/r2v2_description/`. No manufacturer specification or
redistribution permission is inferred from an asset archive.

The validated robot has 28 body hinges and independent dynamic finger joints.
`r2v2_deploy.simulation.robot` builds the floating simulation with 28 body
motors, 12 torque-limited hand channels and ten URDF-derived mimic constraints.
The original kinematic/inertial parameters and collision geometry remain fixed.

The exact resources and evaluation versions are identified by
`r2v2_deploy/compatibility/runtime_profile.json`. External resources must match
these hashes; replacing a robot or its collision geometry requires explicit
validation, not bypassing identity checks.
