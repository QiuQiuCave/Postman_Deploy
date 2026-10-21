# Native runtime and model identity

The runtime accepts an explicit `native_contact_reach_runtime_v1` source and
resource profile. Package versions, observation/action contracts, model hashes,
execution identity, program SHA and numerical parity are checked before use.
Unknown combinations fail; source validation is not disabled.

The supplied checkpoint remains byte-identical. A separate ONNX copy updates
identity metadata only: GraphProto and initializer bytes remain unchanged.
Its migration record pins original/target hashes and 64 native observation
comparisons (maximum action difference 0). Original model and parity files
are not overwritten. New training exports use the current native identity.

The compact reference schema `contact_reach_reference_v1` contains only
`time_s`, `wrist_target` (WXYZ), `phase`, `phase_elapsed_s`, `hand_commands`.
Array and archive hashes identify its lossless projection. It contains no
body qpos, body actions or bottle-state replay. Source identities remain
separate from current implementation identities.

Reset banks are separate training assets: integration state, actually executed
action histories, PD targets, hand Ruckig continuation and source clocks must
remain mutually consistent. Loading checks lineage and continuity evidence;
only episode initialization restores a state. Pickled banks must be trusted
and their hashes verified before loading.
