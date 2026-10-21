# R2V2 Bottle Delivery

Single-network CONTACT simulation: box pickup → transfer → adjacent-table
placement → finger release → arm withdrawal. Model asset `reach-delivery-3191`
uses 1461 observations and 28 body actions at 50 Hz. Fingers have an independent
dynamic controller; body PD, torque limits and native history remain unchanged.

## Install

Python 3.12. Use the pinned evaluation environment; simulation needs no robot SDK.

```bash
python -m pip install -c constraints-preserved-eval.txt -e '.[sim,visualization,test]'
python -m pytest -q
r2v2-delivery --help
```

Models, robot descriptions/meshes and target-reference data are supplied
separately. Set `R2V2_RESOURCE_ROOT` **before importing the package** to the full
resource root containing `r2v2_description/`, `assets/` and `deploy_mujoco/config/`.
The source includes scene builders and default hand/geometry configurations,
not permission to redistribute externally supplied resources.

## Run and render

```bash
r2v2-delivery --assets /data/reach/assets.local.json --check-only
r2v2-delivery --assets /data/reach/assets.local.json --output /data/runs/contact --record
MUJOCO_GL=egl python -m r2v2_deploy.visualization.contact --source /data/runs/contact --output /data/videos/contact
```

Output paths must be new. Video uses actual physical states at normal speed,
H.264/yuv420p, without terminal-frame hold. Use `--view side` for a side view.
Rendering is not another physics validation run.

The asset manifest pins checkpoint/ONNX/parity, the fixed wrist/finger program,
scene settings and a compact 50 Hz target stream. It does **not** require full
demonstration episodes or historical training reports for inference. Runtime
source/resource hashes and explicit model identities reject unknown pairings.
See [asset schema](configs/assets.example.json), [compatibility](docs/COMPATIBILITY.md).
For the example, set `R2V2_ASSET_ROOT` to the separately supplied native model
bundle; its relative layout is `model/`, `program.json`, `reference.json`,
`reference.npz`, `scene.json`, `scene_manifest.json` and `reach.yaml`.

## Modules

`control`: native observations, PD, fingers; `simulation`: scene and contact
measurements; `tasks`: wrist/finger reference clock and physical conditions;
`data`: exact reset/history interfaces; `evaluation`: full/local CONTACT;
`visualization`: recording renderer; `cli`: public run entry.
No runtime teacher or IK controller is constructed. The companion HALO-mj
package provides Reach PPO, export and bounded checkpoint retention.

## Verified scope

The fixed nominal task completes in 78.32 simulated seconds with one body
network, real table support, release and withdrawal. This does not establish
unseen-scene generalization or hardware readiness. No welded bottle, fixed
robot base, extra support forces or runtime state replay is used.

[中文说明](README_zh.md) · [acceptance](docs/ACCEPTANCE.md) ·
[validation](docs/VALIDATION.md) · [source/resource notices](NOTICES.md)
