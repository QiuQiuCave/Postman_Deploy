# CONTACT 分支整合记录（2026-10-23）

## 基线和整合范围

- 原分支 `r2v2-fsm`：`6f03d69`，保存了 9 个 Loco/Reach 可乐闭环文件。
- 回退标签：`baseline/loco-cola-straight-20261023`。
- 独立整合分支：`integration/contact-reach-delivery-20261023`。
- 新代码来自 `origin/feature/contact-reach-delivery` 的 `fa561f3`。

保留原 FSM、`common/`、`deploy_mujoco/`、机器人资产和历史工具，不执行
新分支对旧基础文件的删除。保留原 `pyproject.toml`、`uv.lock` 和
`requirements-r2v2-reach-sim.txt`，不升级 `.venv-r2v2`。
README 合并两套入口；CONTACT 的 44 个模块及哈希协议原样保留。
原分支的 6 处 Git 冲突均采用这一兼容原则处理。

## 两条控制路线不互相替换

| 路线 | 入口 | 控制与环境 |
| --- | --- | --- |
| 已验证的 Loco/Reach FSM | `deploy_mujoco/r2v2_loco_cola.py` | Loco 560→14，Reach 1460→28；Python 3.11、MuJoCo 3.3.7、NumPy 1.26.4、ORT 1.29.0 |
| 新单网络 CONTACT | `tools/r2v2_contact_delivery.py` → `src/r2v2_deploy/` | 1461→28，额外输入是左手闭合轨迹进度；Python 3.12、MuJoCo 3.7.0、NumPy 2.4.4、ORT 1.23.2 |

两者均使用独立手指控制。CONTACT 是固定场景的箱内取瓶、邻桌放置流程，
按预先记录的双腕/手指**目标流**推进，物理接触指标单独验收；不是运行时
身体动作/物体状态回放，也不是我们已经验证的 Loco/Reach 切换 FSM。
因此不能仅替换 ONNX，也不能把其上游 78.32 秒成功记录说成本机复现结果。

## 安装和运行 CONTACT（独立环境）

以下命令从仓库根目录运行。数据盘环境和输出不入 Git；不安装 Torch/SDK，
不改旧虚拟环境。不要运行根目录 `pip install -e .` 或 `uv sync` 来安装 CONTACT。

```bash
uv venv --no-project --python 3.12 /root/autodl-tmp/Postman_Deploy/envs/contact-delivery-20261023
uv pip install \
  --python /root/autodl-tmp/Postman_Deploy/envs/contact-delivery-20261023/bin/python \
  --cache-dir /root/autodl-tmp/Postman_Deploy/uv-cache \
  -r requirements-contact-delivery.txt

/root/autodl-tmp/Postman_Deploy/envs/contact-delivery-20261023/bin/python \
  tools/r2v2_contact_delivery.py --help
/root/autodl-tmp/Postman_Deploy/envs/contact-delivery-20261023/bin/python \
  -m pytest -c pytest-contact.ini
```

专用模型包就绪后，设定实际资产目录：

```bash
export R2V2_RESOURCE_ROOT=/root/code/amo/Postman_Deploy
export R2V2_ASSET_ROOT=/data/reach-delivery-3191
/root/autodl-tmp/Postman_Deploy/envs/contact-delivery-20261023/bin/python \
  tools/r2v2_contact_delivery.py --assets configs/assets.example.json --check-only
/root/autodl-tmp/Postman_Deploy/envs/contact-delivery-20261023/bin/python \
  tools/r2v2_contact_delivery.py --assets configs/assets.example.json \
  --output /data/runs/contact-new --record
PYTHONPATH=/root/code/amo/Postman_Deploy/src MUJOCO_GL=egl \
  /root/autodl-tmp/Postman_Deploy/envs/contact-delivery-20261023/bin/python \
  -m r2v2_deploy.visualization.contact --source /data/runs/contact-new \
  --output /data/videos/contact-new
```

所有输出目录必须是新目录。当前采用源码入口，不宣称混合仓库的 wheel
已验证；上游分支的独立 wheel/安装说明不直接适用于本整合分支。

## 需要同学补充的资产

本机已核对源码 44/44、物理资源 96/96 的 SHA256，全数匹配。
在本机关键权重、数据盘、训练日志及附件目录中未找到新的专用交付包。
`configs/assets.example.json` 要求的 12 项为：

- `model/policy.pt`、`policy.onnx`、`policy.resume.pt`；
- `model/policy.parity.json`、`policy.migration.json`、`policy.candidate.json`；
- `program.json`、`reference.json`、`reference.npz`；
- `scene.json`、`scene_manifest.json`、`reach.yaml`。

这里 checkpoint 的 SHA256 应为
`de6ea90c4e37f037f71d10485f3cf53b0692461ef8d7944c365b99c4a6d3dccf`；
迁移后的 ONNX 应为
`093c3bcb0149ee5f8d05af8c362a125c6b4f7c1ca8ec9bfe4281ff031c2e3fb4`。
即使推理不使用优化器状态，当前上游示例仍把 `policy.resume.pt` 列为必检
文件；本次未绕过资产校验或重写模型元数据。

## 验证

保存基线前 218 项相关测试通过。本次整合完成的本地检查：

- CONTACT 独立环境：12 项测试通过（上游 9 项 + 整合 3 项），包括实际
  `validate_runtime()`、整机模型编译、28 个身体与 12 个手指执行器互斥，
  以及从另一工作目录启动 CLI；渲染入口 `--help` 正常。
- 整合后再次运行 Loco/Reach、手部、front-drop 及新增隔离检查：221 项通过。
- 旧环境全量：1289 项通过、9 项跳过、1 项失败（174.38 秒）。失败为
  `test_payload_archive_preserves_all_original_prefix_arrays_and_has_no_finger_commands`：
  本机缺少历史轨迹
  `/root/autodl-tmp/Postman_Deploy/selected_path_posttrain_preflight_20260911/revised_lift2_rear3_v1/manifest.json`。
  该测试及其三个几何工具与基线提交逐字相同，本次没有删除这个外部目录，
  也没有跳过该失败或改写验收条件。全量测试因此**不是全绿**。
- 旧闭环完整物理回归：`success=true`，60.78 秒完成 20 次状态切换；
  抓握、释放、最终放置标记均为 true。3040 条 `trace.json` 记录和
  `transitions.json` 与之前 `straight_v8_video` 的 JSON 内容逐项完全一致。
  本轮使用 `--no-video`；不是复用旧报告，也没有新渲染视频。
- 最终可乐平面误差 6.87 mm、倾斜 0.00170°；原碰撞/固定场景限制仍然成立。
- CONTACT 专用模型包缺失，因此未执行新单网络的完整物理实验。

旧闭环本次证据：
`/root/autodl-tmp/Postman_Deploy/integration_contact_20261023/legacy_straight_regression/`。
CONTACT 环境约 478 MB，缓存约 469 MB，均位于数据盘。

旧环境默认测试只运行 `tests/`；CONTACT 测试位于 `tests_contact_delivery/`，
通过独立 pytest 配置执行，避免两套不兼容依赖混装。

```bash
MUJOCO_GL=egl .venv-r2v2/bin/python -m pytest
MUJOCO_GL=egl .venv-r2v2/bin/python deploy_mujoco/r2v2_loco_cola.py \
  --config deploy_mujoco/config/r2v2_loco_cola_straight.json \
  --stop-after place --no-video \
  --output /root/autodl-tmp/Postman_Deploy/integration_contact_20261023/legacy_straight_regression
```

单元测试和模型编译不等于 CONTACT 闭环成功；完整新实验需专用资产到齐后运行。
旧路线的成功同样仅为固定场景回归，不表示无碰撞或随机鲁棒性验收。
