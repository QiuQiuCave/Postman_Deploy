# R2V2 箱内取瓶与邻桌放置

模型资产 `reach-delivery-3191`：单个1461→28网络连续控制身体关节，
配合独立动态手指，按照双腕末端参考完成取瓶、转运、邻桌放置、松手及撤离。
没有运行时教师、IK、固定基座、物体焊接、外力托举或状态回放。

## 安装与执行

Python 3.12，物理依赖固定MuJoCo 3.7.0，不需要真机SDK。

```bash
python -m pip install -c constraints-preserved-eval.txt -e '.[sim,visualization,test]'
python -m pytest -q
export R2V2_RESOURCE_ROOT=/data/resources
r2v2-delivery --assets /data/reach/assets.local.json --check-only
r2v2-delivery --assets /data/reach/assets.local.json --output /data/runs/contact --record
MUJOCO_GL=egl python -m r2v2_deploy.visualization.contact --source /data/runs/contact --output /data/videos/contact
```

资源根需包含 `r2v2_description/`、`assets/`、`deploy_mujoco/config/`，
必须在导入包之前设置。模型、机器人资源和参考数据单独提供，不上传本仓库。
资产索引示例见 `configs/assets.example.json`；所有输出目录必须全新。
视频为真实物理记录的H.264 MP4，原速播放；可用 `--view side` 渲染侧面。

推理只需要模型、固定场景、末端/手指目标流，不读取整批身体动作示范或旧训练报告。
训练由配套HALO-mj提供：Reach奖励、合法真实接触初态及邻域目标采样，
不训练瓶子任务决策，也不使用教师动作监督。

## 已验证边界

同高双桌、固定名义场景完整CONTACT成功78.32秒，真实触桌、释放及撤离。
单次名义成功不代表不同初态/物体的成功率，不宣称真机已验证。
网络权重、原PD、手指时序、末端目标和安全条件保持一致。
重新训练还需密集reset库及固定局部评估库，不能只用ONNX恢复训练。

[验收定义](docs/ACCEPTANCE.md) · [兼容协议](docs/COMPATIBILITY.md) ·
[实际验证](docs/VALIDATION.md) · [来源与授权](NOTICES.md)
