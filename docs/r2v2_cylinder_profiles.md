# 圆柱 / 可乐罐物理规格 profile

仅用于 `Postman_Deploy` 的仿真场景；不修改训练任务、策略、手部轨迹或实机部署。
旧展示默认仍是 **直径 40 mm × 高 120 mm、100 g** 的圆柱加可乐外观。
新增真实商品规格级别的 **330 ml Sleek 外形参考**，但不是已完成标定的实物数字孪生。

## 两个独立配置

| profile | 直径 | 高度 | 仿真质量 | 抓握验证状态 |
| --- | --- | --- | --- | --- |
| `baseline_40mm_100g`（默认） | 40 mm | 120 mm | 100 g，原仿真设定 | 保留原固定手腕 75° 抓握记录，不代表全身精度验收 |
| `sleek_330ml_approx_full` | 58 mm | 145.4 mm | **350 g，暂定满罐近似值，非测量值** | 尚未重新标定 / 验证 |

文件在 `deploy_mujoco/config/cylinder_profiles/`，分别为
`baseline_40mm_100g.yaml`、`sleek_330ml_approx_full.yaml`。
数值以米、千克为单位，明确记录 `geometry_source`、`mass_provenance`、
`mass_notes` 和 `grasp_calibration_status`；报告保留完整解析后的 profile。
尺寸依据 [Royal Can 的 330ml Sleek 产品规格](https://www.royalcanco.com/products-cans)
（核查日期 2026-09-10）。这不是对用户某个具体地区/批次 Coca-Cola 罐的实测，
也不要与同为 330 ml、直径 66 mm × 高 115.2 mm 的 Standard 罐混为一谈。

真实物体的空罐/满罐质量尚未提供。`0.350 kg` 仅用于后续仿真测试的显式假设，
不是厂家公布质量、不是称重结果，不能据此宣称实物参数精确。
后续应对实际罐体称重，再创建或更新独立 profile 并重新验证。

## 一个尺寸来源贯穿场景

`common/r2v2_cylinder_test.py` 提供：

- `load_cylinder_profile(source=None)`：按名称、仓库相对/绝对 YAML 路径或已解析字典加载；
  缺省选择原基线，配置缺字段、未知字段、非正/非有限尺寸质量会报错。
- `CylinderParameters.from_profile(source=None, **experiment_parameters)`：生成参数；
  允许选择左右手等固定手腕测试设置，不允许在同一调用中偷偷覆盖 profile 的尺寸/质量。
  原有 `CylinderParameters()` 及显式构造接口保持可用。
- `.half_height` 和 `.upright_center_height(surface_height, clearance=0)`：统一半高与水平面上竖直放置的中心高度。

场景 `build_tabletop_xml` 和 demo 配置都接受 `cylinder_profile`。
碰撞体半径/半高/质量从该 profile 生成，MuJoCo 根据同一个圆柱计算惯量；
可乐外观继续从实际碰撞尺寸缩放，视觉 geom 保持零质量、无碰撞，不额外增加重量。
物理近似仍是均匀实心圆柱，惯量为
`Ixx = Iyy = m(3r²+h²)/12`、`Izz = mr²/2`，并未模拟薄铝壳、凹底或液体晃动。
`330 ml` 是商品名义容量标签，不能当成该完整外包圆柱碰撞体的体积。

demo 初始化圆柱中心为 `桌面高度 + h/2 + 初始间隙`，落放中心为 `桌面高度 + h/2`，
搬运参考中心为 `桌面高度 + h/2 + lift_m`，不再写死 `0.06 m`。
在底层场景中省略 `cylinder_position_xyz` 时同样自动计算桌面上方的出生高度；
显式世界坐标仍优先，并由调用者负责相应姿态/碰撞余量。
竖直底部检测及倾斜圆柱的底部高度继续读取实际 `geom_size`。

## 如何选择（当前先静态检查）

原 `deploy_mujoco/config/r2v2_tabletop_demo.yaml` 明确选择 `baseline_40mm_100g`，
因此旧命令、相机、运动距离和默认控制行为不变。
后续实验请复制该 demo 配置到独立文件，只将如下项改为：

```yaml
object_appearance: cola_can
cylinder_profile: sleek_330ml_approx_full
```

不要覆盖原基线配置或 `reference_motion_bank/r2v2_grasp/thumb75_cylinder40mm_100g/`。
下面代码只构建 CPU 场景，不载入策略、不渲染、不推进物理或执行抓握：

```python
from common.path_config import PROJECT_ROOT
from common.r2v2_tabletop_demo import build_demo_scene_config, load_demo_config
from common.r2v2_tabletop_scene import build_tabletop_model

cfg = load_demo_config()
cfg["cylinder_profile"] = "sleek_330ml_approx_full"
scene = build_demo_scene_config(cfg, table_height=1.11)
model, hand_cfg = build_tabletop_model(cfg["reach"], scene)
```

固定手腕探针可用 `CylinderParameters.from_profile("sleek_330ml_approx_full")`
创建参数；这一步只是选择物体参数，**不能证明该手型抓得住**。
固定手腕 CLI 原有 `--radius/--height/--mass` 入口和默认数值不变；若显式使用这些
参数，调用者需自行记录参数来源，它不会自动附上命名 profile 元数据。

## 必须重新标定的部分

原 `[0.145,-0.035,0] m` 腕柱关系、75° 拇指初始化、各手指闭合量/轨迹及力矩限制，
只针对 40 mm / 100 g 圆柱验证过。新罐更粗且测试质量更大，不能将旧视频、
固定手腕成功结论或 `CLOSED` 状态当成新罐抓握验证。选择未验证 profile 启动
demo 时会在站立预热之前显式警告；该警告不会自动调小物体、改变手型或放宽成功判据。

后续应先检查初始手—罐穿插、闭合接触和关节/力矩余量，重新确定相对位姿与轨迹，
再做真实离台、无外部支持保持、滑移与松手测试，最后接入全身任务。
现有台面近边 `x=.36 m`、圆柱中心 `x=.40 m` 不变时，竖直支撑余量从 20 mm
降至 11 mm；高度增加也会改变腕部/台面与接近路径余量，需要另行验证。
严格 Reach 空载诊断中的 `placement.cylinder_height_m: 0.12` 仍属旧诊断配置，
不是桌面新 profile 的尺寸来源；本次没有把新物体混入该诊断或训练。

## 本次验收范围

仅执行 CPU 配置、MJCF 构建/前向运动学、旧接触测量短测与合成 FSM 单测，覆盖：默认基线参数不变，
新尺寸/质量/解析惯量一致，外观位于碰撞包络内且不改变物理数组，出生/抬升/落放
高度跟随 profile，以及参数来源/未标定状态可追溯。
未执行新罐抓握、物理搬运 rollout、渲染、GPU 推理或训练；不宣称新罐抓取成功。
