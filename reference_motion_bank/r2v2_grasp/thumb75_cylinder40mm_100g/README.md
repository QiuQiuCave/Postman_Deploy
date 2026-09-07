# R2V2 固定手腕圆柱抓取基线

用户确认：75° 拇指对掌初始化；直径 40 mm、高 120 mm、质量 100 g 圆柱。
记录日期：2026-09-07。

`left/` 和 `right/` 是两次独立试验，各 15 s / 100 Hz / 1501 帧。
每侧包含 `grasp_trajectory.npz`、`grasp_record.json`、`metrics.json`、`trace.json`。
NPZ 包含六路参考、11 个物理关节实测轨迹、力矩、接触和完整手腕/圆柱位姿。
JSON 提供名称映射、坐标约定、配置快照、源文件与数据校验值及结果。

坐标、读取、复现和后续 FSM 接口见
[抓取 FSM 交接说明](../../../docs/r2v2_grasp_fsm_handoff.md)。
这是仿真记录，不是可直接在全身或实机上开环回放的控制程序。
