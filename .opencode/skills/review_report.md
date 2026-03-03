# PR #1259 Skill Review Report

Generated: 2026-03-03

## 审查概述
PR #1259 包含两个新 skill：
- `pypto-performance-analyzer`: 解析泳道图及多种 output 产物，识别性能瓶颈并生成评级与优化建议
- `pypto-performance-autotuner`: 搜索最优 NPU 性能调优配置

## P0 问题（必须修复）

### P0-1: autotuner 缺少可执行的迭代搜索脚本
- **问题**: autotuner SKILL.md 是纯知识文档，缺少 `scripts/autotune.py` 可执行脚本。用户明确要求"autotune skill 需要执行迭代搜索策略"。
- **修复**: 创建 `scripts/autotune.py`，实现完整的迭代搜索循环：输入(search_space.json + baseline) → 逐轮生成候选 → 执行 benchmark → 消费 analysis_summary.json → 记录结果 → 选择下一个搜索点。支持 Grid/Random/Bayesian 策略，包含 --dry-run 模式。

### P0-2: "泳道图/目视观察" 措辞与无多模态约束冲突
- **问题**: 两个 SKILL.md 中多处使用"泳道图观察"、"用 Perfetto 打开"等视觉化描述，但 Agent 无多模态能力，不能读取图形界面。
- **修复**: 将所有"泳道图/目视观察"替换为"解析 merged_swimlane.json (Chrome Trace Format JSON)"，在两个 SKILL.md 中添加硬约束声明："本 skill 假设代理**无多模态能力**：禁止要求打开/查看图形界面；只能读取 JSON/CSV/log 并进行程序化分析。"

### P0-3: 废弃 stitch 参数仍作为主要搜索目标
- **问题**: `stitch_function_num_initial`、`stitch_function_outcast_memory`、`stitch_function_inner_memory` 已在 knobs.md 中标记为废弃（应使用 `stitch_function_max_num` 替代），但 autotuner SKILL.md 的优先级排序(Section 二)、搜索策略(Section 四)、搜索空间示例(Section 五)、实战工作流(Section 六) 全部仍将其作为 Priority 1 主要搜索目标。search_space_schema.json 也使用废弃参数。
- **修复**: 在 SKILL.md 和 search_space_schema.json 中，将 stitch 层的主要搜索参数从3个废弃参数替换为 `stitch_function_max_num`。保留对废弃参数的兼容说明，但不作为默认推荐。

### P0-4: analyzer→autotuner 交接协议缺失
- **问题**: analyzer 输出 Markdown 报告，autotuner 无法程序化消费。需要机器可读的交接格式 `analysis_summary.json`。
- **修复**: 在 analyze.py 中添加 `write_analysis_summary()` 函数，在生成 Markdown 报告后同时输出 `analysis_summary.json`，包含：关键指标(aic_util, aiv_util, timeline_length, bubble_ratio)、瓶颈标签(bottleneck_labels)、建议调优旋钮(suggested_knobs)、评级信息(rating)。autotuner 的 autotune.py 消费此 JSON 来决定搜索起点和策略。

## P1 问题（应该修复）

### P1-1: autotuner Section 3.1 与 analyzer 功能重复
- **问题**: autotuner 的 Section 3.1 "采集方式" 详细描述了 profiling 方法，与 analyzer 的功能高度重复。
- **修复**: 将 Section 3.1 精简为引用 analyzer skill 的 profiling 能力，避免维护两份相同的采集指南。

### P1-2: 缺少与现有 skill 的关系说明
- **问题**: 仓库中已有 `pypto-operator-perf-autotune` 和 `pypto-perf-tuning-loop` 两个性能相关 skill，新增的两个 skill 未说明与它们的关系（替代/补充/引用）。
- **修复**: 在两个 SKILL.md 中各添加"与现有技能的关系"章节，明确说明新旧 skill 的定位差异和适用场景。

### P1-3: 可复现性和可追溯性不足
- **问题**: autotuner 未明确要求记录每轮的环境信息(commit hash, NPU型号)、命令行参数、失败原因等。
- **修复**: 在 autotune.py 中实现：每轮记录完整上下文(knobs, env info, command, error/success)写入 tuning_results.jsonl，便于事后审计。

## P2 问题（建议优化）

### P2-1: analyzer 结论缺少标准化标签体系
- **问题**: 分析结论以自由文本形式输出，autotuner 难以程序化判断瓶颈类型。
- **修复**: 在 analysis_summary.json 中定义标准瓶颈标签：compute, memory, scheduling, communication, stitch, control_overhead。

### P2-2: 缺少最小 demo 数据和端到端示例
- **问题**: 没有完整的端到端 demo（1个 baseline + 2轮搜索），用户难以快速验证 autotune 流程。
- **修复**: autotune.py 的 --dry-run 模式模拟 2-3 轮搜索，使用 sample_swimlane.json 作为基础数据。

## 测试结果

| 测试项 | 结果 |
|--------|------|
| analyze.py py_compile | ✅ 通过 |
| analyze.py --dry-run | ✅ 通过（生成完整 Markdown 报告）|
| search_space_schema.json JSON 校验 | ✅ 通过 |
| autotune.py | ❌ 文件不存在（待创建）|

## 修复跟踪

| 编号 | 状态 | 说明 |
|------|------|------|
| P0-1 | 🔧 修复中 | 创建 autotune.py |
| P0-2 | 🔧 修复中 | 替换泳道图措辞 |
| P0-3 | 🔧 修复中 | 替换废弃参数 |
| P0-4 | 🔧 修复中 | 添加 analysis_summary.json |
| P1-1 | 🔧 修复中 | 精简 autotuner Section 3.1 |
| P1-2 | 🔧 修复中 | 添加关系说明 |
| P1-3 | 🔧 修复中 | autotune.py 中实现 |
| P2-1 | 🔧 修复中 | analysis_summary.json 中实现 |
| P2-2 | 🔧 修复中 | autotune.py --dry-run 中实现 |