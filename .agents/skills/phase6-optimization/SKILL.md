---
name: pypto-kernel-phase6-optimization
description: Phase 6 配置级与算法级优化。仅在最终生产设计数值正确后方可开始。
---

# PyPTO 复杂 Kernel — Phase 6：优化

优化仅在最终生产设计数值正确后方可开始。

## Phase 6A：配置级优化

在盲目搜索之前，从现有生产 kernel 中检索 tiling 模式：
```
retrieve_docs(query="<kernel type> set_vec_tile_shapes set_cube_tile_shapes tiling config", chunk_type="source_code")
retrieve_docs(query="<kernel type> tiling strategy loop_unroll stitch", chunk_type="example")
```

搜索空间可能包括：vector TileShape、cube TileShape、运行时选项、stitch 设置、loop unroll 选项、设备调度选项、reuse 设置。

使用约束搜索：
1. 评估 10 个初始候选，
2. 保留最优候选，
3. 局部变异，
4. 当改进停滞时停止。

拒绝任何破坏正确性、超时、超出内存限制或编译失败的候选。

## Phase 6B：算法级优化

仅在配置级调优稳定后进行。

系统性检查：
- 中间 tensor 能否减少？
- 数据搬运能否减少？
- 类型转换次数能否减少？
- reuse 能否增加？
- loop 顺序能否优化？
- memory-bound 阶段能否简化？
- view/reshape/assemble 次数能否减少？

每次只改变一个算法思路。

## 子技能委派：系统性性能调优

如需更系统化的多阶段性能分析与调优工作流，请阅读 `skills/pypto-op-perf-tune/SKILL.md`。它提供了 3 阶段方法：

1. **前端调优**（`skills/tune-frontend/SKILL.md`）：loop 写法模式、TileShape 设置、数据操作优化。
2. **泳道调优**（`skills/tune-swimlane/SKILL.md`）：通过泳道图分析进行 stitch 调优、深度 TileShape 调优、图融合、调度策略优化。
3. **核内调优**（`skills/tune-incore/SKILL.md`）：单 task 指令级优化、核内流水、Operation 实现优化。

此外，`skills/pypto-operator-auto-tuner/SKILL.md` 提供了泳道数据提取和 AIV 依赖链分析的自动化调优脚本。

**复杂 kernel 覆盖：** 下方的正确性守卫仍然适用。每个调优变更在继续之前，必须对所有输出使用 `detailed_tensor_compare` 进行验证。

## 正确性守卫

每次优化变更后：
1. 重新运行 `test_<operator_name>.py` — 所有输出必须仍然通过 `detailed_tensor_compare`。
2. 如果正确性退化，**立即回滚**并在计划中记录失败的尝试。
3. 不要在重新测试之前累积多个优化变更。
