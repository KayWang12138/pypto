---
name: pypto-op-workflow
description: PyPTO 算子开发工作流程。用于开发华为昇腾 AI 处理器自定义算子。在接到算子开发任务时使用，确保开发过程规范、高效、符合官方最佳实践。Triggers: 开发算子、算子开发流程、全流程开发、算子开发工作流、operator workflow。
tag: [PyPTO, 算子开发]
---

# PyPTO 算子开发工作流程

本技能提供 PyPTO 算子开发的完整工作流程指导。

## 工作流程概览

```
需求理解 → 环境准备 → Golden → 设计 → 算子实现 → 精度调试 → 性能分析 → 性能调优
```

## 关键 skill 串联

- 需求理解：调用 `pypto-intent-understanding` skill
- Golden 参考实现：调用 `pypto-golden-generator` skill
- 设计方案：调用 `pypto-op-design` skill
- 算子实现：调用 `pypto-op-implement` skill（承载核心开发流程）
- 精度调试：调用 `pypto-precision-debugger` skill
- 性能分析：调用 `pypto-op-perf-analyzer` skill
- 性能调优：调用 `pypto-op-perf-autotuner` skill

## Checklist

- [ ] 需求规格已明确，`spec.md` 已具备且足以支撑后续开发
- [ ] 环境已满足当前开发要求，或当前环境阻塞点已被明确识别
- [ ] Golden 参考实现已生成，可作为精度基线
- [ ] 设计方案已完成，可指导实现与验证
- [ ] 实现工件已完成（`{op}_impl.py`、`test_{op}.py`、`README.md`）
- [ ] 精度验证已通过；若未通过，误差原因已定位或修复路径已明确
- [ ] 性能分析已完成，且已获得实测性能数据；完成调优后已得到对应的性能结果

## 核心原则

1. **遇问题先定位，不简化代码**
   - 第一步：优先搜索 API 文档、相关 skills 和仓库示例，选择合适的 pypto operation
   - 第二步：综合审视代码，查阅官方示例
   - 第三步：定位问题点后修复
   - 禁止：下意识简化代码、凭直觉实现、遇到错误就推翻重写

2. **充分了解后再下结论**
   - 查阅资料、搜索代码、理解原理
   - 不要轻易下结论

3. **持续搜索更优方案**
   - 方案走通后，继续搜索是否存在更优实现

4. **环境兼容性验证**
   - 确认 API/方法适用于 A3 服务器，CANN 8.5.0

## 开发阶段

### 阶段一：需求检查

优先调用 `pypto-intent-understanding` skill，将用户描述转化为结构化需求，再继续后续开发。

必需信息清单：
- 算子名称
- 数学公式
- 输入/输出规格（shape、dtype）
- 支持的数据类型
- 精度要求
- 服务器类型

### 阶段二：环境准备

由 `pypto-op-implement` skill 的环境准备阶段处理。若遇到环境问题，优先调用 `pypto-environment-setup` skill。

### 阶段三：开发实现

开发顺序：
1. 创建算子目录结构：`custom/{op}/`
2. 调用 `pypto-golden-generator` skill 生成 golden 参考实现
   - 主要输出件：`spec.md`、`{op}_golden.py`
3. 调用 `pypto-op-design` skill 生成设计方案
   - 主要输出件：`design.md`
4. 调用 `pypto-op-implement` skill 生成实现、测试和 README
   - 主要输出件：`{op}_impl.py`、`test_{op}.py`、`README.md`

⚠️ 实现代码与测试代码分开。若 API 约束、tiling 策略或 loop 结构仍不清晰，先调用 `pypto-api-explorer` / `pypto-op-design` skill，不要直接硬写实现。

### 阶段四：测试验证

由 `pypto-op-implement` skill 的测试验证阶段执行首次验证。

如果运行通过但精度不满足预期，转入 `pypto-precision-debugger` skill 继续定位与修复。

⚠️ 有 NPU 卡的情况下，不要使用 `run_mode=sim`。完成基础验证和精度验证后，必须继续进行性能分析。

### 阶段五：高阶参数使能与性能调优 ⭐

当阶段四保证算子基础版本正确后，由 `pypto-op-implement` skill 的高阶参数使能阶段处理 `loop_unroll`、stitch 参数等。

必须先调用 `pypto-op-perf-analyzer` skill 完成性能分析并获取实测性能数据，再调用 `pypto-op-perf-autotuner` skill 做调优与回验，并给出调优前后实测对比。

## 环境兼容性

**当前环境**：A3 服务器，CANN 8.5.0

查阅资料时必须确认 API/方法适用于当前环境。

## 注意事项

1. 当编译或执行时长超过 10 分钟且确认已卡住时，请中断并结束相关进程，再重新检查代码。
2. 优先使用 NPU 模式进行精度验证
