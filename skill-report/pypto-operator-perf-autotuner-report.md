# 技能评审报告

## 评审摘要

| 项目 | 结果 |
|------|------|
| 技能名称 | pypto-operator-perf-autotuner |
| 评审时间 | 2026-03-17 |
| 总分 | 95.00 / 100 |
| 等级 | A |
| S0 否决 | 否 |
| 规则统计 | 通过 37 / 失败 9 / 警告 1 / 跳过 1 |

## 维度得分

| 维度 | 名称 | 权重 | 满分 | 得分 | 扣分详情 |
|------|------|------|------|------|---------|
| D1 | Frontmatter 元数据 | 25% | 25.0 | 25.00 | 无扣分 |
| D2 | 简洁性与效率 | 15% | 15.0 | 15.00 | 无扣分 |
| D3 | 文件结构与导航 | 10% | 10.0 | 9.00 | R18(-5), R19(-5) |
| D4 | 语言与表达 | 10% | 10.0 | 9.00 | R20(-5), R23(-5) |
| D5 | 精确性与可执行性 | 10% | 10.0 | 7.50 | R24(-5), R25(-5), R26(-10), R27(-5) |
| D6 | 工作流完整性 | 10% | 10.0 | 9.50 | R30(-5) |
| D7 | 模式与最佳实践 | 5% | 5.0 | 5.00 | 无扣分 |
| D8 | 反模式检测 | 10% | 10.0 | 10.00 | 无扣分 |
| D9 | 脚本与代码质量 | 5% | 5.0 | 5.00 | 无扣分 (无scripts目录，自动满分) |

## 规则覆盖率

| 指标 | 数值 |
|------|------|
| 期望规则数 | 48 |
| 已评估规则数 | 48 |
| 跳过规则数 | 1 |
| 覆盖率 | 100.0% |

**跳过的规则**：R42（原因：不适用于此技能，不存在scripts目录）

### 规则状态明细

| 规则 | 状态 | 维度 | 严重度 | 类型 |
|------|------|------|--------|------|
| R01 | PASS | D1 | S0 | static |
| R02 | PASS | D1 | S0 | static |
| R03 | PASS | D1 | S0 | static |
| R04 | PASS | D1 | S1 | static |
| R05 | PASS | D1 | S2 | static |
| R06 | PASS | D1 | S3 | static |
| R07 | PASS | D1 | S1 | semantic |
| R08 | PASS | D1 | S2 | semantic |
| R09 | PASS | D1 | S2 | semantic |
| R10 | PASS | D1 | S2 | static |
| R11 | PASS | D2 | S1 | static |
| R12 | PASS | D2 | S2 | static |
| R13 | PASS | D2 | S1 | static |
| R14 | PASS | D2 | S2 | semantic |
| R15 | PASS | D3 | S2 | static |
| R16 | PASS | D3 | S2 | static |
| R17 | PASS | D3 | S2 | static |
| R18 | FAIL | D3 | S2 | static |
| R19 | FAIL | D3 | S2 | semantic |
| R20 | FAIL | D4 | S2 | semantic |
| R21 | PASS | D4 | S3 | semantic |
| R22 | PASS | D4 | S2 | static |
| R23 | FAIL | D4 | S2 | semantic |
| R24 | FAIL | D5 | S2 | semantic |
| R25 | FAIL | D5 | S2 | semantic |
| R26 | FAIL | D5 | S1 | semantic |
| R27 | FAIL | D5 | S2 | semantic |
| R28 | PASS | D6 | S1 | semantic |
| R29 | PASS | D6 | S2 | semantic |
| R30 | FAIL | D6 | S2 | semantic |
| R31 | PASS | D6 | S2 | semantic |
| R32 | PASS | D7 | S3 | semantic |
| R33 | PASS | D7 | S3 | semantic |
| R34 | PASS | D8 | S0 | static |
| R35 | PASS | D8 | S1 | static |
| R36 | PASS | D8 | S1 | static |
| R37 | PASS | D8 | S1 | static |
| R38 | PASS | D8 | S2 | static |
| R39 | PASS | D9 | S2 | static |
| R40 | PASS | D9 | S2 | static |
| R41 | PASS | D9 | S2 | static |
| R42 | SKIP | D9 | S2 | semantic |
| R43 | PASS | D3 | S3 | static |
| R44 | PASS | D1 | S2 | static |
| R45 | PASS | D2 | S2 | static |
| R46 | PASS | D4 | S3 | semantic |
| R47 | PASS | D7 | S2 | semantic |
| R48 | WARN | D1 | S3 | semantic |

## 质量闸门

| 类型 | 数量 | 说明 |
|------|------|------|
| 内部误绑条目 | 0 | 无 |
| 证据不足条目 | 0 | 无 |

## 问题清单

### S0 致命缺陷

无

### S1 重大问题

#### 问题 1：步骤3技能调用缺少输入输出说明

**命中规则**：R26 (S1)

> 规则内容：所有提及的操作都必须提供具体实现方法

**位置**：`SKILL.md:60`

**当前内容**：
> ### 步骤 3：分析性能数据
> 使用pypto-operator-perf-analyzer分析性能，生成性能报告和性能优化建议

**问题说明**：
步骤3调用 pypto-operator-perf-analyzer 技能时未说明预期的输入和输出。虽然技能调用由 AI 自动处理，但缺少对数据流向的说明会导致执行者不清楚该技能需要哪些前置数据以及产出什么结果。

**修改建议**：
> ### 步骤 3：分析性能数据
>
> 调用 pypto-operator-perf-analyzer 技能分析性能。该技能将：
> - **输入**：读取 `output/output_*/` 目录下的 `merged_swimlane.json` 和 `machine_runtime_operator_trace.json` 文件
> - **输出**：生成包含瓶颈分析和优化建议的性能报告

---

### S2 中等问题

#### 问题 2：引用文件路径不存在

**命中规则**：R18 (S2)

> 规则内容：被引用的文件路径必须真实存在

**位置**：`SKILL.md:102-106`

**当前内容**：
> - [性能调优文档](../../docs/tutorials/debug/performance.md)
> - [Matmul 高性能编程](../../docs/tutorials/debug/matmul_performance_guide.md)
> - [性能优化案例](../../docs/tutorials/debug/performance_case_quantindexerprolog.md)
> - [功能调试](../../docs/tutorials/debug/debug.md)
> - [精度调试](../../docs/tutorials/debug/precision.md)

**问题说明**：
参考资料章节中的5个链接指向的文件路径均不存在于当前仓库中，会导致链接失效。

**修改建议**：
> 检查并修正文件路径，或将这些参考文档移至正确位置。如果文档确实存在，请确保相对路径正确；如果文档尚未创建，请先创建文档或移除链接。

---

#### 问题 3：被引用文件缺少用途和加载时机说明

**命中规则**：R19 (S2)

> 规则内容：SKILL.md 应说明每个被引用文件的用途和加载时机

**位置**：`SKILL.md:100`

**当前内容**：
> ## 参考资料
>
> - [性能调优文档](../../docs/tutorials/debug/performance.md)

**问题说明**：
参考资料章节仅列出链接，未说明每个文档的用途（为什么需要参考）和加载时机（在什么情况下应该阅读）。

**修改建议**：
> ## 参考资料
>
> 以下文档在需要深入理解性能调优原理或遇到特定问题时参考：
>
> - [性能调优文档](../../docs/tutorials/debug/performance.md) - 性能调优的基础概念和方法论，初次接触性能调优时阅读
> - [Matmul 高性能编程](../../docs/tutorials/debug/matmul_performance_guide.md) - Matmul 算子的高性能优化技巧，优化 Matmul 类型算子时参考
> - [性能优化案例](../../docs/tutorials/debug/performance_case_quantindexerprolog.md) - 实际性能优化案例分析，寻找优化思路时参考
> - [功能调试](../../docs/tutorials/debug/debug.md) - 功能调试方法，遇到功能问题时查阅
> - [精度调试](../../docs/tutorials/debug/precision.md) - 精度问题排查方法，精度检查失败时参考

---

#### 问题 4：指令使用描述性语气

**命中规则**：R20 (S2)

> 规则内容：指令必须使用祈使语气，而非被动建议式表达

**位置**：`SKILL.md:54`

**当前内容**：
> ### 步骤 2：检查精度
> 进行精度检查，根据检查结果选择执行下面步骤：

**问题说明**：
"进行精度检查"是描述性语气，应改为更明确的祈使语气，让执行者清楚知道需要执行的动作。

**修改建议**：
> ### 步骤 2：检查精度
>
> **执行**精度检查，验证输出结果是否符合预期。根据检查结果选择后续步骤：
> 1. 若检查通过，继续执行步骤3
> 2. 若首轮检查失败，询问用户是否修复精度问题或继续优化
> 3. 若因上一轮修改导致失败，回退修改并尝试其他优化方案

---

#### 问题 5：关键指令缺少原因说明

**命中规则**：R23 (S2)

> 规则内容：指令应解释"为什么"，而不仅是"做什么"

**位置**：`SKILL.md:16`

**当前内容**：
> #### 1.1 启用性能数据采集
>
> 在算子实现文件中，修改 `@pypto.frontend.jit` 装饰器，添加 `debug_options` 参数：

**问题说明**：
指令要求添加 `debug_options` 参数，但未解释为什么要这样做，执行者可能不理解这个修改的目的。

**修改建议**：
> #### 1.1 启用性能数据采集
>
> **为了采集性能追踪数据**，需要在算子实现文件中修改 `@pypto.frontend.jit` 装饰器，添加 `debug_options` 参数以启用运行时调试模式：

---

#### 问题 6：步骤缺少可验证的成功标准

**命中规则**：R24 (S2)

> 规则内容：每个步骤都应具备可验证的成功标准

**位置**：`SKILL.md:54`

**当前内容**：
> ### 步骤 2：检查精度
> 进行精度检查，根据检查结果选择执行下面步骤：

**问题说明**：
步骤2要求"进行精度检查"，但未说明如何判断精度检查是否通过，缺少可量化的成功标准。

**修改建议**：
> ### 步骤 2：检查精度
>
> 执行精度检查，**验证输出结果与预期值的误差是否在允许范围内（如相对误差 < 0.001）**。根据检查结果选择后续步骤：

---

#### 问题 7：命令包含未解析的占位符

**命中规则**：R25 (S2)

> 规则内容：命令和路径必须具体且可执行

**位置**：`SKILL.md:37`

**当前内容**：
> ```bash
> # 设置环境变量
> export TILE_FWK_DEVICE_ID=0
> export PTO_TILE_LIB_CODE_PATH=/mnt/workspace/pto-isa/
>
> # 编译 whl 包
> python3 build_ci.py -f python3 --disable_auto_execute
>
> # 运行算子（生成泳道图数据）
> python3 custom/operator_name/operator.py --run-mode npu
> ```

**问题说明**：
命令中包含未解析的占位符：`/mnt/workspace/pto-isa/` 是硬编码路径，`operator_name` 是未替换的占位符，执行者需要自行推测正确值。

**修改建议**：
> ```bash
> # 设置环境变量
> export TILE_FWK_DEVICE_ID=0
> export PTO_TILE_LIB_CODE_PATH=<PTO_TILE_LIB代码路径>  # 例如：/mnt/workspace/pto-isa/
>
> # 编译 whl 包（在项目根目录执行）
> python3 build_ci.py -f python3 --disable_auto_execute
>
> # 运行算子（生成泳道图数据）
> # 将 <算子名称> 替换为实际的算子目录名
> python3 custom/<算子名称>/<算子名称>.py --run-mode npu
> ```

---

#### 问题 8：缺少完成标准定义

**命中规则**：R27 (S2)

> 规则内容：完成标准必须明确定义

**位置**：`SKILL.md:66`

**当前内容**：
> ### 步骤 5：迭代调优
> 1. 采集性能数据
> 2. 分析性能瓶颈
> ...
> 7. 重复步骤 1-6 直到达到目标性能

**问题说明**：
虽然提到"直到达到目标性能"，但未明确定义整个性能调优任务的完成标准，执行者不清楚何时可以认为任务完成。

**修改建议**：
> 在文档末尾添加完成标准章节：
>
> ## 完成标准
>
> 性能调优任务完成需满足以下条件：
> 1. **精度检查通过**：输出结果误差在允许范围内
> 2. **性能达标**：性能指标达到或超过预设目标
> 3. **环境还原**：已将 `debug_options` 开关还原（禁用运行时调试模式）
> 4. **报告输出**：已生成最终性能报告

---

#### 问题 9：缺少错误处理说明

**命中规则**：R30 (S2)

> 规则内容：必须包含错误处理或失败恢复说明

**位置**：`SKILL.md:34`

**当前内容**：
> #### 1.2 重新编译并运行
> 如果非第一次运行，没有修改framework或python下的代码，则不需要重新编译，跳过此节。

**问题说明**：
文档缺少常见错误场景的处理说明，如编译失败、性能数据文件未生成、环境变量设置错误等情况的应对方法。

**修改建议**：
> 添加错误处理章节：
>
> ## 错误处理
>
> ### 编译失败
> 若编译失败，检查以下内容：
> 1. 确认环境变量 `TILE_FWK_DEVICE_ID` 和 `PTO_TILE_LIB_CODE_PATH` 已正确设置
> 2. 确认 `build_ci.py` 脚本路径正确
> 3. 查看编译日志定位具体错误
>
> ### 性能数据文件未生成
> 若 `output/output_*/` 目录下未生成数据文件：
> 1. 确认 `debug_options` 参数已正确添加到装饰器
> 2. 确认使用 `--run-mode npu` 参数运行
> 3. 检查算子执行是否有运行时错误

---

### S3 轻微建议

#### 问题 10：description 缺少扩展触发短语

**命中规则**：R48 (S3)

> 规则内容：description 应该 pushy 一些，避免在技能本应发挥作用的场景下却不使用

**位置**：`SKILL.md:3`

**当前内容**：
> description: PyPTO算子性能分析和自动调优技能。用于生成泳道图、分析性能数据、查看性能统计和提供优化建议。当用户需要分析 PyPTO 算子的性能、生成性能报告或进行性能调优时使用此技能。

**问题说明**：
description 仅包含显式触发短语（性能分析、性能调优、生成性能报告），缺少扩展触发范围的短语如"whenever"、"including"、"or any related to"等，可能在相关场景下不会被触发。

**修改建议**：
> description: PyPTO算子性能分析和自动调优技能。用于生成泳道图、分析性能数据、查看性能统计和提供优化建议。**当用户涉及任何与 PyPTO 算子性能相关的任务时**，**包括但不限于**性能瓶颈分析、运行时优化、吞吐量提升，**或任何需要深入理解算子执行效率的场景**，都应使用此技能。

---

## 通过项

共 37 条规则通过。

| 维度 | 通过规则 |
|------|---------|
| D1 | R01, R02, R03, R04, R05, R06, R07, R08, R09, R10, R44 |
| D2 | R11, R12, R13, R14, R45 |
| D3 | R15, R16, R17, R43 |
| D4 | R21, R22, R46 |
| D5 | 无 |
| D6 | R28, R29, R31 |
| D7 | R32, R33, R47 |
| D8 | R34, R35, R36, R37, R38 |
| D9 | R39, R40, R41 |
