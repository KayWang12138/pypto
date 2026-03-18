# 技能评审报告

## 评审摘要

| 项目 | 结果 |
|------|------|
| 技能名称 | pypto-operator-perf-analyzer |
| 评审时间 | 2026-03-17 |
| 总分 | 96.75 / 100 |
| 等级 | A |
| S0 否决 | 否 |
| 规则统计 | 通过 42 / 失败 5 / 警告 1 / 跳过 0 |

## 维度得分

| 维度 | 名称 | 权重 | 满分 | 得分 | 扣分详情 |
|------|------|------|------|------|---------|
| D1 | Frontmatter 元数据 | 25% | 25.0 | 23.75 | R08(S2:-5) |
| D2 | 简洁性与效率 | 15% | 15.0 | 15.0 | 无扣分 |
| D3 | 文件结构与导航 | 10% | 10.0 | 9.0 | R18(S2:-5×3), R19(S2:-5) |
| D4 | 语言与表达 | 10% | 10.0 | 9.5 | R23(S2:-5) |
| D5 | 精确性与可执行性 | 10% | 10.0 | 10.0 | 无扣分 |
| D6 | 工作流完整性 | 10% | 10.0 | 9.5 | R30(S2:-5) |
| D7 | 模式与最佳实践 | 5% | 5.0 | 5.0 | 无扣分 |
| D8 | 反模式检测 | 10% | 10.0 | 10.0 | 无扣分 |
| D9 | 脚本与代码质量 | 5% | 5.0 | 5.0 | 无扣分 |

## 规则覆盖率

| 指标 | 数值 |
|------|------|
| 期望规则数 | 48 |
| 已评估规则数 | 48 |
| 跳过规则数 | 0 |
| 覆盖率 | 100.0% |

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
| R08 | FAIL | D1 | S2 | semantic |
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
| R20 | PASS | D4 | S2 | semantic |
| R21 | PASS | D4 | S3 | semantic |
| R22 | PASS | D4 | S2 | static |
| R23 | FAIL | D4 | S2 | semantic |
| R24 | PASS | D5 | S2 | semantic |
| R25 | PASS | D5 | S2 | semantic |
| R26 | PASS | D5 | S1 | semantic |
| R27 | PASS | D5 | S2 | semantic |
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
| R42 | PASS | D9 | S2 | semantic |
| R43 | PASS | D3 | S3 | static |
| R44 | PASS | D1 | S2 | static |
| R45 | PASS | D2 | S2 | static |
| R46 | PASS | D4 | S3 | semantic |
| R47 | PASS | D7 | S2 | semantic |
| R48 | WARN | D1 | S3 | semantic |

## 质量闸门

| 类型 | 数量 | 说明 |
|------|------|------|
| 内部误绑条目 | 0 | 发现与目标技能无关，已从评分中剔除 |
| 证据不足条目 | 0 | snippet 无法在源文件中匹配，已从评分中剔除 |

## 问题清单

### S0 致命缺陷

(无)

### S1 重大问题

(无)

### S2 中等问题

#### 问题 1：引用文件路径不存在

**命中规则**：R18(S2)

> 规则内容：被引用的文件路径必须真实存在

**位置**：`SKILL.md:397-399`

**当前内容**：
> - [性能调优文档](../../docs/tutorials/debug/performance.md)
> - [Matmul 高性能编程](../../docs/tutorials/debug/matmul_performance_guide.md)
> - [性能优化案例](../../docs/tutorials/debug/performance_case_quantindexerprolog.md)

**问题说明**：
参考资料章节中引用的三个文档路径（performance.md、matmul_performance_guide.md、performance_case_quantindexerprolog.md）在当前项目中不存在，会导致用户无法访问这些参考资料。

**修改建议**：
> 删除不存在的引用链接，或创建对应的文档文件。如果这些文档在其他位置，请更新为正确的相对路径。

---

#### 问题 2：脚本文件缺少用途说明

**命中规则**：R19(S2)

> 规则内容：SKILL.md 应说明每个被引用文件的用途和加载时机

**位置**：`SKILL.md:174`

**当前内容**：
> 使用技能中的性能分析脚本自动生成报告

**问题说明**：
被引用文件 'scripts/analyze_perf.py' 缺少明确的用途说明。虽然提到了使用方法，但未在文件引用表格或章节开头说明该脚本的具体用途（如'用于自动解析 bubble_analysis.log 并生成报告'）和加载时机。

**修改建议**：
> 在 '数据提取方法' 章节开头添加脚本用途说明：
> ```
> scripts/analyze_perf.py - 自动解析 bubble_analysis.log 文件，计算核心利用率、气泡率等性能指标，并生成 Markdown 格式的性能分析报告
> ```

---

#### 问题 3：性能评级步骤缺少原因解释

**命中规则**：R23(S2)

> 规则内容：指令应解释"为什么"，而不仅是"做什么"

**位置**：`SKILL.md:81`

**当前内容**：
> ### 步骤 4：性能评级
>
> 根据以下标准进行性能评级

**问题说明**：
步骤 4 '性能评级' 仅列出评级标准表格，但未解释为什么需要对这些指标进行评级、评级结果将如何帮助用户确定优化优先级。

**修改建议**：
> 在步骤 4 开头添加原因说明：
> ```
> 性能评级帮助快速识别算子性能瓶颈严重程度。根据核心利用率、气泡率等指标与目标值的差距，确定优化优先级和预期收益。
> ```

---

#### 问题 4：缺少错误处理说明

**命中规则**：R30(S2)

> 规则内容：必须包含错误处理或失败恢复说明

**位置**：`SKILL.md:22`

**当前内容**：
> ### 步骤 1：定位性能数据文件
>
> 性能数据文件位于 `output/output_*/` 目录下

**问题说明**：
工作流程中缺少错误处理说明。当 bubble_analysis.log 文件不存在、文件格式错误、或无法解析性能数据时，没有说明应如何处理。

**修改建议**：
> 在工作流程中添加错误处理章节或说明：
> ```
> 若 output 目录为空或 bubble_analysis.log 不存在，请先运行算子编译生成性能数据；若文件格式无法解析，检查 PyPTO 版本是否兼容。
> ```

---

#### 问题 5：description 缺少自然触发短语

**命中规则**：R08(S2)

> 规则内容：`description` 应包含用户自然会说出的触发短语

**位置**：`SKILL.md:3`

**当前内容**：
> description: 分析 PyPTO 算子的性能指标。用于分析 PyPTO 算子的性能指标，从性能数据文件中提取关键指标，计算性能评级，并提供性能瓶颈分析和优化建议。

**问题说明**：
description 中缺少用户自然会说出的触发短语。当前描述使用技术化语言（'分析 PyPTO 算子的性能指标'），缺少如 '分析性能'、'性能分析'、'检查算子性能'、'帮我看看性能问题' 等用户可能直接说的短语。

**修改建议**：
> 在 description 中添加用户自然触发短语，例如：
> ```
> description: 分析性能数据、检查算子性能、性能调优。用于分析 PyPTO 算子的性能指标，从性能数据文件中提取关键指标，计算性能评级，并提供性能瓶颈分析和优化建议。
> ```

---

### S3 轻微建议

#### 问题 6：description 缺少扩展触发短语

**命中规则**：R48(S3)

> 规则内容：description 应该 pushy 一些，避免在技能本应发挥作用的场景下却不使用

**位置**：`SKILL.md:3`

**当前内容**：
> description: 分析 PyPTO 算子的性能指标。用于分析 PyPTO 算子的性能指标，从性能数据文件中提取关键指标，计算性能评级，并提供性能瓶颈分析和优化建议。

**问题说明**：
description 仅列出显式触发关键词（'分析 PyPTO 算子的性能指标'），缺少扩展触发短语如 'whenever'、'including'、'or any related to'，可能导致在相关场景下未主动使用该技能。

**修改建议**：
> 扩展 description 触发范围，例如：
> ```
> description: 分析 PyPTO 算子的性能指标。whenever 用户需要评估算子效率，including 核心利用率、气泡率分析，or any performance-related investigation。从性能数据文件中提取关键指标，计算性能评级，并提供性能瓶颈分析和优化建议。
> ```

---

## 通过项

共 42 条规则通过。

| 维度 | 通过规则 |
|------|---------|
| D1 | R01, R02, R03, R04, R05, R06, R07, R09, R10, R44 |
| D2 | R11, R12, R13, R14, R45 |
| D3 | R15, R16, R17, R43 |
| D4 | R20, R21, R22, R46 |
| D5 | R24, R25, R26, R27 |
| D6 | R28, R29, R31 |
| D7 | R32, R33, R47 |
| D8 | R34, R35, R36, R37, R38 |
| D9 | R39, R40, R41, R42 |
