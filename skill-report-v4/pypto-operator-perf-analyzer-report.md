# 技能评审报告

## 评审摘要

| 项目 | 结果 |
|------|------|
| 技能名称 | pypto-operator-perf-analyzer |
| 评审时间 | 2026-03-11 17:22:10 |
| 总分 | 95.20 / 100 |
| 等级 | A |
| S0 否决 | 否 |
| 规则统计 | 通过 41 / 失败 7 / 警告 0 |

## 维度得分

| 维度 | 名称 | 权重 | 满分 | 得分 | 扣分详情 |
|------|------|------|------|------|---------|
| D1 | Frontmatter 元数据 | 25% | 25.0 | 25.00 | 无 |
| D2 | 简洁性与效率 | 15% | 15.0 | 15.00 | 无 |
| D3 | 文件结构与导航 | 10% | 10.0 | 8.00 | R18(-5分), R18(-5分), R18(-5分), R19(-5分) |
| D4 | 语言与表达 | 10% | 10.0 | 8.70 | R46(-2分), R46(-2分), R46(-2分), R46(-2分), R23(-5分) |
| D5 | 精确性与可执行性 | 10% | 10.0 | 9.00 | R24(-5分), R27(-5分) |
| D6 | 工作流完整性 | 10% | 10.0 | 9.50 | R30(-5分) |
| D7 | 模式与最佳实践 | 5% | 5.0 | 5.00 | 无 |
| D8 | 反模式检测 | 10% | 10.0 | 10.00 | 无 |
| D9 | 脚本与代码质量 | 5% | 5.0 | 5.00 | 无 |

## 规则覆盖率

| 指标 | 数值 |
|------|------|
| 期望规则数 | 50 |
| 已评估规则数 | 48 |
| 跳过规则数 | 1 |
| 覆盖率 | 96.0% |

### 规则状态明细

| 规则 | 状态 | 维度 | 严重度 | 类型 |
|------|------|------|--------|------|
| R01 | PASS | D1 | S0 | semantic |
| R02 | PASS | D1 | S0 | semantic |
| R03 | PASS | D1 | S0 | semantic |
| R04 | PASS | D1 | S1 | semantic |
| R05 | PASS | D1 | S2 | semantic |
| R06 | PASS | D1 | S3 | semantic |
| R07 | PASS | D1 | S1 | semantic |
| R08 | PASS | D1 | S2 | semantic |
| R09 | PASS | D1 | S2 | semantic |
| R10 | PASS | D1 | S2 | semantic |
| R11 | PASS | D2 | S1 | semantic |
| R12 | PASS | D2 | S2 | semantic |
| R13 | PASS | D2 | S1 | semantic |
| R14 | PASS | D2 | S2 | semantic |
| R15 | PASS | D3 | S2 | semantic |
| R16 | PASS | D3 | S2 | semantic |
| R17 | PASS | D3 | S2 | semantic |
| R18 | FAIL | D3 | S2 | semantic |
| R19 | FAIL | D3 | S2 | semantic |
| R20 | PASS | D4 | S2 | semantic |
| R21 | PASS | D4 | S3 | semantic |
| R22 | PASS | D4 | S2 | semantic |
| R23 | FAIL | D4 | S2 | semantic |
| R24 | FAIL | D5 | S2 | semantic |
| R25 | PASS | D5 | S2 | semantic |
| R26 | PASS | D5 | S1 | semantic |
| R27 | FAIL | D5 | S2 | semantic |
| R28 | PASS | D6 | S1 | semantic |
| R29 | PASS | D6 | S2 | semantic |
| R30 | FAIL | D6 | S2 | semantic |
| R31 | PASS | D6 | S2 | semantic |
| R32 | PASS | D7 | S3 | semantic |
| R33 | PASS | D7 | S3 | semantic |
| R34 | PASS | D8 | S0 | semantic |
| R35 | PASS | D8 | S1 | semantic |
| R36 | PASS | D8 | S1 | semantic |
| R37 | PASS | D8 | S1 | semantic |
| R38 | PASS | D8 | S2 | semantic |
| R39 | PASS | D9 | S2 | semantic |
| R40 | PASS | D9 | S2 | semantic |
| R41 | PASS | D9 | S2 | semantic |
| R42 | PASS | D9 | S2 | semantic |
| R44 | SKIP | D0 | S2 | semantic |
| R43 | PASS | D3 | S3 | semantic |
| R44 | PASS | D1 | S2 | semantic |
| R45 | PASS | D2 | S2 | semantic |
| R46 | FAIL | D4 | S3 | semantic |
| R47 | PASS | D9 | S2 | semantic |
| R50 | PASS | D7 | S2 | semantic |

## 质量闸门

| 类型 | 数量 | 说明 |
|------|------|------|
| 内部误绑条目 | 0 | 发现与目标技能无关，已从评分中剔除 |
| 证据不足条目 | 0 | snippet 无法在源文件中匹配，已从评分中剔除 |

## 问题清单

### S0 致命缺陷
无 S0 致命缺陷

### S1 重大问题
无 S1 重大问题

### S2 中等问题

#### 问题 1：SKILL.md:397 - - [性能调优文档](../../docs/tutorials/debug/performance....

**命中规则**：R18(5分), R18(5分), R18(5分)

> 规则内容：被引用的文件路径必须真实存在

**位置**：`SKILL.md:397`

**当前内容**：
> - [性能调优文档](../../docs/tutorials/debug/performance.md)

**问题说明**：
引用路径 `../../docs/tutorials/debug/performance.md` 不存在
引用路径 `../../docs/tutorials/debug/matmul_performance_guide.md` 不存在
引用路径 `../../docs/tutorials/debug/performance_case_quantindexerprolog.md` 不存在

**修改建议**：
> 

---

#### 问题 2：SKILL.md:177 - python3 .opencode/skills/pypto-operator-perf-analy...

**命中规则**：R19(5分)

> 规则内容：SKILL.md 应说明每个被引用文件的用途和加载时机

**位置**：`SKILL.md:177`

**当前内容**：
> python3 .opencode/skills/pypto-operator-perf-analyzer/scripts/analyze_perf.py <output_dir>

**问题说明**：
SKILL.md 中引用了 scripts/analyze_perf.py，但没有说明其用途和加载时机

**修改建议**：
> 在 '数据提取方法' 章节中补充说明：'该脚本用于自动解析 bubble_analysis.log 文件，计算性能指标并生成分析报告，在完成性能数据采集后执行。'

---

#### 问题 3：SKILL.md:22 - ### 步骤 1：定位性能数据文件...

**命中规则**：R23(5分), R24(5分), R30(5分)

> 规则内容：指令应解释"为什么"，而不仅是"做什么"

**位置**：`SKILL.md:22`

**当前内容**：
> ### 步骤 1：定位性能数据文件

**问题说明**：
多个步骤缺少理由说明，如步骤 1-4 仅描述了做什么，没有解释为什么
步骤 1-4 缺少可验证的成功标准，仅描述了操作内容
未提及任何错误处理或失败恢复说明，如文件不存在、数据格式错误等场景

**修改建议**：
> 在步骤标题后补充理由说明，例如：'### 步骤 1：定位性能数据文件\n**原因**：性能数据文件包含算子执行时的核心时间、等待时间等关键指标，是性能分析的基础数据源。'
在每个步骤后补充成功标准，例如：'**成功标准**：确认 output/output_*/ 目录下存在 bubble_analysis.log、merged_swimlane.json 和 machine_runtime_operator_trace.json 文件。'
在工作流程章节末尾添加 '错误处理' 子章节，说明：'如果 bubble_analysis.log 文件不存在，请先运行算子性能测试；如果数据格式不正确，检查算子编译是否成功。'

---

#### 问题 4：SKILL.md:6 - # PyPTO 性能指标分析技能...

**命中规则**：R27(5分)

> 规则内容：完成标准必须明确定义

**位置**：`SKILL.md:6`

**当前内容**：
> # PyPTO 性能指标分析技能

**问题说明**：
未显式定义整体任务的完成标准，没有说明用户如何确认性能分析已完成

**修改建议**：
> 在文档末尾添加 '完成标准' 章节：'当成功生成 performance_analysis_report.md 文件，且报告包含完整的性能指标、评级、瓶颈分析和优化建议时，视为任务完成。'

---
### S3 轻微建议

#### 问题 1：SKILL.md:59 - ```...

**命中规则**：R46(2分)

> 规则内容：围栏代码块应带有语言标注

**位置**：`SKILL.md:59`

**当前内容**：
> ```

**问题说明**：
代码块缺少语言标注

**修改建议**：
> 

---

#### 问题 2：SKILL.md:65 - ```...

**命中规则**：R46(2分)

> 规则内容：围栏代码块应带有语言标注

**位置**：`SKILL.md:65`

**当前内容**：
> ```

**问题说明**：
代码块缺少语言标注

**修改建议**：
> 

---

#### 问题 3：SKILL.md:71 - ```...

**命中规则**：R46(2分)

> 规则内容：围栏代码块应带有语言标注

**位置**：`SKILL.md:71`

**当前内容**：
> ```

**问题说明**：
代码块缺少语言标注

**修改建议**：
> 

---

#### 问题 4：SKILL.md:77 - ```...

**命中规则**：R46(2分)

> 规则内容：围栏代码块应带有语言标注

**位置**：`SKILL.md:77`

**当前内容**：
> ```

**问题说明**：
代码块缺少语言标注

**修改建议**：
> 

---
## 通过项

共 41 条规则通过。

| 维度 | 通过规则 |
|------|---------|
| D1 | R01, R02, R03, R04, R05, R06, R10, R44, R07, R08, R09 |
| D2 | R11, R12, R13, R45, R14 |
| D3 | R15, R16, R17, R43 |
| D4 | R22, R20, R21 |
| D5 | R25, R26 |
| D6 | R28, R29, R31 |
| D7 | R32, R33, R50 |
| D8 | R34, R35, R36, R37, R38 |
| D9 | R39, R40, R41, R42, R47 |
