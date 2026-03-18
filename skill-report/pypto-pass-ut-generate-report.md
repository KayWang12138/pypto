# 技能评审报告

## 评审摘要

| 项目 | 结果 |
|------|------|
| 技能名称 | pass-ut-generate |
| 评审时间 | 2026-03-17 |
| 总分 | 96.25 / 100 |
| 等级 | A |
| S0 否决 | 否 |
| 规则统计 | 通过 42 / 失败 4 / 警告 1 / 跳过 1 |

## 维度得分

| 维度 | 名称 | 权重 | 满分 | 得分 | 扣分详情 |
|------|------|------|------|------|---------|
| D1 | Frontmatter 元数据 | 25% | 25.0 | 22.50 | R04(S1,-10): name与目录名不匹配 |
| D2 | 简洁性与效率 | 15% | 15.0 | 15.00 | 无扣分 |
| D3 | 文件结构与导航 | 10% | 10.0 | 9.50 | R19(S2,-5): 参考资料缺少用途说明 |
| D4 | 语言与表达 | 10% | 10.0 | 9.50 | R23(S2,-5): 指令缺少原因解释 |
| D5 | 精确性与可执行性 | 10% | 10.0 | 10.00 | 无扣分 |
| D6 | 工作流完整性 | 10% | 10.0 | 10.00 | 无扣分 |
| D7 | 模式与最佳实践 | 5% | 5.0 | 4.75 | R47(S2,-5): 多选场景无默认推荐 |
| D8 | 反模式检测 | 10% | 10.0 | 10.00 | 无扣分 |
| D9 | 脚本与代码质量 | 5% | 5.0 | 5.00 | 无扣分（不存在scripts/目录，自动满分）|

## 规则覆盖率

| 指标 | 数值 |
|------|------|
| 期望规则数 | 48 |
| 已评估规则数 | 48 |
| 跳过规则数 | 1 |
| 覆盖率 | 100.0% |

**跳过的规则**：R42（原因：不存在 scripts/ 目录）

### 规则状态明细

| 规则 | 状态 | 维度 | 严重度 | 类型 |
|------|------|------|--------|------|
| R01 | PASS | D1 | S0 | static |
| R02 | PASS | D1 | S0 | static |
| R03 | PASS | D1 | S0 | static |
| R04 | FAIL | D1 | S1 | static |
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
| R18 | PASS | D3 | S2 | static |
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
| R30 | PASS | D6 | S2 | semantic |
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
| R47 | FAIL | D7 | S2 | semantic |
| R48 | WARN | D1 | S3 | semantic |

## 质量闸门

| 类型 | 数量 | 说明 |
|------|------|------|
| 内部误绑条目 | 0 | 未发现与目标技能无关的条目 |
| 证据不足条目 | 0 | 所有 snippet 均在源文件中匹配 |

## 问题清单

### S0 致命缺陷

（无）

### S1 重大问题

#### 问题 1：name 字段值与目录名不匹配

**命中规则**：R04 [S1]

> 规则内容：`name` 值应与 skill 目录名一致

**位置**：`SKILL.md:2`

**当前内容**：
> name: pass-ut-generate

**问题说明**：
frontmatter 中的 `name` 字段值为 `pass-ut-generate`，但 skill 目录名为 `pypto-pass-ut-generate`，两者不一致。这可能导致 skill 在加载或引用时出现混淆。

**修改建议**：
> 将 frontmatter 中的 `name` 字段值修改为 `pypto-pass-ut-generate`，使其与目录名保持一致。
> 
> 修改前：`name: pass-ut-generate`
> 修改后：`name: pypto-pass-ut-generate`

---

### S2 中等问题

#### 问题 2：参考资料缺少用途和加载时机说明

**命中规则**：R19 [S2]

> 规则内容：SKILL.md 应说明每个被引用文件的用途和加载时机

**位置**：`SKILL.md:210`

**当前内容**：
> ## 参考资料
> 
> | 类别 | 文件路径 |
> |------|----------|
> | 生成流程一示例 | `pypto/framework/tests/ut/passes/src/test_removeredundantop.cpp` |

**问题说明**：
参考资料表中的文件路径仅列出了路径，未说明每个文件的用途和加载时机。例如 `pypto/framework/tests/ut/passes/src/computational_graph_builder.h` 仅列出了路径，但未说明该文件在什么场景下应该参考。这导致用户不知道何时需要查看这些文件。

**修改建议**：
> 为参考资料表添加'用途'和'参考时机'列，或在表格下方说明每个文件的用途。
> 
> 修改前：
> | 类别 | 文件路径 |
> 
> 修改后：
> | 类别 | 文件路径 | 用途 | 参考时机 |
> |------|----------|------|----------|
> | 生成流程一示例 | `pypto/framework/tests/ut/passes/src/test_removeredundantop.cpp` | 完整的UT用例示例代码 | 流程一步骤3-7需要参考此文件了解代码结构 |

---

#### 问题 3：指令缺少原因解释

**命中规则**：R23 [S2]

> 规则内容：指令应解释"为什么"，而不仅是"做什么"

**位置**：`SKILL.md:150`

**当前内容**：
> ### 步骤 8：利用现有的环境，执行生成的UT，并对错误进行修改(若当前环境正常，需在当前环境中验证)
> 
>     当生成UT用例后，执行Python3 build_ci.py -c -u=xxx.* -j=24来验证用例是否正确

**问题说明**：
步骤8中提供了多个执行命令（python3 build_ci.py -c -u=xxx.* -j=24 等），但未解释为什么需要多个命令、各命令之间的区别是什么。例如'当执行超时时，优先执行...'缺少对超时原因的解释。用户无法理解为什么要选择不同的命令。

**修改建议**：
> 为步骤8的命令添加原因说明。
> 
> 修改前：当生成UT用例后，执行Python3 build_ci.py -c -u=xxx.* -j=24来验证用例是否正确
> 
> 修改后：当生成UT用例后，执行Python3 build_ci.py -c -u=xxx.* -j=24来验证用例是否正确。其中 -c 参数表示在执行前重新编译，确保代码是最新的；当执行超时时，可能是编译阶段耗时过长，因此优先执行不带 -c 的命令来跳过编译

---

#### 问题 4：多选场景无默认推荐

**命中规则**：R47 [S2]

> 规则内容：提供多个选项时，应给出默认推荐

**位置**：`SKILL.md:31`

**当前内容**：
> ## UT生成流程一
> 
> ### 步骤 1：分析业务

**问题说明**：
提供了两个UT生成流程（流程一和流程二），但未说明哪个是默认推荐的流程，用户不知道应该优先使用哪个。这可能导致用户在选择时产生困惑。

**修改建议**：
> 在流程标题或概述中明确标注推荐流程。
> 
> 修改前：## UT生成流程一
> 
> 修改后：## UT生成流程一（推荐）
> 
> 或在概述中添加说明："对于新手用户，推荐使用流程一；对于熟悉ComputationalGraphBuilder的用户，可使用流程二"

---

### S3 轻微建议

#### 问题 5：description 缺少扩展触发短语

**命中规则**：R48 [S3]

> 规则内容：description 应该 pushy 一些，避免在技能本应发挥作用的场景下却不使用

**位置**：`SKILL.md:3`

**当前内容**：
> description: 根据Pass业务描述，生成单元测试用例（UT）。当用户输入业务情况时，能根据业务，生成对应Pass的Ut用例。

**问题说明**：
description 仅列出显式触发短语，缺少扩展触发范围的 pushy 短语。建议添加如'whenever'、'including'、'or any related to'等扩展短语，让AI在相关场景下主动使用该技能。

**修改建议**：
> 扩展 description 以覆盖更多相关场景。
> 
> 修改前：根据Pass业务描述，生成单元测试用例（UT）。当用户输入业务情况时，能根据业务，生成对应Pass的Ut用例。
> 
> 修改后：根据Pass业务描述，生成单元测试用例（UT）。当用户需要测试Pass功能、验证Pass行为、或任何与Pass单元测试相关的任务时，自动生成对应的UT用例。

---

## 通过项

共 42 条规则通过。

| 维度 | 通过规则 |
|------|---------|
| D1 | R01, R02, R03, R05, R06, R07, R08, R09, R10, R44 |
| D2 | R11, R12, R13, R14, R45 |
| D3 | R15, R16, R17, R18, R43 |
| D4 | R20, R21, R22, R46 |
| D5 | R24, R25, R26, R27 |
| D6 | R28, R29, R30, R31 |
| D7 | R32, R33 |
| D8 | R34, R35, R36, R37, R38 |
| D9 | R39, R40, R41 |
