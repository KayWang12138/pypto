# 技能评审报告

## 评审摘要

| 项目 | 结果 |
|------|------|
| 技能名称 | pypto-skill-reviewer |
| 评审时间 | 2026-03-16 |
| 总分 | 99.90 / 100 |
| 等级 | A |
| S0 否决 | 否 |
| 规则统计 | 通过 46 / 失败 1 / 警告 1 / 跳过 0 |

## 维度得分

| 维度 | 名称 | 权重 | 满分 | 得分 | 扣分详情 |
|------|------|------|------|------|---------|
| D1 | Frontmatter 元数据 | 25% | 25.0 | 24.25 | R48(WARN): 0 |
| D2 | 简洁性与效率 | 15% | 15.0 | 15.00 | 无 |
| D3 | 文件结构与导航 | 10% | 10.0 | 10.00 | 无 |
| D4 | 语言与表达 | 10% | 10.0 | 10.00 | 无 |
| D5 | 精确性与可执行性 | 10% | 10.0 | 10.00 | 无 |
| D6 | 工作流完整性 | 10% | 10.0 | 10.00 | 无 |
| D7 | 模式与最佳实践 | 5% | 5.0 | 4.90 | R46(S3): -0.10 |
| D8 | 反模式检测 | 10% | 10.0 | 10.00 | 无 |
| D9 | 脚本与代码质量 | 5% | 5.0 | 5.00 | 无 |

## 规则覆盖率

| 指标 | 数值 |
|------|------|
| 期望规则数 | 48 |
| 已评估规则数 | 48 |
| 跳过规则数 | 0 |
| 覆盖率 | 100.00% |

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
| R18 | PASS | D3 | S2 | static |
| R19 | PASS | D3 | S2 | semantic |
| R20 | PASS | D4 | S2 | semantic |
| R21 | PASS | D4 | S3 | semantic |
| R22 | PASS | D4 | S2 | static |
| R23 | PASS | D4 | S2 | semantic |
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
| R42 | PASS | D9 | S2 | semantic |
| R43 | PASS | D3 | S3 | static |
| R44 | PASS | D1 | S2 | static |
| R45 | PASS | D2 | S2 | static |
| R46 | FAIL | D7 | S3 | semantic |
| R47 | PASS | D7 | S2 | semantic |
| R48 | WARN | D1 | S3 | semantic |

## 质量闸门

| 类型 | 数量 | 说明 |
|------|------|------|
| 内部误绑条目 | 0 | 发现与目标技能无关，已从评分中剔除 |
| 证据不足条目 | 0 | snippet 无法在源文件中匹配，已从评分中剔除 |

## 问题清单

### S0 致命缺陷

无

### S1 重大问题

无

### S2 中等问题

无

### S3 轻微建议

#### 问题 1：代码块缺少语言标注

**命中规则**：R46 (S3)

> 规则内容：代码块应带有语言标注，纯文本块除外

**位置**：`SKILL.md:33-35`

**当前内容**：
> ```
> python3 <reviewer-dir>/scripts/validate_skill.py <skill-path>
> ```

**问题说明**：
第 33-35 行的代码块包含 Python/Bash 命令内容，但缺少语言标注。这使得 AI 助手无法正确识别代码类型，影响代码高亮和理解。

**修改建议**：
> 将代码块标注为 `bash` 或 `shell`：
> ```bash
> python3 <reviewer-dir>/scripts/validate_skill.py <skill-path>
> ```

---

#### 问题 2：description 可增加扩展触发短语

**命中规则**：R48 (S3)

> 规则内容：description 应该 pushy 一些，避免在技能本应发挥作用的场景下却不使用

**位置**：`SKILL.md:3`

**当前内容**：
> description: "对一个 skill 目录进行质量与最佳实践合规性评审并评分。用于需要审计某个 skill、检查 skill 是否遵循规范，或在发布前评估 skill 的场景。通过 Python 脚本执行静态检查，并通过清单执行语义评审，最终产出带有具体修复建议的评分报告。"

**问题说明**：
description 仅列出显式关键词（"评审"、"评分"、"审计"），缺少扩展触发短语如 "whenever"、"including"、"or any related to"。这可能导致在某些相关场景下 AI 不会主动调用该技能。

**修改建议**：
> 可考虑添加扩展触发短语，例如：
> "对一个 skill 目录进行质量与最佳实践合规性评审并评分。**Whenever** 用户需要审计某个 skill、检查 skill 是否遵循规范，**or whenever** 创建、修改或审查任何 skill 相关内容时，都应使用此技能。..."
>
> 注：此为轻微建议，当前 description 已具备基本触发能力。

---

## 通过项

共 46 条规则通过。

| 维度 | 通过规则 |
|------|---------|
| D1 | R01, R02, R03, R04, R05, R06, R07, R08, R09, R10, R44 |
| D2 | R11, R12, R13, R14, R45 |
| D3 | R15, R16, R17, R18, R19, R43 |
| D4 | R20, R21, R22, R23 |
| D5 | R24, R25, R26, R27 |
| D6 | R28, R29, R30, R31 |
| D7 | R32, R33, R47 |
| D8 | R34, R35, R36, R37, R38 |
| D9 | R39, R40, R41, R42 |
