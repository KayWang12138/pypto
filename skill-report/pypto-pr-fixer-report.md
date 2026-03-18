# 技能评审报告

## 评审摘要

| 项目 | 结果 |
|------|------|
| 技能名称 | pypto-pr-fixer |
| 评审时间 | 2026-03-17 |
| 总分 | 99.50 / 100 |
| 等级 | A |
| S0 否决 | 否 |
| 规则统计 | 通过 46 / 失败 1 / 警告 1 / 跳过 0 |

## 维度得分

| 维度 | 名称 | 权重 | 满分 | 得分 | 扣分详情 |
|------|------|------|------|------|---------|
| D1 | Frontmatter 元数据 | 25% | 25.0 | 25.00 | 无扣分 |
| D2 | 简洁性与效率 | 15% | 15.0 | 15.00 | 无扣分 |
| D3 | 文件结构与导航 | 10% | 10.0 | 9.50 | R19(S2): -5分 |
| D4 | 语言与表达 | 10% | 10.0 | 10.00 | 无扣分 |
| D5 | 精确性与可执行性 | 10% | 10.0 | 10.00 | 无扣分 |
| D6 | 工作流完整性 | 10% | 10.0 | 10.00 | 无扣分 |
| D7 | 模式与最佳实践 | 5% | 5.0 | 5.00 | 无扣分 |
| D8 | 反模式检测 | 10% | 10.0 | 10.00 | 无扣分 |
| D9 | 脚本与代码质量 | 5% | 5.0 | 5.00 | 无扣分 |

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

无

### S1 重大问题

无

### S2 中等问题

#### 问题 1：被引用文件缺少用途和加载时机说明

**命中规则**：R19 [S2]

> 规则内容：SKILL.md 应说明每个被引用文件的用途和加载时机

**位置**：`SKILL.md:176`

**当前内容**：
> 详见 [references/review-guide.md](references/review-guide.md)。

**问题说明**：
SKILL.md 引用了 references/review-guide.md、references/codecheck-rules.md、references/error-handling.md，但仅 codecheck-rules.md 在第 334 行有简要说明，其他文件缺少用途和加载时机说明

**修改建议**：
> 在引用 review-guide.md 和 error-handling.md 时添加用途和加载时机说明。例如：
> 
> 修改前：
> `详见 [references/review-guide.md](references/review-guide.md)。`
> 
> 修改后：
> `详见 [references/review-guide.md](references/review-guide.md) — 处理人工 review 评论时参考的修复策略指南，在理解评论意图和定位文件时加载。`

---

### S3 轻微建议

#### 问题 2：description 缺少 pushy 触发短语

**命中规则**：R48 [S3]

> 规则内容：description 应该 pushy 一些，避免在技能本应发挥作用的场景下却不使用

**位置**：`SKILL.md:3`

**当前内容**：
> 触发词：修复codecheck、codecheck问题、codecheck报错、codecheck失败、codecheck不通过、CI失败、CI报错、PR评论修复、review意见修复、修复PR、PR review fixer

**问题说明**：
description 包含触发词列表，但缺少 pushy 短语（如'whenever'、'including'、'or any related to'）来扩展触发范围

**修改建议**：
> 在 description 中添加 pushy 短语，如：
> 
> 修改前：
> `"修复 PyPTO PR 的 CodeCheck CI 失败和 review 评论。自动获取 CodeCheck 违规详情、匹配规则、应用修复。触发词：修复codecheck..."`
> 
> 修改后：
> `"修复 PyPTO PR 的 CodeCheck CI 失败和 review 评论。自动获取 CodeCheck 违规详情、匹配规则、应用修复。whenever encountering CI failures on PyPTO PRs, including codecheck violations or any review comments requiring fixes. 触发词：修复codecheck..."`

---

## 通过项

共 46 条规则通过。

| 维度 | 通过规则 |
|------|---------|
| D1 | R01, R02, R03, R04, R05, R06, R07, R08, R09, R10, R44 |
| D2 | R11, R12, R13, R14, R45 |
| D3 | R15, R16, R17, R18, R43 |
| D4 | R20, R21, R22, R23, R46 |
| D5 | R24, R25, R26, R27 |
| D6 | R28, R29, R30, R31 |
| D7 | R32, R33, R47 |
| D8 | R34, R35, R36, R37, R38 |
| D9 | R39, R40, R41, R42 |
