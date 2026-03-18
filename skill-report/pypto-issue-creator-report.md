# 技能评审报告

## 评审摘要

| 项目 | 结果 |
|------|------|
| 技能名称 | pypto-issue-creator |
| 评审时间 | 2026-03-17 |
| 总分 | 100.00 / 100 |
| 等级 | A |
| S0 否决 | 否 |
| 规则统计 | 通过 46 / 失败 0 / 警告 1 / 跳过 1 |

## 维度得分

| 维度 | 名称 | 权重 | 满分 | 得分 | 扣分详情 |
|------|------|------|------|------|---------|
| D1 | Frontmatter 元数据 | 25% | 25.0 | 25.0 | 无扣分 |
| D2 | 简洁性与效率 | 15% | 15.0 | 15.0 | 无扣分 |
| D3 | 文件结构与导航 | 10% | 10.0 | 10.0 | 无扣分 |
| D4 | 语言与表达 | 10% | 10.0 | 10.0 | 无扣分 |
| D5 | 精确性与可执行性 | 10% | 10.0 | 10.0 | 无扣分 |
| D6 | 工作流完整性 | 10% | 10.0 | 10.0 | 无扣分 |
| D7 | 模式与最佳实践 | 5% | 5.0 | 5.0 | 无扣分 |
| D8 | 反模式检测 | 10% | 10.0 | 10.0 | 无扣分 |
| D9 | 脚本与代码质量 | 5% | 5.0 | 5.0 | 无扣分（不存在 scripts/ 目录，自动满分）|

## 规则覆盖率

| 指标 | 数值 |
|------|------|
| 期望规则数 | 48 |
| 已评估规则数 | 48 |
| 跳过规则数 | 1 |
| 覆盖率 | 100.0% |

**跳过的规则**：R42（原因：不存在 scripts/ 目录，此规则不适用）

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

无

### S2 中等问题

无

### S3 轻微建议

#### 问题 1：description 缺少扩展触发短语

**命中规则**：R48 [S3]

> 规则内容：description 应该 pushy 一些，避免在技能本应发挥作用的场景下却不使用

**位置**：`SKILL.md:3`

**当前内容**：
> 触发词：创建issue、提交issue、反馈问题、报告bug、功能请求、文档问题。

**问题说明**：
description 仅列出显式触发关键词（创建issue、提交issue、反馈问题、报告bug、功能请求、文档问题），缺少扩展触发短语如 "whenever"、"including"、"or any related to" 等，可能在某些应使用该技能的场景下不会被触发。

**修改建议**：
> 建议扩展 description 中的触发范围，添加扩展短语如"或任何与 PyPTO/CANN 相关的问题反馈"，确保在相关场景下主动使用该技能。
>
> 修改前：`触发词：创建issue、提交issue、反馈问题、报告bug、功能请求、文档问题。`
>
> 修改后：`触发词：创建issue、提交issue、反馈问题、报告bug、功能请求、文档问题，或任何与 PyPTO/CANN 开发相关的问题报告场景。`

---

## 通过项

共 46 条规则通过。

| 维度 | 通过规则 |
|------|---------|
| D1 | R01, R02, R03, R04, R05, R06, R07, R08, R09, R10, R44 |
| D2 | R11, R12, R13, R14, R45 |
| D3 | R15, R16, R17, R18, R19, R43 |
| D4 | R20, R21, R22, R23, R46 |
| D5 | R24, R25, R26, R27 |
| D6 | R28, R29, R30, R31 |
| D7 | R32, R33, R47 |
| D8 | R34, R35, R36, R37, R38 |
| D9 | R39, R40, R41 |
