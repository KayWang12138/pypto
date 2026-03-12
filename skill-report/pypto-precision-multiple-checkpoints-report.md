# 技能评审报告

## 评审摘要

| 项目 | 结果 |
|------|------|
| 技能名称 | pypto-precision-multiple-checkpoints |
| 评审时间 | 2026-03-11 |
| 总分 | 59.90 / 100 |
| 等级 | D |
| S0 否决 | 是 |
| 规则统计 | 通过 0 / 失败 1 / 警告 0 / 跳过 46 |

## 维度得分

| 维度 | 名称 | 权重 | 满分 | 得分 | 扣分详情 |
|------|------|------|------|------|---------|
| D1 | Frontmatter 元数据 | 25% | 25.0 | 20.00 | R01(S0): -20 |
| D2 | 简洁性与效率 | 15% | 15.0 | 15.00 | 无扣分 |
| D3 | 文件结构与导航 | 10% | 10.0 | 10.00 | 无扣分 |
| D4 | 语言与表达 | 10% | 10.0 | 10.00 | 无扣分 |
| D5 | 精确性与可执行性 | 10% | 10.0 | 10.00 | 无扣分 |
| D6 | 工作流完整性 | 10% | 10.0 | 10.00 | 无扣分 |
| D7 | 模式与最佳实践 | 5% | 5.0 | 5.00 | 无扣分 |
| D8 | 反模式检测 | 10% | 10.0 | 10.00 | 无扣分 |
| D9 | 脚本与代码质量 | 5% | 5.0 | 5.00 | 无扣分 |

## 规则覆盖率

| 指标 | 数值 |
|------|------|
| 期望规则数 | 47 |
| 已评估规则数 | 47 |
| 跳过规则数 | 46 |
| 覆盖率 | 100.0% |

**跳过的规则**：R02-R47（原因：目标技能目录不存在，无法进行进一步检查）

### 规则状态明细

| 规则 | 状态 | 维度 | 严重度 | 类型 |
|------|------|------|--------|------|
| R01 | FAIL | D1 | S0 | static |
| R02 | SKIP | D1 | S0 | static |
| R03 | SKIP | D1 | S0 | static |
| R04 | SKIP | D1 | S1 | static |
| R05 | SKIP | D1 | S2 | static |
| R06 | SKIP | D1 | S3 | static |
| R07 | SKIP | D1 | S1 | semantic |
| R08 | SKIP | D1 | S2 | semantic |
| R09 | SKIP | D1 | S2 | semantic |
| R10 | SKIP | D1 | S2 | static |
| R11 | SKIP | D2 | S1 | static |
| R12 | SKIP | D2 | S2 | static |
| R13 | SKIP | D2 | S1 | static |
| R14 | SKIP | D2 | S2 | semantic |
| R15 | SKIP | D3 | S2 | static |
| R16 | SKIP | D3 | S2 | static |
| R17 | SKIP | D3 | S2 | static |
| R18 | SKIP | D3 | S2 | static |
| R19 | SKIP | D3 | S2 | semantic |
| R20 | SKIP | D4 | S2 | semantic |
| R21 | SKIP | D4 | S3 | semantic |
| R22 | SKIP | D4 | S2 | static |
| R23 | SKIP | D4 | S2 | semantic |
| R24 | SKIP | D5 | S2 | semantic |
| R25 | SKIP | D5 | S2 | semantic |
| R26 | SKIP | D5 | S1 | semantic |
| R27 | SKIP | D5 | S2 | semantic |
| R28 | SKIP | D6 | S1 | semantic |
| R29 | SKIP | D6 | S2 | semantic |
| R30 | SKIP | D6 | S2 | semantic |
| R31 | SKIP | D6 | S2 | semantic |
| R32 | SKIP | D7 | S3 | semantic |
| R33 | SKIP | D7 | S3 | semantic |
| R34 | SKIP | D8 | S0 | static |
| R35 | SKIP | D8 | S1 | static |
| R36 | SKIP | D8 | S1 | static |
| R37 | SKIP | D8 | S1 | static |
| R38 | SKIP | D8 | S2 | static |
| R39 | SKIP | D9 | S2 | static |
| R40 | SKIP | D9 | S2 | static |
| R41 | SKIP | D9 | S2 | static |
| R42 | SKIP | D9 | S2 | semantic |
| R43 | SKIP | D3 | S3 | static |
| R44 | SKIP | D1 | S2 | static |
| R45 | SKIP | D2 | S2 | static |
| R46 | SKIP | D4 | S3 | static |
| R47 | SKIP | D7 | S2 | semantic |

## 质量闸门

| 类型 | 数量 | 说明 |
|------|------|------|
| 内部误绑条目 | 0 | 无 |
| 证据不足条目 | 0 | 无 |

## 问题清单

### S0 致命缺陷

#### 问题 1：目标技能目录不存在

**命中规则**：R01 (S0)

> 规则内容：SKILL.md 必须以由 `---` 分隔的 frontmatter 块开头

**位置**：`N/A:0`

**当前内容**：
> Directory does not exist: /workspace/code/pypto/.claude/worktrees/pypto-skill-reviewer/.agents/skills/pypto-precision-multiple-checkpoints

**问题说明**：
指定的目标技能路径 `/workspace/code/pypto/.claude/worktrees/pypto-skill-reviewer/.agents/skills/pypto-precision-multiple-checkpoints` 不存在。该目录在文件系统中未找到，因此无法读取 SKILL.md 文件进行评审。

**修改建议**：
> 请确认目标技能路径是否正确。当前指定的路径不存在。可用的技能目录包括：
> - pypto-aicore-error-locator
> - pypto-binary-search-verify
> - pypto-binary-search-without-verify
> - pypto-environment-setup
> - pypto-operator-accuracy-verify
> - pypto-operator-develop-workflow
> - pypto-operator-perf-analyzer
> - pypto-operator-perf-autotuner
> - pypto-pass
> - pypto-pr-creator
> - pypto-pr-fixer
> - pypto-skill-reviewer

---

### S1 重大问题

无

---

### S2 中等问题

无

---

### S3 轻微建议

无

---

## 通过项

共 0 条规则通过。

| 维度 | 通过规则 |
|------|---------|
| D1 | (无) |
| D2 | (无) |
| D3 | (无) |
| D4 | (无) |
| D5 | (无) |
| D6 | (无) |
| D7 | (无) |
| D8 | (无) |
| D9 | (无) |
