# 技能评审报告

## 评审摘要

| 项目 | 结果 |
|------|------|
| 技能名称 | pypto-pr-fixer |
| 评审时间 | 2026-03-11 |
| 总分 | 99.80 / 100 |
| 等级 | A |
| S0 否决 | 否 |
| 规则统计 | 通过 46 / 失败 1 / 警告 0 / 跳过 0 |

## 维度得分

| 维度 | 名称 | 权重 | 满分 | 得分 | 扣分详情 |
|------|------|------|------|------|---------|
| D1 | Frontmatter 元数据 | 25% | 25.0 | 25.00 | 无扣分 |
| D2 | 简洁性与效率 | 15% | 15.0 | 15.00 | 无扣分 |
| D3 | 文件结构与导航 | 10% | 10.0 | 10.00 | 无扣分 |
| D4 | 语言与表达 | 10% | 10.0 | 9.80 | R46(S3): -2分 × 1条 |
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
| R46 | FAIL | D4 | S3 | static |
| R47 | PASS | D7 | S2 | semantic |

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

#### 问题 1：代码块缺少语言标注

**命中规则**：R46 [S3]

> 规则内容：围栏代码块应带有语言标注

**位置**：`SKILL.md:15, 22, 35, 78, 102, 121, 170, 190, 211, 227, 244, 262, 282, 311, 357, 376`

**当前内容**：
> ```
> ```
> ```

**问题说明**：
SKILL.md 中有 16 处代码块使用了 ``` 开头但未指定语言标识符。这些代码块包括 PR URL 模式、三元组格式、核心流程图、Python 示例代码、bash 命令等。缺少语言标注会影响代码高亮和可读性。

**修改建议**：
> 为每个代码块添加语言标识符。例如：
> - PR URL 和三元组示例：使用 `text` 或不标注（保持原样）
> - 流程图：使用 `text` 或 `mermaid`（如果适用）
> - Python 代码：使用 `python`
> - bash 命令：使用 `bash`
> - HTML 示例：使用 `html`
> - YAML 输出：使用 `yaml`
> - JSON 输出：使用 `json`
>
> 修改前：` ``` `
> 修改后：` ```python `

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
