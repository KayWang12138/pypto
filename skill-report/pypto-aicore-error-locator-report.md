# 技能评审报告

## 评审摘要

| 项目 | 结果 |
|------|------|
| 技能名称 | pypto-aicore-error-locator |
| 评审时间 | 2026-03-17 |
| 总分 | 98.55 / 100 |
| 等级 | A |
| S0 否决 | 否 |
| 规则统计 | 通过 44 / 失败 2 / 警告 1 / 跳过 1 |

## 维度得分

| 维度 | 名称 | 权重 | 满分 | 得分 | 扣分详情 |
|------|------|------|------|------|---------|
| D1 | Frontmatter 元数据 | 25% | 25.0 | 23.75 | R08(S2): -5 |
| D2 | 简洁性与效率 | 15% | 15.0 | 15.0 | 无扣分 |
| D3 | 文件结构与导航 | 10% | 10.0 | 10.0 | 无扣分 |
| D4 | 语言与表达 | 10% | 10.0 | 9.8 | R46(S3): -2 |
| D5 | 精确性与可执行性 | 10% | 10.0 | 10.0 | 无扣分 |
| D6 | 工作流完整性 | 10% | 10.0 | 10.0 | 无扣分 |
| D7 | 模式与最佳实践 | 5% | 5.0 | 5.0 | 无扣分 |
| D8 | 反模式检测 | 10% | 10.0 | 10.0 | 无扣分 |
| D9 | 脚本与代码质量 | 5% | 5.0 | 5.0 | 无扣分（无 scripts/ 目录） |

## 规则覆盖率

| 指标 | 数值 |
|------|------|
| 期望规则数 | 48 |
| 已评估规则数 | 48 |
| 跳过规则数 | 1 |
| 覆盖率 | 100.0% |

**跳过的规则**：R42（原因：不存在 scripts/ 目录，规则不适用）

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
| R46 | FAIL | D4 | S3 | semantic |
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

#### 问题 1：Description 缺少用户自然触发短语

**命中规则**：R08 [S2]

> 规则内容：`description` 应包含用户自然会说出的触发短语

**位置**：`SKILL.md:3`

**当前内容**：
> description: 定位测试案例中出现 aicore error 时的问题 CCE 文件。当需要分析 aicore 错误并找到导致错误的 CCE 文件时使用此技能。

**问题说明**：
description 缺少用户自然会说出的触发短语。当前描述较为技术化，缺少如"定位 aicore 错误"、"找 CCE 文件"、"分析 aicore error"等用户口语化表达。这可能导致 AI 助手无法准确识别用户意图并触发该技能。

**修改建议**：
> 在 description 中添加更自然的触发短语，例如：
> ```yaml
> description: 定位测试案例中出现 aicore error 时的问题 CCE 文件。当用户说"aicore error 定位"、"找 CCE 文件"、"aicore 错误分析"、"帮我定位 aicore 报错"时使用此技能。
> ```

---

### S3 轻微建议

#### 问题 2：代码块缺少语言标注

**命中规则**：R46 [S3]

> 规则内容：代码块应带有语言标注，纯文本块除外

**位置**：`SKILL.md:22`

**当前内容**：
> ```
> question:
>   - header: "PyPTO配置"
> ```

**问题说明**：
代码块缺少语言标注。第 22-37 行的 YAML 配置示例代码块没有指定语言标识符 'yaml'，第 84-90 行的 bash 命令代码块没有指定 'bash'，第 102-103 行的格式说明没有标注。缺少语言标注会降低代码可读性和语法高亮效果。

**修改建议**：
> 为代码块添加语言标注：
> - 将第 22 行的 '```' 改为 '```yaml'
> - 将第 84 行的 '```' 改为 '```bash'
>
> 修改前：
> ```
> question:
>   - header: "PyPTO配置"
> ```
>
> 修改后：
> ```yaml
> question:
>   - header: "PyPTO配置"
> ```

---

#### 问题 3：Description 触发范围可进一步扩展

**命中规则**：R48 [S3]

> 规则内容：description 应该 pushy 一些，避免在技能本应发挥作用的场景下却不使用

**位置**：`SKILL.md:3`

**当前内容**：
> description: 定位测试案例中出现 aicore error 时的问题 CCE 文件。当需要分析 aicore 错误并找到导致错误的 CCE 文件时使用此技能。

**问题说明**：
description 仅列出显式触发关键词，缺少扩展触发范围的短语如 'whenever'、'including'、'or any related to'，可能导致在相关场景下不使用该技能。

**修改建议**：
> 扩展 description 的触发范围，添加类似以下短语：
> ```yaml
> description: 定位测试案例中出现 aicore error 时的问题 CCE 文件。whenever aicore error occurs during testing, including any aicore-related failures or kernel errors, use this skill to locate the problematic CCE file.
> ```

---

## 通过项

共 44 条规则通过。

| 维度 | 通过规则 |
|------|---------|
| D1 | R01, R02, R03, R04, R05, R06, R07, R09, R10, R44 |
| D2 | R11, R12, R13, R14, R45 |
| D3 | R15, R16, R17, R18, R19, R43 |
| D4 | R20, R21, R22, R23 |
| D5 | R24, R25, R26, R27 |
| D6 | R28, R29, R30, R31 |
| D7 | R32, R33, R47 |
| D8 | R34, R35, R36, R37, R38 |
| D9 | R39, R40, R41 |
