# 技能评审报告

## 评审摘要

| 项目 | 结果 |
|------|------|
| 技能名称 | pypto-aicore-error-locator |
| 评审时间 | 2026-03-11 |
| 总分 | 96.55 / 100 |
| 等级 | A |
| S0 否决 | 否 |
| 规则统计 | 通过 40 / 失败 6 / 警告 0 / 跳过 1 |

## 维度得分

| 维度 | 名称 | 权重 | 满分 | 得分 | 扣分详情 |
|------|------|------|------|------|---------|
| D1 | Frontmatter 元数据 | 25% | 25.0 | 23.75 | R08(S2): -5 |
| D2 | 简洁性与效率 | 15% | 15.0 | 15.00 | 无扣分 |
| D3 | 文件结构与导航 | 10% | 10.0 | 10.00 | 无扣分 |
| D4 | 语言与表达 | 10% | 10.0 | 9.30 | R23(S2): -5, R46(S3): -2 |
| D5 | 精确性与可执行性 | 10% | 10.0 | 9.00 | R24(S2): -5, R27(S2): -5 |
| D6 | 工作流完整性 | 10% | 10.0 | 9.50 | R30(S2): -5 |
| D7 | 模式与最佳实践 | 5% | 5.0 | 5.00 | 无扣分 |
| D8 | 反模式检测 | 10% | 10.0 | 10.00 | 无扣分 |
| D9 | 脚本与代码质量 | 5% | 5.0 | 5.00 | 无扣分（无 scripts 目录，自动满分）|

## 规则覆盖率

| 指标 | 数值 |
|------|------|
| 期望规则数 | 47 |
| 已评估规则数 | 47 |
| 跳过规则数 | 1 |
| 覆盖率 | 100.0% |

**跳过的规则**：R42（原因：不存在 scripts/ 目录）

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

#### 问题 1：description 缺少自然触发短语

**命中规则**：R08 [S2]

> 规则内容：`description` 应包含用户自然会说出的触发短语

**位置**：`SKILL.md:3`

**当前内容**：
> 定位测试案例中出现 aicore error 时的问题 CCE 文件。当需要分析 aicore 错误并找到导致错误的 CCE 文件时使用此技能。

**问题说明**：
当前 description 仅使用技术化表达，缺少用户自然会说出的触发短语。这可能导致 AI 助手难以准确匹配用户请求与该 skill。

**修改建议**：
> 在 description 中添加触发短语，例如："定位 aicore 错误、分析 aicore 报错、找到出错的 CCE 文件、CCE 文件定位、aicore error 定位"

---

#### 问题 2：部分关键指令缺少理由说明

**命中规则**：R23 [S2]

> 规则内容：指令应解释"为什么"，而不仅是"做什么"

**位置**：`SKILL.md:44`

**当前内容**：
> - 设置 `"fixed_output_path"` 为 `true`
>   - 设置 `"force_overwrite"` 为 `false`

**问题说明**：
配置修改指令未解释为何需要这些设置，用户无法理解其目的和重要性。

**修改建议**：
> 为每个配置修改添加原因说明，例如：
> - 设置 `"fixed_output_path"` 为 `true`（确保输出路径固定不变，便于后续定位）
> - 设置 `"force_overwrite"` 为 `false`（避免覆盖已有日志，保留历史记录）

---

#### 问题 3：多个步骤缺少可验证的成功标准

**命中规则**：R24 [S2], R30 [S2]

> 规则内容：
> - R24: 每个步骤都应具备可验证的成功标准
> - R30: 必须包含错误处理或失败恢复说明

**位置**：`SKILL.md:54`

**当前内容**：
> ### 3. 重新编译和安装
>
> 进入用户提供的 pypto 目录路径，重新编译 pypto 包并pip安装。

**问题说明**：
1. 步骤 3 未说明如何验证编译安装成功
2. 未包含错误处理说明（如编译失败怎么办）

**修改建议**：
> 为每个步骤添加成功标准和错误处理：
> - 成功标准：执行 `pip show pypto` 确认版本已更新
> - 错误处理：若编译失败，检查 CANN 环境是否正确配置

---

#### 问题 4：未明确定义完成标准

**命中规则**：R27 [S2]

> 规则内容：完成标准必须明确定义

**位置**：`SKILL.md:107`

**当前内容**：
> ### 8. 输出结果
>
> 输出找到的 CPP 文件路径。

**问题说明**：
skill 未说明最终交付物是什么以及用户如何确认任务已完成。

**修改建议**：
> 在末尾添加明确的完成标准章节：
> ```markdown
> ## 完成标准
> - 已找到并输出导致 aicore error 的 CCE 文件路径
> - 用户可据此检查对应 CCE 文件的实现
> - 若找到多个 CCE 文件，已全部列出
> ```

---

### S3 轻微建议

#### 问题 5：代码块缺少语言标注

**命中规则**：R46 [S3]

> 规则内容：围栏代码块应带有语言标注

**位置**：`SKILL.md:23`

**当前内容**：
> ```

**问题说明**：
第 23 行的代码块缺少语言标注，影响代码可读性和语法高亮。

**修改建议**：
> 将 ``` 改为 ```yaml，为 YAML 配置示例添加语言标注

---

## 通过项

共 40 条规则通过。

| 维度 | 通过规则 |
|------|---------|
| D1 | R01, R02, R03, R04, R05, R06, R07, R09, R10, R44 |
| D2 | R11, R12, R13, R14, R45 |
| D3 | R15, R16, R17, R18, R19, R43 |
| D4 | R20, R21, R22 |
| D5 | R25, R26 |
| D6 | R28, R29, R31 |
| D7 | R32, R33, R47 |
| D8 | R34, R35, R36, R37, R38 |
| D9 | R39, R40, R41 |
