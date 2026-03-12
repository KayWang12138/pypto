# 技能评审报告

## 评审摘要

| 项目 | 结果 |
|------|------|
| 技能名称 | pypto-aicore-error-locator |
| 评审时间 | 2026-03-11 |
| 总分 | 95.05 / 100 |
| 等级 | A |
| S0 否决 | 否 |
| 规则统计 | 通过 38 / 失败 7 / 警告 0 |

## 维度得分

| 维度 | 名称 | 权重 | 满分 | 得分 | 扣分详情 |
|------|------|------|------|------|---------|
| D1 | Frontmatter 元数据 | 25% | 25.0 | 23.25 | R06(-2), R08(-5) |
| D2 | 简洁性与效率 | 15% | 15.0 | 15.0 | 无 |
| D3 | 文件结构与导航 | 10% | 10.0 | 10.0 | 无 |
| D4 | 语言与表达 | 10% | 10.0 | 9.30 | R46(-2), R23(-5) |
| D5 | 精确性与可执行性 | 10% | 10.0 | 8.00 | R24(-5), R25(-5), R26(-10) |
| D6 | 工作流完整性 | 10% | 10.0 | 9.50 | R30(-5) |
| D7 | 模式与最佳实践 | 5% | 5.0 | 5.0 | 无 |
| D8 | 反模式检测 | 10% | 10.0 | 10.0 | 无 |
| D9 | 脚本与代码质量 | 5% | 5.0 | 5.0 | 无 |

## 规则覆盖率

| 指标 | 数值 |
|------|------|
| 期望规则数 | 50 |
| 已评估规则数 | 45 |
| 跳过规则数 | 5 |
| 覆盖率 | 90.0% |

**跳过的规则**：R42, R47, R44（原因：不适用于此技能）

### 规则状态明细

| 规则 | 状态 | 维度 | 严重度 | 类型 |
|------|------|------|--------|------|
| R01 | PASS | D1 | S0 | static |
| R02 | PASS | D1 | S0 | static |
| R03 | PASS | D1 | S0 | static |
| R04 | PASS | D1 | S1 | static |
| R05 | PASS | D1 | S2 | static |
| R06 | FAIL | D1 | S3 | static |
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
| R25 | FAIL | D5 | S2 | semantic |
| R26 | FAIL | D5 | S1 | semantic |
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
| R42 | SKIP | D9 | S2 | semantic |
| R47 | SKIP | D9 | S2 | semantic |
| R43 | PASS | D0 | S2 | static |
| R44 | SKIP | D0 | S2 | semantic |
| R43 | PASS | D0 | S3 | static |
| R44 | PASS | D1 | S2 | static |
| R45 | PASS | D2 | S2 | static |
| R46 | FAIL | D4 | S3 | static |
| R50 | PASS | D7 | S2 | semantic |

## 质量闸门

| 类型 | 数量 | 说明 |
|------|------|------|
| 内部误绑条目 | 0 | 发现与目标技能无关，已从评分中剔除 |
| 证据不足条目 | 0 | snippet 无法在源文件中匹配，已从评分中剔除 |

## 问题清单

### S0 致命缺陷

无 S0 致命缺陷。

### S1 重大问题

#### 问题 1：重新编译 pypto 包缺少具体命令

**命中规则**：R26 (S1)

> 规则内容：所有提及的操作都必须提供具体实现方法

**位置**：`SKILL.md:56`

**当前内容**：
> 进入用户提供的 pypto 目录路径，重新编译 pypto 包并pip安装。

**问题说明**：
步骤 3 提到"重新编译 pypto 包并pip安装"，但没有提供具体的编译和安装命令。用户无法直接执行此步骤。

**修改建议**：
> 进入用户提供的 pypto 目录路径，执行以下命令重新编译并安装：
> ```bash
> python3 build_ci.py -f python3 --disable_auto_execute
> pip install dist/pypto-*.whl
> ```

---

### S2 中等问题

#### 问题 1：Description 缺少自然触发短语

**命中规则**：R08 (S2)

> 规则内容：`description` 应包含用户自然会说出的触发短语

**位置**：`SKILL.md:3`

**当前内容**：
> description: 定位测试案例中出现 aicore error 时的问题 CCE 文件。当需要分析 aicore 错误并找到导致错误的 CCE 文件时使用此技能。

**问题说明**：
description 使用较正式的技术语言，缺少用户在自然对话中会说出的触发短语，如"定位 aicore 错误"、"找出 aicore 问题"、"aicore 报错定位"等。

**修改建议**：
> description: 定位测试案例中出现 aicore error 时的问题 CCE 文件。当测试出现 aicore 错误、需要找出导致错误的 CCE 文件时使用此技能。

---

#### 问题 2：指令缺少原因解释

**命中规则**：R23 (S2)

> 规则内容：指令应解释"为什么"，而不仅是"做什么"

**位置**：`SKILL.md:44`

**当前内容**：
> - 设置 `"fixed_output_path"` 为 `true`
> - 设置 `"force_overwrite"` 为 `false`

**问题说明**：
步骤 2 中列出了多个配置修改，但没有解释为什么要进行这些修改。用户无法理解这些配置的作用。

**修改建议**：
> - 设置 `"fixed_output_path"` 为 `true`（保持输出路径不变，便于追踪日志）
> - 设置 `"force_overwrite"` 为 `false`（避免覆盖已有日志）

---

#### 问题 3：步骤缺少可验证的成功标准

**命中规则**：R24 (S2)

> 规则内容：每个步骤都应具备可验证的成功标准

**位置**：`SKILL.md:56`

**当前内容**：
> 进入用户提供的 pypto 目录路径，重新编译 pypto 包并pip安装。

**问题说明**：
步骤 3 没有说明如何验证编译和安装是否成功，用户无法确认此步骤是否正确完成。

**修改建议**：
> 进入用户提供的 pypto 目录路径，重新编译 pypto 包并pip安装。验证成功后应能成功导入 pypto 包。

---

#### 问题 4：存在未解析的占位符

**命中规则**：R25 (S2)

> 规则内容：命令和路径必须具体且可执行

**位置**：`SKILL.md:72`

**当前内容**：
> - 设置 device log 落盘路径：`export ASCEND_PROCESS_LOG_PATH=<用户提供的路径>`

**问题说明**：
步骤 5 中使用了 `<用户提供的路径>` 占位符，这不是一个可执行的命令。用户需要知道这里应该填入什么值。

**修改建议**：
> - 设置 device log 落盘路径：`export ASCEND_PROCESS_LOG_PATH=$DEVICE_LOG_PATH`（使用步骤 1 中收集的路径）

---

#### 问题 5：缺少错误处理说明

**命中规则**：R30 (S2)

> 规则内容：必须包含错误处理或失败恢复说明

**位置**：`SKILL.md:1`

**当前内容**：
> ---
> name: pypto-aicore-error-locator

**问题说明**：
整个 skill 没有提及任何错误处理或失败恢复的场景。如果编译失败、日志未找到、找不到 CCE 文件等情况，用户不知道该如何处理。

**修改建议**：
在 skill 末尾添加"错误处理"章节，说明常见错误场景和解决方案。

---

### S3 轻微建议

#### 问题 1：未知的 frontmatter 字段

**命中规则**：R06 (S3)

> 规则内容：未知的 frontmatter 字段应产生告警

**位置**：`SKILL.md:4`

**当前内容**：
> license: 完整条款见 LICENSE.txt

**问题说明**：
frontmatter 中包含 `license` 字段，但该字段不在标准 frontmatter 字段列表中。这可能导致某些工具无法正确解析。

**修改建议**：
> 移除 `license` 字段，或将其内容移至 skill 正文中的"许可证"章节。

---

#### 问题 2：代码块缺少语言标注

**命中规则**：R46 (S3)

> 规则内容：围栏代码块应带有语言标注

**位置**：`SKILL.md:23`

**当前内容**：
> ```

**问题说明**：
步骤 1 中的示例问题配置代码块没有语言标注，影响代码高亮和可读性。

**修改建议**：
> ```yaml
> question: 
>   - header: "PyPTO配置"
>     question: "请提供 pypto 目录的完整路径"
>     options: []
> ```

---

## 通过项

共 38 条规则通过。

| 维度 | 通过规则 |
|------|---------|
| D1 | R01, R02, R03, R04, R05, R07, R09, R10, R44 |
| D2 | R11, R12, R13, R14, R45 |
| D3 | R15, R16, R17, R18, R19 |
| D4 | R20, R21, R22 |
| D5 | R27 |
| D6 | R28, R29, R31 |
| D7 | R32, R33, R50 |
| D8 | R34, R35, R36, R37, R38 |
| D9 | R39, R40, R41 |
| D0 | R43, R43 |
