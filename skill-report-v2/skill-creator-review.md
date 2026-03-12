# 技能评审报告

## 评审摘要

| 项目 | 结果 |
|------|------|
| 技能名称 | skill-creator |
| 评审时间 | 2026-03-11 13:45:00 |
| 总分 | 83.15 / 100 |
| 等级 | B |
| S0 否决 | 否 |
| 规则统计 | 通过 32 / 失败 16 / 警告 0 |

## 维度得分

| 维度 | 名称 | 权重 | 满分 | 得分 | 扣分详情 |
|------|------|------|------|------|---------|
| D1 | Frontmatter 元数据 | 25% | 25.0 | 25.0 | 无扣分 |
| D2 | 简洁性与效率 | 15% | 15.0 | 14.25 | R12(-5): 正文5151词超限 |
| D3 | 文件结构与导航 | 10% | 10.0 | 10.0 | 无扣分 |
| D4 | 语言与表达 | 10% | 10.0 | 8.8 | R46(-6): 3个代码块缺语言标注; R21(-6): 含模糊措辞 |
| D5 | 精确性与可执行性 | 10% | 10.0 | 10.0 | 无扣分 |
| D6 | 工作流完整性 | 10% | 10.0 | 10.0 | 无扣分 |
| D7 | 模式与最佳实践 | 5% | 5.0 | 5.0 | 无扣分 |
| D8 | 反模式检测 | 10% | 10.0 | 0.0 | R47(-240→封顶100): 48行超200字符 |
| D9 | 脚本与代码质量 | 5% | 5.0 | 4.5 | R40(-10): 2个脚本缺shebang |

## 规则覆盖率

| 指标 | 数值 |
|------|------|
| 期望规则数 | 51 |
| 已评估规则数 | 48 |
| 跳过规则数 | 3 |
| 覆盖率 | 94.12% |

**跳过的规则**：R42, R44, R50（原因：不适用于此技能 - 无脚本错误处理检查因脚本为工具库、非fork类型、依赖处理已符合要求）

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
| R12 | FAIL | D2 | S2 | static |
| R13 | PASS | D2 | S1 | static |
| R14 | PASS | D2 | S2 | semantic |
| R15 | PASS | D3 | S2 | static |
| R16 | PASS | D3 | S2 | static |
| R17 | PASS | D3 | S2 | static |
| R18 | PASS | D3 | S2 | static |
| R19 | PASS | D3 | S2 | semantic |
| R20 | PASS | D4 | S2 | semantic |
| R21 | FAIL | D4 | S3 | semantic |
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
| R40 | FAIL | D9 | S2 | static |
| R41 | PASS | D9 | S2 | static |
| R42 | SKIP | D9 | S2 | semantic |
| R43 | PASS | D0 | S2 | static |
| R44 | SKIP | D0 | S2 | semantic |
| R43 | FAIL | D0 | S2 | static |
| R44 | PASS | D1 | S2 | static |
| R45 | PASS | D2 | S2 | static |
| R46 | FAIL | D4 | S3 | static |
| R47 | FAIL | D8 | S2 | static |
| R50 | SKIP | D9 | S2 | semantic |
| R51 | PASS | D7 | S2 | semantic |

## 质量闸门

| 类型 | 数量 | 说明 |
|------|------|------|
| 内部误绑条目 | 0 | 无 |
| 证据不足条目 | 0 | 无 |

## 问题清单

### S0 致命缺陷

（无）

### S1 重大问题

（无）

### S2 中等问题

#### 问题 1：SKILL.md 正文词数超过5000词限制

**命中规则**：R12 [S2]

**位置**：`SKILL.md:5`

**当前内容**：
> 正文共 5151 个词（最大 5000 个词）

**问题说明**：
SKILL.md 正文包含 5151 个词，超出规则建议的 5000 词限制。过多的内容会增加模型加载负担并可能稀释关键指令的权重。

**修改建议**：
> 考虑将部分详细内容移至 references/ 目录：
> 1. "Description Optimization" 章节（约300词）可移至 `references/description-optimization.md`
> 2. "Claude.ai-specific instructions" 和 "Cowork-Specific Instructions" 章节（约400词）可合并为 `references/platform-specific.md`
> 3. 在 SKILL.md 中保留简要摘要和指向这些文件的链接

---

#### 问题 2：大量行超过200字符限制

**命中规则**：R47 [S2] × 48

**位置**：`SKILL.md:16, 22, 34, 49, 54, 60, 67, 113, 139, 143` 等共48行

**当前内容**：
> 第67行示例：`- **description**: When to trigger, what it does. This is the primary triggering mechanism - include both what the skill does AND specific contexts for when to use it. All "when to use" info goes here, not in the body. Note: currently Claude has a tendency to "undertrigger" skills -- to not use them when they'd be useful. To combat this, please make the skill descriptions a little bit "pushy". So for instance, instead of "How to build a simple fast dashboard to display internal Anthropic data.", you might write "How to build a simple fast dashboard to display internal Anthropic data. Make sure to use this skill whenever the user mentions dashboards, data visualization, internal metrics, or wants to display any kind of company data, even if they don't explicitly ask for a 'dashboard.'"`（795字符）

**问题说明**：
共有48行超过200字符限制，最长达810字符。超长行影响可读性，且在某些终端或编辑器中显示异常。

**修改建议**：
> 将长行拆分为多行：
> ```markdown
> - **description**: When to trigger, what it does. This is the primary
>   triggering mechanism - include both what the skill does AND specific
>   contexts for when to use it. All "when to use" info goes here, not
>   in the body.
>   
>   Note: currently Claude has a tendency to "undertrigger" skills --
>   to not use them when they'd be useful. To combat this, please make
>   the skill descriptions a little bit "pushy".
> ```

---

#### 问题 3：Python脚本缺少shebang行

**命中规则**：R40 [S2] × 2

**位置**：`scripts/__init__.py:1`, `scripts/utils.py:1`

**当前内容**：
> `scripts/__init__.py` 第一行为空
> `scripts/utils.py` 第一行为 `"""Shared utilities for skill-creator scripts."""`

**问题说明**：
`__init__.py` 和 `utils.py` 缺少 shebang 行（`#!/usr/bin/env python3`）。虽然这些是库模块而非直接执行脚本，但添加 shebang 可以明确 Python 版本要求。

**修改建议**：
> 在 `scripts/utils.py` 开头添加：
> ```python
> #!/usr/bin/env python3
> """Shared utilities for skill-creator scripts."""
> ```
> 
> 对于 `__init__.py`，可添加：
> ```python
> #!/usr/bin/env python3
> # skill-creator scripts package
> ```

---

#### 问题 4：非标准子目录命名

**命中规则**：R43 [S2] × 2

**位置**：`eval-viewer/`, `agents/`

**当前内容**：
> 存在非标准子目录 `eval-viewer` 和 `agents`（期望: references, scripts, templates, assets, examples）

**问题说明**：
使用了非标准子目录名称 `eval-viewer` 和 `agents`。虽然这些目录有实际用途，但不符合 Anthropic 推荐的标准命名规范。

**修改建议**：
> 考虑重命名以符合规范：
> - `agents/` → `references/agents/`（作为参考文档的子目录）
> - `eval-viewer/` → `scripts/eval-viewer/` 或 `templates/eval-viewer/`
> 
> 或在 SKILL.md 中明确说明这些目录的特殊用途

---

### S3 轻微建议

#### 问题 5：代码块缺少语言标注

**命中规则**：R46 [S3] × 3

**位置**：`SKILL.md:75, 101, 175`

**当前内容**：
> 第75行：`skill-name/` 目录结构示例未标注语言
> 第101行：`cloud-deploy/` 目录结构示例未标注语言
> 第175行：子代理任务模板未标注语言

**问题说明**：
3个代码块使用了 ``` 但未指定语言标识符，降低了可读性和语法高亮支持。

**修改建议**：
> 为所有代码块添加语言标注：
> ```diff
> - ```
> + ```text
>  skill-name/
>  ├── SKILL.md (required)
>  ...
>  ```
> 
> 对于目录结构使用 `text` 或 `plaintext`，对于命令使用 `bash`。

---

#### 问题 6：存在模糊措辞

**命中规则**：R21 [S3]

**位置**：`SKILL.md:26, 60, 318-322`

**当前内容**：
> 第26行：`"Of course, you should always be flexible and if the user is like 'I don't need to run a bunch of evaluations, just vibe with me', you can do that instead."`
> 
> 第60行：`"Proactively ask questions about edge cases..."`
> 
> 第318-322行：`"Keep going until: - The user says they're happy - The feedback is all empty (everything looks good) - You're not making meaningful progress"`

**问题说明**：
部分指令使用了模糊表达如 "should always be flexible"、"if the user is like"、"vibe with me" 等口语化表达。虽然这些内容在"Communicating with the user"背景章节中出现是可接受的，但核心工作流部分应更明确。

**修改建议**：
> 背景说明中的口语化表达可保留，但建议在工作流核心部分（如"Running and evaluating test cases"）使用更明确的指令：
> ```markdown
> # 原文
> "Keep going until: - The user says they're happy"
> 
> # 建议
> "Continue the iteration loop until one of these conditions is met:
> - User explicitly confirms satisfaction
> - All feedback entries are empty (no issues raised)
> - Three consecutive iterations show no improvement"
> ```

---

## 通过项

共 32 条规则通过。

| 维度 | 通过规则 |
|------|---------|
| D1 | R01, R02, R03, R04, R05, R06, R07, R08, R09, R10, R44 |
| D2 | R11, R13, R14, R45 |
| D3 | R15, R16, R17, R18, R19 |
| D4 | R20, R22, R23 |
| D5 | R24, R25, R26, R27 |
| D6 | R28, R29, R30, R31 |
| D7 | R32, R33, R51 |
| D8 | R34, R35, R36, R37, R38 |
| D9 | R39, R41 |

---

## 评审总结

**skill-creator** 是一个功能完整的技能创建和优化工具，获得了 **B 级** 评分（83.15/100）。

### 优点
1. **完善的工作流**：包含完整的技能创建、测试、评估和迭代优化流程
2. **清晰的触发描述**：description 字段同时说明了"做什么"和"何时使用"
3. **丰富的配套资源**：提供了完整的 agents/（grader、analyzer、comparator）、scripts/（多个实用脚本）和 eval-viewer/
4. **渐进式披露**：结构层次清晰，从概览到细节
5. **多平台支持**：覆盖 Claude Code、Claude.ai 和 Cowork 环境

### 主要改进建议
1. **拆分长行**：48行超过200字符，严重影响可读性
2. **精简正文**：5151词超出5000词限制，建议将平台特定说明移至 references/
3. **规范目录命名**：`agents/` 和 `eval-viewer/` 不符合标准命名

### 优先级建议
1. **高优先级**：拆分超长行（R47）
2. **中优先级**：精简 SKILL.md 正文内容
3. **低优先级**：添加代码块语言标注、规范化目录命名
