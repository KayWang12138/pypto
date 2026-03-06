# PyPTO OpenCode 工作流

通过 AI 代理 + 14 个专家技能，自动完成昇腾 NPU 算子开发全流程。

```
需求分析 → 环境准备 → 编码实现 → 精度验证 → 性能调优 → PR 提交
```

本仓库为 [OpenCode](https://opencode.ai) 预配置了项目规范（AGENTS.md）和专家技能（Skills），开箱即用。

---

## 快速开始

在本仓库目录下启动 OpenCode，直接描述你的目标：

```
我需要开发一个名为 sinh 的算子，数学公式是 (e^x - e^(-x)) / 2。
输入是 shape 为 [b, s, n, d] 的 float32 tensor，输出 shape 相同。
精度要求：atol=0.000025, rtol=0.005。
```

OpenCode 会自动加载项目规范，选择合适的技能，按标准流程执行开发任务。无需手动配置。

---

## 我想做…→ 用这个

根据你的目标快速定位合适的技能：

| 我想…                     | 推荐技能                             | 一句话说明               |
| :------------------------ | :----------------------------------- | :----------------------- |
| 开发一个新算子            | `pypto-operator-develop-workflow`     | 全流程引导，从需求到交付 |
| 修复环境报错              | `pypto-environment-setup`            | 诊断 + 修复 + 验证      |
| 分析编译 pass 校验结果    | `pypto-verify`                       | 定位失败的 pass 及原因   |
| 排查精度不一致            | `pypto-verify-binary-search`         | 二分法定位首个误差点     |
| 分析算子性能瓶颈          | `pypto-operator-perf-autotune`       | 泳道图分析 + 优化建议    |
| 迭代调优到目标性能        | `pypto-perf-tuning-loop`             | 参数扫描 + 对比报告      |
| 提交 PR 到 cann/pypto     | `pypto-pr-creator`                   | 自动创建规范 PR          |
| 修复 PR review 意见       | `pypto-pr-fixer`                     | 解析评论 + 自动修复      |
| 审查代码质量              | `code-review`                        | 问题清单 + 严重程度分级  |
| 创建实现计划              | `plan-protocol`                      | 标准计划格式 + 引用      |
| 审查实现计划              | `plan-review`                        | 完整性 / 可执行性评估    |
| 了解编码哲学              | `code-philosophy`                    | 5 Laws of Elegant Defense |
| 了解 UI 设计哲学          | `frontend-philosophy`                | 5 Pillars of Intentional UI |
| 安装 GitCode MCP Server   | `gitcode-mcp-install`                | 安装 + 配置 + 验证       |

> **不确定用哪个？** 直接描述目标，OpenCode 会自动匹配。

---

## 技能详解

### 算子开发

#### `pypto-operator-develop-workflow` — 算子开发全流程

**适用场景**：开发昇腾 NPU 自定义算子

**你需要提供**：算子名称、数学公式、输入输出规格、数据类型、精度要求

**你会得到**：完整的分阶段执行流程——需求检查 → 环境准备 → 方案设计 → 编码实现 → 构建测试 → 高阶参数使能

#### `pypto-environment-setup` — 环境诊断与修复

**适用场景**：环境安装失败、import 报错、NPU 设备检测不到

**你需要提供**：问题描述（如错误信息、Python 版本、已安装的包）

**你会得到**：诊断报告 + 修复步骤 + 验证命令

#### `pypto-verify` — Pass 校验分析

**适用场景**：编译后需要确认各 pass 是否通过

**你需要提供**：编译输出日志或 pass 验证结果

**你会得到**：逐 pass 成功/失败状态 + 失败原因定位

#### `pypto-verify-binary-search` — 精度二分定位

**适用场景**：算子输出与 golden 不一致，需要定位误差来源

**你需要提供**：精度不匹配的算子代码、golden 实现

**你会得到**：第一个出现误差的 op 位置

---

### 性能调优

#### `pypto-operator-perf-autotune` — 性能分析与调优建议

**适用场景**：算子开发完成后，分析性能瓶颈

**你需要提供**：算子代码、输出目录路径

**你会得到**：泳道图分析 + 性能瓶颈定位 + 优化建议

#### `pypto-perf-tuning-loop` — 迭代式性能调优

**适用场景**：需要系统性地搜索最优参数组合

**你需要提供**：算子代码、性能目标、可调参数范围

**你会得到**：基准性能 → 调优后性能对比报告 + 最优参数组合

---

### PR 与贡献

#### `pypto-pr-creator` — 创建 PR

**适用场景**：将开发完成的算子提交到 cann/pypto 仓库

**你需要提供**：分支名、commit 信息、PR 标题和描述

**你会得到**：PR 创建链接 + 结构化报告

#### `pypto-pr-fixer` — 修复 PR 问题

**适用场景**：PR 收到 review 评论或 CodeCheck 失败

**你需要提供**：PR URL 或 `owner/repo/pull_number`、需要修复的评论

**你会得到**：修复方案 + 自动应用 + 同步更新

---

### 工程规范

#### `code-review` — 代码审查

**适用场景**：审查代码的正确性、安全性、性能、风格

**你需要提供**：需要审查的文件或 diff

**你会得到**：问题清单（含严重程度、置信度、`file:line`）

#### `plan-protocol` — 创建实现计划

**适用场景**：多步骤任务需要结构化的实现计划

**你需要提供**：目标、上下文、依赖关系

**你会得到**：带 YAML frontmatter 和引用的标准计划文档

#### `plan-review` — 审查实现计划

**适用场景**：提交计划前检查质量

**你需要提供**：计划文档

**你会得到**：完整性 / 可执行性 / 引用质量审查报告

#### `code-philosophy` — 编码哲学

**适用场景**：编码前加载设计原则作为前置知识

**你会得到**：5 Laws of Elegant Defense + 实践检查清单

#### `frontend-philosophy` — UI 设计哲学

**适用场景**：UI/设计任务前加载设计原则

**你会得到**：5 Pillars of Intentional UI + 设计检查清单

---

### 工具集成

#### `gitcode-mcp-install` — GitCode MCP Server

**适用场景**：安装或配置 GitCode MCP Server

**你需要提供**：安装方式偏好（Go 二进制 / Python 源码）

**你会得到**：安装命令 + 配置模板 + 验证步骤

---

## 工作原理

### AGENTS.md — 项目规范，自动生效

AGENTS.md 是 OpenCode 的项目级自定义指令文件。当你在本仓库中使用 OpenCode 时，它会自动加载并生效——无需手动操作。

该文件定义了：

- **核心原则**：遇问题优先定位修复、基于官方文档实现、优先保证方案可用
- **环境配置**：默认版本（CANN 8.5.0 / PyTorch 2.6.0 / torch_npu 2.6.0.post3）
- **开发规范**：目录结构、分阶段流程、错误处理策略
- **默认值**：输入输出规格、数据类型、精度要求的合理缺省

> 📖 进一步了解：[OpenCode 自定义规则文档](https://opencode.ai/docs/zh-cn/rules/)

### Skills — 专家技能，按需加载

Skills 是定义在 `.opencode/skills/` 目录下的可复用行为模块。每个 skill 包含一个 `SKILL.md` 文件，描述完整的执行流程。OpenCode 会在需要时自动发现并加载。

**调用方式**（三选一，效果等价）：

**自动匹配** — 描述目标，OpenCode 自动选择：

```
我需要开发一个 PyPTO 算子，请帮我完成环境检查和开发流程。
```

**斜杠命令** — 明确指定技能：

```
/pypto-operator-develop-workflow
```

**自然语言点名** — 在对话中提及：

```
请使用 pypto-operator-develop-workflow 技能帮我开发一个算子。
```

> 📖 进一步了解：[OpenCode Skills 文档](https://opencode.ai/docs/zh-cn/skills/)

---

## 安全须知

- **Token 不得泄露**：永远不要在对话、代码、日志中暴露真实令牌。配置文件使用 `$GITCODE_TOKEN` 等环境变量占位符。
- **提交前检查**：确认 PR 中不包含 `.env`、凭证文件或硬编码的密钥。

---

## 常见问题

<details>
<summary><b>没有 NPU 环境怎么办？</b></summary>

1. 使用 `pypto-environment-setup` 诊断当前环境状态
2. 如果是环境缺失，按技能提供的步骤安装 CANN / torch_npu
3. 如果只是验证算法逻辑，可先在 CPU 上用 PyTorch 原生实现验证 golden，再迁移到 NPU

</details>

<details>
<summary><b>Skill 调用失败怎么办？</b></summary>

1. 检查 skill 名称拼写（使用目录名，如 `pypto-operator-develop-workflow`）
2. 确认该 skill 在当前配置的允许列表中
3. 尝试斜杠命令方式：`/<skill_name>`
4. 如果仍然失败，可以让 OpenCode 直接读取对应的 `SKILL.md` 文件并按其流程执行

</details>

<details>
<summary><b>AGENTS.md 和 Skills 有什么区别？</b></summary>

| 维度     | AGENTS.md              | Skills                         |
| :------- | :--------------------- | :----------------------------- |
| 作用     | 项目级自定义规范       | 特定任务的执行流程             |
| 加载方式 | 自动加载，对所有对话生效 | 按需加载，调用时才生效         |
| 内容     | 通用开发规范和原则     | 具体任务的步骤、工具、验证标准 |

两者配合使用：AGENTS.md 定义"怎么做才对"，Skills 定义"怎么一步步做完"。

</details>
