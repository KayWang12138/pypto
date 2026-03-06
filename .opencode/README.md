# 使用指南：AGENTS 与 Skills

本文档面向使用本仓库 OpenCode 工作流的用户，帮助你快速选择并调用合适的 skill。

---

## 总体介绍

- **AGENTS.md**：本仓库的工作流入口与约束集合，定义核心原则、默认值、开发规范。
- **Skills**：可复用的提示词/流程模块，通过"按名称加载"把内容注入上下文，让 AI 按既定流程执行。

**推荐使用方式**：先读 AGENTS.md 了解约束，再按需选择 skill 并调用。

---

## AGENTS.md 是什么，怎么用

### 定位
- 工作流入口与约束集合（不是唯一真理，但优先参考）
- 包含：核心原则、需求检查默认值、Plan 模式使用规范、开发规范

### 使用步骤
1. **阅读核心原则**：遇到问题时优先定位修复，而非推翻重写
2. **确认环境兼容性**：检查默认版本（CANN 8.5.0 / PyTorch 2.6.0 / torch_npu 2.6.0.post3）
3. **按任务类型选 skill**：算子开发 → 环境问题 → 性能调优 → PR/贡献

### 入口
- [../AGENTS.md](../AGENTS.md)

---

## 快速上手（3 步）

### Step 1: 阅读 AGENTS.md
了解本仓库的约束和默认值，避免常见错误。

### Step 2: 选择 Skill
从下方"Skills 索引"中找到适合你任务的 skill。

### Step 3: 调用 Skill
在你的 AI 工具中：
- **强制触发**：`/<skill_name>`（若客户端支持）
- **或自然语言**："请先加载 skill: `<skill_name>`，然后按它的步骤完成 `<目标>`"

---

## Skills 怎么用（两种方式）

### 1. 自动匹配
只描述你的目标和约束，系统会自动选择合适的 skill。

**示例提示词**：
> 我需要开发一个 PyPTO 算子，请帮我完成环境检查和开发流程。

### 2. 强制触发
明确点名要用的 skill。

**方式 A**：`/<skill_name>`（若客户端支持）

**方式 B**：自然语言点名
> 请先加载 skill: `pypto-operator-develop-workflow`，然后按其步骤完成算子开发。输入信息如下：算子名称=sinh，数学公式=...

> ⚠️ **注意**：权限配置可能影响 skill 可用性。如果调用失败，请检查该 skill 是否在允许列表中。

---

## Skills 索引（按领域分组）

### 算子开发

| Skill | 什么时候用 | 如何调用 | 你需要提供什么 | 你会得到什么 | 入口 |
|-------|-----------|---------|--------------|-------------|------|
| pypto-operator-develop-workflow | 开发昇腾 NPU 自定义算子 | `/pypto-operator-develop-workflow` 或 "加载 skill: pypto-operator-develop-workflow" | 算子名称、数学公式、输入输出规格、数据类型、精度要求 | 完整开发流程：需求检查→环境准备→Plan→实现→测试→高阶参数使能 | [SKILL.md](./skills/pypto-operator-develop-workflow/SKILL.md) |
| pypto-environment-setup | 环境安装/诊断/修复 | `/pypto-environment-setup` 或 "加载 skill: pypto-environment-setup" | 问题描述（如 import 错误、NPU 检测失败） | 诊断报告 + 修复步骤 + 验证命令 | [SKILL.md](./skills/pypto-environment-setup/SKILL.md) |
| pypto-verify-pass | 分析 pass 校验结果 | `/pypto-verify-pass` 或 "加载 skill: pypto-verify-pass" | 编译输出日志、pass 验证结果 | 哪些 pass 成功/失败、失败原因定位 | [SKILL.md](./skills/pypto-verify-pass/SKILL.md) |
| pypto-verify-binary-search | 精度问题二分定位 | `/pypto-verify-binary-search` 或 "加载 skill: pypto-verify-binary-search" | 精度不匹配的算子代码、golden 实现 | 第一个出现误差的 op 位置 | [SKILL.md](./skills/pypto-verify-binary-search/SKILL.md) |

### 性能调优

| Skill | 什么时候用 | 如何调用 | 你需要提供什么 | 你会得到什么 | 入口 |
|-------|-----------|---------|--------------|-------------|------|
| pypto-operator-perf-autotune | 分析算子性能、生成调优建议 | `/pypto-operator-perf-autotune` 或 "加载 skill: pypto-operator-perf-autotune" | 算子代码、输出目录路径 | 泳道图分析、性能瓶颈定位、优化建议 | [SKILL.md](./skills/pypto-operator-perf-autotune/SKILL.md) |
| pypto-perf-tuning-loop | 迭代式性能调优 | `/pypto-perf-tuning-loop` 或 "加载 skill: pypto-perf-tuning-loop" | 算子代码、性能目标、可调参数范围 | 基准性能→调优后性能对比报告、最优参数组合 | [SKILL.md](./skills/pypto-perf-tuning-loop/SKILL.md) |

### PR/贡献

| Skill | 什么时候用 | 如何调用 | 你需要提供什么 | 你会得到什么 | 入口 |
|-------|-----------|---------|--------------|-------------|------|
| pypto-pr-creator | 创建 PR 到 cann/pypto 仓库 | `/pypto-pr-creator` 或 "加载 skill: pypto-pr-creator" | 分支名、commit 信息、PR 标题和描述 | PR 创建链接 + 结构化报告 | [SKILL.md](./skills/pypto-pr-creator/SKILL.md) |
| pypto-pr-fixer | 修复 PR review 评论或 CodeCheck 失败 | `/pypto-pr-fixer` 或 "加载 skill: pypto-pr-fixer" | PR URL 或 owner/repo/pull_number、需要修复的评论 | 修复方案 + 自动应用 + 同步更新 | [SKILL.md](./skills/pypto-pr-fixer/SKILL.md) |

### 工程规范

| Skill | 什么时候用 | 如何调用 | 你需要提供什么 | 你会得到什么 | 入口 |
|-------|-----------|---------|--------------|-------------|------|
| code-review | 代码审查（4层：正确性/安全/性能/风格） | `/code-review` 或 "加载 skill: code-review" | 需要审查的文件或 diff | 问题清单（含严重程度、置信度、file:line） | [SKILL.md](./skills/code-review/SKILL.md) |
| code-philosophy | 编码前了解代码哲学（5 Laws） | `/code-philosophy` 或 "加载 skill: code-philosophy" | 无（作为前置知识加载） | 5 Laws of Elegant Defense + 实践检查清单 | [SKILL.md](./skills/code-philosophy/SKILL.md) |
| plan-protocol | 创建多步骤实现计划 | `/plan-protocol` 或 "加载 skill: plan-protocol" | 目标、上下文、依赖 | 带 YAML frontmatter 和引用的标准计划格式 | [SKILL.md](./skills/plan-protocol/SKILL.md) |
| plan-review | 审查实现计划质量 | `/plan-review` 或 "加载 skill: plan-review" | 计划文档 | 完整性/可执行性/引用质量审查报告 | [SKILL.md](./skills/plan-review/SKILL.md) |

### 前端

| Skill | 什么时候用 | 如何调用 | 你需要提供什么 | 你会得到什么 | 入口 |
|-------|-----------|---------|--------------|-------------|------|
| frontend-philosophy | UI/设计任务前了解设计哲学 | `/frontend-philosophy` 或 "加载 skill: frontend-philosophy" | 无（作为前置知识加载） | 5 Pillars of Intentional UI + 设计检查清单 | [SKILL.md](./skills/frontend-philosophy/SKILL.md) |

### 工具集成

| Skill | 什么时候用 | 如何调用 | 你需要提供什么 | 你会得到什么 | 入口 |
|-------|-----------|---------|--------------|-------------|------|
| gitcode-mcp-install | 安装/配置 GitCode MCP Server | `/gitcode-mcp-install` 或 "加载 skill: gitcode-mcp-install" | 安装方式偏好（Go 二进制/Python 源码） | 安装命令 + 配置模板 + 验证步骤 | [SKILL.md](./skills/gitcode-mcp-install/SKILL.md) |

---

## 通用调用模板（可复制）

### 模板
```
请先加载 skill: <skill_name>

目标：<一句话描述你要完成什么>

输入信息：
- <参数1>：<值1>
- <参数2>：<值2>
- ...

约束：
- <特殊要求或限制>

输出格式：
- <你期望的输出形式，如：命令列表/步骤清单/报告>
```

### 示例 1：环境诊断
```
请先加载 skill: pypto-environment-setup

目标：诊断并修复 torch_npu import 失败问题

输入信息：
- 错误信息：ImportError: cannot import name 'torch_npu'
- Python 版本：3.10
- 已安装的包：torch 2.6.0

约束：
- 服务器为 A3，CANN 8.5.0

输出格式：
- 诊断结论
- 修复命令
- 验证步骤
```

### 示例 2：代码审查
```
请先加载 skill: code-review

目标：审查 custom/sinh/ 目录下的算子实现代码

输入信息：
- 目标文件：custom/sinh/sinh.py, custom/sinh/test_sinh.py
- 关注点：正确性、性能

约束：
- 使用本仓库的编码规范
- 仅报告置信度 >= 80% 的问题

输出格式：
- 问题清单（含 file:line、严重程度、建议修复）
```

---

## 安全提醒

### Token 不得泄露
- **永远不要**在对话、代码、文档中打印或记录真实 token（如 `GITCODE_TOKEN`）
- 配置文件中使用环境变量占位符，例如：`"$GITCODE_TOKEN"` 或 `"${GITCODE_TOKEN}"`
- 如果需要分享配置示例，务必替换为占位符

### 敏感信息处理
- API 密钥、访问令牌、证书等均视为敏感信息
- 日志输出前检查是否包含敏感字段
- PR 提交前确认没有意外包含 `.env` 或凭证文件

---

## FAQ

### Q1: 怎么选择合适的 skill？
**A**: 按任务类型选择：
- 算子开发 → `pypto-operator-develop-workflow`
- 环境问题 → `pypto-environment-setup`
- 性能调优 → `pypto-operator-perf-autotune` → `pypto-perf-tuning-loop`
- 精度问题 → `pypto-verify-pass` → `pypto-verify-binary-search`
- 提交 PR → `pypto-pr-creator`
- 代码审查 → `code-review`

### Q2: 如果没有 NPU 环境怎么办？
**A**:
1. 先用 `pypto-environment-setup` 诊断环境状态
2. 如果是环境缺失，按 skill 提供的步骤安装 CANN/torch_npu
3. 如果只是验证逻辑，可以先在 CPU 上用 PyTorch 原生实现验证 golden，再迁移到 NPU

### Q3: `/<skill_name>` 和自然语言调用有什么区别？
**A**:
- `/<skill_name>`：强制触发，适合你明确知道要用哪个 skill
- 自然语言：可以是强制触发（点名 skill）或自动匹配（只描述目标）
- 两种方式效果等价，选择你习惯的即可

### Q4: Skill 调用失败怎么办？
**A**:
1. 检查 skill 名称是否正确（使用目录名，如 `pypto-operator-develop-workflow`）
2. 确认该 skill 在当前配置的允许列表中
3. 尝试自然语言方式："请先加载 skill: <name>"
4. 如果仍然失败，可以让 AI 读取对应的 `./skills/<name>/SKILL.md` 并按其流程执行

---

## 快速参考

- **AGENTS.md**: [../AGENTS.md](../AGENTS.md)
- **Skills 目录**: [./skills/](./skills/)
