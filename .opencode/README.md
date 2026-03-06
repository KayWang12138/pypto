# 使用指南：AGENTS 与 Skills

本文档面向使用本仓库 OpenCode 工作流的用户，帮助你快速选择并调用合适的 skill。

---

## AGENTS.md 是什么

AGENTS.md 用于为 OpenCode 提供项目自定义指令，类似于 Cursor 的规则功能。该文件包含的指令会被纳入 LLM 的上下文中，以便针对本仓库自定义其行为。

### AGENTS.md 包含什么
- **核心原则**：遇到问题时优先定位修复、基于官方文档实现、优先保证方案可用等
- **环境兼容性**：默认版本（CANN 8.5.0 / PyTorch 2.6.0 / torch_npu 2.6.0.post3）
- **开发规范**：目录结构要求、分阶段开发指南、错误处理原则
- **需求检查默认值**：算子名称、数学公式、输入输出规格等默认配置

### OpenCode 如何使用 AGENTS.md
OpenCode 会自动从当前目录向上遍历并加载 AGENTS.md 文件，你不需要手动阅读该文件。当你与 OpenCode 对话时，AGENTS.md 中的指令会自动生效，指导 OpenCode 按本仓库的规范工作。

官方文档：https://opencode.ai/docs/zh-cn/rules/

---

## Skills 是什么

Skills（代理技能）通过 SKILL.md 定义可复用的行为。OpenCode 能够从本仓库的 `.opencode/skills/` 目录发现这些技能，并通过原生的 skill 工具按需加载。代理可以查看可用技能，并在需要时加载完整内容。

### 本仓库提供的 Skills
本仓库在 `.opencode/skills/` 目录下提供了 14 个技能，覆盖算子开发、性能调优、PR 贡献、工程规范等领域。每个技能包含：
- **SKILL.md**：技能的完整说明和使用流程
- **scripts/**（可选）：辅助脚本
- **references/**（可选）：参考文档

官方文档：https://opencode.ai/docs/zh-cn/skills/

---

## 快速上手

### OpenCode 自动加载 AGENTS.md
当你在这个仓库中使用 OpenCode 时，AGENTS.md 中的自定义指令会自动生效。你不需要做任何额外操作。

### 如何使用 Skill
OpenCode 提供三种方式使用 skill：

#### 方式 1：自动匹配
只描述你的目标，OpenCode 会自动选择合适的 skill。

**示例**：
> 我需要开发一个 PyPTO 算子，请帮我完成环境检查和开发流程。

OpenCode 会自动判断需要使用 `pypto-environment-setup` 和 `pypto-operator-develop-workflow` 等技能。

#### 方式 2：强制触发
使用斜杠命令明确指定要用的 skill。

**语法**：`/<skill_name>`

**示例**：
```
/pypto-operator-develop-workflow
```

这会强制 OpenCode 加载并执行 `pypto-operator-develop-workflow` 技能。

#### 方式 3：自然语言点名
在对话中明确提到要使用的 skill。

**示例**：
> 请使用 `pypto-operator-develop-workflow` 技能帮我开发一个算子。算子名称是 sinh，数学公式是 (e^x - e^(-x)) / 2。

---

## Skills 索引（按领域分组）

### 算子开发

| Skill | 什么时候用 | 你需要提供什么 | 你会得到什么 |
|-------|-----------|--------------|-------------|
| pypto-operator-develop-workflow | 开发昇腾 NPU 自定义算子 | 算子名称、数学公式、输入输出规格、数据类型、精度要求 | 完整开发流程：需求检查→环境准备→Plan→实现→测试→高阶参数使能 |
| pypto-environment-setup | 环境安装、诊断、修复 | 问题描述（如 import 错误、NPU 检测失败） | 诊断报告 + 修复步骤 + 验证命令 |
| pypto-verify-pass | 分析 pass 校验结果 | 编译输出日志、pass 验证结果 | 哪些 pass 成功/失败、失败原因定位 |
| pypto-verify-binary-search | 精度问题二分定位 | 精度不匹配的算子代码、golden 实现 | 第一个出现误差的 op 位置 |

### 性能调优

| Skill | 什么时候用 | 你需要提供什么 | 你会得到什么 |
|-------|-----------|--------------|-------------|
| pypto-operator-perf-autotune | 分析算子性能、生成调优建议 | 算子代码、输出目录路径 | 泳道图分析、性能瓶颈定位、优化建议 |
| pypto-perf-tuning-loop | 迭代式性能调优 | 算子代码、性能目标、可调参数范围 | 基准性能→调优后性能对比报告、最优参数组合 |

### PR/贡献

| Skill | 什么时候用 | 你需要提供什么 | 你会得到什么 |
|-------|-----------|--------------|-------------|
| pypto-pr-creator | 创建 PR 到 cann/pypto 仓库 | 分支名、commit 信息、PR 标题和描述 | PR 创建链接 + 结构化报告 |
| pypto-pr-fixer | 修复 PR review 评论或 CodeCheck 失败 | PR URL 或 owner/repo/pull_number、需要修复的评论 | 修复方案 + 自动应用 + 同步更新 |

### 工程规范

| Skill | 什么时候用 | 你需要提供什么 | 你会得到什么 |
|-------|-----------|--------------|-------------|
| code-review | 代码审查（正确性/安全/性能/风格） | 需要审查的文件或 diff | 问题清单（含严重程度、置信度、file:line） |
| code-philosophy | 编码前了解代码哲学（5 Laws） | 无（作为前置知识加载） | 5 Laws of Elegant Defense + 实践检查清单 |
| plan-protocol | 创建多步骤实现计划 | 目标、上下文、依赖 | 带 YAML frontmatter 和引用的标准计划格式 |
| plan-review | 审查实现计划质量 | 计划文档 | 完整性/可执行性/引用质量审查报告 |

### 前端

| Skill | 什么时候用 | 你需要提供什么 | 你会得到什么 |
|-------|-----------|--------------|-------------|
| frontend-philosophy | UI/设计任务前了解设计哲学 | 无（作为前置知识加载） | 5 Pillars of Intentional UI + 设计检查清单 |

### 工具集成

| Skill | 什么时候用 | 你需要提供什么 | 你会得到什么 |
|-------|-----------|--------------|-------------|
| gitcode-mcp-install | 安装或配置 GitCode MCP Server | 安装方式偏好（Go 二进制或 Python 源码） | 安装命令 + 配置模板 + 验证步骤 |

---

## 使用示例

### 示例 1：自动匹配
```
我需要开发一个名为 sinh 的算子，数学公式是 (e^x - e^(-x)) / 2。
输入是 shape 为 [b, s, n, d] 的 float32 tensor，输出 shape 相同。
精度要求：atol=0.000025, rtol=0.005。
```

OpenCode 会自动：
1. 加载 AGENTS.md 中的开发规范
2. 选择并加载 `pypto-operator-develop-workflow` 技能
3. 按技能流程执行开发任务

### 示例 2：强制触发
```
/pypto-environment-setup

我的环境报错：ImportError: cannot import name 'torch_npu'
Python 版本：3.10
已安装：torch 2.6.0
服务器：A3，CANN 8.5.0
```

OpenCode 会：
1. 强制加载 `pypto-environment-setup` 技能
2. 执行环境诊断
3. 提供修复步骤和验证命令

### 示例 3：自然语言点名
```
请使用 code-review 技能审查 custom/sinh/ 目录下的算子实现代码。
关注点：正确性和性能。
只报告置信度 >= 80% 的问题。
```

OpenCode 会：
1. 加载 `code-review` 技能
2. 审查指定代码
3. 输出问题清单（含 file:line、严重程度、建议修复）

---

## 安全提醒

### Token 不得泄露
- 永远不要在对话、代码、文档中打印或记录真实 token（如 GITCODE_TOKEN）
- 配置文件中使用环境变量占位符，例如：`$GITCODE_TOKEN` 或 `${GITCODE_TOKEN}`
- 如果需要分享配置示例，务必替换为占位符

### 敏感信息处理
- API 密钥、访问令牌、证书等均视为敏感信息
- 日志输出前检查是否包含敏感字段
- PR 提交前确认没有意外包含 .env 或凭证文件

---

## FAQ

### Q1: 怎么选择合适的 skill？
按任务类型选择：
- 算子开发 → pypto-operator-develop-workflow
- 环境问题 → pypto-environment-setup
- 性能调优 → pypto-operator-perf-autotune → pypto-perf-tuning-loop
- 精度问题 → pypto-verify-pass → pypto-verify-binary-search
- 提交 PR → pypto-pr-creator
- 代码审查 → code-review

如果不确定，直接描述你的目标，让 OpenCode 自动匹配。

### Q2: 如果没有 NPU 环境怎么办？
1. 使用 pypto-environment-setup 诊断环境状态
2. 如果是环境缺失，按技能提供的步骤安装 CANN/torch_npu
3. 如果只是验证逻辑，可以先在 CPU 上用 PyTorch 原生实现验证 golden，再迁移到 NPU

### Q3: 三种 skill 调用方式有什么区别？
- **自动匹配**：只描述目标，OpenCode 自动选择 skill，适合不确定用哪个技能的情况
- **强制触发**：使用 `/<skill_name>` 明确指定 skill，适合清楚知道要用哪个技能的情况
- **自然语言点名**：在对话中提到 skill 名称，介于自动匹配和强制触发之间

三种方式效果等价，选择你习惯的即可。

### Q4: Skill 调用失败怎么办？
1. 检查 skill 名称是否正确（使用目录名，如 pypto-operator-develop-workflow）
2. 确认该 skill 在当前配置的允许列表中
3. 尝试强制触发方式：`/<skill_name>`
4. 如果仍然失败，可以让 OpenCode 读取对应的 SKILL.md 文件并按其流程执行

### Q5: AGENTS.md 和 Skills 有什么关系？
- **AGENTS.md**：项目级自定义指令，OpenCode 自动加载，对所有对话生效
- **Skills**：可复用的行为模块，按需加载，只在明确调用时生效

AGENTS.md 定义了本仓库的通用规范，Skills 定义了特定任务的执行流程。两者配合使用，让 OpenCode 能够按本仓库的规范完成各类任务。
