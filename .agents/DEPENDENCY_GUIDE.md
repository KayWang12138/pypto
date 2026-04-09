# Agent & Skill 依赖关系指南

> 本文档是 `.agents/skills/` 和 `.opencode/agents/` 的依赖关系黄金指南。
> 新增或修改 agent/skill 前**必须阅读**本文档，确保依赖方向正确、引用路径有效。
>
> 最后更新：2026-03-31

---

## 1. 架构分层原则

```
┌─────────────────────────────────────────────────┐
│  Agent 层 (.opencode/agents/)                   │
│  ─ 可引用 Skill、可调度其他 Agent               │
│  ─ 负责编排、状态管理、重试策略                  │
├─────────────────────────────────────────────────┤
│  Skill 层 (.agents/skills/)                     │
│  ─ 可引用其他 Skill（同层依赖）                  │
│  ─ 禁止引用 Agent（不可向上依赖）                │
│  ─ 负责单一领域任务的完整执行流程                │
└─────────────────────────────────────────────────┘
```

### 核心规则

| 规则 | 说明 | 示例 |
|------|------|------|
| **Agent -> Skill** | 允许 | orchestrator 调用 `pypto-intent-understand` |
| **Agent -> Agent** | 允许（调度子agent） | orchestrator 调度 `@pypto-op-analyst` |
| **Skill -> Skill** | 允许（同层引用） | `pypto-pr-fixer` 引用 `pypto-pr-creator` |
| **Skill -> Agent** | **禁止** | skill 中不得出现任何 agent 名称 |
| **下游 -> 上游** | **禁止** | `gitcode-mcp-install` 不得引用 `pypto-pr-fixer` |

### 依赖方向判定

```
上游（被依赖方）             下游（依赖方）
gitcode-mcp-install    <──   pypto-pr-creator   <──   pypto-pr-fixer
pypto-pass-module-analyzer <── pypto-pass-ut-generate
                            <── pypto-pass-error-fixer
                            <── pypto-pass-workflow-analyzer
                            <── pypto-pass-perf-optimizer
```

**上游不得感知下游的存在。** 如果 A 依赖 B，则 B 的文档/代码中不应出现 A 的名字。

---

## 2. 完整依赖图谱

### 2.1 Agent 层

```
pypto-op-orchestrator (primary, 主控)
├── 直接调用 Skills:
│     ├── pypto-intent-understand     [Stage 1]
│     └── pypto-api-explore             [Stage 2]
├── 调度 Subagents:
│     ├── @pypto-op-analyst              [Stage 3-4]
│     │     ├── pypto-golden-generate
│     │     └── pypto-op-design
│     ├── @pypto-op-developer            [Stage 5-6]
│     │     ├── pypto-op-develop
│     │     └── pypto-precision-debug
│     └── @pypto-op-perf-tuner            [Stage 7]
│           ├── pypto-op-perf-analyzer
│           └── pypto-op-perf-autotuner
└── 建议性引用（非强依赖）:
      └── pypto-aicore-error-locator

pypto-code-merge-agent (subagent, 独立)
├── gitcode-mcp-install
├── pypto-issue-creator
└── pypto-pr-creator
```

### 2.2 Skill 层依赖关系

#### 算子开发主链

```
pypto-op-workflow (编排型 skill，引用以下 skill)
├── pypto-intent-understand
├── pypto-api-explore
├── pypto-golden-generate
├── pypto-op-design
├── pypto-op-develop
├── pypto-precision-debug
├── pypto-op-perf-analyzer
└── pypto-op-perf-autotuner
```

#### 精度工具链

```
pypto-precision-debug
├── pypto-precision-verify          (verify 模式检查点对比定位)
└── pypto-precision-binary-search   (checkpoint 模式二分定位)
```

#### Pass 工具链

```
pypto-pass-module-analyzer          (基础分析，被多个 skill 依赖)
├── pypto-pass-ut-generate          (UT 生成)
├── pypto-pass-error-fixer          (错误修复，还依赖 pypto-environment-setup)
├── pypto-pass-workflow-analyzer    (工作流分析)
└── pypto-pass-perf-optimizer       (性能优化)
```

#### PR 工具链

```
gitcode-mcp-install                 (基础设施，被多个 skill 依赖)
├── pypto-pr-creator                (PR 创建)
│     └── pypto-pr-fixer            (PR 修复)
└── pypto-issue-creator             (Issue 创建，独立使用 MCP)
```

#### 性能工具链

```
pypto-op-perf-analyzer              (分析)
└── pypto-op-perf-autotuner         (调优，依赖 analyzer 的分析结果)

pypto-operator-auto-tune            (独立的算子调优 skill)
├── perf-analyzer                   (子 skill)
├── tune-frontend                   (子 skill)
├── tune-swimlane                   (子 skill)
└── tune-incore                     (子 skill)
```

#### 其他依赖

```
pypto-environment-setup --> pypto-op-develop (使用其 list_idle_chip_ids.sh 脚本)
pypto-golden-generate  --> pypto-intent-understand (格式对齐)
pypto-golden-generate  --> pypto-op-design (格式对齐)
```

### 2.3 完全独立的 Skill（叶节点，无外部依赖）

- `gitcode-mcp-install`
- `pypto-aicore-error-locator`
- `pypto-api-explore`
- `pypto-fracture-point-detector`
- `pypto-intent-understand`
- `pypto-op-design`
- `pypto-op-develop`
- `pypto-op-perf-analyzer`
- `pypto-pass-module-analyzer`
- `pypto-precision-binary-search`
- `pypto-skill-reviewer`

---

## 3. Agent Frontmatter 规范

Agent 文件必须在 YAML frontmatter 中声明所有直接调用的 skill：

```yaml
---
name: agent-name
description: "..."
mode: primary | subagent
skills:
  - skill-name-1
  - skill-name-2
---
```

### 当前各 Agent 的声明状态

| Agent | frontmatter skills | body 引用的 skills | 一致性 |
|-------|-------------------|-------------------|--------|
| pypto-op-orchestrator | pypto-intent-understand, pypto-api-explore | pypto-intent-understand, pypto-api-explore | OK |
| pypto-op-analyst | pypto-golden-generate, pypto-op-design | pypto-golden-generate, pypto-op-design | OK |
| pypto-op-developer | pypto-op-develop, pypto-precision-debug | pypto-op-develop, pypto-precision-debug | OK |
| pypto-op-perf-tuner | pypto-op-perf-analyzer, pypto-op-perf-autotuner | pypto-op-perf-analyzer, pypto-op-perf-autotuner | OK |
| pypto-code-merge-agent | gitcode-mcp-install, pypto-issue-creator, pypto-pr-creator | gitcode-mcp-install, pypto-issue-creator, pypto-pr-creator | OK |

---

## 4. MCP Server 依赖

| Skill | MCP 工具 |
|-------|---------|
| `gitcode-mcp-install` | 安装 GitCode MCP Server |
| `pypto-issue-creator` | `gitcode_search_issues`, `gitcode_create_issue` |
| `pypto-pr-creator` | `gitcode_get_repository`, `gitcode_list_pull_requests`, `gitcode_create_pull_request`, `gitcode_update_pull_request`, `gitcode_get_pull_request` |
| `pypto-pr-fixer` | `gitcode_get_pull_request`, `gitcode_list_pull_request_comments` |

---

## 5. 路径规范

### 目录结构

```
.agents/skills/{skill-name}/
├── SKILL.md              # 必须，技能定义
├── scripts/              # 可选，脚本工具
├── references/           # 可选，参考文档
└── templates/            # 可选，模板文件

.opencode/agents/
└── {agent-name}.md       # Agent 定义
```

### 路径引用规则

| 场景 | 正确写法 | 错误写法 |
|------|---------|---------|
| Skill 引用自身脚本 | `scripts/xxx.py` (相对路径) | - |
| Skill 引用其他 skill 的脚本 | `.agents/skills/{skill}/scripts/xxx.py` | `.opencode/skills/...` |
| Agent 引用 skill 文档 | `.agents/skills/{skill}/references/xxx.md` | `.opencode/skills/...` |
| Skill 引用其他 skill | 只写 skill 名称（如 `pypto-pr-creator`） | 写完整 `@.opencode/skills/.../SKILL.md` |

---

## 6. Skill 名称映射（历史名 -> 现名）

以下旧名称已全部清理，如在任何文件中发现它们，必须替换：

| 旧名称 | 现名称 |
|--------|--------|
| `pypto-binary-search-verify` | `pypto-precision-verify` |
| `pypto-binary-search-without-verify` | `pypto-precision-binary-search` |

---

## 7. 新增 Skill/Agent 检查清单

### 新增 Skill

- [ ] 在 `.agents/skills/{name}/` 下创建 `SKILL.md`
- [ ] SKILL.md 包含 YAML frontmatter（name, description）
- [ ] 如果依赖其他 skill，在 SKILL.md 中使用**正确的 skill 名称**引用
- [ ] **不引用任何 agent 名称**
- [ ] 脚本路径使用 `.agents/skills/` 前缀（非 `.opencode/skills/`）
- [ ] 在 `AGENTS.md` 的 Skills 索引中添加条目
- [ ] 在 `.agents/README.md` 的技能详解中添加说明
- [ ] 更新本文档的依赖图谱

### 新增 Agent

- [ ] 在 `.opencode/agents/` 下创建 `{name}.md`
- [ ] YAML frontmatter 包含 `name`, `description`, `mode`, `skills`
- [ ] frontmatter 的 `skills` 字段**列出所有直接调用的 skill**
- [ ] 在 `.agents/README.md` 的 Agents 表格中添加条目
- [ ] 更新本文档的依赖图谱

### 修改已有 Skill/Agent

- [ ] 如果新增了对其他 skill 的引用，更新本文档
- [ ] 如果新增了 Agent 中的 skill 调用，同步更新 frontmatter `skills` 字段
- [ ] 确认依赖方向正确（不违反分层原则）
- [ ] 运行全局搜索确认无旧名称残留

---

## 8. 常见违规模式及修复方法

### 违规 1：Skill 引用 Agent

```markdown
<!-- 错误：skill 中提到 agent -->
正式开发优先使用 `pypto-op-orchestrator`

<!-- 正确：skill 只描述自身职责，不感知上层 -->
本 Skill 适合手动串联相关 Skills 完成端到端开发。
```

### 违规 2：下游感知上游

```markdown
<!-- 错误：gitcode-mcp-install 中提到 pypto-pr-fixer -->
本 MCP Server 安装后可被 pypto-pr-fixer 使用。

<!-- 正确：下游不提及上游 -->
本 MCP Server 安装后可被其他需要 GitCode API 的技能使用。
```

### 违规 3：使用旧路径

```markdown
<!-- 错误 -->
参考 @.opencode/skills/pypto-pass/pypto-pass-module-analyzer/SKILL.md

<!-- 正确 -->
参考 pypto-pass-module-analyzer 技能
```

### 违规 4：Agent frontmatter 缺失 skill 声明

```yaml
# 错误：body 中调用了 skill 但 frontmatter 未声明
---
name: my-agent
mode: subagent
---
请调用 pypto-op-develop ...

# 正确：frontmatter 中声明所有直接调用的 skill
---
name: my-agent
mode: subagent
skills:
  - pypto-op-develop
---
请调用 pypto-op-develop ...
```
