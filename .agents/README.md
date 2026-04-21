# pypto-kernel-custom-skills

面向 **PyPTO kernel 开发**的 Agent skill 库，采用渐进式信息披露架构。本仓库中除本 README 外的所有工件均位于 `skills/` 目录下。

> **单 Agent / 经典工作流 →** 先阅读 `skills/lead-orchestrator/references/principles.md`，再阅读 `skills/lead-orchestrator/references/agent-plan.md`。该计划是一个分步执行清单，在适当时机引用 `skills/lead-orchestrator/references/rules.md` 和其他 skill。
>
> **多 Agent 团队 →** 从 `skills/lead-orchestrator/SKILL.md` 开始。它是 Lead Agent 的入口点，加载 5 份控制文档（原则、团队、计划、规则、目录）作为渐进式引用资源。

## 架构：三层渐进式信息披露

| 层级 | 内容 | 加载时机 |
|------|------|----------|
| **1. 发现层** | `catalog.yaml` → `_category.yaml` → `metadata.yaml` | 始终（路由） |
| **2. 逻辑层** | `SKILL.md`（每个不超过 500 行） | 由阶段或用户触发时 |
| **3. 资源层** | `references/`、`scripts/`、`templates/` | 由 SKILL.md 显式指示时 |

**路由流程：** `skills/lead-orchestrator/references/catalog.yaml`（7 个类别）→ `_category.yaml`（最多 8 个 skill）→ `metadata.yaml`（触发条件）→ `SKILL.md` → 资源。

## Lead Agent 入口

Lead Agent 的操作手册以单一 skill 形式存在：
`skills/lead-orchestrator/`。它将 5 份控制文档作为第三层引用打包，使 Lead Agent 可以渐进式加载而非一次性全部加载。

| 文件 | 职责 |
|------|------|
| `skills/lead-orchestrator/SKILL.md` | 入口和阅读顺序 |
| `skills/lead-orchestrator/metadata.yaml` | 触发条件和范围 |
| `skills/lead-orchestrator/references/principles.md` | 4 条行为准则（思考、简化、精准、目标驱动） |
| `skills/lead-orchestrator/references/agents.md` | 多 Agent 团队名册、每个 Agent 的活跃 skill（2–5 个）、路由 skill 策略 |
| `skills/lead-orchestrator/references/agent-plan.md` | 分阶段清单（Phase 0–6）、门禁、调试协议、完成标准 |
| `skills/lead-orchestrator/references/rules.md` | 23 条强制规则、逐模块执行、3 条禁止项 |
| `skills/lead-orchestrator/references/catalog.yaml` | 第一层 skill 路由索引（7 个类别） |

## Skill 类别（`skills/`）

| 类别 | 路径 | 数量 | 范围 |
|----------|------|-------|-------|
| **orchestration** | `skills/` | 1 | Lead Agent 入口；打包原则、Agent 计划、团队名册、规则、目录 |
| **workflow** | `skills/` | 7 | 阶段编排（0-6）、模板、代码格式、验证 |
| **development** | `skills/` | 8 | 需求、API、golden、设计、实现、环境搭建 |
| **debugging** | `skills/` | 7 | 精度、aicore 错误、崩溃分析、内存重叠 |
| **performance** | `skills/` | 6 | 三阶段调优：前端、泳道、核内 |
| **ci-and-pr** | `skills/` | 6 | 布局检查、PR、Issue、评审、断裂点检测 |
| **pass** | `skills/` | 5 | Pass 模块分析、错误、编译性能、UT |

**总计：40 个 skill**，覆盖 7 个类别。

## 可用工具

| 工具 | 命令 / 调用 | 用途 |
|:---|:---|:---|
| op_index 查询 | `python3 .agents/skills/pypto-api-explore/scripts/query_op_index.py` | 精确 API 签名、类别、文档指针（CLI 回退） |
| MCP `list_ops` | `list_ops(category="")` | 列出所有类别；`list_ops(category="math")` 列出某类别下的算子 |
| MCP `query_op` | `query_op(names=["matmul", "softmax"])` | 精确 API 签名、参数、约束、内联示例 |
| MCP `retrieve_docs` | `retrieve_docs(query=..., chunk_type=...)` | 对文档、示例、源码进行语义搜索 |
| MCP `validate_kernel_structure` | `validate_kernel_structure(source_code=...)` | AST 静态检查：写回、tile 配置、loop 索引 |
| MCP `diagnose_error` | `diagnose_error(error_log=..., kernel_code=...)` | 将 CANN/NPU 错误匹配到已知模式并提供修复方案 |
| PyPTO 调用点列表 | `python3 .agents/skills/ci-and-layout-check/scripts/extract_pypto_calls.py <kernel.py>` | 按行号排列的 `pypto.*` 调用点有序列表 |
| 布局检查（CI） | `bash .agents/skills/ci-and-layout-check/run_validate_layout.sh` | 验证 `custom/<op>/` 结构 — 无需 NPU |

优先使用 MCP；不可用时回退到 CLI 脚本。
