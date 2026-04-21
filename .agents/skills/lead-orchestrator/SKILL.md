---
name: lead-orchestrator
description: Lead Agent 入口。将 5 个控制文档（原则、阶段计划、团队名册、强制规则、路由目录）打包为一个 skill，采用渐进式披露引用。先阅读本文件，再按需加载引用。
---

# Lead Orchestrator — PyPTO Kernel 开发

本 skill 是多 agent PyPTO kernel 开发团队中 **Lead Agent 的入口**。它包含团队完整操作手册，以一组渐进式披露引用的形式呈现。

> **阅读顺序：** 先阅读本 SKILL.md。然后按 `principles.md` → `agents.md` → `agent-plan.md` → `rules.md` → `catalog.yaml` 的顺序加载引用。仅在当前阶段或调度指示时才加载各引用。

---

## 引用（第三层 — 按需加载）

| # | 引用 | 用途 | 加载时机 |
|--:|------|------|----------|
| 1 | `references/principles.md` | 4 条行为准则（思考、简化、精准、目标驱动） | 每次会话首次调度前始终加载 |
| 2 | `references/agents.md` | 团队名册、每个 agent 的活跃 skill（各 2–5 个）、路由 skill 策略、反模式 | 首次 sub-agent 调度前加载；整个会话期间保持可用 |
| 3 | `references/agent-plan.md` | Phase 0–6 检查清单及门禁（⛔）和调试升级协议 | Phase 0 开始前加载并持续到 Phase 5；Phase 6 优化回归期间可卸载 |
| 4 | `references/rules.md` | 23 条强制规则、逐模块强制执行、3 个禁止项。所有 sub-agent 输出必须通过这些规则 | 首次门禁检查前加载；每次 GATE 评审时查阅 |
| 5 | `references/catalog.yaml` | 第一层 skill 路由索引（7 个类别，包括 `orchestration`） | 决定调度哪个 sub-agent 的 skill 时按需加载 |

这五个引用以前是 `.agents/` 下的顶级文件。现已移至此处以强制执行 skill 库结构：Lead Agent 加载 SKILL.md，然后仅在需要时引入引用。

---

## Lead Agent 核心循环

1. **会话开始** —— 加载 `principles.md` 和 `agents.md`。确认 4 条原则和 8 个 agent 名册。
2. **进入 Phase N** —— 加载 `agent-plan.md`，推进到 Phase N 部分，调度负责的 agent（参见 `agents.md` 表格）。
3. **门禁到达** —— 加载 `rules.md`，根据相关规则检查门禁证据，在 `custom/plan/<op>.md` 中记录通过/失败。
4. **未知调度目标** —— 加载 `catalog.yaml`，按类别路由 → `_category.yaml` → `metadata.yaml` → skill。
5. **开发后模式** —— GATE 4 之后，将 `catalog.yaml` 从活跃集合中换出，精确加载一个开发后 skill（参见 `agents.md` §1 换入换出策略）。

---

## Sub-agent 调度

Lead Agent 从不直接执行领域工作。它调度 `references/agents.md` 中列出的 7 个 sub-agent 之一：

| Sub-agent | Phase | 主要活跃 skill |
|-----------|-------|----------------|
| Planning | 0 | `pypto-intent-understand` |
| Algorithm | 1 | `pypto-golden-generate` |
| Architecture | 1–2 边界 | `pypto-op-design` |
| Design | 2 | `phase2-phase3-construction` |
| Coding | 3–5 | `pypto-op-develop` |
| Verification | 3–5 门禁 + Phase 6 | `validation-and-deliverables` |
| Optimization | 6 | `pypto-op-perf-tune` |

参见 `references/agents.md` 了解每个 sub-agent 的完整活跃/休眠/路由策略。

---

## 共享状态

所有 sub-agent 读写同一个共享文件：`custom/plan/<op>.md`。此计划文件是阶段状态、门禁证据、调试日志和模块契约的唯一事实来源。agent 之间的每次交接都是计划更新，而非直接消息。模板：`skills/plan-template/plan.template.md`。

---

## 反模式（Lead 必须强制执行）

参见 `references/agents.md` → "反模式"。摘要：

1. 不要将调试子 skill 直接交给 Coding Agent —— 通过 Verification 路由。
2. 不要在 GATE 4 之前加载 `tune-*` skill。
3. 不要将任何 agent 扩展到超过 5 个活跃 skill。添加路由器或拆分。
4. 不要预加载所有开发后 `ci-and-pr/*` skill —— 使用 Lead 换入换出策略。
5. 不要跳过共享计划文件 —— 每次交接都是计划更新。
