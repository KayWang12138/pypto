# pypto-kernel-custom-skills

Agent skill library for **PyPTO kernel development** with progressive
disclosure architecture. Every artifact in this repo except this README
lives under `skills/`.

> **Single agent / classic workflow →** read `skills/lead-orchestrator/references/principles.md` first, then `skills/lead-orchestrator/references/agent-plan.md`. The plan is a step-by-step execution checklist that references `skills/lead-orchestrator/references/rules.md` and other skills at the right time.
>
> **Multi-agent team →** start at `skills/lead-orchestrator/SKILL.md`. It is the Lead Agent's entry point and loads the 5 control documents (principles, agents, agent-plan, rules, catalog) as progressive-disclosure references.

## Architecture: 3-Tier Progressive Disclosure

| Tier | What | When loaded |
|------|------|-------------|
| **1. Discovery** | `catalog.yaml` → `_category.yaml` → `metadata.yaml` | Always (routing) |
| **2. Logic** | `SKILL.md` (< 500 lines each) | When triggered by phase or user |
| **3. Resources** | `references/`, `scripts/`, `templates/` | When SKILL.md explicitly directs |

**Routing flow:** `skills/lead-orchestrator/references/catalog.yaml` (7 categories) → `_category.yaml` (max 8 skills) → `metadata.yaml` (triggers) → `SKILL.md` → resources.

## Lead Agent entry point

The Lead Agent's operating manual lives as a single skill:
`skills/lead-orchestrator/`. It bundles the 5 control
documents as Tier-3 references so the Lead Agent loads them progressively
rather than all at once.

| File | Role |
|------|------|
| `skills/lead-orchestrator/SKILL.md` | Entry point and reading order |
| `skills/lead-orchestrator/metadata.yaml` | Triggers and scope |
| `skills/lead-orchestrator/references/principles.md` | 4 behavioral guidelines (Think, Simplify, Surgical, Goal-Driven) |
| `skills/lead-orchestrator/references/agents.md` | Multi-agent team roster, per-agent active skills (2–5), router-skill policy |
| `skills/lead-orchestrator/references/agent-plan.md` | Phased checklist (Phase 0–6), gates, debug protocol, completion criteria |
| `skills/lead-orchestrator/references/rules.md` | 23 mandatory rules, module-at-a-time enforcement, 3 prohibitions |
| `skills/lead-orchestrator/references/catalog.yaml` | Tier-1 skill routing index (7 categories) |

## Skill categories (`skills/`)

| Category | Path | Count | Scope |
|----------|------|-------|-------|
| **orchestration** | `skills/` | 1 | Lead Agent entry point; bundles principles, agent plan, team roster, rules, catalog |
| **workflow** | `skills/` | 7 | Phase orchestration (0-6), templates, code format, validation |
| **development** | `skills/` | 8 | Requirements, API, golden, design, implementation, env setup |
| **debugging** | `skills/` | 7 | Precision, aicore errors, crash analysis, memory overlap |
| **performance** | `skills/` | 6 | 3-stage tuning: frontend, swimlane, incore |
| **ci-and-pr** | `skills/` | 6 | Layout check, PR, Issue, review, fracture detection |
| **pass** | `skills/` | 5 | Pass module analysis, errors, compilation perf, UT |

**Total: 40 skills** across 7 categories.

## Available tools

| Tool | Command / Call | Purpose |
|:---|:---|:---|
| op_index query | `python3 .agents/skills/pypto-api-explorer/scripts/query_op_index.py` | Exact API signatures, categories, doc pointers (CLI fallback) |
| MCP `list_ops` | `list_ops(category="")` | List all categories; `list_ops(category="math")` lists ops in a category |
| MCP `query_op` | `query_op(names=["matmul", "softmax"])` | Exact API signatures, parameters, constraints, inline examples |
| MCP `retrieve_docs` | `retrieve_docs(query=..., chunk_type=...)` | Semantic search over docs, examples, source |
| MCP `validate_kernel_structure` | `validate_kernel_structure(source_code=...)` | AST lint: write-back, tile config, loop idx |
| MCP `diagnose_error` | `diagnose_error(error_log=..., kernel_code=...)` | Match CANN/NPU error to known pattern + fix |
| PyPTO call-site list | `python3 .agents/skills/ci-and-layout-check/scripts/extract_pypto_calls.py <kernel.py>` | Ordered list of `pypto.*` call sites by line number |
| Layout check (CI) | `bash .agents/skills/ci-and-layout-check/run_validate_layout.sh` | Validates `custom/<op>/` structure — no NPU needed |

MCP preferred when available; fall back to CLI scripts otherwise.
