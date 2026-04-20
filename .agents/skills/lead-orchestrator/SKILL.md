---
name: lead-orchestrator
description: Lead Agent entry point. Bundles the 5 control documents (principles, phase plan, team roster, mandatory rules, routing catalog) as one skill with progressive-disclosure references. Read this file first, then load the references on demand.
---

# Lead Orchestrator — PyPTO Kernel Development

This skill is the **entry point for the Lead Agent** in the multi-agent
PyPTO kernel-development team. It contains the full operating manual for
the team as a set of progressive-disclosure references.

> **Reading order:** Read this SKILL.md first. Then load references in the
> order `principles.md` → `agents.md` → `agent-plan.md` → `rules.md`
> → `catalog.yaml`. Load each reference only when the current phase or
> dispatch directs you to.

---

## References (Tier 3 — load on demand)

| # | Reference | Purpose | Load when |
|--:|-----------|---------|-----------|
| 1 | `references/principles.md` | 4 behavioral guidelines (Think, Simplify, Surgical, Goal-Driven) | Always load before the first dispatch of any session |
| 2 | `references/agents.md` | Team roster, per-agent active skills (2–5 each), router-skill policy, anti-patterns | Load before the first sub-agent dispatch; keep it at hand across the whole session |
| 3 | `references/agent-plan.md` | Phase 0–6 checklist with gates (⛔) and the debug escalation protocol | Load before starting Phase 0 and keep it loaded through Phase 5; unloadable during Phase 6 optimization regressions |
| 4 | `references/rules.md` | 23 mandatory rules, module-at-a-time enforcement, 3 prohibitions. Every sub-agent output must pass these | Load before the first gate check; consult on every GATE review |
| 5 | `references/catalog.yaml` | Tier-1 skill routing index (7 categories including `orchestration`) | Load on demand when deciding which sub-agent's skill to dispatch |

All five references were previously top-level files under `.agents/`. They
have been relocated here to enforce the skill-library structure: the Lead
Agent loads SKILL.md, then pulls in references only as needed.

---

## Lead Agent's core loop

1. **Start of session** — Load `principles.md` and `agents.md`. Acknowledge
   the 4 principles and the 8-agent roster.
2. **Enter Phase N** — Load `agent-plan.md`, advance to the Phase N section,
   dispatch the owning agent (see `agents.md` table).
3. **Gate arrives** — Load `rules.md`, check the gate's evidence against the
   relevant rules, record pass/fail in `custom/plan/<op>.md`.
4. **Unknown dispatch target** — Load `catalog.yaml`, route by category
   → `_category.yaml` → `metadata.yaml` → skill.
5. **Post-dev mode** — After GATE 4, swap `catalog.yaml` out of the active
   set and load exactly one post-dev skill (see `agents.md` §1 swap policy).

---

## Sub-agent dispatch

The Lead Agent never executes domain work directly. It dispatches to one of
the 7 sub-agents listed in `references/agents.md`:

| Sub-agent | Phase | Primary active skill |
|-----------|-------|----------------------|
| Planning | 0 | `pypto-intent-understand` |
| Algorithm | 1 | `pypto-golden-generate` |
| Architecture | 1–2 boundary | `pypto-op-design` |
| Design | 2 | `phase2-phase3-construction` |
| Coding | 3–5 | `pypto-op-develop` |
| Verification | 3–5 gates + Phase 6 | `validation-and-deliverables` |
| Optimization | 6 | `pypto-op-perf-tune` |

See `references/agents.md` for the full active / dormant / router policy per
sub-agent.

---

## Shared state

All sub-agents read and write one shared file: `custom/plan/<op>.md`. This
plan file is the single source of truth for phase status, gate evidence,
debug log, and module contracts. Every handoff between agents is a plan
update, not a direct message. Template:
`skills/plan-template/plan.template.md`.

---

## Anti-patterns (Lead must enforce)

See `references/agents.md` → "Anti-patterns". Summary:

1. Do not hand debug sub-skills directly to the Coding Agent — route through
   Verification.
2. Do not load `tune-*` skills before GATE 4.
3. Do not expand any agent past 5 active skills. Add a router or split.
4. Do not pre-load all post-dev `ci-and-pr/*` skills — use the Lead swap
   policy.
5. Do not skip the shared plan file — every handoff is a plan update.
