# Agent Team — Active Skill Allocation

This file defines the multi-agent team that executes PyPTO kernel development
on top of this skill library. It enforces two hard constraints:

1. **Each agent loads 2–5 active skills at a time.** Going beyond 5 causes the
   "35-skill cliff" effect where routing accuracy drops sharply.
2. **Dormant skills are only loaded on-demand via router skills.** Router
   skills know *which* sub-skill to pull in for a given failure or stage, and
   never load all of them eagerly.

> **Reading order for new agents:** `principles.md` → this file →
> `agent-plan.md` → `rules.md`. Then follow the phase checklist and read
> skills only when `agent-plan.md` directs you to.

---

## What counts as a "skill" in this library

Two kinds of documents are treated as skills and count toward the 2–5
active budget:

- **Root control documents** — every file under `.agents/` outside
  `skills/` except `README.md`. These are the Lead Agent's operating
  manual.
- **Skills under `skills/<category>/<skill-id>/`** — the 39 procedural /
  domain skills catalogued in `catalog.yaml`.

The root control documents are **exclusively active for the Lead Agent**.
Other agents do not load them into their active set; instead, Lead Agent
hands down the relevant constraints (rules, gates, routing decisions) as
part of each task dispatch.

## Team roster

| # | Agent | Phase owned | Active skills | Router / pivot skill |
|---|-------|-------------|--------------:|----------------------|
| 1 | **Lead** (Orchestrator) | All phases — gate enforcement | 5 | `agent-plan.md` + `agents.md` |
| 2 | **Planning** | Phase 0 | 4 | `workflow/phase0-phase1-planning` |
| 3 | **Algorithm** (Mathematician) | Phase 1 | 3 | — |
| 4 | **Architecture** | Phase 1–2 boundary | 3 | — |
| 5 | **Design** | Phase 2 | 4 | `workflow/phase2-phase3-construction` |
| 6 | **Coding** | Phase 3–5 | 4 | `workflow/phase2-phase3-construction` |
| 7 | **Verification** | Phase 3–5 gate judge + Phase 6 regression | 2 | — (judge-only, never loads `debugging/*`) |
| 8 | **Debug** (Specialist) | Phase 3–5 failure investigation | 3 | `debugging/debugging` |
| 9 | **Optimization** | Phase 6 (correctness-first) | 3 | `performance/pypto-op-perf-tune` |

**Active skill footprint:** Lead loads 5 root control docs; the 8 sub-agents
together load 26 slots across 13 unique skills in `skills/`. The remaining
skills stay dormant and load on-demand via the router skills listed above.

**Separation of concerns in Phase 3 gate failures:** Verification = judge
(passes or returns a `failure_category`); Debug = specialist investigator
(loads the matching `debugging/*` sub-skill, proposes a patch); Coding =
the only agent that writes production kernel code. Lead drives the loop
`verification → debug → coding → verification`. See `agent-plan.md` Phase 3
inner loop and §8 of this document.

---

## 1. Lead Agent (Orchestrator)

**Responsibility:** Drive `agent-plan.md` phase by phase. Enforce gates.
Dispatch sub-agents. Do *not* execute domain work directly.

**Active skills (the 5 root control documents):**

| # | File | Role |
|--:|------|------|
| 1 | `principles.md` | 4 behavioral guidelines (Think, Simplify, Surgical, Goal-Driven). The ethical / quality contract Lead enforces on every sub-agent dispatch. |
| 2 | `agent-plan.md` | Phase 0–6 checklist, gates, debug protocol, completion criteria. The primary orchestration script. |
| 3 | `agents.md` | Team roster, per-agent active-skill allocation, router policy, sub-agent dispatch rules. |
| 4 | `rules.md` | 23 mandatory rules, module-at-a-time enforcement, 3 prohibitions. Every sub-agent output is audited against this. |
| 5 | `catalog.yaml` | Always-loaded skill routing index (6 categories). The Tier-1 discovery layer Lead uses to locate the right skill for a dispatch. |

These 5 files are Lead-exclusive — other agents receive the relevant
constraints as part of the dispatch payload rather than loading these
documents themselves.

**Dispatched skills (Lead routes these to sub-agents, does not run them):**
`workflow/plan-template` (Planning/Design/Verification), `workflow/
validation-and-deliverables` (Verification), `development/pypto-op-workflow`
(reference), plus every skill in `skills/` via the router policy below.

**Dormant skills Lead loads directly (only when a post-dev condition fires):**

| Condition | Skill Lead loads |
|-----------|------------------|
| Environment / build failure before Phase 0 | `development/pypto-environment-setup` |
| GATE 4 passed, PR requested | `ci-and-pr/pypto-pr-creator` |
| PR CI failure or reviewer comment | `ci-and-pr/pypto-pr-fixer` |
| Bug / feature / doc gap discovered | `ci-and-pr/pypto-issue-creator` |
| Framework or doc gap in this session | `ci-and-pr/pypto-fracture-point-detector` |
| Skill audit requested | `ci-and-pr/pypto-skill-reviewer` |
| Post-correctness model integration | `development/pypto-fused-op-integration` |

Post-dev skills are tagged `scope: post-dev` in
`skills/ci-and-pr/_category.yaml`; only `ci-and-layout-check` (tagged
`scope: in-phase`) is visible in-phase, and it is owned by the Verification
Agent, not by Lead.

**Swap policy to stay at 5.** When Lead enters post-dev mode, it swaps
`catalog.yaml` out of the active set (the remaining dispatch target is the
single post-dev skill, so the Tier-1 routing index is no longer needed) and
loads exactly one post-dev skill in its place. The swap keeps Lead's active
count at **5** under all conditions. Only one post-dev skill is active at a
time.

---

## 2. Planning Agent

**Responsibility:** Produce `SPEC.md`, `API_REPORT.md`, and seed
`custom/plan/<op>.md` with the API map. Owns Phase 0 only.

| Role | Skill | Output |
|------|-------|--------|
| Router | `workflow/phase0-phase1-planning` | Phase 0 section |
| Active | `development/pypto-intent-understand` | `SPEC.md` |
| Active | `development/pypto-api-explore` | `API_REPORT.md` |
| Active | `workflow/plan-template` | Plan skeleton + API map section |

**Exit criterion:** GATE 0 in `agent-plan.md` — API map has zero
`unsupported` rows (or each has a documented workaround).

---

## 3. Algorithm Agent (Mathematician)

**Responsibility:** Produce the PyPTO-friendly `golden.py` and the
**Golden function inventory** in the plan. Owns Phase 1 only.

| Role | Skill | Notes |
|------|-------|-------|
| Active | `development/pypto-golden-generate` | Primary — PyTorch/NumPy golden with confidence score |
| Active | `workflow/phase0-phase1-planning` | Phase 1 normalization rules (no `.T`, explicit `reshape`, shape comments) |
| Active | `workflow/kernel-code-format` | §11 shape annotation conventions |

**Exit criterion:** GATE 1 — zero `.T`/`.t()` in golden, shape comments on all
intermediates, inventory recorded, `allclose` passes against the original
reference.

**Dormant escalation:** `debugging/debugging` (for `DEBUG.md` §9.19
reduction-alignment and matmul caveats — load only if the golden triggers
PyPTO constraints during pre-check).

---

## 4. Architecture Agent

**Responsibility:** Produce `DESIGN.md` and the performance target sheet.
Does *not* perform optimization.

| Role | Skill | Notes |
|------|-------|-------|
| Active | `development/pypto-op-design` | `DESIGN.md`: API mapping, tiling strategy, loop structure |
| Active | `workflow/kernel-code-format` | Layers A–L design format, naming conventions |
| Reference-only | `performance/pypto-op-perf-tune` | **Read the target-metric structure only.** Do NOT load `tune-frontend`/`tune-swimlane`/`tune-incore` — those belong to Optimization Agent. |

**Exit criterion:** `DESIGN.md` exists, Layers A–L populated, perf target
expressed as concrete numbers (baseline, target time, required speedup).

---

## 5. Design Agent

**Responsibility:** Split the kernel into semantic modules, define module
contracts, lay out staged files. Owns Phase 2.

| Role | Skill | Notes |
|------|-------|-------|
| Router | `workflow/phase2-phase3-construction` | Phase 2 module-decomposition section |
| Active | `development/pypto-op-design` | Carry `DESIGN.md` tiling/loop decisions down to per-module level |
| Active | `workflow/kernel-code-format` | `pypto_kernel_template.py` skeleton |
| Active | `workflow/plan-template` | Fill in **Module decomposition**, **Module contracts**, **Staged module files** |

**Exit criterion:** GATE 2 — module decomposition, contracts, and staged
file table all present in the plan.

---

## 6. Coding Agent

**Responsibility:** Implement `custom/<op>/<op>_module<suffix>.py` one module
at a time. Owns Phase 3–5 implementation.

| Role | Skill | Notes |
|------|-------|-------|
| Active | `development/pypto-op-develop` | Generate `impl.py`, `test.py`, `README` |
| Router | `workflow/phase2-phase3-construction` | Phase 3 section + DEBUG §9 lookup table |
| Active | `workflow/phase4-phase5-integration` | Phase 4 integration, Phase 5 structural rules |
| Active | `workflow/kernel-code-format` | Template, write-back patterns, tile config rules |

**Tooling used directly (not skills):** MCP `query_op`, `list_ops`,
`retrieve_docs`, `validate_kernel_structure`.

**Handoff on failure:** On any test/layout failure, hand the failing module
and logs to Verification Agent. Do **not** load debug sub-skills yourself.

---

## 7. Verification Agent (Judge-only)

**Responsibility:** Run `detailed_tensor_compare` and layout checks. Produce
a pass/fail verdict for every gate. On fail, classify the failure into a
`failure_category` so Lead can dispatch the Debug Agent. Does **not**
investigate, bisect, or fix — that is the Debug Agent's job.

| Role | Skill | Notes |
|------|-------|-------|
| Active | `workflow/validation-and-deliverables` | `detailed_tensor_compare` runner, success criteria |
| Active | `ci-and-pr/ci-and-layout-check` | `extract_pypto_calls.py`, `run_validate_layout.sh` |

Active-skill count: **2**. Verification must stay lean because it is
re-invoked on every module patch cycle and must re-run the full GATE 3
checklist from scratch each time.

**Verdict format — always one of:**

```
GATE N passed for <scope>. Evidence: <plan-file row pointer>.
```

or

```
GATE N FAILED for <scope>. failure_category: <cat>.
Failing file: <path>. Evidence: <plan-file row + log excerpt>.
Dispatch @debug.
```

`failure_category` values: `precision`, `aicore`, `host_crash`,
`workspace_overlap`, `oom`, `structure`, `layout`, `other`.

**Forbidden:** loading `debugging/*` skills, editing kernel code, bisecting,
retrying checks without a prior @debug→@coding cycle.

---

## 8. Debug Agent (Specialist investigator)

**Responsibility:** Investigate the root cause of a specific GATE failure
on one staged file, propose a concrete patch, and hand the proposal back to
Lead so @coding can apply it. Loads exactly one `debugging/*` sub-skill per
invocation, matched to the `failure_category` reported by Verification.

| Role | Skill | Notes |
|------|-------|-------|
| Router | `debugging/debugging` | Decision tree → loads exactly ONE sub-skill |
| Active | `debugging/debugging/DEBUG.md` § lookup | §9 quick table for common PyPTO pitfalls |
| Active (on demand) | one `debugging/<sub>` | loaded per `failure_category`, unloaded before switching |

**Router dispatch table** (keyed off Verification's `failure_category`):

| `failure_category` | Sub-skill loaded on demand |
|---|---|
| `precision` | `debugging/pypto-precision-debug` → escalate to `debugging/pypto-precision-compare` for bisection |
| `aicore` | `debugging/pypto-aicore-error-locator` |
| `host_crash` | `debugging/pypto-host-stacktrace-analyzer` |
| `workspace_overlap` | `debugging/pypto-memory-overlap-detector` |
| `oom` | `debugging/pypto-machine-workspace` |
| `tile_shape` | `debugging/pypto-tile-shape-debug` (L0/L1 exceeded, tile align, `set_cube_tile_shapes` misuse, `enable_split_k`) |
| `structure` / `layout` | `debugging/debugging` + `DEBUG.md` §9 alone |
| `other` | `debugging/debugging` → escalate per `SKILL.md` decision tree |

Only one sub-skill is loaded per failure; it is unloaded before moving on.
Worst case active-skill count: **3**.

**Deliverable per invocation:** a patch proposal logged to
`custom/plan/<op>.md` → Development & debug log with (a) file + line range,
(b) current snippet, (c) proposed snippet, (d) expected effect on the
failing Verification check. Debug Agent does **not** modify production
kernel code directly; it may create diagnostic scratch files under
`custom/<op>/_debug/`.

**Iteration cap:** 3 full fix/re-verify cycles per module. If the same
module fails a 4th time, stop and surface the blocker to Lead with all
evidence.

---

## 9. Optimization Agent

**Responsibility:** Run Phase 6 optimization after correctness is frozen.
Coordinate tightly with Verification Agent for regression safety.

| Role | Skill | Notes |
|------|-------|-------|
| Active | `workflow/phase6-optimization` | Phase 6 procedure, rollback rules |
| Router | `performance/pypto-op-perf-tune` | 3-stage orchestrator with stage gating |
| Active | `performance/perf-analyzer` | Metrics extraction, bottleneck identification |

**Router dispatch table** (see `performance/pypto-op-perf-tune/SKILL.md`
"Stage Gating (Router Policy)"):

| Stage entered | Sub-skill loaded on demand | Unload when |
|---------------|----------------------------|-------------|
| Stage 1 (frontend) | `performance/tune-frontend` | Stage 1 exits |
| Stage 2 (swimlane) | `performance/tune-swimlane` | Stage 2 exits |
| Stage 3 (incore) | `performance/tune-incore` | Stage 3 exits |
| Any stage needing automation | `performance/pypto-operator-auto-tuner` | Automation task done |

Stage N+1 must not be entered before stage N exits cleanly. Worst case
active-skill count: **4** (only one `tune-*` sub-skill active at a time).

**Activation precondition:** GATE 4 passed (E2E `all_close: true`, layout
check exit 0). Optimization Agent is dormant until then.

**Regression loop with Verification Agent:**

```
Optimization Agent: apply change N
  ↓
Verification Agent:
  (1) detailed_tensor_compare → all_close?
  (2) layout check            → exit 0?
  (3) perf-analyzer           → delta vs baseline
  → write to Verification Report
  ↓
Optimization Agent:
  - regression       → rollback, log in plan, try next idea
  - no gain          → log, try next idea
  - gain + no regression → adopt, move on
  - target reached   → stop, hand back to Lead
```

---

## Skill-to-agent inverted index

Use this to check the "2–5 active" invariant at a glance.

**Root control documents (Lead Agent only):**

| Document | Purpose |
|----------|---------|
| `principles.md` | Behavioral guidelines |
| `agent-plan.md` | Phase checklist / gates |
| `agents.md` | Team roster / dispatch |
| `rules.md` | Mandatory rules |
| `catalog.yaml` | Tier-1 routing index |

**Skills under `skills/` and their active assignments:**

| Skill | Active for agents |
|-------|-------------------|
| `workflow/plan-template` | Planning, Design |
| `workflow/kernel-code-format` | Algorithm, Architecture, Design, Coding |
| `workflow/validation-and-deliverables` | Verification |
| `workflow/phase0-phase1-planning` | Planning, Algorithm |
| `workflow/phase2-phase3-construction` | Design, Coding |
| `workflow/phase4-phase5-integration` | Coding |
| `workflow/phase6-optimization` | Optimization |
| `development/pypto-intent-understand` | Planning |
| `development/pypto-api-explore` | Planning |
| `development/pypto-golden-generate` | Algorithm |
| `development/pypto-op-design` | Architecture, Design |
| `development/pypto-op-develop` | Coding |
| `debugging/debugging` | Debug (router) |
| `performance/pypto-op-perf-tune` | Architecture (reference-only), Optimization (router) |
| `performance/perf-analyzer` | Optimization |
| `ci-and-pr/ci-and-layout-check` | Verification |

`development/pypto-op-workflow` is referenced by Lead as a dispatch guide
but is **not** loaded into Lead's active set; it lives under `skills/` and
is consulted on-demand. Every other skill in the library is **dormant by
default** and loaded only through the router skills above or by the
explicit conditions in each agent's table.

---

## Anti-patterns

1. **Do not hand a debug sub-skill directly to the Coding Agent.** Route
   through Verification (judge) → Debug (investigate) → Coding (apply).
   Coding Agent's job is to write code, not diagnose failures.
2. **Do not let Verification investigate or fix.** Verification is a judge.
   The moment a `debugging/*` skill is needed, Lead must dispatch the Debug
   Agent instead.
3. **Do not let Debug modify production kernel code.** Debug writes patch
   proposals into the plan file; Coding applies them. This preserves the
   audit trail and stops Debug from accidentally advancing the module.
4. **Do not load Optimization skills before GATE 4.** Phase 6 is gated on
   correctness; pre-loading `tune-*` skills causes premature optimization
   and silent correctness regressions.
5. **Do not expand any agent past 5 active skills.** If a new workflow needs
   more, introduce a new router skill or split into two agents.
6. **Do not pre-load all post-dev `ci-and-pr/*` skills on the Lead Agent.**
   Use the dormant dispatch table above — each post-dev skill activates on a
   specific condition (PR requested, CI failed, etc.).
7. **Do not skip the shared plan file.** `custom/plan/<op>.md` is the single
   source of truth between agents; every handoff is a plan update, not a
   direct message.
