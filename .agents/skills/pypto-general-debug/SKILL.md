---
name: pypto-general-debug
description: Overview of the general debug skill — failure history, strategy switching, op-by-op protocol, and the full debug playbook (references/debug-playbook.md).
---

# PyPTO Complex Kernel — Debugging

This skill covers what to do when the agent is stuck. For the full error playbook and agent-learned patterns, read **`references/debug-playbook.md`**.

## Contents at a glance

| Document | What it covers |
|----------|---------------|
| **This file (SKILL.md)** | Failure history logging, strategy switching rules, op-by-op check protocol |
| **`references/debug-playbook.md` §1–§7** | Opaque error playbook: `FFFFF`, `UNKNOWN`, `F21004`, AICore, error-code quick reference, stop conditions |
| **`references/debug-playbook.md` §8** | Example kernels debug practice (global patterns, recurring failures, integration tracks) |
| **`references/debug-playbook.md` §9** | Agent-learned dev patterns: JIT signatures, dynamic shapes, `pypto.view`, SIM mode, matmul API, tile shapes, reduction alignment, and more |

---

## Failure History

Every iteration must log:
- hypothesis,
- exact changed location,
- result,
- next action,
- whether rollback occurred.

This is mandatory.

---

## Strategy Switching Rules

These rules are mandatory.

- repeated compile error → run `validate_kernel_structure` first; re-check interface, shapes, dtypes, kernel structure
- repeated runtime error → run `diagnose_error(error_log=..., kernel_code=...)` first; if no match, switch to device/runtime error localization (`pypto-aicore-error-locator`)
- repeated accuracy mismatch → run `validate_kernel_structure` to rule out write-back bugs; then switch to binary-search checkpoint debugging (`pypto-binary-search-verify`)
- repeated same failure after three evidence-based attempts → revisit module boundaries or architecture
- integrated graph memory-conflict / copy-pass failure → stop assuming a local line fix; use staged fallback or create a minimal repro
- opaque PyPTO error after the above → run `extract_pypto_calls.py`, then follow the op-by-op check protocol below

---

## PyPTO op-by-op check protocol (when the agent is stuck)

**Goal:** When hitting PyPTO-specific failures not resolved by `validate_kernel_structure`, `diagnose_error`, or docs, use this mechanical sequence so debugging converges.

### Step 0 — Enumerate every `pypto` call (mandatory checklist)

Run on the failing kernel file:

```bash
python3 .agents/skills/pypto-kernel-layout-check/scripts/extract_pypto_calls.py custom/<operator_name>/<kernel_or_impl>.py
```

- Output is an ordered, numbered list: line number + call shape.
- Use this list as the single source of truth for "which PyPTO op comes next."
- Optional: `--json` for machine consumption or to paste into the plan.

### Step 1 — Verify each call site against documentation (in order)

For call sites in the suspected region (or from index 1 upward if the fault is unknown):

1. Map `pypto.<name>` → `docs/api/operation/pypto-<name>.md` or `docs/api/config/...`.
2. Confirm dtype, shape/axes, tile config, transpose flags, and write-back rules match the doc for that line.
3. Record mismatches in the plan with line number from Step 0.

### Step 2 — Wrong numeric result: checkpoint bisection (not random edits)

If the graph runs but outputs are wrong:
- Use `pypto-precision-compare` / checkpoint saves so golden and kernel dump tensors at aligned logical points.
- Binary-search which checkpoint index diverges first; map that index back to the call-site range from Step 0.

### Step 3 — Do **not** "comment out half the file"

Disabling arbitrary `pypto` lines inside one fused `@jit` usually invalidates the graph or hides the real bug.

- Prefer module-at-a-time stubs (see `skills/pypto-decompose-construct/SKILL.md` → Phase 3 hard rule on staged sets): shrink the live region, then re-run Step 0 on the smaller file.
- If you must bisect inside one module, insert one intermediate checkpoint between call sites k and k+1 and binary-search k using the numbered list — do not remove ops unless the minimal repro requires it.

### Step 4 — Plan file log (handoff-safe)

Append to `custom/<op>/MEMORY.md`:
- Path to `extract_pypto_calls.py` output (or paste the table),
- First doc mismatch or first diverging checkpoint index,
- Hypothesis and patch; re-run validation.

This protocol is compatible with fully autonomous runs: the agent applies it without waiting for the user when stuck, unless stop conditions apply.

---

## Before writing PyPTO code — consult debug-playbook.md §9

Read the matching subsection before writing each module's PyPTO code:

| What you are about to write | Read first |
|------------------------------|------------|
| Any `@pypto.frontend.jit` function | `references/debug-playbook.md` §9.1 (`from __future__ import annotations` breaks JIT) |
| `pypto.view` / `pypto.assemble` | `references/debug-playbook.md` §9.4 (golden rule: `len(shape)==len(offsets)`, padding, reshape) |
| `pypto.matmul` | `references/debug-playbook.md` §9.19 (transpose flags `a_trans`/`b_trans`, NOT `.T`; cube+vec tiles required) |
| `.sum()` / reduction ops | `references/debug-playbook.md` §9.19 (32-byte alignment; matmul-based workaround) |
| Dynamic shapes / `pypto.loop` | `references/debug-playbook.md` §9.2 (concrete loop bounds, symbolic offsets) |
| Tensor type hints in JIT signature | `references/debug-playbook.md` §9.13 (use `pypto.Tensor([], dtype)`, not explicit `DYNAMIC` dims) |
| Element-wise ops inside JIT | `references/debug-playbook.md` §9.14 (Python `*`, `+`, `.exp()` work; prefer over verbose `pypto.mul`) |
| Tile shape configuration | `references/debug-playbook.md` §9.15 + §9.19 (vec+cube both needed for matmul; ≥4 vec args) |
| Any error during development | `references/debug-playbook.md` §9.11 (common error → cause → solution quick table) |

---

## Subskill fallback decision tree (router policy)

This skill is the **router** for all debugging sub-skills. When used by the
Debug Agent (`.opencode/agents/pypto-op-debugger.md`), load **exactly one** sub-skill per
failure, and unload it before handling the next failure. This keeps the
active-skill count ≤ 4.

**Dispatch order — evaluate top to bottom; stop at the first match:**

1. **Precision / accuracy mismatch**
   *Signal:* `detailed_tensor_compare` returns `all_close: false`; build and
   layout succeed; no crash, no aicore error in logs.
   → Load `skills/pypto-precision-debug/SKILL.md`
   (code-level workarounds: inplace, unroll, `+0.0`).

2. **Precision bisection needed**
   *Signal:* precision still fails after (1), or the diverging checkpoint is
   not yet localized; multiple modules in scope.
   → Load `skills/pypto-precision-compare/SKILL.md`
   (`pass_verify_save` or checkpoint tensors; binary-search the first
   diverging checkpoint index).

3. **aicore error / CCE file reported**
   *Signal:* log contains `aicore error`, `ERROR_CODE: EE...`, or a path to
   a CCE file.
   → Load `skills/pypto-aicore-error-locator/SKILL.md`
   (map error → CCE file → offending source line).

4. **Host-side crash / backtrace**
   *Signal:* segfault, Python/C++ stack trace, process killed before kernel
   launch finishes.
   → Load `skills/pypto-host-stacktrace-analyzer/SKILL.md`
   (address-to-source mapping, common host crash patterns).

5. **Workspace overlap suspected**
   *Signal:* non-deterministic precision failures that move with tensor
   layout; passes on isolated module, fails only in integrated graph.
   → Load `skills/pypto-memory-overlap-detector/SKILL.md`.

6. **OOM / workspace size anomaly**
   *Signal:* `rtMalloc failed`, OOM error, or workspace size far exceeds
   expectation.
   → Load `skills/pypto-machine-workspace/SKILL.md`.

**Router rules (mandatory):**

- Do **not** load more than one sub-skill at once. If a new failure class
  appears, unload the current sub-skill first.
- Do **not** pre-load sub-skills speculatively.
- If no row matches, stay in this SKILL.md + `references/debug-playbook.md`. Do not escalate.
- The Debug Agent's contract (`.opencode/agents/pypto-op-debugger.md`) takes precedence over any sub-skill guidance on conflict.
- Log the dispatch decision to `custom/<op>/MEMORY.md` under **Development &
  debug log** (which row matched, which sub-skill was loaded, outcome).
