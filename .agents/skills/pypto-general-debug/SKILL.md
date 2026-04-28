---
name: pypto-general-debug
description: Overview of the general debug skill — failure history, strategy switching, op-by-op protocol, and routed access to topic-specific debug references.
---

# PyPTO Complex Kernel — Debugging

This skill covers what to do when the agent is stuck. The full playbook is split into focused topic files under `references/`. **Pick the file that matches the current failure mode — do NOT read every reference file.** The full section→file map (preserved old `§X.Y` numbering for any external references) lives in `references/DEBUG_GUIDEBOOK.md`.

## Contents at a glance

| Document | What it covers |
|----------|---------------|
| **This file (SKILL.md)** | Failure history logging, strategy switching rules, op-by-op check protocol, sub-skill routing |
| **`references/DEBUG_GUIDEBOOK.md`** | Index — maps every old `§X.Y` section number to the new leaf file. Open this if you have an external reference like `DEBUG_GUIDEBOOK.md §9.4`. |
| **`references/error-codes.md`** | Opaque error playbook (`FFFFF`, `UNKNOWN`, `F21004`, AICore, stop conditions) — old §1–§7 |
| **`references/examples-debug.md`** | Example kernels debug practice — old §8 |
| **`references/jit-signature.md`** | JIT signature & `pypto.Tensor([], dtype)` — old §9.1 + §9.13 |
| **`references/dynamic-shapes.md`** | Dynamic shapes / `pypto.loop` / symbolic indexing — old §9.2 |
| **`references/pypto-view.md`** | Comprehensive `pypto.view` guide — old §9.4 |
| **`references/sim-mode.md`** | SIM mode limitations — old §9.6 |
| **`references/tensor-ops.md`** | Tensor ops & transpose — old §9.7 + §9.16 |
| **`references/common-patterns.md`** | Common patterns, quick reference, takeaways — old §9.8 + §9.17 + §9.18 |
| **`references/checklist-and-api.md`** | Debug checklist, testing strategy, error table, API notes — old §9.9–§9.12 |
| **`references/python-operators.md`** | Python operators inside JIT (`*`, `+`, `.exp()`) — old §9.14 |
| **`references/tile-shapes.md`** | Tile shape configuration — old §9.15 |
| **`references/matmul.md`** | matmul API, transpose flags, vec+cube tiles, assemble, reduction alignment, K-dim mismatch, 5D views — old §9.19 |
| **`references/npu-launch-failures.md`** | NPU launch failures from delivered ops — old §9.20 |

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
- Optional: `--json` for machine consumption or to paste into the memory.

### Step 1 — Verify each call site against documentation (in order)

For call sites in the suspected region (or from index 1 upward if the fault is unknown):

1. Map `pypto.<name>` → `docs/api/operation/pypto-<name>.md` or `docs/api/config/...`.
2. Confirm dtype, shape/axes, tile config, transpose flags, and write-back rules match the doc for that line.
3. Record mismatches in the memory with line number from Step 0.

### Step 2 — Wrong numeric result: checkpoint bisection (not random edits)

If the graph runs but outputs are wrong:
- Use `pypto-precision-compare` / checkpoint saves so golden and kernel dump tensors at aligned logical points.
- Binary-search which checkpoint index diverges first; map that index back to the call-site range from Step 0.

### Step 3 — Do **not** "comment out half the file"

Disabling arbitrary `pypto` lines inside one fused `@jit` usually invalidates the graph or hides the real bug.

- Prefer module-at-a-time stubs (see `skills/pypto-decompose-construct/SKILL.md` → Phase 3 hard rule on staged sets): shrink the live region, then re-run Step 0 on the smaller file.
- If you must bisect inside one module, insert one intermediate checkpoint between call sites k and k+1 and binary-search k using the numbered list — do not remove ops unless the minimal repro requires it.

### Step 4 — Plan file log (handoff-safe)

Append to `custom/<operator_name>/MEMORY.md`:
- Path to `extract_pypto_calls.py` output (or paste the table),
- First doc mismatch or first diverging checkpoint index,
- Hypothesis and patch; re-run validation.

This protocol is compatible with fully autonomous runs: the agent applies it without waiting for the user when stuck, unless stop conditions apply.

---

## Before writing PyPTO code — consult the matching reference file

Open only the leaf file that matches what you are about to write. Each leaf is small (≈30–300 lines).

| What you are about to write | Read first |
|------------------------------|------------|
| Any `@pypto.frontend.jit` function | `references/jit-signature.md` (`from __future__ import annotations` breaks JIT) |
| `pypto.view` / `pypto.assemble` | `references/pypto-view.md` (golden rule: `len(shape)==len(offsets)`, padding, reshape) |
| `pypto.matmul` | `references/matmul.md` (transpose flags `a_trans`/`b_trans`, NOT `.T`; cube+vec tiles required) |
| `.sum()` / reduction ops | `references/matmul.md` (32-byte alignment; matmul-based workaround) |
| Dynamic shapes / `pypto.loop` | `references/dynamic-shapes.md` (concrete loop bounds, symbolic offsets) |
| Tensor type hints in JIT signature | `references/jit-signature.md` (use `pypto.Tensor([], dtype)`, not explicit `DYNAMIC` dims) |
| Element-wise ops inside JIT | `references/python-operators.md` (Python `*`, `+`, `.exp()` work; prefer over verbose `pypto.mul`) |
| Tile shape configuration | `references/tile-shapes.md` + `references/matmul.md` (vec+cube both needed for matmul; ≥4 vec args) |
| Any error during development | `references/checklist-and-api.md` §9.11 (common error → cause → solution quick table) |

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
- If no row matches, stay in this SKILL.md + the matching `references/*.md` leaf. Do not escalate.
- The Debug Agent's contract (`.opencode/agents/pypto-op-debugger.md`) takes precedence over any sub-skill guidance on conflict.
- Log the dispatch decision to `custom/<op>/MEMORY.md` under **Development &
  debug log** (which row matched, which sub-skill was loaded, outcome).
