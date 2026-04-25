# Plan: `<op>`

Copy to: `custom/<op>/plan.md` and keep **machine-readable** fields current every turn.

**Agent:** This plan is the operator's coordination file. Do **not** skip sub-skill obligations (staged sets, all-output compare, plan logs). **Code layout:** complex kernels follow the multi-file template under `.agents/skills/pypto-op-develop/templates/complex-kernel-template/` (impl + test) plus the layer rules in `.agents/skills/pypto-op-develop/references/kernel-layer-format.md`. Document exceptions here. After `custom/` changes, run `bash .agents/skills/pypto-kernel-layout-check/run_validate_layout.sh`.

## Agent status (minimal)

```yaml
phase: 0|1|2|3|4|5|6
active_module: M1   # or M2, …, none
current_staged_set:
  impl:   custom/<op>/staged/<op>_module1_impl.py
  golden: custom/<op>/staged/<op>_module1_golden.py
  test:   custom/<op>/staged/test_<op>_module1.py
modules_pypto_verified:
  - id: M1
    suffix: "1"
    evidence: "<command or pointer to Per-module verification log row>"
    detailed_tensor_compare_ok: true   # false until boundary passes
next_mandatory_step: "<one concrete step>"
correctness: not_started | golden_ok | sim_ok | npu_ok
optimization: not_started | … | complete | skipped_user_request
blockers: []
```

## Task summary

- **Operator:**
- **I/O shapes (symbolic):**
- **Reference / golden path:**

## Validation (mandatory)

- **Per-staged-set runner:** `python custom/<op>/staged/test_<op>_module<suffix_k>.py` (after each Coding dispatch)
- **End-to-end runner (after Phase D):** `python custom/<op>/test_<op>.py`
- **Comparison helper:** `from detailed_tensor_compare import detailed_tensor_compare` (bundled: `.agents/skills/pypto-op-validate/detailed_tensor_compare.py`). **Do not** use `pytest` as the default for golden vs PyPTO checks unless documented under **blockers** as an exception.
- **All outputs:** the runner must call `detailed_tensor_compare` on **every** leaf output tensor (tuple/list/dict/nested structures — **not** only `outputs[0]`). If any output is intentionally skipped, document under **blockers** with justification.

## Module decomposition (mandatory)

Document **how** the kernel is split into semantic modules and **why** (not "equal-sized chunks"). Keep this aligned with **Module contracts** below.

### Modules (overview)

| ID | One-line role | Boundary tensors (golden checkpoint names) | Depends on |
|----|---------------|---------------------------------------------|------------|
| M1 | … | … | — |
| M2 | … | … | M1 |

### Rationale

- **Semantic boundaries:** (e.g. matmul vs norm vs recurrence — what meaning each block carries)
- **Why this order:** (dependency / debuggability)
- **Alternatives considered and rejected:** (optional; e.g. "single fused block rejected until boundaries stable")

## Staged set table (mandatory)

Development progress is **materialized as 3-file staged sets** under `custom/<op>/staged/`. Suffix after `_module` is **concatenated module indices** (`1` → M1 only, `12` → M1+M2 cumulative, `1234` → M1–M4 for N=4). Each set must pass `detailed_tensor_compare` on **all** outputs before creating the next set. The **last** row's set is renamed to canonical top-level files in Phase D.

| Suffix | impl | golden | test | All outputs verified |
|--------|------|--------|------|----------------------|
| `1` | `staged/<op>_module1_impl.py` | `staged/<op>_module1_golden.py` | `staged/test_<op>_module1.py` | ☐ |
| `12` | `staged/<op>_module12_impl.py` | `staged/<op>_module12_golden.py` | `staged/test_<op>_module12.py` | ☐ |
| `123` | `staged/<op>_module123_impl.py` | `staged/<op>_module123_golden.py` | `staged/test_<op>_module123.py` | ☐ |
| `1234` | `staged/<op>_module1234_impl.py` | `staged/<op>_module1234_golden.py` | `staged/test_<op>_module1234.py` | ☐ — **full kernel** |

*Add or remove rows to match **N** modules. Per-set test command:*
`python custom/<op>/staged/test_<op>_module12.py`

After GATE 4, the M_N row's 3 files are renamed to canonical top-level (see "Phase D canonical rename" section).

## Per-module verification log (mandatory)

Every time a **module boundary** is validated (Phase 3: golden vs PyPTO at that checkpoint), **append** a row. Comparisons **must** use `detailed_tensor_compare` (same helper as E2E). Record enough that a reviewer can see **pass/fail** without re-running.

| When | Module | Suffix | Tensors compared (name / role) | rtol | atol | `all_close` | Key stats (`out_of_tolerance_ratio`, `max_diff`) | Command |
|------|--------|--------|-------------------------------|------|------|-------------|--------------------------------------------------|---------|
| | M1 | `1` | | 1e-3 | 1e-3 | | | `python custom/<op>/staged/test_<op>_module1.py` |

*On failure:* add a **Development & debug log** entry and keep the failed row or add a follow-up row after fix.

## Vector tile config

- `pypto.set_vec_tile_shapes`: use tile dimensions as required by `docs/api/config/pypto-set_vec_tile_shapes.md` — see `skills/pypto-op-develop/SKILL.md` §实现注意点 #7 / 常见错误 #4.

## API map (Phase 0)

| Formula step | PyPTO | supported / substitute / unsupported |

## Golden function inventory (Phase 1 — cross-check)

List **every operation** in the PyPTO-friendly golden. In Phase 3/4, cross-check line-by-line against the staged `*_impl.py` and mark ✅/❌. **Do not run tests or advance modules while any ❌ remains.**

| # | Golden operation | Shape transformation | PyPTO implementation (file:line) | Status |
|---|------------------|----------------------|----------------------------------|--------|
| 1 | `torch.matmul(q, k^T)` | `[B,H,T,K]@[B,H,K,T]->[B,H,T,T]` | `staged/<op>_module1_impl.py:42` | ✅ |
| 2 | `torch.softmax(scores, -1)` | `[B,H,T,T]->[B,H,T,T]` | | ❌ |
| … | | | | |

**When to cross-check:**
- Phase 3 (GATE 3): every row in the current staged set's scope must be ✅
- Phase 4 (GATE 4): every row must be ✅

## Module contracts (Phase 2)

| Module | Inputs | Outputs | PyPTO APIs | Verified? |

## Layer format compliance

- **Templates:**
  - golden (Layers B–F): `.agents/skills/pypto-golden-generate/templates/golden-template.py`
  - impl (Layers G–K): `.agents/skills/pypto-op-develop/templates/complex-kernel-template/impl-template.py`
  - test (Layer L): `.agents/skills/pypto-op-develop/templates/complex-kernel-template/test-template.py`
- **Layer rules:** `.agents/skills/pypto-op-develop/references/kernel-layer-format.md`
- Which layers apply, which omitted, why:

## `skills/debugging/DEBUG.md` §9 — pre-write checklist

Before writing each module's PyPTO code, review the matching subsections from `.agents/skills/debugging/DEBUG.md` §9. Check off after reading. See full lookup table in `skills/pypto-decompose-construct/SKILL.md` → **Phase 3 → Before writing PyPTO code**.

| Subsection | Applies to this kernel? | Reviewed? |
|------------|------------------------|-----------|
| §9.1 JIT signature (`from __future__` ban) | ☐ yes / ☐ no | ☐ |
| §9.2 Dynamic shapes, symbolic loop bounds | ☐ yes / ☐ no | ☐ |
| §9.4 `pypto.view` / `pypto.assemble` guide | ☐ yes / ☐ no | ☐ |
| §9.13 Tensor shape specs (`[]` vs `DYNAMIC`) | ☐ yes / ☐ no | ☐ |
| §9.14 Python operators inside JIT | ☐ yes / ☐ no | ☐ |
| §9.15 Tile shape configuration | ☐ yes / ☐ no | ☐ |
| §9.19 matmul API / reduction / assemble | ☐ yes / ☐ no | ☐ |
| §9.11 Common error quick table | ☐ yes / ☐ no | ☐ |

## Opaque error codes (FFFFF, UNKNOWN, Errcode: F…!)

**Do not** abandon the run on these alone. Follow `.agents/skills/debugging/DEBUG.md` §1–§8, capture full logs, and iterate. When debugging, also consult §9.11 (common error → cause → solution quick table). Log each attempt below.

## Development & debug log

| When | Action | Result | Next |

## Phase D canonical rename log (after GATE 4)

Record the rename commands the Verification Agent ran to produce canonical top-level files:

| When | Command | Result |
|------|---------|--------|
| | `git mv custom/<op>/staged/<op>_module1...N_impl.py custom/<op>/<op>_impl.py` | |
| | `git mv custom/<op>/staged/<op>_module1...N_golden.py custom/<op>/<op>_golden.py` | |
| | `git mv custom/<op>/staged/test_<op>_module1...N.py custom/<op>/test_<op>.py` | |
| | (update imports in `test_<op>.py`) | |
| | (generate `README.md`) | |

After Phase D, `staged/` is **retained** for traceability — never delete.

## Human review milestones (optional)

MS1 … MS7 as in full workflow — or link to milestones in the relevant sub-skill under `skills/`.
