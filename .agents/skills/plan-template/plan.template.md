# Plan: `<operator_name>`

Copy to: `custom/plan/<operator_name>.md` and keep **machine-readable** fields current every turn.

**Agent:** Do **not** skip **`skills/lead-orchestrator/references/rules.md`** or sub-skill obligations (staged files, all-output compare, plan logs). See **`.agents/skills/lead-orchestrator/references/rules.md`** — *Zero tolerance*. **Code layout:** every staged file and the full kernel **must** follow **`.agents/skills/kernel-code-format/pypto_kernel_template.py`** (layers A–L); document exceptions here. After **`custom/`** changes, run **`bash .agents/skills/ci-and-layout-check/run_validate_layout.sh`** (see **`skills/ci-and-layout-check/CI.md`**).

## Agent status (minimal)

```yaml
phase: 0|1|2|3|4|5|6
active_module: M1   # or M2, …, none
current_staged_file: custom/<operator_name>/<operator_name>_module1.py   # update each milestone
modules_pypto_verified:
  - id: M1
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

- **Runner:** `custom/<operator_name>/test_<operator_name>.py`
- **Command (repo root):** `PYTHONPATH=.agents/skills/validation-and-deliverables python custom/<operator_name>/test_<operator_name>.py`
- **Comparison:** `from detailed_tensor_compare import detailed_tensor_compare` (bundled: `.agents/skills/validation-and-deliverables/detailed_tensor_compare.py`). **Do not** use `pytest` as the default for this golden vs PyPTO check unless documented under **blockers** as an exception.
- **All outputs:** the runner must call **`detailed_tensor_compare`** on **every** leaf output tensor (tuple/list/dict/nested structures — **not** only `outputs[0]`). If any output is intentionally skipped, document under **blockers** with justification.

## Module decomposition (mandatory)

Document **how** the kernel is split into semantic modules and **why** (not “equal-sized chunks”). Keep this aligned with **Module contracts** below.

### Modules (overview)

| ID | One-line role | Boundary tensors (golden checkpoint names) | Depends on |
|----|---------------|---------------------------------------------|------------|
| M1 | … | … | — |
| M2 | … | … | M1 |

### Rationale

- **Semantic boundaries:** (e.g. matmul vs norm vs recurrence — what meaning each block carries)
- **Why this order:** (dependency / debuggability)
- **Alternatives considered and rejected:** (optional; e.g. “single fused block rejected until boundaries stable”)

## Staged module files (mandatory)

Development progress is **materialized as Python files** under **`custom/<operator_name>/`**. Suffix after `_module` is **concatenated module indices** (`1` → M1 only, `12` → M1+M2 in one JIT, `1234` → M1–M4 for N=4). Each file contains **golden + PyPTO** for that cumulative scope in **one** `@jit`, and must pass **`detailed_tensor_compare`** on **all** outputs before creating the next file. The **last** row’s file is the **full** end-to-end PyPTO kernel.

| Staged file | Modules in one `@jit` | Golden + PyPTO verified (all outputs) |
|-------------|------------------------|--------------------------------------|
| `<operator_name>_module1.py` | M1 | ☐ |
| `<operator_name>_module12.py` | M1, M2 | ☐ |
| `<operator_name>_module123.py` | M1, M2, M3 | ☐ |
| `<operator_name>_module1234.py` | M1–M4 (example N=4) | ☐ — **full kernel** |

*Add or remove rows to match **N** modules. Command example for a stage:*  
`PYTHONPATH=.agents/skills/validation-and-deliverables python custom/<operator_name>/<operator_name>_module12.py`

## Per-module verification log (mandatory)

Every time a **module boundary** is validated (Phase 3: golden vs PyPTO at that checkpoint), **append** a row. Comparisons **must** use **`detailed_tensor_compare`** (same helper as E2E). Record enough that a reviewer can see **pass/fail** without re-running.

| When | Module | Staged file (e.g. `<op>_module12.py`) | Tensors compared (name / role) | rtol | atol | `all_close` | Key stats (e.g. `out_of_tolerance_ratio`, `max_diff` from return dict) | Command or script |
|------|--------|----------------------------------------|-------------------------------|------|------|-------------|------------------------------------------------------------------------|-------------------|
| | M1 | `<operator_name>_module1.py` | | 1e-3 | 1e-3 | | | |

*On failure:* add a **Development & debug log** entry and keep the failed row or add a follow-up row after fix.

## Vector tile config

- **`pypto.set_vec_tile_shapes`:** use tile dimensions as required by **`docs/api/config/pypto-set_vec_tile_shapes.md`** — see **`skills/pypto-op-develop/SKILL.md`** §实现注意点 #7 / 常见错误 #4.

## API map (Phase 0)

| Formula step | PyPTO | supported / substitute / unsupported |

## Golden function inventory (Phase 1 — cross-check)

List **every operation** in the PyPTO-friendly golden. In Phase 3/4, cross-check line-by-line against the PyPTO implementation and mark ✅/❌. **Do not run tests or advance modules while any ❌ remains.**

| # | Golden operation | Shape transformation | PyPTO implementation | Line | Status |
|---|------------------|----------------------|------------------------|------|--------|
| 1 | `torch.matmul(q, k^T)` | `[B,H,T,K]@[B,H,K,T]->[B,H,T,T]` | `pypto.matmul(q, k, dtype, b_trans=True)` | L.42 | ✅ |
| 2 | `torch.softmax(scores, -1)` | `[B,H,T,T]->[B,H,T,T]` | | | ❌ |
| … | | | | | |

**When to cross-check:**
- Phase 3 (GATE 3): every row in the current module’s scope must be ✅
- Phase 4 (GATE 4): every row must be ✅

## Module contracts (Phase 2)

| Module | Inputs | Outputs | PyPTO APIs | Verified? |

## Design format compliance

- **Template:** `.agents/skills/kernel-code-format/pypto_kernel_template.py` — **mandatory** skeleton for each `*_module*.py` and the integrated kernel (same layers; trim unused sections only with justification below).
- Layers A–L from `docs/pypto-kernel-design-format.md` (copy in this bundle: `.agents/skills/kernel-code-format/pypto-kernel-design-format.md`): which apply, which omitted, why.

## PyPTO call-site checklist (when debugging)

Paste output of:

`python3 .agents/skills/ci-and-layout-check/scripts/extract_pypto_calls.py custom/<operator_name>/<operator_name>_module1234.py` *(or current staged / final kernel file)*

| # | Line | Call | doc OK? | notes |

## `skills/debugging/DEBUG.md` §9 — pre-write checklist

Before writing each module's PyPTO code, review the matching subsections from **`.agents/skills/debugging/DEBUG.md` §9**. Check off after reading. See full lookup table in **`skills/phase2-phase3-construction/SKILL.md`** → **Phase 3 → Before writing PyPTO code**.

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

**Do not** abandon the run on these alone. Follow **`.agents/skills/debugging/DEBUG.md`** §1–§8, capture full logs, and iterate (token budget is not a limit). When debugging, also consult **§9.11** (common error → cause → solution quick table). Log each attempt below.

## Development & debug log

| When | Action | Result | Next |

## Human review milestones (optional)

MS1 … MS7 as in full workflow — or link to milestones in the relevant sub-skill under `skills/`.
