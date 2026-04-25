---
name: pypto-kernel-phase2-phase3
description: Phase 2 (semantic module decomposition — split by meaning, define contracts, freeze) and Phase 3 (module construction — one staged set at a time, validate, cross-check golden inventory).
---

# PyPTO Complex Kernel — Phase 2–3: Decomposition and Construction

## Phase 2: Semantic Module Decomposition

Goal: split the kernel into semantically meaningful, verifiable blocks.

### Write decomposition into the plan (mandatory)

As soon as the module split is known, write Module decomposition in `custom/<op>/plan.md`: named modules, boundary tensors, and rationale. See `skills/plan-template/plan.template.md` for log format.

### Rule: split by meaning, not by equal complexity

Prefer boundaries such as:
- matmul block,
- normalization / softmax block,
- recurrent state update block,
- reduction block,
- writeback / assemble block,
- layout conversion block,
- decay / gating block,
- per-step recurrence block.

Avoid:
- arbitrary equal-sized chunks,
- splitting a single semantic operation across modules,
- combining unrelated layout and compute transforms into one module,
- modules that cannot be verified independently.

### For each module, define a contract

Before writing any code for a module, look up the exact PyPTO API signatures:

```
query_op(names=["<op1>", "<op2>"])
```

CLI fallback:
```bash
python3 .agents/skills/pypto-api-explore/scripts/query_op_index.py --op <op1> --op <op2>
```

For constraint details not captured in the index:
```
retrieve_docs(query="<op name> constraints dtype tile shape", chunk_type="api_doc")
```

Every module must specify:
- name, purpose,
- inputs, outputs, shapes, dtypes,
- invariants,
- semantic predecessor and successor,
- whether it contains loop-carried state,
- whether it contains reduction,
- whether it contains alignment-sensitive tensors,
- PyPTO APIs used (with exact signatures from op_index).

### Module freeze rule

Once a module is verified, mark it frozen. Do not edit frozen modules because a later stage fails. First inspect the earliest unfrozen failing boundary.

### Subskill delegation: DESIGN.md generation (optional)

To produce a standalone design document with API mapping, tiling strategy, loop structure, and verification plan, read `skills/pypto-op-design/SKILL.md` and generate `DESIGN.md`. The design document supplements (does not replace) the plan file's module decomposition and contracts.

---

## Phase 3: Module Construction

Goal: build each module in isolation before integration, as a **staged set of 3 files** (impl + golden + test) per dispatch.

**Hard rule:** In each iteration, extend the production kernel by at most one new semantic module's real PyPTO logic. Everything downstream remains stubbed or fed from golden boundary tensors. The Coding Agent is responsible for enforcing one-staged-set-per-dispatch.

### Before writing PyPTO code — consult `skills/debugging/DEBUG.md` §9

Read the relevant subsections before writing each module's PyPTO code:

| What you are about to write | Read first |
|------------------------------|------------|
| Any `@pypto.frontend.jit` function | §9.1 (`from __future__ import annotations` breaks JIT) |
| `pypto.view` / `pypto.assemble` | §9.4 (golden rule: `len(shape)==len(offsets)`, padding, reshape) |
| `pypto.matmul` | §9.19 (transpose flags `a_trans`/`b_trans`, NOT `.T`; cube+vec tiles required) |
| `.sum()` / reduction ops | §9.19 (32-byte alignment; matmul-based workaround) |
| Dynamic shapes / `pypto.loop` | §9.2 (concrete loop bounds, symbolic offsets) |
| Tensor type hints in JIT signature | §9.13 (use `pypto.Tensor([], dtype)`, not explicit `DYNAMIC` dims) |
| Element-wise ops inside JIT | §9.14 (Python `*`, `+`, `.exp()` work; prefer over verbose `pypto.mul`) |
| Tile shape configuration | §9.15 + §9.19 (vec+cube both needed for matmul; ≥4 vec args) |
| Any error during development | §9.11 (common error → cause → solution quick table) |

### Subskill reference: implementation templates and execution constraints

When writing module code, the **canonical reference is `skills/pypto-op-develop/SKILL.md`** (templates, execution-constraints, error code troubleshooting).

For complex kernels (attention-class, recurrent, fused), use the multi-file template set:
- Layers G–K (impl): `skills/pypto-op-develop/templates/complex-kernel-template/impl-template.py`
- Layer L (test): `skills/pypto-op-develop/templates/complex-kernel-template/test-template.py`
- Layers B–F (golden): `skills/pypto-golden-generate/templates/golden-template.py`

Layer organization rules: `skills/pypto-op-develop/references/kernel-layer-format.md`.

The staged-set file convention (3 files per dispatch) and module-at-a-time enforcement override the simple single-file approach for complex kernels.

### Staged sets (mandatory — do this in code)

Materialize each step as a **staged set of 3 files** under `custom/<op>/staged/`:

```
staged/<op>_module<suffix_k>_impl.py     ← cumulative PyPTO implementation
staged/<op>_module<suffix_k>_golden.py   ← cumulative torch reference
staged/test_<op>_module<suffix_k>.py     ← test driver (uses detailed_tensor_compare)
```

Suffix progression: `1` → `12` → `123` → … → `1...N`. Each staged set is the milestone artifact for that cumulative scope.

After GATE 4 passes for the final M_N staged set, the Verification Agent renames the 3 files to canonical top-level names:
- `staged/<op>_module1...N_impl.py` → `custom/<op>/<op>_impl.py`
- `staged/<op>_module1...N_golden.py` → `custom/<op>/<op>_golden.py`
- `staged/test_<op>_module1...N.py` → `custom/<op>/test_<op>.py`

The `staged/` directory is **retained after delivery** for incremental-build traceability.

### Step 1. Express the module as kernel semantics

Design the module to fit into the final production kernel.

Before writing, retrieve the exact signature for every API:
```
query_op(names=["<op_name>"])
```

If alignment or tiling constraints are unclear:
```
retrieve_docs(query="<op_name> tile shape alignment constraint", chunk_type="api_doc")
```

Allowed forms: semantic pseudocode, helper functions, temporary checkpoint logic, optional temporary validation kernels.

Disallowed as default: separate production `@jit` kernel per semantic module. The final M_N staged set contains exactly **one** production `@pypto.frontend.jit` entry.

### Step 2. Build a module-level validation path

Every module must be verifiable before the next module begins.

Validation is provided by the staged set itself: `test_<op>_module<suffix_k>.py` imports both `<op>_module<suffix_k>_golden` and `pypto_function` from `<op>_module<suffix_k>_impl`, then runs `detailed_tensor_compare` at every output boundary.

Use `detailed_tensor_compare` (bundled in `skills/validation-and-deliverables/detailed_tensor_compare.py`) at module boundaries. After each boundary run, append a row to the Per-module verification log in `custom/<op>/plan.md`.

### Step 2b. Cross-check Golden function inventory (mandatory before running)

Before executing the staged set's test for the first time, open `custom/<op>/plan.md` → Golden function inventory and cross-check every operation in this module's scope:

- For each golden operation belonging to the current module, mark ✅ with the PyPTO call and line number (in `<op>_module<suffix_k>_impl.py`), or ❌ if not yet implemented.
- **If any ❌ remains, do not run the test.** Implement the missing operation first.

This step is the primary defense against precision errors caused by forgotten operations.

### Step 3. Validate module correctness

**Run AST lint first — before compiling or executing:**
```
validate_kernel_structure(source_code=<full impl source>)
```

Fix all `error`-severity findings before proceeding.

Then validate: compile/structural, shape, dtype, and boundary tensor comparison against golden (via the staged `test_*.py` running `detailed_tensor_compare`).

### Step 4. Freeze and log

If the module passes:
- freeze the staged set (it is now part of `modules_pypto_verified`),
- log the passing boundary in Per-module verification log,
- move to the next module (Coding Agent will produce the M_{k+1} staged set on next dispatch).

If the module fails:

1. Check for known error patterns first:
   ```
   diagnose_error(error_log=<full error output>, kernel_code=<module source>)
   ```
   If a match is found, apply the fix and re-run.

2. If no pattern matched:
   - inspect shape/dtype/interface in the staged `*_impl.py`,
   - inspect internal intermediate checkpoints,
   - switch to binary-search-style debugging.

### Subskill delegation: debugging escalation

When `skills/debugging/DEBUG.md` strategies and `diagnose_error` do not resolve the issue, escalate to the following subskills in order:

1. **Precision workarounds**: Read `skills/pypto-precision-debug/SKILL.md` — try the workaround checklist (frontend switch, avoid inplace, unroll_list=[1], submit_before_loop, +0.0, shape adjustment).
2. **Precision bisection**: Read `skills/pypto-precision-compare/SKILL.md` — use `pass_verify_save` or checkpoint tensors to pinpoint the diverging operation.
3. **Memory overlap**: If precision failure is suspected to be caused by workspace issues, read `skills/pypto-memory-overlap-detector/SKILL.md`.
4. **AICore error**: If the error log contains `aicore error`, read `skills/pypto-aicore-error-locator/SKILL.md` to locate the CCE file and problem line.
5. **Host crash**: If the process crashes with a stack trace, read `skills/pypto-host-stacktrace-analyzer/SKILL.md` to resolve addresses to source lines.
6. **MACHINE workspace**: For workspace-related analysis, read `skills/pypto-machine-workspace/SKILL.md`.
