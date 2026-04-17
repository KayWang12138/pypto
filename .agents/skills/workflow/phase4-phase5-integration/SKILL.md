---
name: pypto-kernel-phase4-phase5
description: Phase 4 (progressive integration — assemble modules one at a time, earliest-failing-boundary rule) and Phase 5 (PyPTO-specific structural rules — loops, state, broadcast, alignment, tiles, write-back).
---

# PyPTO Complex Kernel — Phase 4–5: Integration and Structural Rules

## Phase 4: Progressive Integration

Goal: assemble the final kernel without waiting until the very end to discover integration errors.

### Required integration order

Do not wait until all modules are independently built and then do one final merge.

Instead (mirror in staged files — see `skills/orchestration/lead-orchestrator/references/rules.md` rule 14):
1. Validate M1 in `<op>_module1.py`
2. Integrate M1+M2 in `<op>_module12.py`, validate boundary
3. Integrate M1+M2+M3 in `<op>_module123.py`, validate boundary
4. Continue until `<op>_module1…N.py` is the full kernel

### Earliest failing boundary rule

When integrated behavior fails:
- locate the earliest mismatch,
- fix only the module or boundary immediately before that mismatch,
- do not patch the entire graph,
- do not edit upstream frozen modules without evidence.

### Production architecture target

The target is:
- one production `@pypto.frontend.jit` kernel,
- with internal loops represented as `pypto.loop`,
- with semantic modules integrated as kernel structure.

If this is impossible because of framework/toolchain limitations:
- document the failure,
- produce the smallest reliable staged fallback,
- explain exactly why fusion failed.

---

## Phase 5: PyPTO-Specific Structural Rules

These rules come from observed failure modes. The `validate_kernel_structure` tool detects several automatically (marked ✦).

**Cross-reference:** `skills/debugging/debugging/DEBUG.md §9` expands on many of these rules with concrete error messages, root causes, and fix patterns. In particular: §9.2 (dynamic shapes / `pypto.loop`), §9.4 (`pypto.view` dimensions), §9.15 (tile shape config), §9.19 (matmul API, reduction alignment, assemble shape).

### 5.1 Loop rule

For algorithmic loops inside the target kernel:
- use `pypto.loop` inside the kernel design,
- do not replace it with Python `for` loops in the host script.

### 5.2 Recurrent state rule

For recurrent buffers/states:
- slice the semantically correct state,
- do not arbitrarily `view` a higher-rank tensor into a convenient rank just because PyTorch allowed it.

### 5.3 Broadcast rule

If a broadcast is needed:
- prefer one-axis-at-a-time broadcast shapes,
- avoid hidden two-axis implicit expansion,
- make row/column scale tensors explicit.

### 5.4 Alignment rule

For vector-sensitive paths:
- respect alignment constraints on the last dimension,
- if a narrow logical tensor is problematic, use an alignment-friendly representation only if the mapping back to semantics is explicit.

Retrieve alignment rules if needed:
```
retrieve_docs(query="<op_name> 32-byte alignment last dimension constraint", chunk_type="api_doc")
```

### 5.4b `set_vec_tile_shapes` — valid tile dimensions

**Context:** Vector tile shape errors are common; standardize `set_vec_tile_shapes` during kernel coding and debugging.

**Rule:**
1. Pass positive integer tile sizes as required by your PyPTO version's `docs/api/config/pypto-set_vec_tile_shapes.md`.
2. Document tile dimensions used and the reasoning in the plan if non-standard.

### 5.5 Pad/concat/layout rule

Do not assume PyTorch-style padding semantics map directly. If layout/pad operations are fragile:
- prefer explicit reshape/concat/aligned forms,
- keep the golden and kernel mapping clear.

### 5.6 SIM vs NPU rule

Never claim numerical correctness from SIM alone.

Use SIM for: compile, graph structure, early validation.
Use NPU for: real tensor comparison, final correctness.

### 5.7 Write-back rule ✦

Always use `output[:] = expr` or `pypto.assemble(expr, offset, output)`. Never use `output = expr`.

`validate_kernel_structure` detects this automatically — run it after every module is written.

### 5.8 Tile configuration rule ✦

- Vector ops require `pypto.set_vec_tile_shapes(...)` before the first vector op — use tile dimensions per doc (see 5.4b).
- `pypto.matmul` requires `pypto.set_cube_tile_shapes(...)` before the call.
- Loop `idx_name` values must be unique within a function.

`validate_kernel_structure` detects all three automatically.

---

## Subskill delegation: debugging during integration

When integration tests fail in Phase 4 or structural checks fail in Phase 5, and the standard debugging workflow (`skills/debugging/debugging/SKILL.md` + `DEBUG.md §9`) does not resolve the issue, escalate to these subskills:

- **Precision workarounds**: Read `skills/debugging/pypto-precision-debug/SKILL.md`
- **Precision bisection**: Read `skills/debugging/pypto-precision-compare/SKILL.md`
- **Memory overlap**: Read `skills/debugging/pypto-memory-overlap-detector/SKILL.md`
- **AICore error**: Read `skills/debugging/pypto-aicore-error-locator/SKILL.md`
- **Host crash**: Read `skills/debugging/pypto-host-stacktrace-analyzer/SKILL.md`
- **MACHINE workspace**: Read `skills/debugging/pypto-machine-workspace/SKILL.md`

Refer to `skills/workflow/phase2-phase3-construction/SKILL.md` → "Subskill delegation: debugging escalation" for the full prioritized order.
