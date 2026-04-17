---
name: pypto-kernel-code-format
description: Kernel code structure — layers A-L design document, naming conventions, shape annotation convention, and the mandatory Python code skeleton (pypto_kernel_template.py).
---

# PyPTO Complex Kernel — Kernel Code Format

This skill defines **how to structure kernel code**. It contains the design document and the mandatory Python template that every staged file and production kernel must follow.

## Contents

| File | Purpose |
|------|---------|
| **`pypto-kernel-design-format.md`** | Layers A-L design doc, naming conventions, reference constraints, shape annotation convention (§11), checklist |
| **`pypto_kernel_template.py`** | Mandatory Python code skeleton implementing layers A-L — copy into `custom/<op>/` as the starting point |

---

## When to read

- **Phase 0:** Understand which layers apply to your kernel (forward-only, backward, fused, etc.)
- **Phase 2:** Map module decomposition to layers and stage markers
- **Phase 3-5:** Every staged file (`<op>_module*.py`) and the final kernel must follow this template
- **Any time:** Shape annotation convention (§11 of the design doc) applies to all kernel code

## Layers A-L (quick reference)

| Layer | Role | Typical names |
|-------|------|---------------|
| A | Utilities (optional) | `tensor_compare_report`, helpers |
| B | Small math building blocks | `norm_fwd`, `softmax_chunk` |
| C | Forward reference | `forward_ref` |
| D | Host-side constants | `make_chunk_constants` |
| E | Backward reference helpers | `_slice_chunk_inputs`, `_stage_attn` |
| F | Golden backward/forward | `torch_golden_*` |
| G | Cache / bridge | `prepare_cache_for_npu` |
| H | PyPTO sub-kernels | `pypto_slice_inputs`, `pypto_fused_stage_ab` |
| I | Kernel implementation | `_your_op_kernel_impl` (contains `pypto.loop`) |
| J | JIT entry | `@pypto.frontend.jit` `your_op_kernel_npu` |
| K | Host wrapper | `pypto_function` (I/O only, no `for...in range`) |
| L | Driver / test | `main()` or test runner |

Not every kernel needs every layer. Pick layers that match your complexity.

## Key rules

- **Mandatory skeleton:** Every deliverable must use `pypto_kernel_template.py` as the starting layout (`skills/orchestration/lead-orchestrator/references/rules.md` rule 17)
- **No `for...in range` in Layer K** (`pypto_function`): kernel iteration belongs in Layer I using `pypto.loop` (`skills/orchestration/lead-orchestrator/references/rules.md` rule 18)
- **Shape annotation:** Every tensor assignment must carry an inline shape comment (design doc §11)
- **Naming:** Follow the naming conventions in the design doc §2
