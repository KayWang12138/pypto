---
name: pypto-kernel-validation
description: Validation runner requirements, detailed_tensor_compare usage, success criteria, required deliverables, and required output structure for the PyPTO Complex Kernel Workflow.
---

# PyPTO Complex Kernel — Validation and Deliverables

## Validation runner and `detailed_tensor_compare` (mandatory)

**Purpose:** End-to-end correctness is golden vs PyPTO in one process. Do **not** use `pytest` as the default driver. Use a normal Python script that the user runs explicitly.

### Runner file and command

| Item | Requirement |
|------|-------------|
| **Path** | `custom/<operator_name>/test_<operator_name>.py` |
| **CWD** | Repository root |
| **Command** | `PYTHONPATH=.agents/skills/validation-and-deliverables python custom/<operator_name>/test_<operator_name>.py` |
| **Import** | `from detailed_tensor_compare import detailed_tensor_compare` (provided by `PYTHONPATH`; must be the bundled implementation) |

### What the runner must do

1. Build inputs, run golden (reference), run PyPTO (`pypto_function` / kernel).
2. For **every** output tensor — including all elements of a tuple/list, all keys of a dict, and nested structures after flattening to leaf tensors — call `detailed_tensor_compare(golden_tensor, pypto_tensor, tensor_name, ...)` and require `all_close` for each. **Forbidden:** comparing only one output when there are multiple.
3. Exit non-zero or raise after any mismatch; at minimum print a clear PASS/FAIL summary listing each compared tensor name.
4. `if __name__ == "__main__":` entrypoint — not a `pytest` test function as the only runnable path.

### Module boundaries (Phase 3)

Intermediate module-boundary checks must use the same bundled `detailed_tensor_compare`. Record each run in `custom/plan/<operator_name>.md` → Per-module verification log (see `skills/plan-template/plan.template.md`).

### `pytest`

- **Forbidden** as the default mechanism for golden vs PyPTO end-to-end comparison.
- **Allowed** only for small, optional extras if documented in `custom/plan/<operator_name>.md`.

## User-Facing Runner Requirement

The user should not have to manually orchestrate validation. The script `test_<operator_name>.py` must:
- prepare inputs,
- run the production kernel,
- run the golden,
- compare all outputs using `detailed_tensor_compare`,
- optionally expose debug/checkpoint mode,
- print a concise validation summary.

Default command from repo root:
```bash
PYTHONPATH=.agents/skills/validation-and-deliverables python custom/<operator_name>/test_<operator_name>.py
```

---

## Success Criteria

The task is complete only when all of the following are true:
- the normalized golden matches the original golden within tolerance,
- every module matches its expected golden outputs,
- `custom/plan/<operator_name>.md` documents module decomposition (rationale), the staged file chain, and an up-to-date per-module verification log with `detailed_tensor_compare` evidence,
- staged module files exist for each milestone and the final file equals the full integrated PyPTO kernel,
- progressive integration preserves correctness at each boundary,
- the final production design is either one integrated `@pypto.frontend.jit` kernel or a clearly documented staged fallback,
- the user can run one script to execute validation, and that script compares every kernel output tensor.

---

## Required Deliverables

The agent must produce all of the following:

1. **`custom/plan/<operator_name>.md`** — including Module decomposition, Staged module files table, and Per-module verification log; boundary checks must use `detailed_tensor_compare`.
2. **Staged module files** — `<op>_module1.py`, `…_module12.py`, …, `…_module1…N.py`; each stage passes before the next exists.
3. **Normalized golden reference** — consistent with the final kernel (may live inside staged files and/or shared helpers).
4. **Validation runner script** — `custom/<operator_name>/test_<operator_name>.py` run from repo root with `PYTHONPATH=.agents`; must compare all outputs. Do not use `pytest` as the default.
5. **Production kernel implementation** — typically the final staged file and/or a thin re-export; document in the plan.
6. **Optional debug helper files** only if necessary.
7. **Summary** of: module contracts, frozen checkpoints, current known limitations, and whether the final architecture is fused or staged fallback.

---

## Required Output Structure

At the end, the agent must be able to report:
- normalized golden summary,
- semantic module map,
- module verification status,
- frozen checkpoints,
- production architecture decision,
- integrated validation result,
- optimization status,
- known limitations,
- exact user command to run.
