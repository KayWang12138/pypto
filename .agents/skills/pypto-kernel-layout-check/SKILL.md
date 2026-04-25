---
name: pypto-kernel-layout-check
description: Automated layout validation for custom/<operator>/ — CI, pre-commit, and agent-runnable checks. Includes extract_pypto_calls.py for op-by-op debugging.
---

# PyPTO Complex Kernel — CI and Layout Check

This skill bundles the automated validation tools and CI integration for the `custom/<operator>/` directory.

## Contents

| File | Purpose |
|------|---------|
| **`references/CI.md`** | Full CI documentation — commands, pre-commit hook, GitHub Actions setup |
| **`scripts/run_validate_layout.sh`** | Agent entrypoint: run from repo root to check layout |
| **`scripts/validate_custom_kernel_layout.py`** | Implementation: plan file, test runner, staged naming, `for...in range` in kernel code, template layer headers |
| **`scripts/extract_pypto_calls.py`** | List every `pypto.*` call site by line number — used for op-by-op debugging |

---

## Quick reference

### Layout check (run after every `custom/` change)

```bash
bash .agents/skills/pypto-kernel-layout-check/scripts/run_validate_layout.sh
```

Equivalent:
```bash
python3 .agents/skills/pypto-kernel-layout-check/scripts/validate_custom_kernel_layout.py \
  --repo-root "$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
```

- No `custom/` work yet → **exit 0** (nothing to validate).
- Violations found → **exit 1** with details.

### What the layout check validates

| Check | Failure means |
|-------|---------------|
| `custom/<op>/plan.md` exists | Missing plan file |
| `custom/<op>/test_<op>.py` (after Phase D) or `custom/<op>/staged/test_<op>_module<k>.py` (during Phase 3) exists and imports `detailed_tensor_compare` | Missing or incomplete runner |
| Staged sets use cumulative suffixes (`1`, `12`, `123`, …) and the 3-file naming (`*_impl.py`, `*_golden.py`, `test_*`) | Wrong naming convention |
| No `for ... in range(...)` in `pypto_function` / kernel functions | Use `pypto.loop` instead (`skills/pypto-op-develop/references/kernel-layer-format.md` §7.1) |

### Extract PyPTO call sites (for debugging)

```bash
python3 .agents/skills/pypto-kernel-layout-check/scripts/extract_pypto_calls.py \
  custom/<operator_name>/<kernel_file>.py
```

Add `--json` for machine-readable output. See `skills/debugging/SKILL.md` → op-by-op check protocol.

---

## When to run

- **After every meaningful edit** under `custom/` (per the Verification Agent's GATE 3 / GATE 4 contract in `.opencode/agents/verification.md`)
- **Before claiming layout is complete**
- The same validator runs in CI (`.github/workflows/validate-custom-kernel-layout.yml`) and pre-commit (`.pre-commit-config.yaml`)
