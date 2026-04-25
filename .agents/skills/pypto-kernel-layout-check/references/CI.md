# Automated layout check (agent-runnable)

This skill ships a **no-NPU** validator for `custom/<operator_name>/` (plan, `test_<op>.py`, staged `*_module*.py` naming, and **no `for ... in range(...)` inside `pypto_function`** in `custom/<op>/**/*.py`).

## Agent: run this from the repository root

```bash
bash .agents/skills/pypto-kernel-layout-check/scripts/run_validate_layout.sh
```

Equivalent:

```bash
python3 .agents/skills/pypto-kernel-layout-check/scripts/validate_custom_kernel_layout.py --repo-root "$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
```

- If there is no `custom/` work yet, the script **exits 0** (nothing to validate).
- If `custom/<op>/` has `test_<op>.py` or `<op>_module*.py`, checks apply — **exit 1** on violations.

**When to run:** after adding or changing files under `custom/`, and **before** claiming the layout is complete (per the Verification Agent contract in `.opencode/agents/verification.md` — GATE 3 step 5 and GATE 4).

## Repository integration (human / CI)

| Mechanism | Location |
|-----------|----------|
| **pre-commit** | Repo root **`.pre-commit-config.yaml`** — hook `validate-custom-kernel-layout` |
| **GitHub Actions** | **`.github/workflows/validate-custom-kernel-layout.yml`** |

To add the hook in another clone, merge the `repo: local` block from the root **`.pre-commit-config.yaml`** or run `pre-commit install` after pulling.
