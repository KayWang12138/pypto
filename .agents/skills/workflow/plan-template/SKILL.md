---
name: pypto-kernel-plan-template
description: Template for the operator-specific plan file (custom/plan/<op>.md). Defines required sections, machine-readable fields, and update cadence.
---

# PyPTO Complex Kernel — Plan Template

This skill contains the plan template that agents copy to `custom/plan/<operator_name>.md` at the start of each kernel implementation.

## Contents

| File | Purpose |
|------|---------|
| **`plan.template.md`** | The actual template — copy to `custom/plan/<op>.md` and fill in |

---

## When to use

- **Phase 0:** Copy `plan.template.md` to `custom/plan/<operator_name>.md` as the first action
- **Every turn:** Update `active_module`, `modules_pypto_verified`, `current_staged_file`, `next_mandatory_step`
- **Phase 1:** Fill in Golden function inventory
- **Phase 2:** Fill in Module decomposition, Module contracts, Staged module files table
- **Phase 3:** Append to Per-module verification log after each boundary check
- **Phase 4:** Final Golden function inventory cross-check
- **Debugging:** Paste `extract_pypto_calls.py` output, append to Development & debug log

## Key sections in the template

| Section | When to fill | Mandatory? |
|---------|-------------|------------|
| Agent status (YAML) | Every turn | Yes |
| Task summary | Phase 0 | Yes |
| Validation | Phase 0 | Yes |
| Module decomposition + rationale | Phase 2 | Yes |
| Staged module files table | Phase 2 (create), Phase 3 (update) | Yes |
| Per-module verification log | Each GATE 3 pass | Yes |
| API map | Phase 0 | Yes |
| Golden function inventory | Phase 1 (create), Phase 3-4 (cross-check) | Yes |
| Module contracts | Phase 2 | Yes |
| Design format compliance | Phase 2 | Yes |
| DEBUG.md §9 pre-write checklist | Before Phase 3 | Yes |
| Development & debug log | Every error/fix | Yes |
| Human review milestones | Optional | No |
