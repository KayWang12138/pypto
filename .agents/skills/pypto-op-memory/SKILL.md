---
name: pypto-op-memory
description: Template for the operator-specific plan file (custom/<op>/plan.md). Defines required sections, machine-readable fields, and update cadence for the staged-set workflow.
---

# PyPTO Complex Kernel — Plan Template

This skill contains the plan template that agents copy to `custom/<op>/plan.md` at the start of each kernel implementation. The plan lives **inside** the operator folder, not in a separate `custom/plan/` directory.

## Contents

| File | Purpose |
|------|---------|
| **`templates/memory.md`** | The actual template — copy to `custom/<op>/plan.md` and fill in |

---

## When to use

- **Phase 0:** Copy `templates/memory.md` to `custom/<op>/plan.md` as the first action
- **Every turn:** Update `active_module`, `modules_pypto_verified`, `current_staged_set`, `next_mandatory_step`
- **Phase 1:** Fill in Golden function inventory
- **Phase 2:** Fill in Module decomposition, Module contracts, Staged set table
- **Phase 3:** Append to Per-module verification log after each boundary check
- **Phase 4:** Final Golden function inventory cross-check + Phase D rename log
- **Debugging:** Append to Development & debug log

## Key sections in the template

| Section | When to fill | Mandatory? |
|---------|-------------|------------|
| Agent status (YAML) | Every turn | Yes |
| Task summary | Phase 0 | Yes |
| Validation | Phase 0 | Yes |
| Module decomposition + rationale | Phase 2 | Yes |
| Staged set table | Phase 2 (create), Phase 3 (update) | Yes |
| Per-module verification log | Each GATE 3 pass | Yes |
| API map | Phase 0 | Yes |
| Golden function inventory | Phase 1 (create), Phase 3-4 (cross-check) | Yes |
| Module contracts | Phase 2 | Yes |
| Layer format compliance | Phase 2 | Yes |
| debug-playbook.md §9 pre-write checklist | Before Phase 3 | Yes |
| Development & debug log | Every error/fix | Yes |
| Phase D canonical rename log | After GATE 4 | Yes |
| Human review milestones | Optional | No |
