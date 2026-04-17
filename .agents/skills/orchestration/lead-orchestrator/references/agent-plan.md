# Agent Execution Plan — PyPTO Kernel Development

**Reading order:** `principles.md` → this file → `rules.md` → then follow this checklist phase by phase. Use `catalog.yaml` to locate skill categories.

**Do not pre-read all skills.** Read each skill when this checklist directs you to — not before.

Replace `<op>` with the operator name everywhere.

---

## Skill Catalog

Skills are organized into 6 categories under `skills/`. Read `catalog.yaml` for the full index. Read a category's `_category.yaml` to find the right skill. Read a skill only when directed.

| Category | Path | Skills | When used |
|----------|------|--------|-----------|
| **workflow** | `skills/workflow/` | Phase 0-6 skills, templates, validation | Every phase |
| **development** | `skills/development/` | Intent, API, golden, design, impl, env | Phase 0-3, post |
| **debugging** | `skills/debugging/` | Precision, aicore, crash, memory | Phase 3+ (on failure) |
| **performance** | `skills/performance/` | 3-stage tuning, swimlane, auto-tuner | Phase 6 |
| **ci-and-pr** | `skills/ci-and-pr/` | Layout check, PR, Issue, review | Phase 3+, post |
| **pass** | `skills/pass/` | Pass analysis, errors, UT, perf | Reference |

---

## Forbidden (violations require restarting from the failed gate)

1. **Do not advance before tests pass.** Do not write the next module or full integration until the current module's `detailed_tensor_compare` returns `all_close: true` for all outputs.
2. **Do not start kernel implementation without a PyPTO-friendly golden.** Do not enter Phase 3 before Phase 1 is complete.
3. **Do not leave `.T` / `.t()` or implicit transpose forms in the golden.** PyPTO cannot use `.T` (`skills/debugging/debugging/DEBUG.md §9.19`). Normalize the golden to `torch.transpose(t, dim0, dim1)` and document transpose intent for matmul (e.g. comment `# a @ b^T` → `pypto.matmul(a, b, dtype, a_trans=False, b_trans=True)`).
4. **Do not claim done without cross-checking the golden operation list against the PyPTO kernel.** Precision errors are often missing operations.
5. **Do not implement all modules in one shot.** One module at a time.

---

## Checklist

You may not proceed past any **⛔ GATE** until it passes. Record evidence in `custom/plan/<op>.md`.

---

### Phase 0: Preparation

**Read now:** `skills/workflow/phase0-phase1-planning/SKILL.md`, `skills/workflow/plan-template/SKILL.md`, `skills/workflow/kernel-code-format/SKILL.md`

- [ ] **0.1** Create `custom/plan/<op>.md` by copying `skills/workflow/plan-template/plan.template.md`
- [ ] **0.2** List every operation in the target formula (add, matmul, softmax, transpose, etc.)
- [ ] **0.3** Confirm each operation exists in PyPTO → record in plan **API map** (`skills/workflow/phase0-phase1-planning/SKILL.md` Phase 0)
  ```
  query_op(names=["matmul", "softmax", "exp", ...])
  ```
- [ ] **0.4** Search for similar kernel examples → record in plan
  ```
  retrieve_docs(query="<kernel type> example", chunk_type="example")
  ```

⛔ **GATE 0:** API map exists in plan; zero `unsupported` rows (or each has a documented workaround).

---

### Phase 1: Build a PyPTO-friendly golden

**Read now (if not already):** `skills/workflow/phase0-phase1-planning/SKILL.md` (Phase 1 section), `skills/workflow/kernel-code-format/pypto-kernel-design-format.md` §11 (shape annotation)

- [ ] **1.1** Identify the primary reference implementation
- [ ] **1.2** Write the **PyPTO-friendly golden** and apply **all** rules below:

| Rule | Rationale |
|------|-----------|
| Replace `.T` / `.t()` → `torch.transpose(t, dim0, dim1)` | PyPTO has no `.T` (`skills/debugging/debugging/DEBUG.md §9.19`) |
| `.sum(dim)` may stay; be aware of 32-byte alignment issues in PyPTO | `skills/debugging/debugging/DEBUG.md §9.19` |
| No implicit broadcast → explicit `reshape` then op | `skills/workflow/phase4-phase5-integration/SKILL.md` Phase 5.3 |
| Name every intermediate (`scores`, `weights`, …) | Debugging |
| Shape comment on every intermediate `# [B, H, T, K]` | `rules.md` rule 6; `skills/workflow/kernel-code-format/pypto-kernel-design-format.md` §11 |
| Mark semantic boundaries with `# --- Module M1: ... ---` | Prepares Phase 2 split |

- [ ] **1.3** Build the **golden operation inventory**:

```
List every operation in the golden (one op per line):
  1. torch.matmul(q, k^T)     # scores  [B,H,T,T]
  2. torch.softmax(scores, -1) # weights [B,H,T,T]
  3. torch.matmul(weights, v)  # out     [B,H,T,V]
  ...
```

→ Record this list in `custom/plan/<op>.md` under **Golden function inventory**.

- [ ] **1.4** Compare normalized golden vs original golden
  ```python
  assert torch.allclose(original_out, normalized_out, rtol=1e-5, atol=1e-5)
  ```
- [ ] **1.5** Freeze golden → set `correctness: golden_ok` in plan

⛔ **GATE 1:** (a) zero `.T`/`.t()` in golden (b) shape comments on all intermediates (c) Golden function inventory exists in plan (d) original-vs-normalized test passes. **Do not enter Phase 2 until all hold.**

---

### Phase 2: Module decomposition

**Read now:** `skills/workflow/phase2-phase3-construction/SKILL.md` (Phase 2 section), `skills/debugging/debugging/DEBUG.md §9` (pre-write checklist — consult §9 subsections matching your kernel's operations)

- [ ] **2.1** Define modules along the `# --- Module M1 ---` boundaries from Phase 1
  - Split by **semantic** boundaries (matmul / norm / recurrence, …). Equal-size splits are forbidden.
  - Define each module's input/output tensor names, shapes, dtypes
  - → Record in plan **Module decomposition** + **Module contracts** (`skills/workflow/phase2-phase3-construction/SKILL.md` Phase 2)

- [ ] **2.2** Fix staged file sequence → fill **Staged module files** table in plan
  ```
  <op>_module1.py     → M1 only
  <op>_module12.py    → M1 + M2
  ...
  <op>_module1…N.py   → all modules = complete kernel
  ```

- [ ] **2.3** Complete `skills/debugging/debugging/DEBUG.md §9` pre-write checklist in plan (`skills/workflow/plan-template/plan.template.md` section)

⛔ **GATE 2:** Module decomposition + contracts + staged file table exist in plan.

---

### Phase 3: Implement modules (repeat for each M_k)

**Read now:** `skills/workflow/phase2-phase3-construction/SKILL.md` (Phase 3 section + DEBUG §9 lookup table), `skills/workflow/validation-and-deliverables/SKILL.md`

**One module at a time. Do not advance until the current module passes.**

- [ ] **3.1** Read the relevant `skills/debugging/debugging/DEBUG.md §9` subsections (`skills/workflow/phase2-phase3-construction/SKILL.md` Phase 3 table)
- [ ] **3.2** From `skills/workflow/kernel-code-format/pypto_kernel_template.py`, create `custom/<op>/<op>_module<suffix>.py`
  - Golden for this cumulative scope
  - PyPTO: **one** `@pypto.frontend.jit`; stub later modules with golden-fed tensors
  - Runner: `if __name__ == "__main__":` + `detailed_tensor_compare`
- [ ] **3.3** Run AST lint
  ```
  validate_kernel_structure(source_code=<full source>)
  ```
  → Fix until zero errors
- [ ] **3.4** **Cross-check Golden function inventory vs PyPTO**

  ```
  Golden function inventory (from Phase 1.3):
    1. matmul(q, k^T)        → ✅ pypto.matmul(q, k, dtype, b_trans=True)  L.42
    2. softmax(scores, -1)    → ✅ pypto.softmax(scores, dim=-1)           L.45
    3. matmul(weights, v)     → ❌ not implemented — skipping causes precision error
  ```

  **Do not run tests until every operation in M_k's scope has ✅.**

- [ ] **3.5** Run tests
  ```bash
  PYTHONPATH=.agents/pypto-kernel-custom-skills/skills/workflow/validation-and-deliverables python custom/<op>/<op>_module<suffix>.py
  ```
- [ ] **3.6** Layout check
  ```bash
  bash .agents/pypto-kernel-custom-skills/skills/ci-and-pr/ci-and-layout-check/run_validate_layout.sh
  ```

⛔ **GATE 3 (M_k):** (a) `detailed_tensor_compare` → `all_close: true` on **all** outputs (b) Golden inventory: all ops in M_k scope ✅ (c) layout check exit 0 (d) Per-module verification log updated. **Do not start M_{k+1} until all hold.**

- [ ] **3.7** Freeze M_k → update plan: append to `modules_pypto_verified`, set `active_module: M_{k+1}`

**After GATE 3:** return to 3.1 for the next module, or go to Phase 4 when all modules are done.

---

### Phase 4: Integration and E2E

**Read now:** `skills/workflow/phase4-phase5-integration/SKILL.md` (Phase 4 section), `skills/workflow/validation-and-deliverables/SKILL.md`

- [ ] **4.1** Confirm the last staged file (`<op>_module1…N.py`) integrates all modules
- [ ] **4.2** Create `test_<op>.py` (`skills/workflow/validation-and-deliverables/SKILL.md` — Validation runner)
  - Compare **every** output tensor with `detailed_tensor_compare` (single-output-only compare is forbidden)
- [ ] **4.3** **Final Golden function inventory cross-check**

  Line-by-line: golden operation list vs final PyPTO kernel.  
  **If any ❌ remains, do not run E2E — implement first.**

- [ ] **4.4** Run E2E
  ```bash
  PYTHONPATH=.agents/pypto-kernel-custom-skills/skills/workflow/validation-and-deliverables python custom/<op>/test_<op>.py
  ```
- [ ] **4.5** Layout check
  ```bash
  bash .agents/pypto-kernel-custom-skills/skills/ci-and-pr/ci-and-layout-check/run_validate_layout.sh
  ```

⛔ **GATE 4:** (a) E2E `all_close: true` on all outputs (b) Golden inventory 100% ✅ (c) layout check exit 0.

---

### Phase 5: Final structural rules

**Read now:** `skills/workflow/phase4-phase5-integration/SKILL.md` (Phase 5 section), `skills/debugging/debugging/DEBUG.md §9`

- [ ] `validate_kernel_structure` → zero errors
- [ ] No `for ... in range(...)` inside `pypto_function` (layout check)
- [ ] `set_vec_tile_shapes` with valid tile dimensions per doc
- [ ] `set_cube_tile_shapes` before matmul
- [ ] Write-back via `output[:] = expr` or `pypto.assemble(...)`

(Details: `skills/workflow/phase4-phase5-integration/SKILL.md` Phase 5 + `skills/debugging/debugging/DEBUG.md §9`)

---

### Phase 6: Optimization (only after correctness)

**Read now:** `skills/workflow/phase6-optimization/SKILL.md`

Follow the skill. Roll back immediately if correctness regresses.

---

## When precision errors appear (required order)

**Read now:** `skills/debugging/debugging/SKILL.md` (op-by-op protocol), `skills/debugging/debugging/DEBUG.md §9.11` (error table)

1. Open **Golden function inventory**
2. Line-by-line vs PyPTO kernel → find missing or wrong ops
3. Run `extract_pypto_calls.py` on the kernel and reconcile with golden
4. If still unclear → `skills/debugging/debugging/DEBUG.md` §9.11 error table → op-by-op protocol (`skills/debugging/debugging/SKILL.md`)

---

## Required plan sections (`skills/workflow/plan-template/plan.template.md`)

| Section | When to update |
|---------|----------------|
| API map | Phase 0 |
| **Golden function inventory** | Phase 1 (create); Phase 3/4 (append cross-check results) |
| Module decomposition + rationale | Phase 2 |
| Module contracts | Phase 2 |
| Staged module files table | Phase 2 (create) → Phase 3 (check each milestone) |
| Per-module verification log | Each GATE 3 pass in Phase 3 |
| `skills/debugging/debugging/DEBUG.md §9` pre-write checklist | Before Phase 3 |
| Development & debug log | Every error and fix |


---

## Post-development (after correctness confirmed)

After Phase 6 is complete (or skipped), execute the following subskills as needed:

- **Model integration**: Read `skills/development/pypto-fused-op-integration/SKILL.md` to integrate the fused operator into a model, replacing small-op combinations.
- **PR creation**: Read `skills/ci-and-pr/pypto-pr-creator/SKILL.md` to create a PR to the cann/pypto repository with proper commit conventions.
- **PR CI fix**: Read `skills/ci-and-pr/pypto-pr-fixer/SKILL.md` to fix CodeCheck CI failures or address reviewer comments on an existing PR.
- **Issue creation**: Read `skills/ci-and-pr/pypto-issue-creator/SKILL.md` to create a GitCode Issue for bugs, features, or tasks discovered during development.
- **Fracture point detection**: Read `skills/ci-and-pr/pypto-fracture-point-detector/SKILL.md` to identify framework or documentation gaps encountered during this session.
