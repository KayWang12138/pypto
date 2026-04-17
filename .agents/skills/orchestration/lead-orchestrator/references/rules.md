# PyPTO kernel — mandatory rules

> **Navigation:** You were directed here by `agent-plan.md`. After reading this file, return to `agent-plan.md` and follow it phase by phase — it tells you which skills to read at each step.

## Zero tolerance — do not skip, do not shortcut (read first)

**These are not "best effort."** Violating them to save time, tokens, or context is **forbidden**. Claiming a step is done without the artifacts and commands it requires is **non-compliant**.

| Forbidden | Required instead |
|-----------|------------------|
| **Skipping** sections of this bundle you judge "optional" | Follow **`rules.md`**, **`skills/debugging/debugging/DEBUG.md`** when debugging, **`skills/workflow/plan-template/plan.template.md`** fields, and the staged files / validation / per-module log rules end-to-end for the current task. |
| **Shortcutting** the staged chain (`_module1.py` → `_module12.py` → …) | Create and **pass** each staged file **before** the next; no jumping to "full kernel only." |
| **Omitting** `detailed_tensor_compare` or comparing **one** output only | Use the bundled helper; compare **every** leaf output at each stage and in **`test_<op>.py`**. |
| **Skipping** plan updates (`custom/plan/<op>.md`) after runs | Update **every turn** per **Plan file (every turn)** below. |
| **Replacing** real runs with verbal "should pass" / "aligned" | Run the commands; paste evidence into the plan or logs. |
| **Skipping** the **layout check** after changing `custom/` | Run **`bash .agents/pypto-kernel-custom-skills/skills/ci-and-pr/ci-and-layout-check/run_validate_layout.sh`** from repo root (or **`skills/ci-and-pr/ci-and-layout-check/CI.md`**); fix **exit 1** before claiming the layout is done. |
| **`set_vec_tile_shapes` with valid dimensions** | Pass positive tile sizes (use **`1`** as needed); dimensions per **`docs/api/config/pypto-set_vec_tile_shapes.md`** for your version. |
| **Saving tokens** by not reading docs/skills that apply | Read the relevant skill under **`skills/`**. Token cost is **not** an excuse to omit steps. |
| **Ad-hoc kernel file layout** (no layers A–L, random function names) | Use **`skills/workflow/kernel-code-format/pypto_kernel_template.py`** as the **mandatory skeleton** for **each** staged file **and** the full kernel — see **`skills/workflow/kernel-code-format/pypto-kernel-design-format.md`**. Document any deliberate deviation in **`custom/plan/<op>.md`**. |
| **`for ... in range(...)` inside `pypto_function`** (host Python loop over tiles/batch/seq) | Express algorithmic iteration in **`_your_op_kernel_impl`** / **`your_op_kernel_npu`** with **`pypto.loop`** (+ `pypto.view`). `pypto_function` is for I/O pack/unpack only — see **Prohibition B** below. CI: **`skills/ci-and-pr/ci-and-layout-check/scripts/validate_custom_kernel_layout.py`** flags this pattern. |

If you cannot complete a step, **document the blocker** in the plan — do not silently skip.

## Non-negotiable

1. **One module at a time in PyPTO** — Only one semantic module's real `pypto` logic may be unfrozen at a time; later stages **stub** or use **golden boundary tensors** until the current module passes.
2. **No full fused `@jit` in one shot** before per-module boundary checks pass.
3. **Host Python `for` is not the kernel tile loop** — algorithmic tiling lives in `pypto.loop` + `view` + `assemble` (when required).
4. **Single production `@jit`** — not one JIT per module unless documented staged fallback.
5. **Golden frozen** before PyPTO implementation; **do not** change it without evidence and plan log.
6. **Shape comments** on tensor lines in kernel code (see **Shape Annotation Convention** in `skills/workflow/kernel-code-format/pypto-kernel-design-format.md`).
7. **Stuck on PyPTO errors** — read **`skills/debugging/debugging/DEBUG.md`**, then run `skills/ci-and-pr/ci-and-layout-check/scripts/extract_pypto_calls.py`, then **op-by-op protocol** in **`skills/debugging/debugging/SKILL.md`**.
7b. **Before writing PyPTO code** — consult **`skills/debugging/debugging/DEBUG.md` §9** for the subsection matching what you are about to write (JIT signatures §9.1, `pypto.view` §9.4, `matmul` §9.19, reductions §9.19, dynamic shapes §9.2, tensor type hints §9.13, Python ops inside JIT §9.14, tile config §9.15). See the full lookup table in **`skills/workflow/phase2-phase3-construction/SKILL.md`** → **Phase 3 → Before writing PyPTO code**. Skipping this is non-compliant.
8. **End-to-end validation runner** — **`custom/<operator_name>/test_<operator_name>.py`** (not `pytest` as the default driver). From repo root: **`PYTHONPATH=.agents/pypto-kernel-custom-skills/skills/workflow/validation-and-deliverables python custom/<operator_name>/test_<operator_name>.py`** (see **`skills/workflow/validation-and-deliverables/SKILL.md`**).
9. **Golden vs PyPTO comparison** — use **`detailed_tensor_compare`** from **`.agents/pypto-kernel-custom-skills/skills/workflow/validation-and-deliverables/detailed_tensor_compare.py`** (`from detailed_tensor_compare import detailed_tensor_compare`); do not substitute a different implementation for the primary report.
10. **Every output** — **`test_<operator_name>.py`** must compare **all** kernel outputs (tuple/list/dict/nested → every leaf tensor). **Forbidden:** validating only one output when the kernel returns several. Exceptions only in **`custom/plan/<operator_name>.md`** → **blockers** with justification.
11. **Module decomposition in plan** — **`custom/plan/<operator_name>.md`** must record **how** semantic modules are split and **why** (rationale: boundaries, checkpointability, ordering — not "balanced complexity"). See **`skills/workflow/plan-template/plan.template.md`** → **Module decomposition**.
12. **Per-module verification log** — for **each** module boundary check (golden vs PyPTO), append a row to the plan's **Per-module verification log** using **`detailed_tensor_compare`** results (`all_close` and key fields from the returned dict). End-to-end and per-module checks use the **same** bundled helper.
13. **Do not stop on cryptic errors alone** — `FFFFF`, `UNKNOWN`, `0x3FFFF`, or other opaque **`Errcode: F…!`** lines are **not** a reason to abandon the task. Follow **`skills/debugging/debugging/DEBUG.md`**, gather logs, apply **`skills/ci-and-pr/ci-and-layout-check/scripts/extract_pypto_calls.py`** + op-by-op protocol, and iterate. Token/turn cost is not a limiting factor. Stop only when true blockers apply (see **Stop Conditions** below).
14. **Staged module Python files** — Under **`custom/<operator_name>/`**, create **`<operator_name>_module1.py`**, then **`<operator_name>_module12.py`**, **`…_module123.py`**, …, **`…_module1…N.py`** (suffix = digits **1**, **12**, **123**, … = cumulative M1..Mk). Each file: **golden + one `@jit`** for that scope; **`detailed_tensor_compare`** on **all** outputs must pass **before** the next staged file. Final **`…_module1…N.py`** = full end-to-end kernel.
15. **Automated layout check** — After meaningful edits under **`custom/`**, run **`bash .agents/pypto-kernel-custom-skills/skills/ci-and-pr/ci-and-layout-check/run_validate_layout.sh`** from the **repository root** (see **`skills/ci-and-pr/ci-and-layout-check/CI.md`**). **Do not** claim completion while this exits **1**. Same logic as CI/pre-commit.
16. **`set_vec_tile_shapes` — valid tile dimensions** — When coding or debugging, pass positive tile arguments as required by **`docs/api/config/pypto-set_vec_tile_shapes.md`** for your PyPTO version. See **`skills/workflow/phase4-phase5-integration/SKILL.md`** → **5.4b**.
17. **`skills/workflow/kernel-code-format/pypto_kernel_template.py` — mandatory code skeleton** — Structure **every** deliverable (`<op>_module1.py` … `<op>_module1…N.py` and the integrated kernel) using layers **A–L** from **`skills/workflow/kernel-code-format/pypto_kernel_template.py`**. **Do not** drop the template because debugging is hard. See **`skills/workflow/kernel-code-format/pypto-kernel-design-format.md`**.
18. **No `for ... in range(...)` inside `pypto_function`** — The host wrapper **`pypto_function`** must **not** implement kernel tile/batch/sequence loops with Python `for` + `range`. Put those loops in **`_your_op_kernel_impl`** / JIT entry using **`pypto.loop`**. **`validate_custom_kernel_layout.py`** rejects this pattern under **`custom/<op>/`**.
19. **No `.T` / `.t()` in golden** — PyPTO-friendly golden must use `torch.transpose(t, dim0, dim1)` instead of `.T`/`.t()`. PyPTO tensors do not support `.T` (`skills/debugging/debugging/DEBUG.md §9.19`). For matmul `a @ b.T`, write `torch.matmul(a, b.transpose(-2, -1))` and comment the transpose intent.
20. **Golden function inventory** — After writing the PyPTO-friendly golden, list every mathematical operation (one per line) in **`custom/plan/<op>.md` → Golden function inventory**. In Phase 3/4, cross-check each line against the PyPTO implementation: mark ✅ with pypto call + line number, or ❌ if missing. **Do not run tests or advance modules while any ❌ remains in scope.** Precision errors are most often caused by operations that were never implemented.

---

## Module-at-a-time enforcement

**Problem:** A single large `@jit` that combines every semantic stage triggers compound failures (tiling, `view`/`assemble`, write-back, dtype, graph limits). If the agent implements all modules at once, errors become unlocalizable.

| Rule | Requirement |
| --- | --- |
| **One active module** | At most one semantic module's PyPTO logic may be new or unfrozen at a time. Later stages must be stubbed (identity, zeros, or pass-through tensors from the golden). |
| **Boundary before next** | Do not add the next module's real ops inside `kernel_impl` until the current module's outputs match the golden. |
| **Plan file** | Keep `active_module: Mk` and `modules_pypto_verified: [M1, …]` in `custom/plan/<operator_name>.md`. Change `active_module` only after logging `detailed_tensor_compare` evidence. |
| **User prompt default** | "Implement the kernel" = implement the next unverified module only, unless the user explicitly requests full integration. |
| **Stubs must be explicit** | Comment every stub: `# STUB: until M2 verified; golden-fed tensor`. |

**Compliant pattern:** Implement M1 only → validate → freeze → set `active_module: M2` → repeat.

---

## The Three Architectural Prohibitions

### Prohibition A: No one-shot implementation

Do not jump from reference code to one integrated PyPTO kernel. Normalize the golden → split by semantic boundaries → validate each boundary → integrate progressively.

### Prohibition B: No Python host loops for kernel tiling logic

Do not use Python `for` loops to simulate the kernel's algorithmic tiled execution. Algorithmic loops must use `pypto.loop` or explicit semantic staging. Allowed host loops: iterating over test cases, candidate configs, modules for bookkeeping, or host-side validation inputs.

### Prohibition C: No one-JIT-per-module production architecture

Modules are semantic blocks, not separate production JIT entrypoints. Assemble into one production `@pypto.frontend.jit` kernel. Staged multi-kernel fallback is allowed only when fusion is blocked by framework limitations — label it clearly as fallback.

---

## Stop Conditions

Pause only if one of these is true:
- the reference code is missing,
- the normalized golden cannot be made equivalent,
- the framework fundamentally blocks the required integrated form,
- required user runtime logs are missing,
- further progress would be blind speculation.

**Not** valid pause reasons: a single cryptic error code, fear of using more tokens, or unwillingness to try another documented strategy — use `skills/debugging/debugging/DEBUG.md` and keep iterating.

---

## Plan file (every turn)

Update `custom/plan/<operator_name>.md`:

- `active_module`, `modules_pypto_verified`, **`current_staged_file`** (e.g. `custom/<op>/<op>_module12.py`)
- **Module decomposition** (overview + rationale) when the split is known or changes
- **Golden function inventory** — update ✅/❌ status after each module implementation (rule 20)
- **Staged module files** table — checkmarks / filenames as stages complete
- **Per-module verification log** after each boundary run (with **`detailed_tensor_compare`** evidence)
- `next_mandatory_step`
- **Development & debug log** entry after each run or edit
- After **`custom/`** changes: run **`skills/ci-and-pr/ci-and-layout-check/run_validate_layout.sh`** (or fix until exit 0)

---

## Skill library rules

21. **Skill priority** — When a category skill's instructions conflict with these rules, this rules.md takes precedence. In particular: module-at-a-time enforcement, staged file chain, `detailed_tensor_compare` mandatory for all outputs, Golden function inventory cross-check, and Layer A-L template structure.
22. **Skill read timing** — Read a skill's SKILL.md only when `agent-plan.md` or a Phase SKILL.md explicitly directs you to. Use `catalog.yaml` → `_category.yaml` to locate skills. Do not pre-read all skills. Token cost is not an excuse to skip reading a skill when directed.
23. **Skill files are read-only** — Do not edit files under `skills/`. If a skill needs adaptation, add the override in the calling Phase SKILL.md or in `custom/plan/<operator_name>.md`.
