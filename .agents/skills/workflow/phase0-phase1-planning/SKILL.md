---
name: pypto-kernel-phase0-phase1
description: Phase 0 (triage, API availability, planning file) and Phase 1 (golden preparation — audit, normalize, freeze). Covers everything before PyPTO code is written.
---

# PyPTO Complex Kernel — Phase 0–1: Planning and Golden Preparation

## Contents

| File | Purpose |
|------|---------|
| **This file (SKILL.md)** | Phase 0-1 workflow guide |
| **`op_index.json`** | Optional local op index snapshot — offline fallback for `query_op` / `list_ops` (see `pypto-api-explorer` for live query) |

## Phase 0: Triage and Planning

Before any code generation, decide whether the complex kernel workflow applies.

Use this workflow if any of the following are true:
- backward kernel,
- recurrent or stateful kernel,
- scan-like kernel,
- multiple dependent math stages,
- multiple layout transitions,
- multiple reductions,
- nested loop structure,
- previous failed attempts,
- temptation to "just implement the whole thing now".

### Step 0.1: API availability check

Decompose the formula into atomic operations and verify each one exists in PyPTO:

```
# MCP preferred
list_ops(category="")
query_op(names=["<op1>", "<op2>"])
```

CLI fallback:
```bash
python3 .agents/skills/pypto-api-explorer/scripts/query_op_index.py --list-categories
python3 .agents/skills/pypto-api-explorer/scripts/query_op_index.py --op <op1> --op <op2> --compact
```

For any operation not found by name, search by category or semantically:
```
list_ops(category="<relevant_category>")
retrieve_docs(query="<formula step> PyPTO API", chunk_type="api_doc")
```

Mark each formula step as **supported**, **needs substitute**, or **unsupported** before proceeding. Do not begin implementation for unsupported operations without a confirmed substitute.

### Subskill delegation: requirements and environment (optional)

**If requirements are unstructured:** Read `skills/development/pypto-intent-understand/SKILL.md` and follow its workflow to produce a `SPEC.md`. Use the SPEC.md output to populate the plan file's task summary and API map.

**If environment issues arise:** Read `skills/development/pypto-environment-setup/SKILL.md` and follow its workflow to diagnose and fix CANN, torch_npu, or build-chain problems.

### Subskill delegation: enhanced API exploration

After completing Step 0.1, if deeper constraint verification is needed (3-layer validation, reference implementation search across `models/` and `examples/`), read `skills/development/pypto-api-explore/SKILL.md` and produce an `API_REPORT.md`. Merge the API_REPORT.md findings into the plan's API map section.

### Step 0.2: Find structurally similar examples

Search for existing kernels with similar structure:
```
retrieve_docs(query="<kernel type> example loop structure tiling", chunk_type="example")
```

Record the most relevant example path in the plan file.

### Mandatory planning file

Create `custom/plan/<operator_name>.md` first, then continue implementation immediately.

The plan must contain:
- task summary,
- reference locations (including example paths from Step 0.2),
- API availability map (from Step 0.1),
- normalized golden status,
- module list,
- module contracts,
- frozen items,
- attempt history,
- integration status,
- optimization status,
- blocker list,
- **design format compliance:** which layers from `skills/workflow/kernel-code-format/pypto-kernel-design-format.md` (A–L) apply, which are omitted and why,
- **`active_module`** and **`modules_pypto_verified`** (see `skills/orchestration/lead-orchestrator/references/rules.md` → Module-at-a-time enforcement).

---

## Phase 1: Golden Preparation

Goal: define one trusted reference that is mathematically correct and structurally mappable to PyPTO.

### Step 1. Start from the strongest available reference

Preferred order:
1. PyTorch forward/backward reference
2. NumPy reference
3. New mathematical reference only if no existing reference exists

Before writing a new reference:
```
retrieve_docs(query="<operator name> golden reference implementation", chunk_type="example")
```

### Step 2. Audit the reference for PyPTO-unfriendly patterns

Actively search for:
- implicit multi-axis broadcasting,
- opaque library operations,
- difficult composite calls,
- hidden layout changes,
- control flow that must become explicit,
- 4D/5D manipulations that may be fragile in PyPTO,
- host-side conveniences that do not map cleanly to tile_fwk IR.

For each suspicious pattern, query the op_index:
```
query_op(names=["<op>"])
```

### Step 3. Normalize the golden

Rewrite the golden into a PyPTO-friendly reference.

Rules:
- preserve semantics, not source syntax,
- make shapes explicit,
- make dtype transitions explicit,
- expose intermediate tensors with meaningful names,
- expose semantic module boundaries (mark with `# --- Module M1: <role> ---` comments),
- rewrite hidden broadcast chains into one-axis-at-a-time forms if needed,
- rewrite narrow vectors or awkward layouts into alignment-friendly representations,
- **Do not use `.T` / `.t()`:** replace with `torch.transpose(t, dim0, dim1)`. For matmul `a @ b.T`, write `torch.matmul(a, b.transpose(-2, -1))` and comment `# a @ b^T → pypto: b_trans=True`,
- annotate every intermediate tensor with a shape comment `# [B, H, T, K]`.

### Subskill delegation: golden generation (optional)

To leverage automated golden generation with confidence scoring and auto-repair, read `skills/development/pypto-golden-generate/SKILL.md`. The subskill produces `{op}_golden.py` with a validation suite.

**Kernel-complex overrides apply:** regardless of the subskill's output, the golden must comply with all Phase 1 rules above (no `.T`/`.t()`, shape comments on every intermediate, `# --- Module M1 ---` boundary markers). Apply these manually after subskill execution if needed.

### Step 3a. Golden implementation strategy: full vs. tiled

The normalized golden can adopt **two equivalent strategies**:

#### Strategy 1: Full computation (default)
Process the entire input tensor at once. Simplest and most straightforward.
```python
def attention_golden(q, k, v):
    scores = torch.matmul(q, k.transpose(-2, -1))
    probs = torch.softmax(scores, dim=-1)
    return torch.matmul(probs, v)
```

#### Strategy 2: Tiled computation (optional, recommended for complex kernels)
Split input into small tiles, compute each tile independently, and concatenate or accumulate results. This implementation pattern:
- Mirrors how PyPTO kernels actually execute (tile-by-tile)
- Allows early verification of boundary handling, padding, and accumulation logic
- Can expose tile-size effects on numerical precision before full PyPTO implementation
- Is essential for kernels with inherent tiling structure (attention with window size, blockwise matmul, FlashAttention patterns)

Example (batched tiling):
```python
def attention_golden_tiled(q, k, v, window_size=None):
    """Tiled attention golden (matches PyPTO kernel tile-by-tile execution)."""
    outputs = []
    for b in range(q.shape[0]):
        q_tile = q[b:b+1, ...]  # [1, h, t, d]
        k_tile = k[b:b+1, ...]
        v_tile = v[b:b+1, ...]
        scores = torch.matmul(q_tile, k_tile.transpose(-2, -1))  # [1, h, t, t]
        probs = torch.softmax(scores, dim=-1)
        out_tile = torch.matmul(probs, v_tile)  # [1, h, t, d]
        outputs.append(out_tile)
    return torch.cat(outputs, dim=0)
```

**When to choose tiled implementation:**
- Kernel spec explicitly describes tiling or loop-based computation
- Algorithm involves splitting, partial results, or state accumulation
- Need to verify tile-boundary edge cases before full PyPTO implementation

**Both strategies must produce identical numerical results** (within floating-point tolerance). If implementing both, include both in `{op}_golden.py` and verify equivalence in the validation suite.

### Step 3b. Build Golden function inventory (mandatory)

After the normalized golden is written, list every mathematical operation in `custom/plan/<operator_name>.md` → Golden function inventory:

```
| # | Golden operation          | Shape transformation              | PyPTO implementation | Line | Status |
|---|---------------------------|-----------------------------------|---------------------|------|--------|
| 1 | matmul(q, k^T)            | [B,H,T,K]@[B,H,K,T]->[B,H,T,T]  | pypto.matmul(...)   | L.42 | ✅     |
| 2 | softmax(scores, dim=-1)   | [B,H,T,T]->[B,H,T,T]             |                     |      | ❌     |
```

**Gate:** Do not proceed to Phase 2 until the inventory exists and the golden contains zero `.T`/`.t()` calls.

### Step 4. Validate normalized golden against original golden

Always validate with:
- same random seed,
- small shape,
- representative shape,
- boundary/edge shape,
- dtype-aware comparison,
- NaN/Inf checks,
- `assert_allclose` with the required tolerance policy.

If the normalized golden does not match, stop and fix. Do not begin PyPTO implementation.

### Step 5. Freeze the normalized golden

After the normalized golden matches:
- mark it frozen in the plan,
- use it as the single reference going forward,
- do not change it unless there is evidence the normalization itself is wrong.
