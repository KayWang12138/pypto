# PyPTO kernel source layout and design format

This document defines a **file organization pattern** and **documentation conventions** for PyPTO custom kernels. It applies to any operator (elementwise, reduction, attention-like recurrence, fusion, etc.): use the layers that match your complexity and omit the rest.

Goals:

- Human readers and LLM agents can **navigate by role** (reference vs PyPTO vs host glue vs test).
- **Debugging** stays tractable: each numbered stage matches a small function with a clear contract.
- **Portability**: swap math, tiling, or fusion without losing the overall structure.

**Canonical Python skeleton (complex-kernel workflow):** `.agents/pypto-kernel-custom-skills/skills/workflow/kernel-code-format/pypto_kernel_template.py` — agents **must** use this file as the starting layout for **each** staged module file and the **full** kernel under `custom/<op>/` (see **`skills/orchestration/lead-orchestrator/references/rules.md`** rule **17** and **`skills/orchestration/lead-orchestrator/references/rules.md`** rule **17**). This document (layers A–L) and that template stay aligned.

---

## 1. Recommended vertical layers (top to bottom)

Order sections in the source file roughly as follows. Each layer only depends on layers above it.

| Layer | Purpose | Typical names (examples) |
| --- | --- | --- |
| **A. Utilities** | Reusable diagnostics (optional). | `tensor_compare_report`, logging helpers |
| **B. Small math building blocks** | Pure PyTorch (or numpy) fragments of the algorithm, reused by reference and sometimes mirrored on device. | `norm_fwd`, `softmax_chunk`, … |
| **C. Forward reference** | Ground-truth forward in PyTorch, written under **explicit constraints** (see §3). | `forward_ref` |
| **D. Host-side constants** | Matrices and masks that replace forbidden ops in the reference (e.g. prefix sum via `matmul`). | `make_chunk_constants` |
| **E. Backward reference (decomposed)** | Private helpers that mirror one loop-body or one pipeline stage; the full backward stitches them. | `_slice_chunk_inputs`, `_stage_attn`, … |
| **F. Golden backward** | Single entry that implements the full backward in PyTorch for numeric verification. | `torch_golden_*_backward_ref` |
| **G. Cache / bridge** | Convert Python lists, nested caches, or layouts into flat tensors for the NPU path. | `prepare_cache_for_npu`, … |
| **H. PyPTO sub-kernels** | Small, named PyPTO regions: views, matmuls, fused stages. One conceptual step per function. | `pypto_slice_inputs`, `pypto_fused_stage_ab`, … |
| **I. Kernel implementation** | The actual `pypto.loop` nest, calling (H); no `@jit` here if you split impl vs entry. | `_your_op_kernel_impl` |
| **J. JIT entry** | `@pypto.frontend.jit` function: tensor signatures, `runtime_options`, `debug_options`; delegates to (I). | `your_op_kernel_npu` |
| **K. Host wrapper** | Allocates outputs, packs torch tensors, calls (J), reshapes results to user layout. | `pypto_function` |
| **L. Driver / test** | `main()` or pytest: config, forward ref, golden, reshape for PyPTO, compare. | `main` |

Not every kernel needs every layer. A minimal unary op might skip (C), (G), and most of (E)–(H); a training backward kernel typically needs (C)–(L).

---

## 2. Naming conventions (apply to any kernel)

| Pattern | Meaning |
| --- | --- |
| `forward_ref` / `*_forward_ref` | PyTorch reference for the forward pass. |
| `torch_golden_*` / `*_golden_*` | Full reference for backward or end-to-end numeric check. |
| Leading `_` | Private helper: one logical step, not intended as the public API. |
| `pypto_*` | Code that uses `pypto` APIs (views, matmul, loops, tile shapes). |
| `*_kernel_impl` | Implementation body containing `pypto.loop` and sub-kernel calls. |
| `*_kernel_npu` (or `*_jit`) | The `@pypto.frontend.jit` entry point with typed tensor signatures. |
| `pypto_function` (or `launch_*`, `run_*`) | Torch-side launcher: layout conversion + `kernel_npu(...)` + reshape. |

Keep **one primary “golden” name** per direction (e.g. one backward golden) so tests and docs stay grep-friendly.

---

## 3. Reference implementation constraints (document in code)

The reference is not “any PyTorch”; it should follow rules you **state in a header comment**, for example:

- Allowed: `matmul`, elementwise ops, `sum` over last dim, explicit loops over batch/head/chunk.
- Disallowed (if they complicate NPU lowering or differ from your PyPTO path): `cumsum`, `masked_fill`, `tril`/`triu` factory ops, `flip`, or high-rank tensors beyond what the device supports.

Reproduce the same math using **explicit matrices** (e.g. `C_cum @ x` instead of `cumsum`) when the PyPTO side uses that pattern. This keeps **diff debugging** one-to-one between reference steps and `pypto_*` blocks.

---

## 4. Stage markers inside long functions

For long `forward_ref` or backward golden loops, use **labeled stages** so they map to PyPTO modules:

```text
# ===== (A) stage description =====
# ===== (B) stage description =====
# ===== (C) stage description =====
```

Use the **same letters or names** in the PyPTO fusion comments (e.g. “Fused Module A + B”) so a mismatch narrows to one pair of functions.

---

## 5. Contract tables (fill for your kernel)

Maintain a short **tensor contract** (in the module docstring or a dedicated comment block):

| Name | Shape (symbolic) | dtype | Producer | Consumer | Notes |
| --- | --- | --- | --- | --- | --- |
| … | … | … | forward_ref / cache | backward ref / pypto | e.g. “flattened BT×BT per chunk” |

For **forward → backward** dependencies, list **cache keys** and whether each tensor is required for backward:

| Cache key | Shape | Needed for bwd? |
| --- | --- | --- |
| … | … | yes / no |

---

## 6. Mapping reference steps to PyPTO

Use a **one-to-many table** (in documentation, not necessarily code):

| Ref helper / stage | PyPTO function(s) | Notes |
| --- | --- | --- |
| `_ref_stage_alpha` | `pypto_stage_alpha` | Same math, different layout or views. |
| … | … | … |

When precision diverges, compare **stage outputs** (saved tensors) instead of only the final output.

---

## 7. PyPTO sub-kernel responsibilities

Each `pypto_*` function should:

1. **Do one thing** (e.g. “build decay matrix”, “fuse local attn + recurrence”).
2. **Set tile / pass options locally** if needed (`set_vec_tile_shapes`, `set_pass_options`, `set_cube_tile_shapes`) and document why.
3. **Return** all tensors the next stage needs (avoid hidden globals).

The `*_kernel_impl` then reads as a **high-level recipe**: slice → stage1 → stage2 → write outputs.

---

## 8. JIT entry vs implementation split

- **`_your_op_kernel_impl`**: all dynamic indexing, `pypto.view`, loops, and calls to `pypto_*` helpers.
- **`your_op_kernel_npu`**: `@pypto.frontend.jit`, static signatures (`pypto.DYNAMIC` / `pypto.STATIC`), `runtime_options`, `debug_options`; body is a thin call to the impl.

This split makes it easier to **swap options** or **reuse the impl** from tests without recompiling different JIT shells.

---

## 9. Host wrapper (`pypto_function`)

Responsibilities:

1. Move/cache tensors to the **device and layout** the JIT entry expects (flatten, transpose, `expand`, dtype).
2. **Allocate** output buffers.
3. Invoke `*_kernel_npu(*inputs, *outputs)`.
4. **Reshape** outputs back to the user-facing layout (e.g. `[B, T, H, D]`).

Keep I/O reshaping **out** of the JIT function when possible.

---

## 10. Test driver (`main` or pytest)

Suggested structure:

1. **Config**: shapes, dtypes, chunk size `BT`, seeds, device id, `run_mode`.
2. **Inputs**: random tensors with `requires_grad` if testing autograd-related paths.
3. **Constants**: `make_chunk_constants` or equivalent.
4. **Forward reference**: run `forward_ref`, get outputs + cache.
5. **Upstream grads**: random `do`, `dht`, etc.
6. **Golden backward**: `torch.no_grad()` + golden function.
7. **PyPTO path**: adapt cache tensors to PyPTO layout; call `pypto_function`.
8. **Compare**: a detailed per-tensor report helper (or per-tensor `torch.allclose`) for each output.

---

## 11. Shape Annotation Convention

Every tensor assignment and tiling configuration line **must** carry an inline shape comment.

**1. Every tensor assignment gets a shape comment**

```python
q = pypto.view(q_in, [B, N, Sq, D])          # [B, N, Sq, D]
k = pypto.view(k_in, [B, N, Skv, D])         # [B, N, Skv, D]
scores = pypto.matmul(q, k, pypto.DT_BF16)   # [B, N, Sq, Skv]
```

**2. Matmul uses the contraction form**

```python
# [M, K] @ [K, N] -> [M, N]
out = pypto.matmul(a, b, pypto.DT_BF16)      # [M, N]
```

**3. Tile config lines show tile shape**

```python
pypto.set_vec_tile_shapes(1, 1, 8, 8)                         # tile dimensions per doc
pypto.set_cube_tile_shapes([128, 128], [64, 128], [128, 256]) # each list is [L0, L1]; see below
```

> **`set_cube_tile_shapes` parameter rules** — each of `m`, `k`, `n` is a **2-element list `[L0, L1]`**,
> NOT a single-element list. Constraints (from `docs/api/config/pypto-set_cube_tile_shapes.md`):
>
> - `0 < mL0 <= mL1` and `mL1 % mL0 == 0` (same for `k`, `n`).
> - `kL0, kL1, nL0, nL1`: 32-byte aligned (**FP32 input: 16-element aligned** instead).
> - L0A/L0B/L0C and L1 buffer budgets must fit; a common safe baseline for FP16/BF16/FP32 is
>   `[128, 128], [64, 128], [128, 256]`, tuned per shape.
> - `enable_split_k=True` is only valid when inputs are 2D (not 3D/4D).
>
> ❌ `pypto.set_cube_tile_shapes([16], [32], [64])` — WRONG (1-element lists).
> ✅ `pypto.set_cube_tile_shapes([16, 32], [32, 64], [64, 128])` — 2-element lists.

**4. Loop body tensors show the slice shape, not the full shape**

```python
for i in pypto.loop(range(Sq // tile_s), idx_name="i"):
    q_tile = pypto.view(q, [tile_s, D], ...)  # [tile_s, D]  (slice of [Sq, D])
    s_tile = pypto.matmul(q_tile, k_t, ...)   # [tile_s, Skv]
```

**5. Dynamic axes use the symbolic name, not `?`**

```python
x = pypto.view(x_in, [B, S, H])              # [B, S, H]  S=dynamic
```

**6. Reductions annotate both input and output shape**

```python
row_max = pypto.amax(scores, dim=-1)          # [B, N, Sq, Skv] -> [B, N, Sq, 1]
```

**Do not annotate:** import lines, `print`/logging, or plain Python scalars (`tile_m = 16`).

---

## 12. Checklist for new kernels (generic)

- [ ] Math spec and **symbolic shapes** written down.
- [ ] `forward_ref` (if applicable) respects **documented constraints**.
- [ ] Backward golden matches forward cache contract.
- [ ] Each non-trivial loop body chunk extracted as `_helper` or `pypto_*` with a **one-line role comment**.
- [ ] Ref stage labels align with PyPTO module comments.
- [ ] JIT signatures match actual buffer ranks (dynamic vs static dimensions).
- [ ] Host wrapper documents **layout** assumptions (row-major flatten order, head grouping, etc.).
- [ ] Test compares **all** outputs relevant to the API.

---

## 12. Minimal template (skeleton only)

```text
# --- utilities (optional) ---

# --- small torch helpers ---

# --- forward_ref (constraints in header comment) ---

# --- constants ---

# --- backward ref helpers (_*) ---

# --- torch_golden_* backward ---

# --- prepare_* cache bridge ---

# --- pypto_* sub-kernels ---

def _my_kernel_impl(...):
    for ... in pypto.loop(...):
        ...
        # call pypto_* helpers

@pypto.frontend.jit(...)
def my_kernel_npu(... typed tensors ...):
    _my_kernel_impl(...)

def pypto_function(... torch tensors ...):
    # allocate, pack, call my_kernel_npu, reshape
    ...

def main():
    # config → forward_ref → golden → pypto_function → compare
```

Adapt depth: a forward-only inference kernel omits backward/golden sections; a fused elementwise kernel may inline “sub-kernels” into a single `pypto_*` or the impl.
