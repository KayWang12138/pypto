# Debug playbook — do not stop on cryptic error codes

**Audience:** Agents working on complex PyPTO kernels under this bundle.

**Do not skip:** This file is part of the mandatory path when stuck — see the Debug Agent contract (`.opencode/agents/debug.md`).

**Policy:** **`Errcode: FFFFF!`**, **`UNKNOWN`**, **`0x3FFFF`**, or other opaque / scary-looking error codes are **not** a reason to end the session, report failure, or "give up" until the Debug Agent's Hard rules (`.opencode/agents/debug.md`, "If after 3 fix/re-verify cycles the module still fails, stop and report the blocker") actually apply. **Token and turn budget are not a constraint** for this workflow: keep iterating, gathering evidence, and trying the next structured strategy until the problem is resolved or a true blocker is proven.

**What “give up” looks like (forbidden here):** stopping after only printing the error code, declaring the task impossible without consulting docs, skipping logs, or refusing to try another angle.

---

## 1. First actions (always)

1. **Capture the full message** — stderr, Python traceback, and any **`pypto-log*.log`** / device log lines. Search for `Errcode`, `ErrCode`, `F` + digits, `aicore`.
2. **Route the error code** — read `docs/trouble_shooting/README.md` and open the component doc for that prefix (e.g. FUNCTION / `docs/trouble_shooting/function.md` for many `F2xxxx`-style codes). For **`F21004`** / **`REGISTER_COPY`** / invalid vector tile, see **§4** below first.
3. **Append to `custom/<op>/plan.md` → Development & debug log** — what failed, command, hypothesis, next step. No empty "stopped" endings.

---

## 2. UNKNOWN / FFFFF / insufficient detail

- Enable **verbose logging** as documented in the relevant `docs/trouble_shooting/*.md` (e.g. `ASCEND_GLOBAL_LOG_LEVEL`, log paths — see `function.md` for FUNCTION-related errors).
- If the graph is suspect: follow the **computation-graph / program-dump** guidance linked from troubleshooting docs (upstream docs may label this “view computation graph”).
- **Do not** treat “unknown” as terminal; treat it as **need more signal** (logs, smaller repro, earlier checkpoint).

---

## 3. Kernel-specific: narrow the blast radius

1. Confirm which **staged set** is current (e.g. `staged/<op>_module12_impl.py` + `_golden.py` + `test_*.py`; see `skills/phase2-phase3-construction/SKILL.md` → Staged sets). Run `extract_pypto_calls.py` on the staged `*_impl.py` (or, after Phase D, on the canonical `custom/<op>/<op>_impl.py`).
2. Run **`python3 .agents/skills/ci-and-layout-check/scripts/extract_pypto_calls.py <kernel.py>`** (canonical path in this repo; some checkouts document `.agents/skills/ci-and-layout-check/scripts/` — use the path that exists) and follow the **op-by-op check protocol** in `skills/debugging/SKILL.md`.
3. Stay in **one active module** at a time; stub downstream with golden-fed tensors if needed.
4. Re-run **module boundary** checks with **`detailed_tensor_compare`** and log results in **Per-module verification log**.
5. After fixes, re-run the current staged set's `test_<op>_module<suffix_k>.py` (or, after Phase D, `test_<op>.py`) and confirm **every** kernel output — not only the first tensor.

### 3a. Layout CI and pass regressions (quick pointers)

- **Layout / `pypto_function` loops:** After meaningful edits under `custom/`, run `skills/ci-and-layout-check/run_validate_layout.sh` from the repository root using the `bash` line in `skills/ci-and-layout-check/CI.md`. **Exit 1** means fix plan / test / staged naming issues **or** remove `for ... in range(...)` inside `pypto_function` — express that iteration with `pypto.loop` in `_<op>_kernel_impl` / the JIT kernel per `skills/pypto-op-develop/references/kernel-layer-format.md` §7.1.
- **Pass / compile failure right after a graph edit:** If logs show **PASS**-range codes (**`F4` / `F5`**, see **`docs/trouble_shooting/README.md`** → **`pass.md`**), follow **`.agents/skills/pypto-pass-error-fixer/SKILL.md`**, **bisect** the PyPTO graph (e.g. last known-good **staged** file vs current), and re-check API constraints (**`query_op_index` / `docs/`**) before large rewrites. Re-run **`extract_pypto_calls.py`** on the failing file to see whether a new op ordering triggered the pass.

---

## 4. F21004 (`INVALID_VAL` / `TileShape::Current().GetVecTile()` invalid) — `REGISTER_COPY` and vector tile shapes

**Cause (framework):** **`F21004`** is raised in **`Operation`’s constructor** when **`TileShape::Current().GetVecTile()`** is invalid (e.g. `framework/src/interface/operation/operation.cpp` ~191–195).

**Why `REGISTER_COPY` shows up:** **`REGISTER_COPY`** is an **AIV** op. It needs a **valid vector tile** even when the **compiler inserts** that op (e.g. memory-conflict passes). There is **no** separate “REGISTER_COPY tiling” registration — the same **vec tile** rules apply.

**When is `VecTile` valid?** The stored list must be **non-empty** and **every value must be &gt; 0** (`tile_shape.cpp`).

### Fix — what to do in PyPTO Python

1. **Before any vector/tensor work in that scope** (including code that eventually leads to **`REGISTER_COPY`**), set **vector** tile sizes:

   ```python
   import pypto
   pypto.set_vec_tile_shapes(1, 1, 128, 128)   # positive ints per doc
   ```

2. Call it at the **start** of your **`@jit` / kernel function body**, and **again after any scope change** if your API uses **nested scopes**.

3. **Do not rely only on `set_cube_tile_shapes`** — cube tile does **not** replace vector tile for AIV ops like **`REGISTER_COPY`**.

4. If you use **both** cube and vector ops:

   ```python
   pypto.set_cube_tile_shapes([128, 128], [128, 128], [128, 128])
   pypto.set_vec_tile_shapes(1, 1, 128, 128)
   ```

### If it still fails

- **Invalid args or zeros** — ensure all args are positive integers. Refer to **`docs/api/config/pypto-set_vec_tile_shapes.md`** for your PyPTO version's requirements (see **`skills/pypto-op-develop/SKILL.md`** §实现注意点 #7 / 常见错误 #4).
- **Wrong order** — anything that **adds ops** (reshape, view, passes inserting copy) must run **after** **`set_vec_tile_shapes`** on that execution path.
- **Symbolic / dynamic shapes** — ensure tile args resolve to **concrete positive integers** (see **`set_vec_tile_shapes`** + **`SymbolicScalar`** in `python/pypto/_controller.py` in your tree).

**Bottom line:** **`F21004` on `REGISTER_COPY`** is almost always **“no valid `vec_tile_shapes` in the current scope when this op was created.”** Fix it with **`pypto.set_vec_tile_shapes(...)`** with **all positive sizes** before those ops run.

---

## 5. When logs point to device / AICore

- Load **`.agents/skills/pypto-aicore-error-locator/SKILL.md`** and follow its steps when the failure is an **aicore error** / device-side trace problem.

---

## 6. Error-code quick reference (repo)

- `.agents/skills/pypto-op-develop/references/error-code-troubleshooting.md` — flow for `Errcode: Fxxxxx!`.
- `docs/trouble_shooting/function.md` — e.g. **INVALID_VAL (0x21004)**, **UNKNOWN (0x3FFFF)**.
- **F21004 / vec tile / `REGISTER_COPY`:** see **§4** above.

---

## 7. When stopping is allowed

Only align with the Debug Agent's stop conditions (`.opencode/agents/debug.md` Hard rules: 3 fix/re-verify cycles exhausted, missing reference, impossible golden, fundamental framework block, missing user-provided logs when required, or proven blind speculation after exhaustive structured attempts). **A single cryptic error line is never enough.**

---

## 8. Example kernels — debug practice (from `examples/`)

This section condenses **`.agents/pypto-example-debug-practice/DEBUG_PRACTICE.md`**: notes from re-deriving runnable kernels under **`examples/`** (blind scratch tree **`custom/debug_scratch_examples/`** vs official scripts). It complements **§1–§7**; run examples per **`examples/README.md`** (`--list`, `--run_mode sim`, **`TILE_FWK_DEVICE_ID`** for NPU). Error routing remains **`docs/trouble_shooting/README.md`**.

### 8.1 Global patterns (deduplicated)

- **Run mode and decorators:** Most examples set `global_run_mode = pypto.RunMode.NPU` then override from **`_peek_run_mode_from_argv`** so module-level `@pypto.frontend.jit(runtime_options={"run_mode": global_run_mode})` sees **`sim`** when `python3 script.py --run_mode sim`. Parsing **`--run_mode`** only inside **`main()`** leaves the decorator bound to import-time mode — wrong or surprising behavior. When **`--run_mode sim` seems ignored**, compare your pattern to **`examples/README.md`** and the **`_peek_run_mode_from_argv`** idiom in beginner examples.
- **Vector tile before AIV / implicit copy:** Call `pypto.set_vec_tile_shapes` with positive integers as required by your PyPTO version's `docs/api/config/pypto-set_vec_tile_shapes.md` before ops that may trigger AIV / `REGISTER_COPY` (see §4 / `F21004` in `docs/trouble_shooting/function.md`). When expanding shapes or hitting cryptic vec-tile errors, consult the documentation for your version and `skills/pypto-op-develop/SKILL.md` §实现注意点 #7.
- **`pypto.loop` vs host `for`:** Graph iteration belongs in `pypto.loop`; do not drive tile/batch work with plain Python `for range` inside the traced kernel in ways that violate layout rules (see `skills/pypto-op-develop/references/kernel-layer-format.md` §7.1 — Layer K forbids host loops driving algorithmic iteration). Official transform/loop examples nest `pypto.loop` and use `view` / `assemble`.
- **`out.move(...)` vs slicing assign:** Many kernels use **`out.move(pypto.op(...))`** instead of **`out[:] = ...`**. Both appear in-tree; when debugging shape/dtype mismatches, check which pattern the reference uses and match it.
- **Cube vs vec:** **`set_cube_tile_shapes`** for matmul-like cube ops; still set vec tiles when vector ops or compiler-inserted copies participate.
- **Error routing:** Map **`Errcode: F2…`** → `docs/trouble_shooting/function.md`, **`F4/F5`** → `pass.md`, **`F9`** → `simulation.md`, etc. (`docs/trouble_shooting/README.md` table).
- **Multi-output and large scripts:** Use **`extract_pypto_calls.py`** on the failing file (see **§3** step 2 for path) to see op ordering at a glance.
- **Environment / exit codes:** If **`import pypto`** fails, fix the install per **`docs/install/build_and_install.md`** before treating kernel logic as broken. When capturing whether a script failed, remember a shell pipeline can mask Python’s exit code unless you use **`set -e`**, **`set -o pipefail`**, or **`python3 ...; echo $?`** without masking.
- **Golden reference implementation patterns:** Golden functions (used for precision verification) can adopt **two equivalent strategies**:
  - **Full computation:** Process the entire input tensor at once (default, simpler).
  - **Tiled computation:** Split input into small tiles, compute each tile independently, and concatenate results. Tiled golden is closer to how PyPTO kernels actually execute (tile-by-tile), making it better for verifying boundary handling, accumulation logic, and tile-size effects on numerical precision. See **`skills/pypto-golden-generate/SKILL.md`** → **§4 実装策略：全量 vs 分块** for patterns and when to use each.

### 8.2 Example inventory and recurring failures

- **Scope (typical tree):** **23** runnable example kernel **`*.py`** files under **`examples/`** (excluding **`validate_examples.py`** / harness scripts). Optional scratch mirrors live under **`custom/debug_scratch_examples/<sanitized_path>/`**.
- **Top recurring failure families** when PyPTO is available: (1) **Invalid / missing vec tile** → **`F21004` / REGISTER_COPY**; (2) **`run_mode` / decorator binding** vs **`--run_mode sim`**; (3) **Cube vs vec** tiling confusion on matmul; (4) **Pass / stitch** on large fused graphs (**`F4` / `F5`**); (5) **View/assemble + dynamic** shape mismatches.

### 8.3 Integration tracks (examples vs ACL / cost model)

- **`examples/03_advanced/aclgraph/aclgraph.py`:** Uses **`@pypto.frontend.jit()`** with **Torch Dynamo** **`@allow_in_graph`** and **`FakeTensor`** guards — a different integration path from scripts that only pass **`runtime_options={"run_mode": ...}`**. See **`docs/tutorials/network_integration/pytorch_integration.md`** for return/assign patterns compatible with graph capture.
- **`examples/03_advanced/cost_model/cost_model.py`:** Uses **`runtime_options`** such as **`stitch_cfgcache_size`** and **`run_mode: pypto.RunMode.SIM`**; swimlane / JSON artifacts may appear under **`./output`**. If “cost model produced nothing”, verify SIM options and output paths before blaming the softmax math.

---

## Final reminder

**Prefer ten documented failed attempts with log citations over one early exit.** Unknown error codes mean **escalate evidence and method**, not **stop**.

---

## 9. General PyPTO Kernel Development Debug Guide

*Agent-learned patterns from GDR kernel development. Add to this section when discovering new patterns.*

### 9.1 JIT Signature Parsing

#### Issue: `from __future__ import annotations` breaks JIT

**Symptom:** `RuntimeError: Non-tensor parameter 'q_in' must not be a torch.Tensor. Use positional arguments for tensors.`

**Root Cause:** PEP 563 string annotations cause all type hints to be stored as strings instead of objects.

**Diagnosis:**
```python
# Check annotations - they should be pypto.Tensor objects, not strings
func = kernel._original_func
print(func.__annotations__)  # If strings, it's the import issue
```

**Solution:** Remove `from __future__ import annotations` from files using `@pypto.frontend.jit`.

```python
# WRONG - causes JIT to fail
from __future__ import annotations
@pypto.frontend.jit()
def kernel(x: pypto.Tensor(...)):
    pass

# CORRECT
@pypto.frontend.jit()
def kernel(x: pypto.Tensor(...)):
    pass
```

---

### 9.2 Dynamic Shapes

#### Issue: `set_vec_tile_shapes` requires concrete values

**Symptom:** `ValueError: Not concrete value` when using symbolic dimensions.

**Solution:** Use module-level constants or function parameters for tile shapes:

```python
# WRONG
@pypto.frontend.jit()
def kernel(x: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC], pypto.DT_FP32)):
    B, T = x.shape
    pypto.set_vec_tile_shapes(B, T, 32, 32)  # FAILS - B, T are symbolic

# CORRECT - use concrete constants
TILE_M, TILE_N = 32, 32
@pypto.frontend.jit()
def kernel(x: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC], pypto.DT_FP32)):
    pypto.set_vec_tile_shapes(TILE_M, TILE_N, 32, 32)  # WORKS
```

---

#### Issue: `pypto.loop` with symbolic bounds

**Symptom:** 
```
ValueError: Invalid value type
Errcode: F21004!
op [MUL]tile shape not set
```

**Root Cause:** `pypto.loop` requires **concrete integer** start/stop/step values. Using symbolic expressions like `B * H` from tensor shapes fails.

```python
# WRONG - B and H are symbolic from tensor shape
for session in pypto.loop(range(B * H), name="sessions"):
for i in pypto.loop(range(nt), name="chunks"):  # nt = T // bt is symbolic
```

**Solution:** Pass loop bounds as **concrete integer parameters**:

```python
# Kernel signature: pass B, H, nt as concrete parameters
@pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.NPU})
def kernel(
    q_in: pypto.Tensor([], pypto.DT_FP32),
    ...
    B: int, H: int, nt: int,  # Concrete loop bounds
):
    for session in pypto.loop(0, B * H, 1, name="sessions"):
        for c in pypto.loop(0, nt, 1, name="chunks"):
            ...

# Caller: pass concrete values
kernel(q, k, ..., B, H, nt)
```

**Note:** Inside the loop body, `b = session // H` and `h = session % H` are still symbolic but usable in `pypto.view` offsets.

---

#### Issue: Tensor indexing with symbolic indices

**Symptom:** `TypeError: Cannot convert symbols to int` when using `tensor[symbolic_index]`.

**Solution:** Use `pypto.view` with symbolic offsets instead of direct indexing:

```python
# WRONG
result = tensor[idx, :, :]  # idx is symbolic - FAILS

# CORRECT - use view with offsets
view = pypto.view(tensor, [1, T, K], [idx, 0, 0])
result = pypto.matmul(...)  # operate on the view directly
```

---

#### Issue: pypto.view shape/offsets dimension mismatch

**Symptom:**
```
RuntimeError: Errcode: F21004!
Their size actually are 4 and 2, func GetViewValidShape
```

**Root Cause:** The `shape` and `offsets` must have the **same number of elements**.

```python
# WRONG - shape has 2 dims, offsets has 4 elements
pypto.view(tensor, [K, V], [b, h, 0, 0])

# CORRECT - use matching dimensions, then reshape
pypto.view(tensor, [1, 1, K, V], [b, h, 0, 0]).reshape([K, V])
```

**Rule:** `len(shape) == len(offsets)` always.

**Common patterns:**
```python
# Tensor [B, T, H, K], view [bt, K] at [b, t0, h, 0]
view = pypto.view(tensor, [1, bt, 1, K], [b, t0, h, 0]).reshape([bt, K])

# Tensor [B, H, K, V], view [K, V] at [b, h, 0, 0]
view = pypto.view(tensor, [1, 1, K, V], [b, h, 0, 0]).reshape([K, V])

# Tensor [B, T, H], view [bt] at [b, t0, h]
view = pypto.view(tensor, [1, bt, 1], [b, t0, h]).reshape([bt])

# Tensor [B, H, nt, bt, bt], view [bt, bt] at [b, h, c, 0, 0]
view = pypto.view(tensor, [1, 1, 1, bt, bt], [b, h, c, 0, 0]).reshape([bt, bt])
```

---

### 9.4 Comprehensive pypto.view Guide

#### Signature
```python
pypto.view(
    input: pypto.Tensor,
    shape: List[int],           # Must be concrete integers
    offsets: List[Union[int, pypto.SymbolicScalar]],
    valid_shape: Optional[List[Union[int, pypto.SymbolicScalar]]] = None
) -> pypto.Tensor
```

#### Golden Rule
**`len(shape) == len(offsets)`** - This is mandatory!

#### Best Practices

**1. Always match dimensions:**
```python
# Tensor shape: [B, T, H, K] (4D)
# Offsets: [b, t0, h, 0] (4 elements)
# View shape must be 4D: [1, bt, 1, K]
qc = pypto.view(q_norm, [1, bt, 1, K], [b, t0, h, 0]).reshape([bt, K])
```

**2. Use 1s to pad unused dimensions:**
```python
# Tensor [B, H, K, V] → view at [b, h]
# Use [1, 1, K, V] to match 4-element offsets [b, h, 0, 0]
s = pypto.view(state, [1, 1, K, V], [b, h, 0, 0]).reshape([K, V])
```

**3. For 1D/2D tensors with multi-dim offsets:**
```python
# Tensor [B, T, H] with 3D offsets [b, t0, h]
# Use [1, bt, 1] to match 3 elements
betac = pypto.view(beta_in, [1, bt, 1], [b, t0, h]).reshape([bt])
```

**4. For 5D tensors:**
```python
# Tensor [B, H, nt, bt, bt] with 5D offsets [b, h, c, 0, 0]
# Use [1, 1, 1, bt, bt] to match 5 elements
A_c = pypto.view(A_in, [1, 1, 1, bt, bt], [b, h, c, 0, 0]).reshape([bt, bt])
```

**5. Assemble back with reshape:**
```python
# When assembling, reshape to match original tensor's view dimensions
pypto.assemble(s.reshape([1, 1, K, V]), [b, h, 0, 0], output)
```

#### Common Mistakes

| Mistake | Error | Fix |
|---------|-------|-----|
| Shape dims ≠ offset dims | `Their size actually are X and Y` | Pad with 1s |
| Using `[bt, K]` with 4 offsets | Dimension mismatch | Use `[1, bt, 1, K]` |
| Forgetting reshape after view | Wrong shape in computation | Add `.reshape([bt, K])` |

#### Dimension Padding Pattern
When the desired view has fewer dims than the offsets:
```
Original:  [K, V]  desired
Offsets:   [b, h, 0, 0]  has 4 elements
Solution:  Pad shape: [1, 1, K, V]
Result:    pypto.view(tensor, [1, 1, K, V], [b, h, 0, 0]).reshape([K, V])
```

---

### 9.6 SIM Mode Limitations

#### Issue: SIM mode produces garbage values

**Symptom:** Extreme errors (10^20+) in SIM mode, but kernel may work on NPU.

**Root Cause:** SIM mode has fundamental limitations with:
- Dynamic shape handling
- Cube tile configuration
- Memory operations

**Solution:**
1. Verify mathematical correctness against golden reference
2. Test on actual NPU hardware
3. Don't rely on SIM mode for precision validation

```python
@pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.NPU})
def kernel(...):
    ...
```

**Warning:** SIM mode is for basic execution testing, NOT precision validation.

---

#### Issue: `operand1 dim[0] = -1` in SIM mode

**Symptom:** Dimension becomes -1 when using dynamic shapes in SIM mode.

**Solution:** Use static shapes for SIM mode testing, or accept SIM limitations for dynamic shapes.

---

#### Issue: `Invalid tile values: kL0=0, kL1a=0...`

**Symptom:** Cube tiling validation fails in SIM mode.

**Solution:** This is a SIM mode limitation. The kernel may work on actual NPU hardware.

---

### 9.7 Tensor Operations

#### pypto.matmul requires 2D+ tensors

**Error:**
```
RuntimeError: Tensor dimension mismatch. Expect input_dim == mat2_dim and both in [2, 3, 4], got input_dim: 2, mat2_dim: 1.
```

**Cause:** `pypto.matmul` requires both input tensors to have 2+ dimensions. 1D tensors must be reshaped to 2D.

**Common case - vector-matrix multiplication:**
```python
# c_cum is [bt, bt] (2D), gc_raw is [bt] (1D)
# WRONG
g_cum = pypto.matmul(c_cum, gc_raw, ...)

# CORRECT - reshape to 2D
gc_raw_2d = gc_raw.reshape([bt, 1])
g_cum = pypto.matmul(c_cum, gc_raw_2d, ...).reshape([bt])
```

**Pattern for 1D results from matmul:**
```python
# When result should be [bt] but matmul gives [bt, 1]
result = pypto.matmul(matrix, vector_2d, ...).reshape([bt])
```

#### Broadcasting
Use `pypto.reshape` to add/remove dimensions for broadcasting:

```python
# WRONG
result = tensor * scalar  # scalar needs explicit reshape

# CORRECT
scalar_reshaped = pypto.reshape(scalar, [bt, 1])
result = pypto.mul(tensor, scalar_reshaped)
```

#### Transpose
```python
transposed = pypto.transpose(tensor, dim0, dim1)
```

#### Zeros initialization
```python
zeros = pypto.zeros([M, N], pypto.DT_FP32)
```

#### Element-wise operations
```python
negated = pypto.mul(tensor, -1.0)  # multiply by negative one
```

---

### 9.8 Common Patterns

#### Pattern: Multi-session with state carry
```python
for session in pypto.loop(range(B * H), name="sessions"):
    b = session // H
    h = session % H
    
    state = pypto.view(initial_state, [K, V], [b, h, 0, 0])
    
    for c in pypto.loop(range(nt), name="chunks"):
        # process chunk
        ...
        state = updated_state
    
    output[b, h, :, :] = state
```

#### Pattern: Constant tile shapes at module level
```python
# Module-level constants for tile shapes
TILE_SESSIONS = 16
TILE_CHUNKS = 4
TILE_BT = 16
TILE_KV = 64

@pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.NPU})
def kernel(...):
    pypto.set_vec_tile_shapes(TILE_SESSIONS, TILE_CHUNKS, TILE_BT, TILE_KV)
    ...
```

#### Pattern: Precompute on host
For operations not supported by PyPTO (e.g., `torch.linalg.solve_triangular`):
```python
def make_host_constants(bt, k, g_raw, beta, device):
    # Precompute on CPU/GPU
    A = torch.linalg.solve_triangular(...)
    return A

# In kernel, receive as constant
@pypto.frontend.jit()
def kernel(A_in: pypto.Tensor(...)):
    A_c = pypto.view(A_in, [bt, bt], [b, h, c, 0, 0])
    ...
```

#### Pattern: Reverse iteration for backward
```python
for i in pypto.loop(range(nt), name="chunks_reverse"):
    c = nt - 1 - i  # reverse chunk index
    ...
```

---

### 9.9 Debug Checklist

| Check | Command/Method |
|-------|---------------|
| JIT signature | Check `_original_func.__annotations__` are not strings |
| Dynamic shapes | Verify tile shapes are concrete |
| SIM mode | Accept limitations, test on NPU |
| Imports | Ensure `pypto` is imported correctly |
| Type hints | No `from __future__ import annotations` |

---

### 9.10 Testing Strategy

1. **Syntax check:** `python -m py_compile module.py`
2. **Golden comparison:** Compare against PyTorch reference
3. **SIM mode:** For basic execution (not precision)
4. **NPU mode:** For actual precision validation

---

### 9.11 Common Error Messages

| Error | Cause | Solution |
|-------|-------|----------|
| `Non-tensor parameter 'x' must not be a torch.Tensor` | String annotations | Remove `from __future__ import annotations` |
| `Not concrete value` | Symbolic in tile shapes | Use concrete constants |
| `Cannot convert symbols to int` | Symbolic in loop/indexing | Use `pypto.view` with offsets |
| `ValueError: Invalid value type` | Symbolic in `pypto.loop` | Pass loop bounds as concrete int parameters |
| `Errcode: F21004 tile shape not set` | Missing `set_vec_tile_shapes` before ops | Call `set_vec_tile_shapes` first |
| `Errcode: F21004 Their size actually are X and Y` | `pypto.view` shape/offsets mismatch | Ensure `len(shape) == len(offsets)` |
| `operand1 dim[0] = -1` | SIM mode dynamic shape issue | Test on NPU |
| `Invalid tile values` | SIM mode tiling issue | Test on NPU |

---

### 9.12 Key PyPTO API Notes

- **`pypto.DYNAMIC`** - Dynamic dimension marker for tensor type hints
- **`pypto.DT_FP32`, `pypto.DT_BF16`, etc.** - Data type enums
- **`pypto.RunMode.NPU`** - Run on actual NPU hardware
- **`pypto.RunMode.SIM`** - Run in simulation mode (limited)
- **`pypto.loop(start, stop, step)`** - Kernel loop construct
- **`pypto.view(tensor, shape, offsets)`** - Tensor view/slice
- **`pypto.matmul(a, b, out_dtype=...)`** - Matrix multiplication
- **`pypto.exp`, `pypto.mul`, `pypto.add`, etc.** - Element-wise ops
- **`pypto.transpose(t, dim0, dim1)`** - Transpose
- **`pypto.reshape(t, shape)`** - Reshape for broadcasting
- **`pypto.zeros(shape, dtype)`** - Create zeros tensor
- **`pypto.set_vec_tile_shapes(...)`** - Set vector tile configuration
- **`pypto.set_cube_tile_shapes(...)`** - Set cube tile configuration

---

### 9.13 Tensor Shape Specifications

#### Issue: Shape Size Exceeds INT32_MAX

**Error:**
```
RuntimeError: Errcode: FFFFFF!
The shape size of tensor must less than or equal to INT32_MAX(2,147,483,647)
```

**Root Cause:** Using explicit shape specs with `pypto.DYNAMIC` in tensor annotations:
```python
# WRONG - causes shape size error
x: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC, pypto.DYNAMIC, pypto.DYNAMIC], pypto.DT_FP32)
```

**Solution:** Use empty brackets `[]` for shape inference:
```python
# CORRECT - shape inferred from actual tensor
x: pypto.Tensor([], pypto.DT_FP32)
```

**Reference:** All working PyPTO examples use `pypto.Tensor([], dtype)` pattern.

---

### 9.14 Python Operators Inside PyPTO JIT

**Discovery:** Python operators **DO work** inside `@pypto.frontend.jit` decorated functions!

**Evidence from working examples:**
```python
# From pypto_l2norm_bwd
dot_q = (dyq * yq).sum(-1, keepdim=True)
d = dyq * rstd_yq - dot_q * yq * rstd_yq
```

**Recommendation:** Use Python operators for cleaner code:
```python
# VERBOSE (unnecessary)
result = pypto.mul(pypto.mul(a, b), pypto.add(c, d))
result = pypto.sum(x, dim=-1, keepdim=True)
result = pypto.rsqrt(pypto.add(sum_sq, eps))

# PREFERRED (clean and works!)
result = a * b * (c + d)
result = x.sum(-1, keepdim=True)
result = (sum_sq + eps).rsqrt()
```

**Supported Method Chains:**
```python
tensor.T              # transpose
tensor.exp()          # element-wise exp
tensor.reshape([...]) # reshape
tensor.sum(-1)        # sum along last dim
tensor.rsqrt()       # reciprocal square root
tensor.abs()         # absolute value
tensor.sqrt()        # square root
tensor.neg()         # negation
```

---

### 9.15 Tile Shape Configuration

**Issue:** Fixed tile shapes may not match actual tensor dimensions.

**For simple kernels without loops:**
```python
B, T, H, K = x.shape
pypto.set_vec_tile_shapes(B, H, T, K)
```

**For complex kernels with loops:**
```python
TILE_0 = 16   # first tile dimension
TILE_1 = 4    # second tile dimension
TILE_2 = 8    # third tile dimension
TILE_3 = 32   # fourth tile dimension
pypto.set_vec_tile_shapes(TILE_0, TILE_1, TILE_2, TILE_3)
```

**Rule:** Tile shape values should divide evenly into tensor dimensions for best performance.

---

### 9.16 Transpose Operations

Both approaches work:
```python
# Method 1: .T property (preferred - cleaner)
transposed = tensor.T

# Method 2: explicit function
transposed = pypto.transpose(tensor, 0, 1)
```

---

### 9.17 Pattern Quick Reference

| Operation | Verbose Form | Preferred Form |
|-----------|-------------|----------------|
| Tensor shape | `pypto.Tensor([pypto.DYNAMIC, ...], dtype)` | `pypto.Tensor([], dtype)` |
| Multiply | `pypto.mul(x, y)` | `x * y` |
| Square | `pypto.mul(x, x)` | `x * x` |
| Sum | `pypto.sum(x, dim=-1, keepdim=True)` | `x.sum(-1, keepdim=True)` |
| Rsqrt | `pypto.rsqrt(x)` | `x.rsqrt()` |
| Exp | `pypto.exp(x)` | `x.exp()` |
| Add scalar | `pypto.add(x, scalar)` | `x + scalar` |
| Sub scalar | `pypto.sub(x, scalar)` | `x - scalar` |
| Transpose | `pypto.transpose(t, 0, 1)` | `t.T` |
| Cast | `pypto.cast(x, pypto.DT_FP32)` | `x.float()` |
| Reshape | `pypto.reshape(t, [a, b])` | `t.reshape([a, b])` |
| Matmul | `pypto.matmul(a, b, ...)` | `pypto.matmul(a, b, ...)` (keep explicit) |

---

### 9.18 Key Takeaways

1. **Shape Inference**: Use `pypto.Tensor([], dtype)` for automatic shape inference
2. **Python Operators**: Use Python operators inside JIT (`*`, `+`, `-`, `/`)
3. **Method Chaining**: PyPTO tensors support method chaining (`.exp()`, `.T`, `.rsqrt()`)
4. **Tile Shapes**: Match tile shapes to actual tensor dimensions or use even divisors
5. **Keep matmul explicit**: `pypto.matmul()` is preferred over Python `@` operator

---

### 9.19 matmul API and Tile Shapes

#### Issue: Wrong matmul syntax causes errors

**Symptom:** matmul operations fail with cryptic errors.

**Root Cause:** Wrong API usage for `pypto.matmul`.

**Correct matmul syntax:**
```python
# WRONG - this syntax does not work
result = pypto.matmul(a, b.T, out_dtype=pypto.DT_FP32)
result = pypto.matmul(a, b, out_dtype=pypto.DT_FP32)

# CORRECT - use a_trans and b_trans parameters
result = pypto.matmul(a, b, pypto.DT_FP32, a_trans=False, b_trans=True)
result = pypto.matmul(a, b, pypto.DT_FP32)  # both False by default
```

**Transpose patterns:**
```python
# a @ b.T  →  a_trans=False, b_trans=True
result = pypto.matmul(a, b, dtype, a_trans=False, b_trans=True)

# a.T @ b  →  a_trans=True, b_trans=False
result = pypto.matmul(a, b, dtype, a_trans=True, b_trans=False)

# a @ b    →  both False (default)
result = pypto.matmul(a, b, dtype)

# a.T @ b.T  →  both True
result = pypto.matmul(a, b, dtype, a_trans=True, b_trans=True)
```

**Note:** Do NOT use `.T` on tensors before passing to matmul - use the transpose flags instead.

#### Issue: Both vec and cube tile shapes needed

**Symptom:** matmul or other ops fail with tiling errors.

**Root Cause:** Need to set BOTH `set_vec_tile_shapes` AND `set_cube_tile_shapes`.

**Solution:**
```python
@pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.NPU})
def kernel(...):
    # BOTH are required for matmul to work
    pypto.set_vec_tile_shapes(TILE_B, TILE_H, TILE_T, TILE_K)
    pypto.set_cube_tile_shapes([128, 128], [128, 128], [128, 128])
    
    # Now matmul operations will work
    result = pypto.matmul(a, b, pypto.DT_FP32)
    ...
```

**Common tile configurations:**
```python
# For forward kernels
TILE_B = 1
TILE_H = 2
TILE_T = 8
TILE_K = 32
pypto.set_vec_tile_shapes(TILE_B, TILE_H, TILE_T, TILE_K)
pypto.set_cube_tile_shapes([128, 128], [128, 128], [128, 128])

# For backward kernels
TILE_BH = 16
TILE_NT = 4
TILE_BT = 8
TILE_KV = 32
pypto.set_vec_tile_shapes(TILE_BH, TILE_NT, TILE_BT, TILE_KV)
pypto.set_cube_tile_shapes([128, 128], [128, 128], [128, 128])
```

**Rule:** Always set BOTH tile shape configurations when using matmul or complex tensor operations.

#### Issue: pypto.assemble shape mismatch

**Symptom:**
```
CHECK FAILED: dest.GetShape().size() == tensor.GetShape().size()
Assemble: src and dest requires same shape
```

**Root Cause:** The src tensor shape doesn't match the expected view dimensions for the destination.

**Solution:** Reshape the output tensor to match the expected view shape:
```python
# WRONG - out_chunk is [bt, V] but assemble expects [1, bt, 1, V]
pypto.assemble(out_chunk, [b, t0, h, 0], out_out)

# CORRECT - reshape to match view dimensions
pypto.assemble(out_chunk.reshape([1, bt, 1, V]), [b, t0, h, 0], out_out)
```

**Rule:** `pypto.assemble(tensor, offsets, dest)` requires tensor shape to have same number of dimensions as the view at those offsets.

#### Issue: reshape([1]) on multi-element tensor

**Symptom:**
```
CHECK FAILED: capacity == 1
Shape size not match, func CheckAndInferShape
```

**Root Cause:** Trying to reshape a tensor with multiple elements to a single element shape.

**Solution:** Use `pypto.view` with offsets to extract single elements, then reshape:
```python
# WRONG - bt=4 tensor has capacity 4, can't reshape to [1]
gl = g_cum.reshape([1])[bt - 1:bt].reshape([1])

# CORRECT - use view to get last element, then reshape
gl = pypto.view(g_cum, [1], [bt - 1]).reshape([1])
```

**Common pattern for getting last element:**
```python
# For getting last element of a 1D tensor of size bt
last_elem = pypto.view(tensor, [1], [bt - 1]).reshape([1])
```

#### Issue: `.T` attribute doesn't exist on PyPTO tensors

**Symptom:**
```
AttributeError: 'Tensor' object has no attribute 'T'
```

**Root Cause:** PyPTO tensors don't support the `.T` property like PyTorch tensors.

**Solution:** Use `pypto.transpose(tensor, 0, 1)` or use matmul's transpose flags:
```python
# WRONG - .T doesn't work on PyPTO tensors
kc_t = kc.T
result = pypto.matmul(qc, kc_t, dtype)

# CORRECT - use matmul transpose flags
result = pypto.matmul(qc, kc, dtype, a_trans=False, b_trans=True)

# OR use explicit transpose
kc_t = pypto.transpose(kc, 0, 1)
result = pypto.matmul(qc, kc_t, dtype)
```

**For 2D matrix transpose:** `pypto.transpose(tensor, 0, 1)` does NOT work on 2D tensors with the tiling system - it causes "TileShape dim num should same to input" error.

**Instead, use matmul transpose flags:**
```python
# Instead of transpose, use a_trans/b_trans in matmul:
kc_t = pypto.transpose(kc, 0, 1)  # WRONG - doesn't work!
qk = pypto.matmul(qc, kc, dtype, a_trans=False, b_trans=True)  # CORRECT

# For symmetric sum m_mat + m_mat.T, split into two matmuls:
dk_c = dk_c + pypto.matmul(m_mat, kc, dtype)  # m_mat @ kc
dk_c = dk_c + pypto.matmul(m_mat, kc, dtype, a_trans=True, b_trans=False)  # m_mat.T @ kc
```

#### Issue: Reduction axis needs 32-byte alignment

**Symptom:**
```
Reduce op: the tileShape of last axis need to 32Byte align!
```

**Root Cause:** PyPTO reduction operations (`.sum()`) require tensor dimensions to be 32-byte aligned.

**Calculation:** For FP32 (4 bytes), dimension * 4 must be divisible by 32.
- `bt=4` → 4*4=16 bytes ❌ Not aligned
- `bt=8` → 8*4=32 bytes ✅ Aligned
- `V=16` → 16*4=64 bytes ✅ Aligned

**Solution:** Use dimensions that are multiples of 8 for reduction axes:
```python
# WRONG - bt=4 causes alignment error
bt = 4
result = tensor.sum(-1)

# CORRECT - bt=8 is 32-byte aligned
bt = 8
result = tensor.sum(-1)
```

**Rule:** Any tensor dimension involved in `.sum()`, `.mean()`, or other reduction operations must satisfy `(dim * bytes_per_element) % 32 == 0`. For FP32, use dimensions that are multiples of 8.

**Common alignment examples (FP32):**
- V=16 → 16*4=64 bytes ✅ Aligned
- V=32 → 32*4=128 bytes ✅ Aligned
- K=16 → 16*4=64 bytes ✅ Aligned
- K=32 → 32*4=128 bytes ✅ Aligned
- bt=8 → 8*4=32 bytes ✅ Aligned

#### Issue: Sum reduction fails even with aligned dimensions

**Symptom:**
```
Reduce op: the tileShape of last axis need to 32Byte align!
```

**Problem:** Even when dimensions are theoretically aligned (e.g., V=32), `.sum(-1)` on the last axis may still fail due to how PyPTO tiles the tensor internally.

**Solution:** Replace `.sum()` with matmul-based reduction using a precomputed ones vector:

```python
# Create ones vectors on host (for any dimension, not just aligned)
ones_v = torch.ones(V, 1, device=device, dtype=torch.float32)  # [V, 1]
ones_k = torch.ones(K, 1, device=device, dtype=torch.float32)  # [K, 1]

# In kernel signature, add ones vectors as parameters:
@pypto.frontend.jit(...)
def kernel(..., ones_v: pypto.Tensor([], pypto.DT_FP32), ones_k: pypto.Tensor([], pypto.DT_FP32), ...):
    
    # Replace .sum(-1) with matmul:
    db_c = (dvb * vc).sum(-1)  # WRONG - may fail
    
    db_c = pypto.matmul(dvb * vc, ones_v, pypto.DT_FP32).reshape([bt])  # CORRECT
    
    # Replace .sum(0) and .sum(1) with matmul:
    d_l_l_mat_0 = (d_l * l_mat).sum(0)  # WRONG
    d_l_l_mat_0 = pypto.matmul(d_l * l_mat, ones_k, pypto.DT_FP32, a_trans=True, b_trans=False).reshape([bt])  # CORRECT
    
    d_l_l_mat_1 = (d_l * l_mat).sum(1)  # WRONG
    d_l_l_mat_1 = pypto.matmul(d_l * l_mat, ones_k, pypto.DT_FP32).reshape([bt])  # CORRECT
```

**Key insight:** Matmul-based reduction works regardless of alignment because it uses cube operations, while `.sum()` uses vector operations that require strict 32-byte alignment.

**Rule:** When in doubt, use matmul with ones vector for reduction instead of `.sum()` on any axis.

#### Issue: K-dimension valid shape mismatch in matmul

**Symptom:**
```
RuntimeError: K-dimension valid shape mismatch. Got input valid shape: [SymbolicScalar(8), SymbolicScalar(8)], mat2 valid shape: [SymbolicScalar(32), SymbolicScalar(1)], a_trans: False, b_trans: False.
```

**Problem:** Using wrong ones vector for matmul reduction. The ones vector must match the dimension being reduced.

**Solution:** Use different ones vectors for different dimensions:

```python
# Create different ones vectors for different dimensions
ones_v = torch.ones(V, 1, device=device, dtype=torch.float32)   # [V, 1] - for reducing V dimension
ones_k = torch.ones(K, 1, device=device, dtype=torch.float32)  # [K, 1] - for reducing K dimension
ones_bt = torch.ones(bt, 1, device=device, dtype=torch.float32) # [bt, 1] - for reducing bt dimension

# In kernel signature, add all ones vectors:
def kernel(..., ones_v, ones_k, ones_bt, ...):
    
    # For [bt, V] tensor reducing V dim (sum over last axis):
    db_c = pypto.matmul(tensor, ones_v, dtype).reshape([bt])  # CORRECT
    
    # For [bt, bt] tensor reducing bt dim (sum over rows/cols):
    d_l_l_mat_0 = pypto.matmul(tensor, ones_bt, dtype, a_trans=True, b_trans=False).reshape([bt])  # row sums
    d_l_l_mat_1 = pypto.matmul(tensor, ones_bt, dtype).reshape([bt])  # col sums
```

**Rule:** The ones vector's first dimension must equal the dimension being reduced in the matmul operation.

#### Issue: 5D tensor views with 4D vec tile shapes causes "Run pass failed"

**Symptom:**
```
Errcode: FFFFFF!
Run pass failed., func CompileFunction
```

**Root Cause:** Using 5D tensor views (e.g., `pypto.view(A_in, [1, 1, 1, bt, bt], [b, h, c, 0, 0])`) when `pypto.set_vec_tile_shapes` only sets 4D tile shapes. The framework cannot handle 5D operations with 4D tile configuration.

**Solution:** Reshape 5D tensors to 2D before passing to the kernel, then use matching view shape/offsets:

```python
# In pypto_function, reshape before kernel call:
# Original: [B, H, nt, bt, bt] -> Reshape to 2D: [B*H*nt, bt*bt]
A_2d = A_5d.reshape([B * H * nt, bt * bt])
w_2d = w_4d.reshape([B * H * nt, bt * K])
S_before_2d = S_before_4d.reshape([B * H * nt, K * V])
v_new_2d = v_new_4d.reshape([B * H * nt, bt * V])
g_cum_2d = g_cum_3d.reshape([B * H * nt, bt])

# In kernel, use 2D views with 2 offsets (len(shape) == len(offsets)):
session_base = session * nt
for c in pypto.loop(0, nt, 1):
    cache_idx = session_base + c
    # For 2D tensor [N, M] with view [a, b]: offsets = [cache_idx, 0]
    a = pypto.view(A_2d, [bt, bt], [cache_idx, 0]).reshape([bt, bt])
    w = pypto.view(w_2d, [bt, K], [cache_idx, 0]).reshape([bt, K])
    s_before = pypto.view(S_before_2d, [K, V], [cache_idx, 0]).reshape([K, V])
    v_new = pypto.view(v_new_2d, [bt, V], [cache_idx, 0]).reshape([bt, V])
    # For 1D result from 2D tensor: shape = [1, size], offsets = [cache_idx, 0]
    g_cum = pypto.view(g_cum_2d, [1, bt], [cache_idx, 0]).reshape([bt])
```

**Rule:** `len(shape) == len(offsets)` is mandatory for pypto.view. Keep all tensors ≤4D and reshape to 2D before passing to kernel. For 1D views from 2D tensor, use shape [1, size] with 2 offsets.
