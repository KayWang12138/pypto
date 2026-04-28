# Dynamic shapes, pypto.loop, symbolic indexing (DEBUG_GUIDEBOOK.md §9.2)

*Agent-learned patterns from GDR kernel development. Add to this file when discovering new patterns.*

## Issue: `set_vec_tile_shapes` requires concrete values

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

## Issue: `pypto.loop` with symbolic bounds

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

## Issue: Tensor indexing with symbolic indices

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

## Issue: pypto.view shape/offsets dimension mismatch

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

For the comprehensive `pypto.view` reference (signature, padding rules, common mistakes), see `pypto-view.md`.
