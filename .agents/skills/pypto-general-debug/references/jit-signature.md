# JIT signature & tensor type hints (DEBUG_GUIDEBOOK.md §9.1, §9.13)

*Agent-learned patterns from GDR kernel development. Add to this file when discovering new patterns.*

## §9.1 JIT Signature Parsing

### Issue: `from __future__ import annotations` breaks JIT

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

## §9.13 Tensor Shape Specifications

### Issue: Shape Size Exceeds INT32_MAX

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
