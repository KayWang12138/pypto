# Block Store NZ-Out Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement layout-driven `NZ` output for block `plm.store` without adding a new store parameter.

**Architecture:** Keep the user-facing API unchanged and let the destination tensor view drive copy-out layout for the `plm.store -> manual.store` route only. Update `manual.store` to read the destination layout, keep `ND` behavior unchanged, synthesize an NZ `pto.make_tensor_view` only inside the `manual.store` lowering path for the v1 2D/no-custom-stride case, and reject unsupported `DN` or `NZ + custom stride` combinations with explicit errors.

**Tech Stack:** Python frontend/block IR tests, C++ PTO codegen (`pto.make_tensor_view`, `pto.tstore`), pytest-style unit tests, lightweight local validation (`python -m py_compile`, `git diff --check`).

---

### Task 1: Add failing manual-store NZ codegen tests

**Files:**
- Modify: `python/tests/ut/block/codegen/test_pto_codegen.py`
- Test: `python/tests/ut/block/codegen/test_pto_codegen.py`

- [ ] **Step 1: Write the failing `manual.store` NZ tests**

```python
def test_manual_store_with_nz_output_emits_layout_attr_and_pre_quant():
    ...
    assert "{layout = #pto.layout<nz>}" in mlir_code
    assert "arith.constant 7 : i64" in mlir_code

def test_manual_store_rejects_nz_output_with_custom_stride():
    ...
    with pytest.raises(ValueError, match="manual.store: NZ output does not support custom stride in v1"):
        codegen_obj.generate(transformed_program)
```

- [ ] **Step 2: Try to run the focused test file**

Run: `python -m pytest python/tests/ut/block/codegen/test_pto_codegen.py -k "nz or pre_quant" -v`
Expected: local environment may fail before execution if `pytest` is unavailable; if it runs, new tests should fail before implementation.

### Task 2: Lower NZ output inside `manual.store`

**Files:**
- Modify: `framework/src/interface/block/backend/910B_PTO/backend_910b_pto_manual_ops.cpp`
- Test: `python/tests/ut/block/codegen/test_pto_codegen.py`

- [ ] **Step 1: Read destination layout and v1 constraints**

```cpp
const auto layout = tensor_type->tensor_view_.has_value() ? tensor_type->tensor_view_->layout
                                                          : ir::TensorLayout::ND;
const bool has_custom_stride = tensor_type->tensor_view_.has_value() &&
                               !tensor_type->tensor_view_->stride.empty();
```

- [ ] **Step 2: Reject unsupported v1 combinations early**

```cpp
if (layout == ir::TensorLayout::DN) {
  throw pypto::ValueError("manual.store: DN layout is not supported for store output");
}
if (layout == ir::TensorLayout::NZ && has_custom_stride) {
  throw pypto::ValueError("manual.store: NZ output does not support custom stride in v1");
}
```

- [ ] **Step 3: Build an NZ tensor_view only for the manual-store lowering path**

```cpp
row_off = codegen.GetExprAsCode(offsets_tuple->elements_[0]);
col_off = codegen.GetExprAsCode(offsets_tuple->elements_[1]);
std::string raw_ptr = codegen.GetTensorPtr(output_tensor);
...
tv_line << " {layout = #pto.layout<nz>}";
```

- [ ] **Step 4: Keep `pre_quant_scalar` lowering unchanged**

```cpp
if (op->HasKwarg("pre_quant_scalar")) {
  int pre_quant = op->GetKwarg<int>("pre_quant_scalar");
  ...
}
```

### Task 3: Lightweight validation and review

**Files:**
- Review: `framework/src/interface/block/backend/910B_PTO/backend_910b_pto_manual_ops.cpp`
- Review: `python/tests/ut/block/codegen/test_pto_codegen.py`

- [ ] **Step 1: Run lightweight syntax checks**

Run: `python -m py_compile python/tests/ut/block/codegen/test_pto_codegen.py`
Expected: PASS

- [ ] **Step 2: Run diff hygiene check**

Run: `git diff --check`
Expected: PASS

- [ ] **Step 3: Summarize the validation gap explicitly**

```text
Focused pytest execution could not be completed locally because pytest is unavailable in this environment, so only static checks and diff hygiene were run here.
```
