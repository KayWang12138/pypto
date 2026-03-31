---
description: Checklist for adding or removing distributed system tests (AllReduce, AllGather, etc.)
globs: framework/tests/st/distributed/**
alwaysApply: false
---

# Distributed ST Test Management

Use this document when **adding** or **removing** a distributed system test variant.
Each workflow touches exactly **6 artifacts**. Missing any one causes a linker error or silent dead code.

---

## Data to Collect First

Before touching any file, fix these values for the variant being added or removed.
Run the **Quick Verification** script at the bottom to compute `SortIndex`.

| Field | Description | Example |
|-------|-------------|---------|
| `OpName` | CamelCase string used in C++ / Python | `AllReduceV10` |
| `VarName` | Mixed-case JSON filename prefix (note lowercase 'r' in 'reduce') | `AllreduceV10` |
| `JsonFile` | Full JSON filename | `AllreduceV10_st_test_cases.json` |
| `NumCases` | Number of entries in `"test_cases"` array | `1` |
| `NumRanks` | Rank count (default `4`; specify override for multi-rank tests) | `2` |
| `SortIndex` | 0-based index after case-insensitive sort | computed |
| `FilterGroups` | Subset shortcuts the variant belongs to | `oneshot`, `ircheck` |
| `DedicatedShortcut` | Per-variant CMake keyword, or empty if none | `v10` |

---

## Removing a Distributed ST Test Variant

### Step 1 — Test function
**File:** `framework/tests/st/distributed/ops/src/test_allreduce.cpp`

- Delete the entire `TestAllReduce_vN<T>(...)` template function body.
- Delete all 4 explicit instantiations: `template void TestAllReduce_vN<int32_t>`, `<float>`, `<float16>`, `<bfloat16>`.

### Step 2 — Header declaration
**File:** `framework/tests/st/distributed/ops/include/distributed_op_test_suite.h`

- Remove the `template<typename T> void TestAllReduce_vN(...);` forward declaration.

### Step 3 — Op registration
**File:** `framework/tests/st/distributed/src/test_distributed.cpp`

- Remove the `reg.RegisterOp("AllReduceVN", ...)` call inside `RegisterOps()`.

### Step 4 — JSON test case
**Directory:** `framework/tests/st/distributed/ops/test_case/`

- Delete `{VarName}_st_test_cases.json`.

### Step 5 — Golden dispatcher
**File:** `framework/tests/st/distributed/ops/script/distributed_golden.py`

- Remove the `'OpName': generate_..._golden` entry from `OPERATOR_DISPATCHERS`.

### Step 6 — CMakeLists.txt
**File:** `framework/tests/st/distributed/CMakeLists.txt`

See **CMakeLists.txt Anatomy** below for the full list of locations:

1. **Index-map comment**: remove the `#  N  VarName ...` line; renumber all subsequent entries.
2. **`_num_cases`**: decrement by `NumCases`.
3. **Rank override** (if `NumRanks != 4`): remove `set(_rank_override_{SortIndex} ...)`;
   renumber remaining `_rank_override_*` variable names for all indices that shifted down.
4. **Subset shortcut blocks** (each `elseif` that listed `SortIndex`): remove the old index;
   decrement all indices higher than `SortIndex` in every `_distributed_filter_config(...)` call
   and update the `message(STATUS ...)` range text.
5. **Dedicated shortcut block** (if `DedicatedShortcut` is non-empty): delete the entire
   `elseif ("${ENABLE_STEST_DISTRIBUTED}" STREQUAL "{DedicatedShortcut}")` block.

---

## Adding a New Distributed ST Test Variant

### Step 1 — Test function
**File:** `framework/tests/st/distributed/ops/src/test_allreduce.cpp`

- Implement `TestAllReduce_vN<T>(...)` (model after an existing variant of the same family).
- Add 4 explicit instantiations: `template void TestAllReduce_vN<int32_t>`, `<float>`, `<float16>`, `<bfloat16>`.

### Step 2 — Header declaration
**File:** `framework/tests/st/distributed/ops/include/distributed_op_test_suite.h`

- Add `template<typename T> void TestAllReduce_vN(...);` forward declaration alongside the other variant declarations.

### Step 3 — Op registration
**File:** `framework/tests/st/distributed/src/test_distributed.cpp`

- Add `reg.RegisterOp("AllReduceVN", TestAllReduce_vN<float>)` (and required dtypes) inside `RegisterOps()`.
- Use an existing variant's registration block as a template.

### Step 4 — JSON test case
**Directory:** `framework/tests/st/distributed/ops/test_case/`

- Create `{VarName}_st_test_cases.json` with `"operation": "OpName"`.
- Copy an existing variant's JSON and update: `operation`, `dtype`, shape fields, rank count.

### Step 5 — Golden dispatcher
**File:** `framework/tests/st/distributed/ops/script/distributed_golden.py`

- Add `'OpName': generate_all_reduce_golden` (or a new dispatcher if the golden logic differs)
  to `OPERATOR_DISPATCHERS`.

### Step 6 — CMakeLists.txt
**File:** `framework/tests/st/distributed/CMakeLists.txt`

See **CMakeLists.txt Anatomy** below:

1. **Index-map comment**: insert `#  SortIndex  VarName ...` at the correct position;
   renumber all subsequent entries.
2. **`_num_cases`**: increment by `NumCases`.
3. **Rank override** (if `NumRanks != 4`): add `set(_rank_override_{SortIndex} "{NumRanks}")`;
   renumber existing `_rank_override_*` variable names for all indices that shifted up.
4. **Subset shortcut blocks**: for each group in `FilterGroups`, insert `SortIndex` into the
   `_distributed_filter_config(...)` call at the right position and increment all higher indices;
   update the `message(STATUS ...)` range text.
5. **Dedicated shortcut block** (if `DedicatedShortcut` is non-empty): add a new
   `elseif ("${ENABLE_STEST_DISTRIBUTED}" STREQUAL "{DedicatedShortcut}")` block with
   the correct index in the message and config call.

---

## CMakeLists.txt Anatomy

Every location that encodes test indices — all must be consistent after any add or remove:

| Location in `framework/tests/st/distributed/CMakeLists.txt` | What to update |
|-------------------------------------------------------------|----------------|
| Index-map comment block (`# N  VarName ...`) | Insert/remove line; renumber all subsequent `N` values |
| `set(_num_cases "N")` | Increment or decrement by `NumCases` |
| `set(_rank_override_N "R")` | Add/remove for multi-rank tests; keep variable names in sync with current index values |
| `oneshot` `elseif` block | Insert/remove index; increment/decrement all higher indices; update message text |
| `twoshot` `elseif` block | Same |
| `multirank` `elseif` block | Same |
| `ircheck` `elseif` block | Same |
| `twoshotir` `elseif` block | Same |
| `v7`, `v8`, `v9`, … dedicated `elseif` blocks | Update index value in message + config; add or remove entire block for new/removed dedicated shortcuts |

**Index source of truth**: the sorted JSON filenames. Always run Quick Verification to confirm before committing.

---

## Sort Order

The C++ test runner sorts JSON filenames **case-insensitively** (`std::string::operator<` on lowercased names). Key facts:
- `_` (ASCII 95) sorts **before** lowercase letters (97+)
- Digits (48–57) sort **before** both letters and `_`
- Consequence: `AllreduceV5_st…` sorts **before** `AllreduceV5MultiRank_st…`

Verify the sort order before editing CMakeLists.txt:

```bash
ls framework/tests/st/distributed/ops/test_case/*.json \
  | xargs -I{} basename {} | tr '[:upper:]' '[:lower:]' | LC_ALL=C sort
```

---

## Quick Verification

Run from the repo root to regenerate the full index map and confirm `_num_cases`:

```bash
python3 -c "
import json; from pathlib import Path
d = Path('framework/tests/st/distributed/ops/test_case')
files = sorted([f for f in d.iterdir() if f.suffix=='.json'], key=lambda p: p.name.lower())
idx = 0
for f in files:
    n = len(json.load(open(f))['test_cases'])
    name = f.stem.replace('_st_test_cases','')
    for c in range(n):
        print(f'  {idx:2d}  {name}' + (f'  case_{c}' if n>1 else ''))
        idx += 1
print(f'\n_num_cases = \"{idx-1}\"  (indices 0-{idx-1}, {idx} total)')
"
```
