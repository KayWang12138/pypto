---
description: Checklist for adding or removing distributed test variants (AllReduce, AllGather, etc.) — covers implementation, UT, and ST layers.
globs: framework/tests/st/distributed/**,framework/tests/ut/codegen/**,framework/src/interface/operation/distributed/**
alwaysApply: false
---

# Distributed Test Variant Management

Use this document when **adding** or **removing** a distributed operation variant (e.g. `OneShotAllReduce_vN`).

A full removal touches up to **11 artifacts** across three layers. Missing any one causes a compile error, linker error, or silent dead code.

| Layer | Artifacts |
|-------|-----------|
| **Implementation** | function body, header declaration, communicator class (if any) |
| **Unit tests** | IR structure tests, codegen smoke test |
| **System tests** | test function, header, op registration, JSON, golden dispatcher, CMakeLists |

---

## Data to Collect First

Before touching any file, fix these values for the variant being added or removed.
Run the **Quick Verification** script at the bottom to compute `SortIndex`.

| Field | Description | Example |
|-------|-------------|---------|
| `FuncName` | C++ function name | `OneShotAllReduce_v10` |
| `OpName` | CamelCase string used in C++ / Python registration | `AllReduceV10` |
| `VarName` | Mixed-case JSON filename prefix (note lowercase 'r' in 'reduce') | `AllreduceV10` |
| `JsonFile` | Full JSON filename | `AllreduceV10_st_test_cases.json` |
| `NumCases` | Number of entries in `"test_cases"` array | `1` |
| `NumRanks` | Rank count (default `4`; specify override for multi-rank tests) | `2` |
| `SortIndex` | 0-based index after case-insensitive sort | computed |
| `FilterGroups` | Subset shortcuts the variant belongs to | `oneshot` |
| `DedicatedShortcut` | Per-variant CMake keyword, or empty if none | `v10` |
| `CommunicatorClass` | Communicator class used exclusively by this variant, or empty | `OneShotCommunicatorV2` |

---

## Removing a Distributed Test Variant

### Step 1 — Function implementation
**File:** `framework/src/interface/operation/distributed/shmem_operation_impl.cpp`

- Delete the entire function body for `FuncName`.
- If the function uses a communicator class listed in `CommunicatorClass`, also delete that class's
  implementation here (if it has one) and check whether any other variant still references it.

### Step 2 — Function and communicator declarations
**File:** `framework/include/tilefwk/tilefwk_op.h`

- Remove the forward declaration `FuncName(...)`.
- Remove the `class CommunicatorClass;` forward declaration (if `CommunicatorClass` is set and
  no other variant uses it).

**File:** `framework/include/tilefwk/distributed_communicator.h`

- If `CommunicatorClass` is set and unused after removal: delete the entire class definition.

### Step 3 — Unit tests
**File:** `framework/tests/ut/codegen/src/test_dynamic/test_codegen_dyn_distributed/test_allreduce_ir.cpp`

Variants can appear in **multiple fixture classes** — search all of them:

- `AllReduceIRTest`: remove named `TEST_F` blocks (e.g. `V2_IRStructure`, `V4_IRStructure`).
- `AllReduceIRTest`: remove the variant's arms from `AllVariants_ShmemOpcodeEquivalence` (the
  cross-variant IR equivalence test). If all non-base variants are gone, remove the entire test.
- `AllReduceIRMultiRankTest`: remove the `RunOneShotVN` helper method and the four
  `TEST_F(..., OneShotVN_W*_IRStructure)` blocks.
- `AllReduceCorrectnessTest`: remove `RunOneShotVNCorrectness` helper and its `TEST_F` blocks;
  remove the variant's lambda from `RunOneShotVariantsSameResults` (delete the whole method if
  no variants remain).
- Update the file-level `\brief` comment to remove the variant version range (e.g. change
  `OneShot (base, v2–v6, v10, v10_pipe_ge)` → `OneShot (base, v10, v10_pipe_ge)`).

**File:** `framework/tests/ut/codegen/src/test_dynamic/test_codegen_dyn_distributed/test_shmem_operation_impl.cpp`

- Remove the `TEST_F(TestDistributedShmemImpl, TestFuncName)` codegen smoke test.

### Step 4 — ST test function
**File:** `framework/tests/st/distributed/ops/src/test_allreduce.cpp`

- Delete the entire `TestAllReduce_vN<T>(...)` template function body.
- Delete all 4 explicit instantiations: `template void TestAllReduce_vN<int32_t>`, `<float>`, `<float16>`, `<bfloat16>`.
- If the variant has an associated IR-equivalence ST test (`TestAllReduceIREquivalence`), delete
  that function and its 4 instantiations too; also remove it from Step 5 below.

### Step 5 — ST header declaration
**File:** `framework/tests/st/distributed/ops/include/distributed_op_test_suite.h`

- Remove the `template<typename T> void TestAllReduce_vN(...);` forward declaration.

### Step 6 — Op registration
**File:** `framework/tests/st/distributed/src/test_distributed.cpp`

- Remove the `reg.RegisterOp("AllReduceVN", ...)` call inside `RegisterOps()`.

### Step 7 — JSON test case
**Directory:** `framework/tests/st/distributed/ops/test_case/`

- Delete `{VarName}_st_test_cases.json`.
- For multi-rank variants: also delete `{VarName}MultiRank_st_test_cases.json` if present.

### Step 8 — Golden dispatcher
**File:** `framework/tests/st/distributed/ops/script/distributed_golden.py`

- Remove the `'OpName': generate_..._golden` entry from `OPERATOR_DISPATCHERS`.

### Step 9 — CMakeLists.txt
**File:** `framework/tests/st/distributed/CMakeLists.txt`

See **CMakeLists.txt Anatomy** below for the full list of locations:

1. **Index-map comment**: remove the `#  N  VarName ...` line(s); renumber all subsequent entries.
2. **`_num_cases`**: decrement by `NumCases`.
3. **Rank override** (if `NumRanks != 4`): remove `set(_rank_override_{SortIndex} ...)`;
   renumber remaining `_rank_override_*` variable names for all indices that shifted down.
4. **Subset shortcut blocks** (each `elseif` that listed `SortIndex`): remove the old index;
   decrement all indices higher than `SortIndex` in every `_distributed_filter_config(...)` call
   and update the `message(STATUS ...)` range text.
   > **If this was the last variant in a subset group** (e.g. the last multi-rank test), delete
   > the entire `elseif` block for that group rather than leaving it with an empty index list.
5. **Dedicated shortcut block** (if `DedicatedShortcut` is non-empty): delete the entire
   `elseif ("${ENABLE_STEST_DISTRIBUTED}" STREQUAL "{DedicatedShortcut}")` block.

---

## Adding a New Distributed Test Variant

### Step 1 — Function implementation
**File:** `framework/src/interface/operation/distributed/shmem_operation_impl.cpp`

- Implement `FuncName(...)`. Model after an existing variant of the same family.
- If the variant requires a new communicator class, implement it in the appropriate `.cpp`/`.h`.

### Step 2 — Declarations
**File:** `framework/include/tilefwk/tilefwk_op.h`

- Add the function declaration for `FuncName`.
- Add `class CommunicatorClass;` forward declaration if introducing a new communicator.

**File:** `framework/include/tilefwk/distributed_communicator.h`

- Add the new communicator class definition if applicable.

### Step 3 — Unit tests
**File:** `framework/tests/ut/codegen/src/test_dynamic/test_codegen_dyn_distributed/test_allreduce_ir.cpp`

- Add a `TEST_F(AllReduceIRTest, VN_IRStructure)` block.
- If the variant should produce the same IR as the base, add it to `AllVariants_ShmemOpcodeEquivalence`.
- Add `RunOneShotVN` helper + four `TEST_F` blocks in `AllReduceIRMultiRankTest`.
- Add `RunOneShotVNCorrectness` helper + four `TEST_F` blocks in `AllReduceCorrectnessTest`.
- Update the `\brief` comment to include the new version.

**File:** `framework/tests/ut/codegen/src/test_dynamic/test_codegen_dyn_distributed/test_shmem_operation_impl.cpp`

- Add a `TEST_F(TestDistributedShmemImpl, TestFuncName)` codegen smoke test.

### Step 4 — ST test function
**File:** `framework/tests/st/distributed/ops/src/test_allreduce.cpp`

- Implement `TestAllReduce_vN<T>(...)` (model after an existing variant of the same family).
- Add 4 explicit instantiations: `template void TestAllReduce_vN<int32_t>`, `<float>`, `<float16>`, `<bfloat16>`.

### Step 5 — ST header declaration
**File:** `framework/tests/st/distributed/ops/include/distributed_op_test_suite.h`

- Add `template<typename T> void TestAllReduce_vN(...);` forward declaration alongside the other variant declarations.

### Step 6 — Op registration
**File:** `framework/tests/st/distributed/src/test_distributed.cpp`

- Add `reg.RegisterOp("AllReduceVN", ...)` inside `RegisterOps()`.
- Use an existing variant's registration block as a template.

### Step 7 — JSON test case
**Directory:** `framework/tests/st/distributed/ops/test_case/`

- Create `{VarName}_st_test_cases.json` with `"operation": "OpName"`.
- Copy an existing variant's JSON and update: `operation`, `dtype`, shape fields, rank count.

### Step 8 — Golden dispatcher
**File:** `framework/tests/st/distributed/ops/script/distributed_golden.py`

- Add `'OpName': generate_all_reduce_golden` (or a new dispatcher if the golden logic differs)
  to `OPERATOR_DISPATCHERS`.

### Step 9 — CMakeLists.txt
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
   > **If this is the first variant in a new subset group**, add a new `elseif` block for it.
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
| `multirank` `elseif` block | Same — **delete the entire block** if the last multi-rank test is removed |
| `ircheck` `elseif` block | Same — **delete the entire block** if the last IR-check test is removed |
| `twoshotir` `elseif` block | Same |
| Dedicated `elseif` blocks (`v10`, `v10pipege`, …) | Update index value in message + config; add or remove entire block for new/removed dedicated shortcuts |

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
