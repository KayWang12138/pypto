# PyPTO Dynamic Online Softmax Compiler Repro Report

Date: 2026-03-27  
Author: Cursor coding agent  
Scope: Minimal reproductions for dynamic-shape + online-softmax related compile failures.

## Environment

- Host: `linux 5.10.0-182.0.0.95.r1941_123.hce2.aarch64`
- Python: `3.9.9`
- PyTorch: `2.8.0+cpu`
- `torch_npu`: available
- NPU card used for repro: `TILE_FWK_DEVICE_ID=6` (free at run time)

## Repro Artifacts

- Repro 1 (online-softmax + dynamic axes):
  - Script: `pypto/models/experimental/attention/online_attention_dynamic_repro.py`
  - Log: `pypto/models/experimental/attention/online_attention_dynamic_repro.log`
- Repro 2 (branch scope + loop begin/end):
  - Script: `pypto/models/experimental/attention/online_state_scope_repro.py`
  - Log: `pypto/models/experimental/attention/online_state_scope_repro.log`
- Negative control (loop index codegen only):
  - Script: `pypto/models/experimental/attention/dynamic_loop_index_codegen_repro.py`
  - Log: `pypto/models/experimental/attention/dynamic_loop_index_codegen_repro.log`
  - Result: exits successfully in this environment.

## Repro Commands

```bash
cd /data/z00885570/pypto-master/pypto

# Repro 1: online-softmax + dynamic axes
TILE_FWK_DEVICE_ID=6 python3 "models/experimental/attention/online_attention_dynamic_repro.py" \
  --batch 2 --seq 64 --tile_b 2 --tile_q 64 --tile_k 32 \
  > "models/experimental/attention/online_attention_dynamic_repro.log" 2>&1

# Repro 2: scope issue in loop-begin/else branch
TILE_FWK_DEVICE_ID=6 python3 "models/experimental/attention/online_state_scope_repro.py" \
  > "models/experimental/attention/online_state_scope_repro.log" 2>&1

# Negative control
TILE_FWK_DEVICE_ID=6 python3 "models/experimental/attention/dynamic_loop_index_codegen_repro.py" \
  > "models/experimental/attention/dynamic_loop_index_codegen_repro.log" 2>&1
```

## Observed Failures

### Failure A: Host compile pass failure (`OoOSchedule`)

From `online_attention_dynamic_repro.log`:

- `Errcode: FFFFFF! Run pass failed., func CompileFunction, file host_machine.cpp, line 179`
- `Run pass [OoOSchedule] failed.`

This failure occurs before runtime execution, during PyPTO compile pipeline.

### Failure B: Parser scope / symbol visibility in loop branches

From `online_state_scope_repro.log`:

- `NameError: name 'm_acc' is not defined` (inside `if pypto.is_loop_begin(idx) ... else ...`)
- Followed by framework-level secondary error:
  - `Errcode: F21004! op [REGISTER_COPY] tile shape not set`

This suggests control-flow branch variable state is not preserved as expected for parser symbolic evaluation in this pattern.

## Symptoms Summary

- Dynamic-shape + online recurrence patterns are unstable across compile stages:
  - parser scope handling (`NameError`)
  - host pass (`OoOSchedule`)
  - occasional secondary diagnostics (`REGISTER_COPY tile shape not set`) that appear after primary failure.

## What Was Already Tried

- Replacing explicit transpose with `matmul(..., b_trans=True)`
- Adjusting `set_vec_tile_shapes` around scalar/vector ops
- Reducing loop nesting depth and moving initialization outside loops
- Avoiding branch-local state updates where possible

The two minimal scripts isolate failures without application-level dependencies.

## Suggested Next Steps

- PyPTO compiler team:
  - inspect parser branch symbol scope for loop-begin/loop-end condition paths
  - inspect OoOSchedule pass behavior with dynamic `valid_shape` + reduction recurrence patterns
- Operator side workaround candidates:
  - avoid branch-local first-use variables (`if is_loop_begin ... else ...`) in online recurrence
  - split compile graph into two kernels (init pass + update pass) to simplify control-flow graph

