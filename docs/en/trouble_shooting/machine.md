# MACHINE Component Error Codes

- **Range**: F7-F8XXXX
- This document describes the error code definitions, scenario descriptions, and troubleshooting recommendations for the MACHINE component.
---

## Error Code Definitions and Usage

The unified definitions for related error codes can be found in the [machine_error.h](../../framework/src/machine/utils/machine_error.h) file.

## Troubleshooting Recommendations

### AIC ERROR/The aicore execution is abnormal

1. **Comment out CallSubFuncTask and related code to rule out machine framework scheduling issues**

`framework/src/interface/machine/device/tilefwk/aicore_entry.h`

```cpp
    INLINE void ExecDynCoreFunctionKernel(ExecuteContext *ctx, uint32_t taskId) {
        uint64_t t1 = get_sys_cnt();
        SetStatus(ctx->args, ((uint64_t)taskId << 32) | STAGE_PRE_EXEC_COREFUNC_KERNEL); // high 32 bits used for taskId
        auto funcData = &ctx->funcDataList[npu::tile_fwk::FuncID(taskId)];
        auto opAttrs = &funcData->opAttrs[funcData->opAtrrOffsets[npu::tile_fwk::TaskID(taskId)]];
    #if ENABLE_AICORE_PRINT
        CoreFuncParam param = {funcData, opAttrs, funcData->exprTbl, taskId, ctx->logger.context()};
    #else
        CoreFuncParam param = {funcData, opAttrs, funcData->exprTbl, taskId, nullptr};
    #endif
        CallSubFuncTask(opAttrs[0] + funcData->exprTbl[0], &param, funcData->stackWorkSpaceAddr + ctx->blockIdx * funcData->stackWorkSpaceSize,
                        (__gm__ int64_t *)funcData->startArgs->commContexts);
        SetStatus(ctx->args, STAGE_FINISH_EXEC_COREFUNC_KERNEL);
        PipeSync();
        ...
    }
```

```cpp
    INLINE void ExecDynCoreFunctionKernel(ExecuteContext *ctx, uint32_t taskId) {
        uint64_t t1 = get_sys_cnt();
        SetStatus(ctx->args, ((uint64_t)taskId << 32) | STAGE_PRE_EXEC_COREFUNC_KERNEL); // high 32 bits used for taskId
        auto funcData = &ctx->funcDataList[npu::tile_fwk::FuncID(taskId)];
        auto opAttrs = &funcData->opAttrs[funcData->opAtrrOffsets[npu::tile_fwk::TaskID(taskId)]];
    // #if ENABLE_AICORE_PRINT
    //     CoreFuncParam param = {funcData, opAttrs, funcData->exprTbl, taskId, ctx->logger.context()};
    // #else
    //     CoreFuncParam param = {funcData, opAttrs, funcData->exprTbl, taskId, nullptr};
    // #endif
    //     CallSubFuncTask(opAttrs[0] + funcData->exprTbl[0], &param, funcData->stackWorkSpaceAddr + ctx->blockIdx * funcData->stackWorkSpaceSize,
    //                     (__gm__ int64_t *)funcData->startArgs->commContexts);
        SetStatus(ctx->args, STAGE_FINISH_EXEC_COREFUNC_KERNEL);
        PipeSync();
        ...
    }
```

Recompile and install, run to verify. If the issue still reproduces, it is a machine scheduling framework problem — stop subsequent steps. If the issue does not reproduce, revert the above changes and continue.

2. **Enable trace logging**

`framework/src/interface/configs/tile_fwk_config.json`
```cpp
"fixed_output_path": true,
"force_overwrite": false,
```

`framework/src/interface/machine/device/tilefwk/aicore_print.h`
```cpp
#define ENABLE_AICORE_PRINT 1
```

`framework/src/machine/utils/device_switch.h`
```cpp
#define ENABLE_COMPILE_VERBOSE_LOG 1
```

Recompile and install.

3. **Clean logs and run tests**

(1) Clean logs:
```bash
rm -rf ./my_log/*
rm -rf ./kernel_aic*
```

(2) Enable DEBUG logging and specify the log persistence path:
```bash
export ASCEND_GLOBAL_LOG_LEVEL=0
export ASCEND_PROCESS_LOG_PATH=./my_log
```

(3) Execute the test case.

4. **Analyze trace logs and locate CCE files**

(1) Find trace logs, analyze missing leaf indices, and locate the problematic CCE file:
```bash
python3 .agents/skills/pypto-aicore-error-locator/scripts/analyze_trace.py ./my_log run_path/kernel_aicore
```
Result description: This script will provide the path to the problematic CCE file. If multiple problematic CCE files are output, you need to verify which CCE file is the actual problematic one — proceed to step 2. If only one CCE file is output, check whether it is the problematic file — proceed to step 2.

(2) Test and verify the CCE file:
```bash
python3 .agents/skills/pypto-aicore-error-locator/scripts/test_cce_file.py <cce_file> test_cmd run_path
```
Result description: This script will make a judgment to clearly determine whether the input CCE file is the problematic file.

Note: run_path is the running directory path, test_cmd is the command to run the test.

5. **Binary search to locate the problematic code line in the CCE file**

(1) Check if the error is in a T operation:

```bash
python3 .agents/skills/pypto-aicore-error-locator/scripts/determine_error_scope.py <cce_file> test_cmd run_path
```
Result description: This script will comment out all operation lines (e.g., TLoad, TMatmul, etc.) and test. If the issue does not reproduce, the problem is in the operation lines, and ERROR_IN_T is output as True; otherwise ERROR_IN_T is False.

(2) Get the initial range for binary search:
```bash
python3 .agents/skills/pypto-aicore-error-locator/scripts/get_commentable_range.py <cce_file> ERROR_IN_T
```
Result description: This script provides left and right values for the binary search range based on the value of ERROR_IN_T. If ERROR_IN_T is True, the investigation range is all operation lines; if ERROR_IN_T is False, the investigation range is all lines except sync lines.

(3) Execute binary search iterations until the problematic code line in the CCE file is found:

```bash
python3 .agents/skills/pypto-aicore-error-locator/scripts/binary_search_iteration.py <cce_file> test_cmd run_path <left> <right> ERROR_IN_T
```
Result description: This script outputs new left and right values. Continue executing this script with the new left and right values until "problematic code line found" appears.

**Associated Skill**: [pypto-aicore-error-locator](../../.agents/skills/pypto-aicore-error-locator/SKILL.md)


### Suspected Precision Issues Related to MACHINE Memory Handling

1. **Check input initialization**:
Ensure that inputs/outputs are initialized.

2. **Check Tensor contiguity**:
Some MACHINE-related operators or interfaces require that input/output tensors are **contiguous** under the specified memory layout (`tensor.is_contiguous()` is True). Non-contiguous tensors (e.g., results of certain view, transpose, or slice operations) may cause precision anomalies or runtime errors.
e.g.: Tensors after reshape/view, dimension swapping, or indexing/slicing may be non-contiguous. If the preceding operation is COPY_IN or a tail-axis reduce, ensure that the tensor passed in at the frontend or call site is contiguous under the corresponding format.

3. **Increase workspace size**:
`python/pypto/frontend/parser/entry.py`
```python
workspace_tensor = torch.empty(workspace_size, dtype=torch.uint8, device=device)
```

```python
workspace_tensor = torch.empty(workspace_size * 10, dtype=torch.uint8, device=device)
```
If the issue no longer reproduces, it is a workspace calculation problem.

4. **Change workspace management from torch to internal self-management**:
`framework/src/machine/runtime/device_launcher.cpp`
```cpp
    static void PrepareDevProgArgs(DevAscendProgram *devProg, DeviceLauncherConfig &config,
                                  [[maybe_unused]]bool isDevice) {
        ...
        if (config.workspaceAddr) {
            kArgs.workspace = (int64_t *)config.workspaceAddr;
        } else if (kArgs.workspace == nullptr && (devProg->workspaceSize != 0)) {
            kArgs.workspace = (int64_t *)devMem.AllocDev(devProg->workspaceSize, CachedOperator::GetWorkspaceDevAddrHolder(cachedOperator));
        }
        ...
    }
```

```cpp
    static void PrepareDevProgArgs(DevAscendProgram *devProg, DeviceLauncherConfig &config,
                                  [[maybe_unused]]bool isDevice) {
        ...
        if (0) {
            kArgs.workspace = (int64_t *)config.workspaceAddr;
        } else if (kArgs.workspace == nullptr && (devProg->workspaceSize != 0)) {
            kArgs.workspace = (int64_t *)devMem.AllocDev(devProg->workspaceSize, CachedOperator::GetWorkspaceDevAddrHolder(cachedOperator));
        }
        ...
    }
```
If the issue no longer reproduces, there is a workspace usage problem, such as memory stomping.

5. **Memory overlap detection at the leaf function granularity**:
(1) Enable VERBOSE logging:
`framework/src/machine/utils/device_switch.h`
```cpp
#define ENABLE_COMPILE_VERBOSE_LOG 1
```

(2) Enable DEBUG logging and specify the log persistence path:
```bash
export ASCEND_GLOBAL_LOG_LEVEL=0
export ASCEND_PROCESS_LOG_PATH=./my_log
```

(3) Recompile the pypto whl package and install it.

(4) Enable performance data collection to ensure dyn_topo.txt is generated:

    ```python
    @pypto.frontend.jit(
        debug_options={"runtime_debug_mode": 1}
    )
    ```

(5) Execute the test case.

(6) Execute the memory detection script:

Command format:
```bash
python3 tools/schema/schema_memory_check.py -d <device_log_dir_absolute_path> -t <topo_file_absolute_path>
```

Example (replace with actual absolute paths):
```bash
python3 tools/schema/schema_memory_check.py -d /path/to/my_log/debug/device-8/ -t /path/to/output/output_20260314_112655_964781_3025352/dyn_topo.txt
```

If no anomaly: prompts that device tasks have no memory overlap.
If anomaly exists: prompts the device task and leaf function with memory overlap.

Notes:
(1) If the error `memory reuse must happen for full match.` is reported, the ranges of two rawtensors that require memory reuse are inconsistent.
(2) If the error `memory reuse must happen for same dimension.` is reported, the shapes of two memory-reused rawtensors are inconsistent.
The above two cases are not memory overlaps. The script's memory check dependency will not encounter these situations, so the script will assert directly and promptly report log information errors.

6. **Rule out complex features**:
Use the `pypto-precision-debugger` skill to disable unroll_list, axis-merging features, configure `submit_before_loop=True` to execute loops serially, confirm the correctness of valid_shape configuration, +0.0, etc., to narrow down the investigation scope.


### encode Phase actualRawMagic Assertion Triggered

**Problem characteristics**: An assertion is triggered during the encode phase when running a test case. The error message contains `Shape size mismatch` or `Data size mismatch`, accompanied by output of fields such as `rawTensor->actualRawmagic`, `rawShape`, and `actualrawShape`. The error location is the `InitRawTensorAndMemoryRequirement` function in `framework/src/machine/utils/dynamic/dev_encode.cpp`.

**Problem background**: The encode phase performs memory reuse consistency checks on all RawTensors with `actualRawmagic` — requiring that both ends of the reuse chain (the current rawTensor and the actualRaw pointed to by `actualRawmagic`) are strictly consistent in rawShapeSize (product of element counts) or rawDataSize (byte size). When memory reuse operations such as `reshape` and `assemble` are involved, if the pass side does not synchronously update the feature information of the related tensors, this type of assertion will be triggered.

**Troubleshooting steps**:

1. **Obtain basic information from the first error occurrence**:

   When an assertion is triggered, the error log will contain the following key fields (corresponding to lines 412~420 of `dev_encode.cpp`):

   ```
   Shape size mismatch: <rawShapeSize> != <actualRawShapeSize>,
   rootMagic=<...>, rootHash=<...>,
   rawShape=<...>, actualrawShape=<...>,
   rawTensor->rawMagic=<...>, rawTensor->actualRawmagic=<...>, actualRaw->rawMagic=<...>
   ```

   Record the `rawMagic`, `actualRawmagic`, `rawShape`, and `actualrawShape` for use in subsequent computation graph location.

   If the current error information is incomplete (missing rawMagic and other fields), refer to the context of this ASSERT in `dev_encode.cpp`, add print statements for the relevant fields before the assertion, recompile the whl package, and reproduce the issue.

2. **Enable the pass computation graph dump switch**:

   Modify `framework/src/interface/configs/tile_fwk_config.json`, changing `dump_graph` under `global.pass.default_pass_configs` to `true`:

   ```json
   "default_pass_configs": {
       "print_graph": false,
       "print_program": false,
       "dump_graph": true,
       ...
   }
   ```

   Recompile the whl package and install it.

3. **Reproduce the test case and obtain the pass computation graph**:

   Execute the test case again. Computation graph files for each pass stage will be generated in the `build/output/pass/` directory.

4. **Locate the target pass range**:

   It is recommended to focus on tilegraphs between **pass4 ~ pass27**:
   - pass4 ExpandFunction: Before this is the tensorgraph stage, which has not yet entered the tile expansion granularity analysis of the pass.
   - pass27 SubgraphToFunction: After this, the framework begins to split the root function into leaf functions, and the graph structure is more scattered, making overall location less convenient.

   Within this range, use the `rawMagic` / `actualRawmagic` obtained in step 1 to locate the corresponding tensor node in the computation graph.

5. **Analyze the root cause in combination with the computation graph**:

   After identifying the tensor node, analyze its upstream and downstream operations along the data flow, with a focus on the following scenarios:

   - **reshape operation**: reshape sets `actualRawmagic` for the output tensor, allowing it to reuse the memory address of the input tensor.
   - **assemble operation**: During graph optimization, assemble replaces the raw pointer of the input tensor with the raw of the target large tensor. If reshape has already set `actualRawmagic` before this, that field may not be synchronously updated with the raw replacement.
   - **Other memory reuse operations**: Operations such as view and inplace also involve the setting and passing of `actualRawmagic`.

   If the rawShapeSize at both ends of the reuse chain becomes inconsistent after a certain pass, it indicates that the pass rewrote the rawshape on one side but did not synchronize the other side.

6. **Determine the issue owner**:

   (1) **Test case writing issue**: Check whether the combination of reshape, assemble, and other operations in the test case meets the API constraints (e.g., limitations of `inplace=True`, consistency requirements of valid_shape and rawshape, etc.). If non-compliant usage is found, adjust the test case.

   (2) **Framework-side issue**: If the test case writing is correct, contact the **pass team** for further analysis to investigate whether there are omissions or incorrect update scenarios in the framework's handling of actualRawmagic propagation.

Notes:
- An actualRawmagic assertion failure is essentially a compile-time consistency check. If precision is still correct after removing the assertion, it indicates that the runtime read/write on that path did not exceed bounds. This is a case of overly strict validation or a missed synchronization on the pass side, and requires joint analysis with the pass team.
- If this type of issue is triggered in a dynamic shape (with negative dimension) scenario, `dev_encode.cpp` will skip validation of dynamic dimensions (the `isDynamicShape` branch). Be aware of the distinction between static and dynamic shape scenarios when troubleshooting.

---

### F70006 HANDSHAKE_TIMEOUT

1. **Confirm device and driver**: The NPU device is available and driver is normal; `npu-smi info` shows no anomalies.
2. **Confirm resources and load**: Check whether NPU utilization is too high in the current process/container, and whether multiple processes are competing for the same device.
3. **Confirm timeout configuration**: If there is a handshake/sync timeout configuration item, check whether it is too short or does not match the environment.
4. **Check log context**: Combined with the logs before and after in the same thread (e.g., after "Schedule run init succ" and related to AbnormalStop) to determine whether this is an initial handshake failure or an anomaly during operation.

**Associated Skill**: [pypto-environment-setup](../../.agents/skills/pypto-environment-setup/SKILL.md) (environment and NPU device diagnostics, `npu-smi`, driver and compilation/runtime)
