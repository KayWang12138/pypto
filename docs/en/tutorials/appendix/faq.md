# Frequently Asked Questions

## Output Parameter of Kernel Function Not Written Back, Causing Computation to Have No Effect

### Problem Description

In the current PyPTO framework, kernel functions decorated with `pypto.jit` do not support return values. Output must be passed in as a parameter and written back using `[:]` or similar slice operations. If a direct assignment with `=` is used, data cannot be written into the output tensor.

Example code:

```python
@pypto.jit
def add_kernel(x, y):
    pypto.set_vec_tile_shapes(4, 4)
    y = x + 1 # This creates a new tensor y

torch.npu.set_device(0)
x = torch.ones(4, 4, dtype=torch.float32)
y = torch.empty(4, 4, dtype=torch.float32)
add_kernel(pypto.from_torch(x), pypto.from_torch(y))
print(y) # Outputs the uninitialized random values created by torch.empty
```

Output data:

```python
tensor([[2.0703e-19, 7.1833e+22, 1.8502e+28, 6.8608e+22],
        [4.8011e+30, 1.2123e+25, 4.7418e+30, 1.8465e+25],
        [1.2122e+25, 4.6114e+24, 1.7836e+31, 1.7591e+22],
        [1.1306e+24, 4.2245e-39, 6.8664e-44, 0.0000e+00]])
```

### Root Cause

When `y = x + 1` is executed inside the `add_kernel` function, the `y` here is a local variable of the function (equivalent to creating a new variable `y`), which overrides the reference to the passed-in parameter `y`. In other words, this line of code only makes the `y` inside the function point to the new tensor `x + 1`, and does not modify the contents of the externally passed-in tensor `y`.

### Resolution

Use the full-slice operator `[:]` to write the computation result into the original memory space of the function parameter `y`.

Example code:

```python
@pypto.jit
def add_kernel(x, y):
    pypto.set_vec_tile_shapes(4, 4)
    y[:] = x + 1 # Write the result of x+1 into the original memory space of parameter y

torch.npu.set_device(0)
x = torch.ones(4, 4, dtype=torch.float32)
y = torch.empty(4, 4, dtype=torch.float32)
add_kernel(pypto.from_torch(x), pypto.from_torch(y))
print(y) # Outputs the result of x + 1
```

Output data:

```python
tensor([[2., 2., 2., 2.],
        [2., 2., 2., 2.],
        [2., 2., 2., 2.],
        [2., 2., 2., 2.]])
```

Note that `y[:] = x + 1` can also be replaced with `y.move(x + 1)` or `y.assemble(x + 1, [0, 0])`.

## Device ID for Operator Execution Not Set

### Problem Description

When executing an operator on an Ascend AI processor, the execution fails with the following error message:

```text
2025-12-17 14:31:32.491 E | fail get device id, check if set device id
2025-12-17 14:31:32.492 E | RuntimeAgent::AllocDevAddr failed for size 20448
2025-12-17 14:31:32.493 E | RuntimeAgent::AllocDevAddr failed for size 20448
2025-12-17 14:31:32.493 E | aclmdlRICaptureGetInfo failed, return[100000]
```

### Possible Cause

The user-defined operator is not decorated with `@jit`, and the Device ID for the current operator execution is not explicitly set using the `torch_npu` interface.

### Resolution Steps

Set the Device ID before operator execution, for example:

```python
def test_onboard():
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0)) # Get the desired device id from environment variable
    torch.npu.set_device(device_id) # Explicitly set device id
    ....
```

## Incompatible CANN Package

### Problem Description

The following error appears when executing an operator on the device:

```text
ErrorTracking callback in, task_id = 0, stream_id = 3.
[ERROR] Exception Type: exception invalid error
taskid: 0, streamid: 3, tid: 6495, deviceid: 0, retcode: 507018
kernelName = (null)
ErrorTracking callback in, task_id = 1, stream_id = 3.
[ERROR] Exception Type: exception invalid error
taskid: 1, streamid: 3, tid: 6495, deviceid: 0, retcode: 507018
kernelName = (null)
```

And in the device log, errors similar to the following null interface appear:

```text
~/ascend/log/debug/device-0/device-6495_20251222194004973.log
[ERROR] CCECPU(5670,aicpu_scheduler):2025-12-22-19:40:01.899.541 [ae_kernel_lib_aicpu_kfc.cpp:105][CallKernelApi][tid:5680][AICPU_PROCESSER] Get KFC DynTileFwkKernelServerInit api success, but func is nullptr: (null)
[ERROR] CCECPU(5670,aicpu_scheduler):2025-12-22-19:40:01.902.745 [ae_kernel_lib_aicpu_kfc.cpp:105][CallKernelApi][tid:5681][AICPU_PROCESSER] Get KFC DynTileFwkKernelServer api success, but func is nullptr: (null)
```

### Root Cause

The PyPTO driver package supports version 25.2.0 and above, and the CANN package supports version 8.5.0 and above.

### Resolution

Check the version information in the driver package installation directory, for example:

```text
/usr/local/Ascend/driver/version.info
    Version=25.3.rc1
    ascendhal_version=7.35.23
    aicpu_version=1.0
    tdt_version=1.0
    log_version=1.0
    prof_version=2.0
    dvppkernels_version=1.1
    tsfw_version=1.0
    Innerversion=V100R001C23SPC002B212
    compatible_version=[V100R001C19],[V100R001C20],[V100R001C21],[V100R001C22],[V100R001C23]
    compatible_version_fw=[7.0.0,8.9.9]
    package_version=25.3.rc1
```

You can also check the version information in the opp package within the CANN package installation directory, for example:

```text
/usr/local/Ascend/ascend-toolkit/latest/opp/version.info
    Version=8.5.0.2.220
    version_dir=8.5.0
    timestamp=20251117_000024591
    required_package_amct_acl_version="8.5"
```

Check whether the driver package and CANN package meet the version requirements using the above method, and upgrade the corresponding version if they do not.

## TileShape Does Not Match Tensor Dimensions

### Problem Description

The following error occurs during operator execution:

```text
2025-12-18 10:33:06.107 E | [ExpandFunction][Function][ERROR]: FUnction[TENSOR_b_loop_Unroll1_PATH0_hiddenfunc0] ExpandFunction failed: Tile shape size 1 is not matched the output shape size 2.
2025-12-18 10:33:06.107 E | Run pass [ExpandFunction] failed.
2025-12-18 10:33:06.107 E | Run pass <ExpandFunction> failed
```

### Possible Cause

The TileShape dimension set for a certain operation is too small, smaller than the shape dimension of the output tensor of that operation, causing an error.

### Resolution Steps

Based on the error message, locate the corresponding loop. As indicated, the problem code is in the b\_loop loop.

```text
FUnction[TENSOR_b_loop_Unroll1_PATH0_hiddenfunc0]
```

After finding the corresponding loop, based on the error dimension indicated in the log and the code logic, determine that the TileShape dimension in the code is 1 while the output shape dimension is 2, then reset the TileShape to 2 dimensions.

## Accuracy Issues Caused by Not Passing valid_shape to view

### Problem Description

Accuracy issues occur in certain scenarios when `view` is used without passing `valid_shape`.

### Root Cause

When the input tensor of the `view` interface does not have a correct `validshape`, the framework cannot correctly infer the output's `validshape`.

### Resolution Steps

When suspecting issues with `validshape` inference in the view part, first pass a `validshape` to `view` and observe whether the output result meets expectations.

A typical scenario that requires passing `validshape`:

When the input `validShape` depends on another tensor's identifier, `dynValidShape` must be passed. In the following scenario, `q0`'s `validshape` `curSeq` comes from another tensor and cannot be obtained through inference:

```python
# Input: input [B, S, H]
# Input: act_seqs [B]
# Output: out [B, S, H]
# Computation: AddS
# Code as follows:
for b_idx in pypto.loop(B, name="b_loop", idx_name="b"):
    cur_seq = act_seqs[b_idx]
    a0 = pypto.view(input, [1, S, H], [b_idx, 0, 0], valid_shape=[1, cur_seq, H])
    a1 = a0 + 1.0
    out[b_idx:, :, :] = a1
```

## Validation Error for Last Dimension of set_xxx_tile_shapes Not 32-Byte Aligned

### Problem Description

```python
with pypto.function("TENSOR_SUM_FP32", [x], [res]):
        for _ in pypto.loop(1, name="LOOP_L0", idx_name="a_idx"):
            pypto.set_vec_tile_shapes(4, 8)
            res.move(x.sum())
```

When setting the TileShape size using `pypto.set_xxx_tile_shapes`, the last dimension must be 32-byte aligned, otherwise a validation error will occur.

```text
C++ exception with description "ASSERTION FAILED: vecTile[lastDim] % alignNum == 0
Sum op: the tileShape of last axis need to 32Byte align!, func Sum, file reduction.cpp, line 374
libtile_fwk_interface.so(npu::tile_fwk::Sum(npu::tile_fwk::Tensor const&, int, bool)+0x620) [0xffff9ff2e090]

```

### Root Cause

Hardware instruction constraints require processed data to be 32-byte aligned.

### Resolution Steps

When setting the TileShape size using `pypto.set_xxx_tile_shapes`, the last dimension must be set to a value that is 32-byte aligned, i.e., TileShape\[-1\] \* sizeof\(dtype\) % 32 == 0.

## Stack Overflow Error During Operator Compilation

### Problem Description

An error similar to the following appears during operator compilation:

```text
error: stack frame size (*****) exceeds limit (32768) in function '*****'
```

A sample log output is as follows:

```text
error: stack frame size (47928) exceeds limit (32768) in function 'TENSOR_nLoop_Unroll1_PATH0_6_0_4503599627370496'
error: stack frame size (47928) exceeds limit (32768) in function 'TENSOR_nLoop_Unroll1_PATH0_6_0_4503599627370496'
2 errors generated.
terminate called after throwing an instance of 'npu::tile_fwk::Error'
  what():  ASSERTION FAILED: ret == 0
CompileCCE failed. errCode = 256, cce file: output/output_20251111_175724_806073/kernel_aicore/TENSOR_nLoop_Unroll1_PATH0_6_17699850674043372772_0_aic.cpp
```

### Root Cause

Due to hardware constraints, the maximum stack space for the scalar processing unit inside the Ascend AI processor core is 32K. Therefore, during operator compilation, the underlying Bisheng compiler analyzes and verifies the stack usage of operator core functions. If the function implementation is complex — for example due to many variables or long variable lifetimes — the analysis results may be affected. If the final analysis result exceeds the 32K limit, the Bisheng compiler will intercept and report this error.

Under the PTO programming model, the main reasons for complex operator core function implementations are as follows:

-   The subgraph scale is large, resulting in too many variables.
-   The subgraph computation logic is complex, resulting in long variable lifetimes.

### Resolution

To address the above possible causes, the following measures can be taken:

-   Adjust the tensor tiling strategy by using larger tile blocks for partitioning. By increasing the amount of computation data per tile block while keeping the total data volume constant, the number of computation steps required for the subgraph can be reduced, effectively reducing the subgraph scale. The relevant configuration interfaces are: [pypto.set\_vec\_tile\_shapes](../../api/config/pypto-set_vec_tile_shapes.md) and [pypto.set\_cube\_tile\_shapes](../../api/config/pypto-set_cube_tile_shapes.md).
-   For matrix multiplication (MATMUL) computation scenarios, it is recommended to perform multi-core partitioning along the K axis.
-   Modify the option "cycle\_upper\_bound" that controls the subgraph size to limit the maximum scale of a single subgraph to a specified range. The relevant configuration interface is: [pypto.set\_pass\_options](../../api/config/pypto-set_pass_options.md).

## Using Python print Function Inside Loops

### Problem Description

```python
@pypto.jit
def add_kernel_0(a, b, c):
    for i in pypto.loop(20):
        print("i = ", i)
        c[:] = a + b
>>>
i = 0

@pypto.jit
def add_kernel_1(a, b, c):
    for i in pypto.loop(20):
        print("i = ", i)
        if pypto.cond(i == 0):
            c[:] = a + b
        else:
            c[:] = a - b
>>>
i = 0
i = 1
```

### Possible Cause

The user operator describes a graph construction process, not the actual execution logic. During the graph construction phase, loop execution is only used to traverse all execution paths. In the example code `add_kernel_0`, there is only one execution path, so the loop executes only once. In `add_kernel_1`, there are two paths (if/else), so the loop executes twice.

### Resolution Steps

N/A

## Loop Principles and Description

During compilation, the loop is compiled into a control flow, and the loop body is converted into a computation flow, corresponding to kernel_aicpu/kernel_aicore respectively.
kernel_aicpu is responsible for executing the control flow, creating kernel_aicore tasks, including memory allocation for the resources used and execution parameter preparation.
It also analyzes the input/output dependencies to merge kernel_aicore tasks into a larger scheduling unit, then submits them to the scheduling unit for scheduling.

### Prototype

```python
def loop(start, end, step=1, name=None, idx_name=None,
         unroll_list = [1], submit_before_loop=False):

def loop_roll(start, end, step=1, name=None, idx_name=None,
         unroll_list = [1], submit_before_loop=False):
```

### Description

1. `start`, `end`, and `step` represent the start value, end value, and step size of the loop respectively. Their types can be SymbolicScalar or int, and are basically consistent with Python's `range` syntax.

2. `name` represents the name of the loop. The default value is `loop_{id}`. It has no effect on the actual runtime behavior and is only used for debugging and comment information in generated code.

3. `idx_name` represents the name of the loop index. The default value is `loop_idx_{id}`. For nested loops using the same `idx`, an override behavior will occur, and the frontend will currently perform a check and report an error.

4. `unroll_list` is primarily used for loop unrolling to produce a larger loop body and reduce scheduling overhead. For `unroll_list=2`, the loop roughly produces the following code:

    ```python
    new_start = start
    for k in unroll_list:
        left = (stop - start) % k
        for idx in loop(new_start, stop - left, k):
            for i in range(k):
                body(idx) # User needs to process a step of 1 at a time
        new_start = stop - left
    ```

    `loop_unroll` roughly produces the following code:

    ```python
    new_start = start
    for k in unroll_list:
        left = (stop - start) % k
        for idx in loop(new_start, stop - left, k):
            body(idx, k) # User needs to process a step of k at a time
        new_start = stop - left
    ```

    In principle, if multiple `i` can be processed at once, using `loop_unroll` is more efficient; if only one `i` can be processed at a time, `loop` should be used.

5. `submit_before_loop` indicates whether to submit tasks before the loop starts. The default value is False. If set to True, tasks before the loop are submitted to the scheduling queue first, waiting for subsequent tasks to complete before starting execution. *Setting `submit_before_loop` too frequently increases scheduling overhead.* It is recommended to set it to True only when necessary.

6. Impact of `unroll_list` on `pypto.cond`: Typically, a loop with one `pypto.cond` produces two branches. When the unroll count is 4, it produces 2^4 = 16 path branches. In general, each branch needs to be compiled separately, which significantly increases compilation time and the amount of compiled code. To support the compilation optimization of the key operator FA, two special functions `pypto.is_loop_begin()` and `pypto.is_loop_end()` are provided to optimize conditional branches.

7. Considering that performing `loop_unroll` on outer loops cannot increase the loop body size, currently only the innermost loop supports unrolling.

8. For coding convenience, you may sometimes see that frontend operators do not directly express a loop. How is the loop and loop body produced? In fact, during the graph construction phase, the frontend implicitly inserts a loop at the beginning of the function with an iteration count of 1. The loop continues until the start of the next loop. An example is as follows:

    ```python
    @pypto.jit
    def foo(a, b, c):
        c[:] = a + b
    # Equivalent to
    @pypto.jit
    def foo(a, b, c):
        for i in pypto.loop(1):
            c[:] = a + b

    @pypto.jit
    def foo(a, b, c):
        t = a + 1
        for i in pypto.loop(1):
            c[:] = t + b
    # Equivalent to
    @pypto.jit
    def foo(a, b, c):
        for i in pypto.loop(1):
            t = a + 1
        for i in pypto.loop(1):
            c[:] = t + b
    ```

   The framework currently does not automatically merge `loop(1)`, so in practice it is recommended that users manually merge `loop(1)` to improve efficiency.
