# Known Issues

## Using an Uninitialized Tensor

### Problem Description

A tensor is declared using `pypto.tensor`, mistakenly assuming it behaves like `torch.empty`, which allocates a block of uninitialized memory. Reading it directly (e.g., using `view`) before writing to it again will cause framework validation errors or accuracy issues.

### Root Cause

In PyPTO, in addition to tensor declarations that explicitly include initialization behavior (such as `pypto.full`, `pypto.zeros`, etc.), tensors can also be declared using `pypto.tensor`, but this interface does not include initialization behavior. In PyPTO, every tensor must be written before it is read — i.e., it must have a producer before it can have a consumer. Uninitialized tensors do not allocate memory. The framework typically validates and reports an error when a tensor with no producer is used directly, but sometimes due to validation oversights, this may result in on-device accuracy errors.

### Resolution Steps

Avoid using uninitialized tensors.

## Cycle Error When Performing View and Assemble on the Same Tensor

### Problem Description

Example code:

```python
@pypto.jit
def foo_kernel(x, y):
    pypto.set_vec_tile_shapes(16, 16)
    a = pypto.zeros([32, 32])
    b = a[:16, :16] # Get data from a via view
    a[16:, 16:] = b.exp() # Compute and assemble back to a
    y[:] = x + a

torch.npu.set_device(0)
x = torch.ones(32, 32, dtype=torch.float32)
y = torch.empty(32, 32, dtype=torch.float32)
foo_kernel(pypto.from_torch(x), pypto.from_torch(y))
```

An ASSERTION FAILED error occurs during execution:

```text
ERROR:root:Record function foo_kernel failed: ASSERTION FAILED: outDegree[opToIndex[op.get()]] == 0
```

Detailed error:

```text
ERROR:root:Record function foo_kernel failed: ASSERTION FAILED: outDegree[opToIndex[op.get()]] == 0
Operation not fully processed: /* /home/pypto-dev/a.py:9 */
<32 x 32 x DT_FP32 / 32 x 32 x DT_FP32> %0@2#(-1)MEM_UNKNOWN::MEM_UNKNOWN = !10000 VEC_DUP(g:-1, s:-1) #SCALAR{0.000000} #op_attr_shape{[32, 32]} #op_attr_validShape{[32,32]}
, func GetSortedOperations, file function.cpp, line 1105
libtile_fwk_interface.so(npu::tile_fwk::Function::GetSortedOperations() const+0xb3c) [0xffff9c2f6650]
libtile_fwk_interface.so(npu::tile_fwk::Function::SortOperations()+0x38) [0xffff9c2f6f28]
libtile_fwk_interface.so(npu::tile_fwk::Function::EndFunction(std::shared_ptr<npu::tile_fwk::TensorSlotScope> const&)+0x960) [0xffff9c31b8d0]
libtile_fwk_interface.so(npu::tile_fwk::Program::FinishCurrentFunction(std::shared_ptr<npu::tile_fwk::TensorSlotScope> const&, bool)+0x1b0) [0xffff9c532274]
libtile_fwk_interface.so(npu::tile_fwk::Program::EndFunction(std::string const&, bool)+0x10c) [0xffff9c536dcc]
libtile_fwk_interface.so(npu::tile_fwk::Program::EndHiddenLoop(npu::tile_fwk::Function*, bool)+0xb0) [0xffff9c537384]
libtile_fwk_interface.so(npu::tile_fwk::Program::EndFunction(std::string const&, bool)+0x5c) [0xffff9c536d1c]
libtile_fwk_interface.so(npu::tile_fwk::RecordLoopFunc::IterationEnd()+0x44) [0xffff9c5399c4]
libtile_fwk_interface.so(npu::tile_fwk::RecordLoopFunc::Iterator::operator!=(npu::tile_fwk::RecordLoopFunc::IteratorEnd const&)+0xfc) [0xffff9c539ea0]
```

### Root Cause

This error occurs because a cycle is detected internally during topological sorting of the basic operators.

This is caused by data being read from `a` and then written back to `a`.

Since PyPTO describes a graph expression, when reading and writing, `a` is currently considered as a whole. Therefore, the connection relationship created forms a cycle: a->b->b.exp()->a. PyPTO does not allow cycles in the constructed graph — it must be a DAG (Directed Acyclic Graph). This is why the error occurs.

![](../figures/zh-cn_image_0000002499301464.png)

### Resolution

-   Currently, the logic for reading from and writing to `a` must be split into two separate graph definitions to avoid cycles within a single graph.

-   This issue will be resolved after the SSA semantics for Assemble are released.

![](../figures/zh-cn_image_0000002530981685.png)

## Passing Different Runtime Values to Static Axes When Executing the Same Operator Multiple Times (or Missing Annotations for Dynamic Axes)

### Problem Description

Accuracy errors, or AI CPU/AI Core exceptions.

### Possible Cause

Since compilation performs fixed-size partitioning for static axes, the number of partitions is also fixed. This means that code compiled once can only handle one specific static axis. If the static axis passed in is different from the one used during the first compilation, the memory address accessed after partitioning may exceed the actual size of the passed-in tensor, leading to memory access errors or interfering with other tensors, thereby causing accuracy anomalies.

### Resolution Steps

-   Option 1: Define different operators for each different static value.

    ```python
    def handler(in): # Define common processing function
        return pypto.add(in, in)

    @pypto.jit
    def adder_256(in_shape_256): # Define operator for handling in axis size of 256
        return handler(in_shape_256)

    @pypto.jit
    def adder_1024(in_shape_1024): # Define operator for handling in axis size of 1024
        return handler(in_shape_1024)

    adder_256(in_256)
    adder_1024(in_1024)
    ```

-   Option 2: Define as dynamic axis.

    ```python
    @pypto.jit
    def adder(in_shape): # Define operator for handling in axis size of 256
        out = Tensor(in_shape.shape[0])
        for k in pypto.loop(in_shape.shape[0] / 256):
            out[k * 256: k *256 + 256] = pypto.add(
                in[k * 256 : k * 256 + 256],
                in[k * 256 : k * 256 + 256])
        return handler(in_shape_256)
    ```

## Tensor Memory Spanning Multiple Sub-loops Within a Parent Loop Not Supported for Allocation in Each Parent Loop Iteration

### Problem Description

In a two-level or deeper nested loop, a tensor is defined in the parent loop, written in one sub-loop, and used in another sub-loop — resulting in accuracy errors.

### Possible Cause

For tensors on GM, the memory address allocated in each iteration of the parent loop is the same. This means that different iterations should use different temporary memory to store data, but actually use the same address. Since different loop iterations are executed in parallel, when multiple iterations run simultaneously, later-executing iterations overwrite the temporary memory of earlier iterations, causing accuracy issues.

### Resolution Steps

Add `submit_before_loop = True` to the second sub-loop to submit tasks before starting that sub-loop, forcing multiple iterations to run serially and avoiding accuracy issues caused by memory overwriting and conflicts during parallel execution.

```python
for outer in pypto.loop(...): # Parent loop, executes at least twice; if it only executes once, there is no parallel overwriting issue
    t = pypto.Tensor(...)  # Define a temporary tensor
    for inner0 in pypto.loop(...): # First sub-loop, assigns value to temporary tensor t
        ...
        t[...] = ...
    # Add submit_before_loop to ensure multiple parent loop iterations are not in the same parallel execution block
    for inner1 in pypto.loop(..., submit_before_loop=True): # Second sub-loop, uses temporary tensor t
        x[:] = t[:] + t[:]
```

## SymbolicScalar Does Not Support Auto-Increment Inside Loops

### Problem Description

```python
@pypto.jit
def add_kernel_1(a, b, c):
    count = 0
    for i in pypto.loop(20):
        count = count + 1
```

When actually executing at `i = 1`, `count` will not increment from 0 to 20 as the user expects.

### Possible Cause

The current PyPTO framework only captures the user's tensor operations, not the user's scalar operations. Therefore, `count` will not be processed as a variable. Currently, only loop variables can implement auto-increment.

### Resolution Steps

Use the loop variable to express auto-increment logic.

## Simulation Exception When torch_npu Is Installed but CANN Is Not Installed

### Problem Description

When executing an operator in the simulation environment, the execution fails with the following error message:

```text
ImportError: libhccl.so: cannot open shared object file: No such file or directory
```

### Root Cause

When the program starts, torch (version > 2.5) automatically loads all extensions named "torch.backends" (e.g., torch_npu). If torch_npu is installed in the environment but CANN is not, an exception will be raised due to missing dependencies.

### Resolution Steps

Add the following environment variable before executing the operator to avoid the above exception:

```bash
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
```

