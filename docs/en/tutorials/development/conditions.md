# Conditions and Branches

Conditions and branches are used in programs to implement conditional logic, executing different code paths based on different conditions. The programming framework supports two types of conditional branch functionality:

-   Static conditional branches: Configure conditional branches at compile time, generating fixed instructions for execution. Multiple JIT compilations can generate different kernels.
-   Dynamic conditional branches: Evaluate conditions and branches at runtime, executing the corresponding functionality.

## Static Conditional Branches

```python
# Generate kernel using parameter add1_flag=False
@pypto.frontend.jit
def add_kernel_false(
    input0: pypto.Tensor([pypto.DYNAMIC, 4, 1, 64], pypto.DT_FP32),
    input1: pypto.Tensor([pypto.DYNAMIC, 4, 1, 64], pypto.DT_FP32),
    output: pypto.Tensor([pypto.DYNAMIC, 4, 1, 64], pypto.DT_FP32),
    val: int
):
    add_core(input0, input1, output, val, False)

#Generate kernel using parameter add1_flag=True
@pypto.frontend.jit
def add_kernel_true(
    input0: pypto.Tensor([pypto.DYNAMIC, 4, 1, 64], pypto.DT_FP32),
    input1: pypto.Tensor([pypto.DYNAMIC, 4, 1, 64], pypto.DT_FP32),
    output: pypto.Tensor([pypto.DYNAMIC, 4, 1, 64], pypto.DT_FP32),
    val: int
):
    add_core(input0, input1, output, val, True)
```

Code example:

```python
def add_core(input0: pypto.Tensor, input1: pypto.Tensor, output: pypto.Tensor, val: int, add1_flag: bool = False):
    # Tiling configuration and loop logic
    pypto.set_vec_tile_shapes(1, 4, 1, 64)

    #calculate the loop parameters
    b = input0.shape[0]
    tile_b = 1
    b_loop = b // tile_b

    for idx in pypto.loop(b_loop):
        b_offset = idx * tile_b
        b_offset_end = (idx + 1) * tile_b
        t0_sub = input0[b_offset:b_offset_end, ...]
        t1_sub = input1[b_offset:b_offset_end, ...]
        t3_sub = t0_sub + t1_sub
        if add1_flag:
            output[b_offset:b_offset_end, ...] = t3_sub + val
        else:
            output[b_offset:b_offset_end, ...] = t3_sub
```

This example adds an optional parameter `add1_flag` to the `add_kernel` function and uses it to handle different cases. If `add1_flag` is True, the output result has `val` added to it; otherwise, the result of the previous processing step is output directly.

For a complete example, refer to: [condition.py](../../../examples/02_intermediate/controlflow/condition/condition.py).

## Dynamic Conditional Branches

Evaluate conditions and branches at runtime to execute the corresponding functionality. The core interfaces include:

-   `pypto.cond`\(condition\): Evaluate a condition at runtime.
-   `pypto.is_loop_begin`\(idx\): Check whether this is the first iteration of the loop.
-   `pypto.is_loop_end`\(idx\): Check whether this is the last iteration of the loop.

```python
@pypto.frontend.jit
def add_kernel(
    input0: pypto.Tensor([pypto.DYNAMIC, 4, 1, 64], pypto.DT_FP32),
    input1: pypto.Tensor([pypto.DYNAMIC, 4, 1, 64], pypto.DT_FP32),
    output: pypto.Tensor([pypto.DYNAMIC, 4, 1, 64], pypto.DT_FP32),
    val: int
):
    ...
    for idx in pypto.loop(b_loop):
        t3_sub = t0_sub + t1_sub
        if idx < 2:  # Dynamic conditional check
            output[b_offset:b_offset_end, ...] = t3_sub + val
        else:
            output[b_offset:b_offset_end, ...] = t3_sub

        # Or conditions based on loop position
        if pypto.is_loop_begin(idx):
            output[b_offset:b_offset_end, ...] = t3_sub + val
        elif pypto.is_loop_end(idx):
            output[b_offset:b_offset_end, ...] = t3_sub + val + 1
        else:
            output[b_offset:b_offset_end, ...] = t3_sub
```

For a complete example, refer to:

[condition.py](../../../examples/02_intermediate/controlflow/condition/condition.py)

