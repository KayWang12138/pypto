# Loops and Data Tiling

In the previous sections, after tiling configuration, the tensor inputs and outputs are partitioned inside the kernel according to the tiling configuration scheme. For example, if the original tensor is \(1, 32, 1, 256\) and the tiling scheme is configured as \(1, 4, 1, 64\), the kernel will tile and iterate using the \(1, 4, 1, 64\) scheme.

If the data volume increases and the tensor configuration changes to \(32, 32, 1, 256\), the first axis (shape\[0\] or the Batch axis) can use `pypto.loop` to add loop logic, enabling the framework to unroll multiple batches for parallel processing.

## Basic Loop Structure

```python
SHAPE = (32, 32, 1, 256)

@pypto.frontend.jit
def add_kernel(
    input0: pypto.Tensor(SHAPE, pypto.DT_FP32),
    input1: pypto.Tensor(SHAPE, pypto.DT_FP32),
    out: pypto.Tensor(SHAPE, pypto.DT_FP32),
    val: int
):
    pypto.set_vec_tile_shapes(1, 4, 1, 64)

    #calculate the loop parameters
    b, n, s, d = SHAPE
    tile_b = 1
    b_loop = b // tile_b

    for idx in pypto.loop(b_loop):
        b_offset = idx * tile_b
        b_offset_end = (idx + 1) * tile_b
        t0_sub = input0[b_offset:b_offset_end, ...]
        t1_sub = input1[b_offset:b_offset_end, ...]
        t3_sub = t0_sub + t1_sub
        out[b_offset:b_offset_end, ...] = t3_sub + val
```

The full interface parameters for `pypto.loop` are as follows:

```python
for idx in pypto.loop(start, end, step, name="label", idx_name="idx_label", submit_before_loop=False)
```

-   Parameters:
    -   start, end, step: Optional parameters, supporting flexible loop range configuration.
    -   name, idx\_name: Loop label and index variable name, used for debugging.
    -   submit\_before\_loop: Controls the loop execution order.

It can also be simplified as needed:

```python
for idx in pypto.loop(start, end, step)   #without a loop name label
for idx in pypto.loop(start, end)         # default step=1
for idx in pypto.loop(end)                # default start=0, step=1
```

For a complete example of slicing syntax sugar, refer to: [loop.py](../../../examples/02_intermediate/controlflow/loop/loop.py).

## Using view/assemble Interfaces in Loop Structures

The example above shows how to use Python's slice operation \[:, :, :\] inside a loop to extract small data blocks for computation. After computation, the data is output. Alternatively, you can use the `pypto.view` interface to extract small data blocks for computation, and then use `pypto.assemble` to assemble and output the data after computation is complete.

```python
SHAPE = (32, 32, 1, 256)

@pypto.frontend.jit
def add_kernel(
    input0: pypto.Tensor(SHAPE, pypto.DT_FP32),
    input1: pypto.Tensor(SHAPE, pypto.DT_FP32),
    out: pypto.Tensor(SHAPE, pypto.DT_FP32),
):
    pypto.set_vec_tile_shapes(1, 4, 1, 64)

    #calculate the loop parameters
    b, n, s, d = SHAPE
    tile_b = 1
    b_loop = b // tile_b

    for idx in pypto.loop(b_loop):
        b_offset = idx * tile_b
        t0_sub = pypto.view(input0, [tile_b, n, s, d], [b_offset, 0, 0, 0])
        t1_sub = pypto.view(input1, [tile_b, n, s, d], [b_offset, 0, 0, 0])
        t3_sub = t0_sub + t1_sub
        pypto.assemble(t3_sub, [b_offset, 0, 0, 0], out)
```

For a complete example of the view/assemble interface, refer to: [add_scalar_loop_view_assemble.py](../../../examples/01_beginner/transform/add_scalar_loop_view_assemble.py).

## Data Dependencies and Loop Order

By default, `pypto.loop` unrolls and dispatches iterations to multiple cores for parallel processing, which is suitable for scenarios without data dependencies. If there are data dependencies between loop iterations (e.g., the output of a previous iteration is the input for the next), you need to set `submit_before_loop = True` to ensure that the result of each loop iteration is written back to the tensor before the next iteration starts:

```python
for idx in pypto.loop(0, b_loop, 1, name="LOOP_L0_bIdx", idx_name="idx", submit_before_loop=True):
```

