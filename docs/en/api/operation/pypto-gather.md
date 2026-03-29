# pypto.gather

## Supported Products

| Product          | Supported |
|:-----------------|:---------:|
| Atlas A3 Training Series/Atlas A3 Inference Series |    √     |
| Atlas A2 Training Series/Atlas A2 Inference Series |    √     |

## Description

Gathers values from the input Tensor along the specified dimension dim using the indices in index and returns the result. For a 3-dimensional Tensor, the formula is:

$$
\begin{cases}
output[i,j,k] = input[index[i,j,k], j, k] & \text{if } dim = 0; \\
output[i,j,k] = input[i, index[i,j,k], k] & \text{if } dim = 1; \\
output[i,j,k] = input[i,j, index[i,j,k]] & \text{if } dim = 2.
\end{cases}
$$

## Function Prototype

```python
gather(input: Tensor, dim: int, index: Tensor) -> Tensor
```

## Parameters


| Parameter | Input/Output | Description                                                                 |
|-----------|--------------|-----------------------------------------------------------------------------|
| input     | input        | Source operand. <br> Supported type: Tensor. <br> Supported tensor data types: DT_FP32, DT_FP16, DT_BF16, DT_INT16, DT_INT32. <br> Empty tensors are not supported; Shape supports 2–4 dimensions; shape size must not exceed 2147483647 (i.e., INT32_MAX). |
| dim       | input        | Source operand. <br> Supports any valid dimension index in the range: -input.dim to input.dim - 1. |
| index     | input        | Source operand. <br> Supported type: Tensor. <br> Supported tensor data types: DT_INT32, DT_INT64. <br> Empty tensors are not supported; Shape supports 2–4 dimensions; all axes of index must not exceed the corresponding axes of input; values must be valid indices, i.e., must not exceed input's shape along the dim axis. |

## Return Value

Returns the output Tensor. The data type of the output Tensor matches the data type of input; the shape of the output Tensor matches the shape of index.

## Constraints

1. index.dim = input.dim, and index.shape\[i\] <= input.shape\[i\] (i != dim); values must be valid indices, i.e., must not exceed input.shape\[dim\];

2. dim: -input.dim <= dim < input.dim;

3. The dim axis of input.shape cannot be partitioned; viewshape\[dim\] \>= max\( input.shape\[dim\], index.shape\[dim\] \) is required; no restriction on other dimension sizes;

4. The TileShape dimensions match result and are used to partition result and index. TileShape\[dim\] = viewshape\[dim\]; the total size of all input and output TileShapes must not exceed UB memory size.

## Example

### TileShape Configuration Example

Before calling this operation interface, set the TileShape via set_vec_tile_shapes.

The TileShape dimensions must match the output dimensions.

If input is [x, y, z], dim is 1, index is [m, t, p], and output is [m, t, p], where m <= x, p <= z, setting TileShape to [m1, t1, p1] means m1, t1, and p1 are used to partition the m, t, and p axes respectively. The y axis cannot be partitioned; it must be fully loaded.

```python
pypto.set_vec_tile_shapes(4, 16, 32)
```

### Interface Call Example

```python
x = pypto.tensor([3, 5], pypto.DT_INT32)        # shape (3, 5)
index = pypto.tensor([3, 4], pypto.DT_INT32)   # shape (3, 4)
dim = 0
y = pypto.gather(x, dim, index)
```

Example output:

```python
input data x: [[0,  1,  2,  3,  4],
             [5,  6,  7,  8,  9],
             [10, 11, 12, 13, 14]]
     index: [[0, 1, 2, 0],
             [1, 2, 0, 1],
             [2, 2, 1, 0]]
output data y: [[0,  6,  12, 3],
             [5,  11, 2,  8],
             [10, 11, 7,  3]]
```

