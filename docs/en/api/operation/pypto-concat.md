# pypto.concat

## Supported Products

| Product          | Supported |
|:-----------------|:---------:|
| Atlas A3 Training Series/Atlas A3 Inference Series |    √     |
| Atlas A2 Training Series/Atlas A2 Inference Series |    √     |

## Description

Concatenates multiple input tensors along the specified dimension (dim) and returns the concatenated tensor.

## Function Prototype

```python
concat(tensors: List[Tensor], dim: int = 0) -> Tensor
```

## Parameters


| Parameter | Input/Output | Description                                                                 |
|-----------|--------------|-----------------------------------------------------------------------------|
| tensors   | Input        | Source operand. <br> Supported type: Tensor. <br> Supported tensor data types: DT_BF16, DT_FP32, DT_FP16, DT_INT8, DT_INT16, DT_INT32. <br> Empty tensors are not supported; shape supports 2–4 dimensions only; shape size must not exceed 2147483647 (INT32_MAX). |
| dim       | Input        | Source operand. <br> Supported data type: int, default is 0.                |

## Return Value

Returns the output tensor. The tensor's data type is the same as any tensor in `tensors`. The shape is the same as any tensor in `tensors` (except along `dim`); the size of the `dim` dimension equals the sum of the corresponding dimension sizes across all tensors in `tensors`.

## Constraints

1. The size of source operand `tensors` must be greater than or equal to 2, i.e., len\(tensors\)\>=2; and less than or equal to 128. (A single-tensor input is supported, but precision is not guaranteed at this time.)

2. Input tensors must have the same data type, the same number of dimensions, and equal sizes in every dimension except the concatenation dimension (dim).

3. dim: -input.dim <= dim < input.dim (where input refers to any tensor in `tensors`).

4. When setting viewshape, the dimension corresponding to `dim` must not be split into blocks (i.e., the viewshape value for that dimension must be >= the corresponding value of any tensor in `tensors`).

## Example

### TileShape Configuration Example

Before calling this operation interface, set the TileShape via `set_vec_tile_shapes`.

The TileShape dimensions must match the output.

For example, if the input tensors have dimensions [m, c1, p] and [m, c2, p], with output [m, c1+c2, p], and TileShape is set to [m1, n1, p1], then m1 and p1 are used to split the m and p axes respectively, and n1 is used to split the c1 and c2 axes.

```python
pypto.set_vec_tile_shapes(4, 16, 32)
```

### Interface Call Example

```python
a = pypto.tensor([2, 2], pypto.DT_FP32)  # 2x2 tensor with all 1s
b = pypto.tensor([2, 2], pypto.DT_FP32)  # 2x2 tensor with all 0s
out = pypto.concat([a, b], dim = 0)
```

Example result:

```python
Input a:   [[1.0 1.0],
             [1.0 1.0]]
Input b:   [[0.0 0.0],
             [0.0 0.0]]
Output out: [[1.0 1.0],
              [1.0 1.0],
              [0.0 0.0],
              [0.0 0.0]]

```
