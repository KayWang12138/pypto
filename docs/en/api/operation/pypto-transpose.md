# pypto.transpose

## Supported Products

| Product          | Supported |
|:-----------------|:---------:|
| Atlas A3 Training Series/Atlas A3 Inference Series |    √     |
| Atlas A2 Training Series/Atlas A2 Inference Series |    √     |

## Description

Returns a tensor that is a transposed version of the input tensor. The specified dimensions `dim0` and `dim1` are swapped.

## Function Prototype

```python
transpose(input: Tensor, dim0: int, dim1: int) -> Tensor
```

## Parameters


| Parameter | Input/Output | Description                                                                 |
|-----------|--------------|-----------------------------------------------------------------------------|
| input     | Input        | Source operand.<br> Supported type: Tensor.<br> Supported tensor data types: DT_FP16, DT_BF16, DT_INT16, DT_UINT16, DT_FP32, DT_INT32, DT_UINT32.<br> Empty tensors are not supported; shape supports only 2–5 dimensions; shape size must not exceed 2147483647 (INT32_MAX).<br> The operator supports different shapes; see Constraints for details. |
| dim0      | Input        | Source operand, the index of the first dimension to swap, starting from 0. |
| dim1      | Input        | Source operand, the index of the second dimension to swap, starting from 0. |

## Return Value

Returns a tensor with the same data type as the input, where the positions of `dim0` and `dim1` are swapped.

## Constraints

1. TileShape must match the dimensions of the input `input`, used to tile `input`.

2. Input dimensions `dim0` and `dim1` must be greater than 0 and less than the number of dimensions of `input`.

3. The current transpose implementation only supports the following transpose scenarios:

-   2D: any axes
-   3D: any axes
-   4D: Supported: axis 0 and 2, axis 1 and 3, axis 2 and 3, axis 1 and 2. Not supported: axis 0 and 3, axis 0 and 1.
-   5D: Supported: axis 3 and 4 only; all others are not supported.

4. Scenarios involving transposing the last axis require reserving a temporary memory space for data movement.

Example:

input : \[a, b, c, d\]  TileShape is \[t0, t1, t2, t3\], data type is DT\_FP32

dim0: 2

dim1: 3

The reserved temporary space is: t0 \* t1 \* align\(t2, 16\) \* align\(t3, 32 / sizeof\(DT\_FP32\)\)

## Example

### TileShape Configuration Example

Note: Before calling this operation interface, set the TileShape via `set_vec_tile_shapes`.

The TileShape dimensions must match the input `input` dimensions.

Example 1: If the input `input` shape is `[m, n, p]`, `dim0` is 1, and `dim1` is 2, the output is `[m, p, n]`. Set TileShape to `[m1, n1, p1]`, where `m1`, `n1`, `p1` tile the `m`, `n`, `p` axes respectively.

```python
pypto.set_vec_tile_shapes(4, 16, 32)
```

### Interface Call Example

```python
x = pypto.tensor([2, 3], pypto.DT_FP32)
y = pypto.transpose(x, 0, 1)
```

Example result:

```python
Input x: [[ 1.0028, -0.9893,  0.5809],
            [-0.1669,  0.7299,  0.4942]]
Output y: [[ 1.0028, -0.1669],
            [-0.9893,  0.7299],
            [0.5809,  0.4942]]
```
