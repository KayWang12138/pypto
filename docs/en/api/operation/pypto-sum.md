# pypto.sum

## Supported Products

| Product          | Supported |
|:-----------------|:---------:|
| Atlas A3 Training Series/Atlas A3 Inference Series |    √     |
| Atlas A2 Training Series/Atlas A2 Inference Series |    √     |

## Description

Accumulates data along a specified dimension of a multi-dimensional tensor.

The specified computation dimension (Reduce axis) is denoted as the R axis, and the non-specified dimensions (Normal axes) are denoted as the A axes. As shown below, for a 2D matrix with shape \(2, 3\), accumulating along the first dimension yields \[5, 7, 9\]; accumulating along the second dimension yields \[6, 15\].

**Figure 1**  Example of sum computed along the first dimension
![](../figures/pypto.sum_1.png)

**Figure 2**  Example of sum computed along the last dimension
![](../figures/pypto.sum_2.png)

## Function Prototype

```python
sum(input: Tensor,  dim: int, keepdim: bool = False) -> Tensor:
```

## Parameters


| Parameter | Input/Output | Description                                                                 |
|-----------|--------------|-----------------------------------------------------------------------------|
| input     | Input        | Source operand. <br> Supported type: Tensor. <br> Supported tensor data types: DT_FP32, DT_INT32, DT_INT16. <br> Empty tensors are not supported; shape supports only 2–4 dimensions; shape size must not exceed 2147483647 (INT32_MAX). |
| dim       | Input        | Source operand. <br> Supports any single axis. |
| keepdim   | Input        | Source operand. <br> Controls whether to retain the reduced dimension after the reduction. <br> Default: False. |

## Return Value

Returns an output tensor whose shape depends on the `keepdim` parameter.

If `keepdim` is True, the reduced dimension is retained after the reduction operation. The output tensor has the same shape as the input tensor on all dimensions except the one specified by `dim`, while the dimension specified by `dim` has size 1.

If `keepdim` is False (default), the reduced dimension is removed from the output tensor, while the corresponding dimension in the tileshape remains unchanged. It is therefore recommended to reset the tileshape before calling other operations.

## Constraints

1. TileShape size must not exceed 64 KB;

2. The last axis must be 32-byte aligned;

3. The second-to-last axis of TileShape must be less than or equal to 255, i.e., TileShape\[-2\] <= 255.

## Example

### TileShape Configuration Example

Note: Before calling this operation interface, set the TileShape via `set_vec_tile_shapes`.

The TileShape dimensions must match the input `input` dimensions.

Example 1: If the input `input` shape is `[m, n]` and the output is `[m, 1]`, set TileShape to `[m1, n1]`, where `m1` and `n1` tile the `m` and `n` axes respectively.

```python
pypto.set_vec_tile_shapes(4, 16)
```

### Interface Call Example

```python
x = pypto.tensor([2, 3], pypto.DT_FP32)
y = pypto.sum(x, -1, True)
```

Example result:

```
Input x: [[1.0 2.0 3.0],
             [1.0 2.0 3.0]]
Output y: [[6.0],
             [6.0]]
```
