# pypto.amax

## Supported Products

| Product          | Supported |
|:-----------------|:---------:|
| Atlas A3 Training Series/Atlas A3 Inference Series |    √     |
| Atlas A2 Training Series/Atlas A2 Inference Series |    √     |

## Description

Computes the maximum value of a multi-dimensional tensor along a specified dimension.

The specified computation dimension (Reduce axis) is defined as the R axis, and the non-specified dimension (Normal axis) is defined as the A axis. As shown in the figures below, for a 2D matrix with shape \(2, 3\), computing the maximum along the first dimension yields \[4, 5, 6\]; computing the maximum along the second dimension yields \[3, 6\].

**Figure 1**  amax computation example along the first dimension
![](../figures/pypto.amax_1.png)

**Figure 2**  amax computation example along the last dimension
![](../figures/pypto.amax_2.png)

## Function Prototype

```python
amax(input: Tensor, dim: int, keepdim: bool = False) -> Tensor:
```

## Parameters


| Parameter | Input/Output | Description                                                                 |
|-----------|--------------|-----------------------------------------------------------------------------|
| input     | input        | Source operand. <br> Supported type: Tensor. <br> Tensor supported data types: DT_FP16, DT_BF16, DT_FP32, DT_INT32, DT_INT16. <br> Empty Tensor not supported; shape supports only 2–4 dimensions; shape size must not exceed 2147483647 (i.e., INT32_MAX). |
| dim       | input        | Source operand. <br> Supports any single axis.                              |
| keepdim   | input        | Source operand. <br> Controls whether to retain the reduced dimension after reduction. <br> Default value: False. |

## Return Value

Returns the output Tensor. The shape of the output Tensor depends on the keepdim parameter.

If keepdim is True, the reduced dimension is retained after the reduction operation. The output Tensor has the same shape as the input Tensor for all dimensions except the one specified by dim, while the size of the dimension specified by dim is 1.

If keepdim is False (default), the reduced dimension is removed from the output Tensor, while the corresponding dimension in tileshape remains unchanged. It is therefore recommended to reset the tileshape before calling other operations.

## Constraints

1. TileShape size must not exceed 64KB;

2. The last axis must be 32-byte aligned;

3. The second-to-last axis of TileShape must be less than or equal to 255, i.e., TileShape\[-2\]<=255.

## Example

### TileShape Configuration Example

Before calling this operation interface, set the TileShape via set_vec_tile_shapes.

The TileShape dimensions should be consistent with the input.

For example, if input shape is [m, n] and output is [m, 1], setting TileShape to [m1, n1] means m1 and n1 are used to tile the m and n axes respectively.

```python
pypto.set_vec_tile_shapes(4, 16)
```

Note: If keepdim is set to false, the reduced dimension is removed from the output Tensor, while the corresponding dimension in tileshape remains unchanged. It is therefore recommended to reset the tileshape before calling other operations.

### Interface Call Example

```python
x = pypto.tensor([2, 3], pypto.DT_FP32)
y = pypto.amax(x, -1, True)
```

Example result:

```python
Input data x: [[1.0 2.0 3.0],
               [1.0 2.0 3.0]]
Output data y: [[3.0],
                [3.0]]
```

