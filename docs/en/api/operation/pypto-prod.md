# pypto.prod

## Supported Products

| Product          | Supported |
|:-----------------|:--------:|
| Atlas A3 Training Series/Atlas A3 Inference Series |    √     |
| Atlas A2 Training Series/Atlas A2 Inference Series |    √     |

## Description

Computes the cumulative product of elements in a multi-dimensional tensor along a specified dimension.

## Function Prototype

```python
prod(input: Tensor,  dim: int, keepdim: bool = False) -> Tensor:
```

## Parameters


| Parameter | Input/Output | Description                                                                 |
|-----------|--------------|-----------------------------------------------------------------------------|
| input     | input        | Source operand. <br> Supported type: Tensor. <br> Tensor supported data types: DT_FP32, DT_INT32, DT_INT16. <br> Empty Tensor not supported; shape supports only 2–4 dimensions; Shape Size must not exceed 2147483647 (i.e., INT32_MAX). |
| dim       | input        | Source operand. <br> Supports any single axis. |
| keepdim   | input        | Source operand. <br> Controls whether the reduced dimension is retained after the reduction. <br> Default value is False. |

## Return Value

Returns the output Tensor. The shape of the output Tensor depends on the keepdim parameter.

If keepdim is True, the reduced dimension is retained after the reduction operation. The output Tensor has the same shape as the input Tensor in all dimensions except the one specified by dim, which has size 1.

If keepdim is False (default), the reduced dimension is removed from the output Tensor, while the corresponding dimension in tileshape remains unchanged. It is therefore recommended to reset tileshape before calling other operations.

## Constraints

1. TileShape size must not exceed 64KB;

2. The last axis must be 32-byte aligned;


## TileShape Configuration Example

The TileShape dimensions should match the input dimensions.

For example, if the input shape is [m, n] and the output is [m, 1], TileShape is set to [m1, n1], where m1 and n1 are used to tile the m and n axes respectively.

```python
pypto.set_vec_tile_shapes(m1, n1)
```

## Example

```python
x = pypto.tensor([2, 3], pypto.DT_FP32)
y = pypto.prod(x, -1, True)
```

Example result:

```
input data x: [[1.0 2.0 3.0],
               [1.0 2.0 3.0]]
output data y: [[6.0],
                [6.0]]
```

