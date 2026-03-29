# pypto.isfinite

## Supported Products

| Product          | Supported |
|:-----------------|:---------:|
| Atlas A3 Training Series/Atlas A3 Inference Series |    √     |
| Atlas A2 Training Series/Atlas A2 Inference Series |    √     |

## Description

Determines whether the element values in the tensor are finite.

When the tensor type is integer, returns a boolean tensor of all True values with the same shape as the input tensor.

When the tensor type is floating-point, only inf/nan/-inf are not finite values; the corresponding element positions in the result are False, all others are True.

## Function Prototype

```python
isfinite(self: Tensor) -> Tensor
```

## Parameters


| Parameter | Input/Output | Description                                                                 |
|-----------|--------------|-----------------------------------------------------------------------------|
| self      | input        | Source operand. <br> Supported type: Tensor. <br> Supported tensor data types: DT_FP16, DT_BF16, DT_FP32, DT_UINT8, DT_INT8, DT_UINT16, DT_INT16, DT_UINT32, DT_INT32, DT_UINT64, DT_INT64. <br> Empty tensors are not supported; shape supports 1–5 dimensions; the number of elements in the shape must not exceed 2147483647 (i.e., INT32_MAX). |

## Return Value

Returns the output tensor. The data type of the tensor is boolean type DT_BOOL; the shape is the same as the input tensor.

## Constraints

1. Only the following data types are supported: DT_FP16, DT_BF16, DT_FP32, DT_UINT8, DT_INT8, DT_UINT16, DT_INT16, DT_UINT32, DT_INT32, DT_UINT64, DT_INT64.
2. The last axis of TileShape and ViewShape must be 32B aligned according to the output tensor type. Since the output tensor is of boolean type, the last axis of TileShape and ViewShape must be a multiple of 32.

## Example

### TileShape Configuration Example

Note: Before calling this operation interface, set the TileShape via set_vec_tile_shapes.

The TileShape dimensions must match the output dimensions.

Example 1: If input shape is [m, n] and output is [m, n], setting TileShape to [m1, n1] means m1 and n1 are used to partition the m and n axes respectively.

```python
pypto.set_vec_tile_shapes(4, 32)
```

### Interface Call Example

```python
self = pypto.tensor([3, 3], pypto.data_type.DT_FP32)
out = pypto.isfinite(self)
```

Example output:

```python
input data self: [[1 nan 3],
               [inf 1 1],
               [1 1 -inf]]
output data out: [[True False True],
             [False True True],
             [True True False]]
```

