# pypto.le

## Supported Products

| Product          | Supported |
|:-----------------|:---------:|
| Atlas A3 Training Series/Atlas A3 Inference Series |    √     |
| Atlas A2 Training Series/Atlas A2 Inference Series |    √     |

## Description

Element-wise less-than-or-equal comparison operation.

## Function Prototype

```python
le(input: Tensor, other: Union[Tensor, float, Element]) -> Tensor
```

## Parameters


| Parameter | Input/Output | Description                                                                 |
|-----------|--------------|-----------------------------------------------------------------------------|
| input     | input        | Source operand. <br> Supported type: Tensor. <br> Supported tensor data types: DT_FP16, DT_BF16, DT_FP32; the data types of the two source operands must be consistent. <br> Empty tensors are not supported; Shape supports only 2–4 dimensions; Shape Size must not exceed 2147483647 (i.e., INT32_MAX). |
| other     | input        | Source operand. <br> Supported types: Tensor, float, Element. <br> When of type float, it will be automatically converted to Element type, where float corresponds to DT_FP32. For other data types, use Element to construct. <br> Tensor and Element supported data types: DT_FP16, DT_BF16, DT_FP32; the data types of the two source operands must be consistent. <br> Empty tensors are not supported; Shape supports only 2–4 dimensions; Shape Size must not exceed 2147483647 (i.e., INT32_MAX). |

## Return Value

Returns a Tensor with the same shape as the input Tensor and data type DT\_BOOL. If the element value at the corresponding position in input is less than or equal to the element value at the corresponding position in other, the return value at that position is True; otherwise it is False.

## Constraints

1.  input and other must have the same type.
2.  One-dimensional broadcasting is supported.

## Example

### TileShape Configuration Example

Note: Before calling this operation interface, set the TileShape via set_vec_tile_shapes.

The TileShape dimensions must match the output dimensions.

Example 1: Non-broadcast scenario, input shape is [m, n], other is [m, n], output is [m, n]. Setting TileShape to [m1, n1] means m1 and n1 are used to partition the m and n axes respectively.

```python
pypto.set_vec_tile_shapes(4, 16)
```

Example 2: Broadcast scenario, input shape is [m, n], other is [m, 1], output is [m, n]. Setting TileShape to [m1, n1] means m1 and n1 are used to partition the m and n axes respectively.

```python
pypto.set_vec_tile_shapes(4, 16)
```

### Interface Call Example

```python
a = pypto.tensor([3], pypto.DT_FP32)
b = pypto.tensor([3], pypto.DT_FP32)
out = pypto.le(a, b)
```

Example output:

```python
input data a: [1.0 2.0 3.0]
input data b: [2.0 2.0 2.0]
output data out: [True, True, False]
```

