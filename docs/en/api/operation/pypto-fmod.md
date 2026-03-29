# pypto.fmod

## Supported Products

| Product          | Supported |
|:-----------------|:---------:|
| Atlas A3 Training Series/Atlas A3 Inference Series |    √     |
| Atlas A2 Training Series/Atlas A2 Inference Series |    √     |

## Description

Computes the element-wise remainder of each element in input divided by the corresponding element in other. The formula is as follows:

$$
res_i = input_i \;\%\; other_i
$$

## Function Prototype

```python
fmod(input: Tensor, other: Union[Tensor, float]) -> Tensor
```

## Parameters


| Parameter | Input/Output | Description                                                                 |
|-----------|--------------|-----------------------------------------------------------------------------|
| input     | input        | Source operand. <br> Supported type: Tensor. <br> Supported tensor data types: DT_FP32, DT_FP16, DT_BF16. <br> Empty tensors are not supported; Shape supports only 2–4 dimensions and supports broadcasting along a single dimension to the same shape; Shape Size must not exceed 2147483647 (i.e., INT32_MAX). |
| other     | input        | Source operand. <br> Supported types: float and Tensor. <br> Supported tensor data types: DT_FP32, DT_FP16, DT_BF16. <br> Empty tensors are not supported; Shape supports only 2–4 dimensions and supports broadcasting along a single dimension to the same shape; Shape Size must not exceed 2147483647 (i.e., INT32_MAX). |

## Return Value

Returns the output Tensor. The data type is the same as input and other; the Shape is the broadcasted size of input and other.

## Constraints

1.  input and other must have the same type.
2.  When other is a scalar, implicit type conversion is not supported.
3.  other does not support special values such as nan and inf.

## Example

### TileShape Configuration Example

Before calling this operation interface, set the TileShape via set_vec_tile_shapes.

The TileShape dimensions must match the output dimensions.

For non-broadcast scenarios, if input shape is [m, n], other is [m, n], and output is [m, n], setting TileShape to [m1, n1] means m1 and n1 are used to partition the m and n axes respectively.

For broadcast scenarios, if input shape is [m, n], other is [m, 1], and output is [m, n], setting TileShape to [m1, n1] means m1 and n1 are used to partition the m and n axes respectively.

```python
pypto.set_vec_tile_shapes(4, 16)
```

### Interface Call Example

```python
a = pypto.tensor([1, 3], pypto.DT_FP32)
b = pypto.tensor([1, 3], pypto.DT_FP32)
out = pypto.fmod(a, b)
```

Example output:

```python
input data a:    [[7.0 8.0 9.0]]
input data b:    [[3.0 3.0 3.0]]
output data out:  [[1.0 2.0 0.0]]
```

