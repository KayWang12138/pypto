# pypto.hypot

## Supported Products

| Product          | Supported |
|:-----------------|:---------:|
| Atlas A3 Training Series/Atlas A3 Inference Series |    √     |
| Atlas A2 Training Series/Atlas A2 Inference Series |    √     |

## Description

Computes the element-wise square root of the sum of squares of input and other (i.e., the hypotenuse of a right triangle). The formula is as follows:

$$
res_i = \sqrt{input_i^2 + other_i^2}
$$

## Function Prototype

```python
hypot(input: Tensor, other: Tensor) -> Tensor
```

## Parameters

| Parameter | Input/Output | Description                                                                 |
|-----------|--------------|-----------------------------------------------------------------------------|
| input     | input        | Source operand. <br> Supported type: Tensor. <br> Supported tensor data types: DT_FP32, DT_FP16, DT_BF16. <br> Empty tensors are not supported; Shape supports only 2–4 dimensions; supports broadcasting along a single dimension to the same shape; Shape Size must not exceed 2147483647 (i.e., INT32_MAX). |
| other     | input        | Source operand. <br> Supported type: Tensor. <br> Supported tensor data types: DT_FP32, DT_FP16, DT_BF16. <br> Empty tensors are not supported; Shape supports only 2–4 dimensions; supports broadcasting along a single dimension to the same shape; Shape Size must not exceed 2147483647 (i.e., INT32_MAX). |

## Return Value

Returns the output Tensor. The data type is the same as input and other; the Shape is the broadcasted size of input and other.

## Constraints

1.  input and other must have the same type.
2.  other does not support special values such as nan and inf.
3.  For BF16 and FP16 types, the internal computation may promote precision to avoid intermediate overflow.

## Example

### TileShape Configuration Example

Note: Before calling this operation interface, set the TileShape via set_vec_tile_shapes.

The TileShape dimensions must match the output dimensions.

Example 1: If input shape is [m, n] and output is [m, n], setting TileShape to [m1, n1] means m1 and n1 are used to partition the m and n axes respectively.

```python
pypto.set_vec_tile_shapes(4, 16)
```

### Interface Call Example

```python
# Example: compute the hypotenuse for two sets of right-angle sides
# First set: (3, 4) -> 5
# Second set: (5, 12) -> 13
a = pypto.tensor([3.0, 5.0], pypto.DT_FP32)
b = pypto.tensor([4.0, 12.0], pypto.DT_FP32)
out = pypto.hypot(a, b)
```

Example output:

```python
input data a:   [3.0, 5.0]
input data b:   [4.0, 12.0]
output data out: [5.0, 13.0]
```
