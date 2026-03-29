# pypto.gcd

## Supported Products

| Product                                     | Supported |
| :------------------------------------------ | :-------: |
| Atlas A3 Training Series/Atlas A3 Inference Series |    √    |
| Atlas A2 Training Series/Atlas A2 Inference Series |    √    |

## Description

Computes the greatest common divisor (GCD) of corresponding elements of input and other.

## Function Prototype

```python
gcd(input: Tensor, other: Union[Tensor, int]) -> Tensor:
```

## Parameters

| Parameter | Input/Output | Description                                                                                                                                                                                                                                                   |
| --------- | ------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| input     | input        | Source operand. <br> Supported type: Tensor. <br> Supported tensor data types: DT_INT8, DT_INT16, DT_INT32, DT_UINT8. <br> Empty tensors are not supported; Shape supports only 2–4 dimensions and supports broadcasting along a single dimension to the same shape; Shape Size must not exceed 2147483647 (i.e., INT32_MAX).           |
| other     | input        | Source operand. <br> Supported types: int and Tensor. <br> Supported tensor data types: DT_INT8, DT_INT16, DT_INT32, DT_UINT8. <br> Empty tensors are not supported; Shape supports only 2–4 dimensions and supports broadcasting along a single dimension to the same shape; Shape Size must not exceed 2147483647 (i.e., INT32_MAX). |

## Return Value

Returns the output Tensor. The data type is the same as input and other; the Shape is the broadcasted size of input and other.

## Constraints

1. input and other must have the same type.
2. When other is a scalar, implicit type conversion is not supported.
3. other does not support special values such as nan and inf.

## Example

### TileShape Configuration Example

Note: Before calling this operation interface, set the TileShape via set_vec_tile_shapes. The last axis of TileShape must be 32B aligned.

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
x = pypto.tensor([2, 3], pypto.DT_INT32)
y = pypto.tensor([2, 3], pypto.DT_INT32)
z = pypto.gcd(x, y)
# Using a scalar
c = pypto.gcd(x, 2)
```

Example output:

```python
input data x: : [[9 9 9],
             [6 6 6]]
input data y:   [[1 2 3],
             [1 2 3]]
output data z:   [[1 1 3],
             [1 2 3]]
output data c:   [[1 1 1],
             [2 2 2]]
```
