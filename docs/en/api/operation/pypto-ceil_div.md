# pypto.ceil_div

## Supported Products

| Product          | Supported |
|:-----------------|:---------:|
| Atlas A3 Training Series/Atlas A3 Inference Series |    √     |
| Atlas A2 Training Series/Atlas A2 Inference Series |    √     |

## Description

Divides each element of self by the corresponding element in other and rounds up (ceiling division). The formula is:

$$
res_i = ceil(self_i \div other_i)
$$

## Function Prototype

```python
ceil_div(self: Tensor, other: Tensor) -> Tensor
```

## Parameters


| Parameter | Input/Output | Description                                                                 |
|-----------|--------------|-----------------------------------------------------------------------------|
| self      | input        | Source operand. <br> Supported type: Tensor. <br> Tensor supported data types: DT_INT32. <br> Empty Tensor not supported; shape supports only 2–4 dimensions, and supports broadcast along a single dimension to the same shape; shape size must not exceed 2147483647 (i.e., INT32_MAX). |
| other     | input        | Source operand. <br> Supported type: Tensor. <br> Tensor supported data types: DT_INT32. <br> Empty Tensor not supported; shape supports only 2–4 dimensions, and supports broadcast along a single dimension to the same shape; shape size must not exceed 2147483647 (i.e., INT32_MAX). |

## Return Value

Returns the output Tensor. The data type is the same as input and other, and the shape is the broadcasted size of input and other.

## Constraints

1.  input and other must have the same type.
3.  other does not support special values such as nan or inf.

## Example

### TileShape Configuration Example

Before calling this operation interface, set the TileShape via set_vec_tile_shapes.

The TileShape dimensions should be consistent with the output.

For non-broadcast scenarios, if input shape is [m, n], other is [m, n], and output is [m, n], setting TileShape to [m1, n1] means m1 and n1 are used to tile the m and n axes respectively.

For broadcast scenarios, if input shape is [m, n], other is [m, 1], and output is [m, n], setting TileShape to [m1, n1] means m1 and n1 are used to tile the m and n axes respectively.

```python
pypto.set_vec_tile_shapes(4, 16)
```

### Interface Call Example

```python
a = pypto.tensor([1, 3], pypto.DT_INT32)
b = pypto.tensor([1, 3], pypto.DT_INT32)
out = pypto.ceil_div(a, b)
```

Example result:

```python
Input data a:    [[2 4 6]]
Input data b:    [[4 2 5]]
Output data out: [[1 2 2]]
```
