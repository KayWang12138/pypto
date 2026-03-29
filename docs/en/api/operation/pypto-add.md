# pypto.add

## Supported Products

| Product          | Supported |
|:-----------------|:---------:|
| Atlas A3 Training Series/Atlas A3 Inference Series |    √     |
| Atlas A2 Training Series/Atlas A2 Inference Series |    √     |

## Description

Adds the values of input and other element-wise. The formula is:

$$
res_i = input_i + other_i
$$

## Function Prototype

```python
add(input: Tensor, other: Union[Tensor, float]) -> Tensor
```

## Parameters


| Parameter | Input/Output | Description                                                                 |
|-----------|--------------|-----------------------------------------------------------------------------|
| input     | input        | Source operand. <br> Supported type: Tensor. <br> Tensor supported data types: DT_FP16, DT_BF16, DT_INT16, DT_INT32, DT_FP32. <br> Empty Tensor not supported; shape supports only 2–4 dimensions, supports multi-dimensional broadcast to the same shape; when data type is DT_FP32 or DT_FP16, supports automatic inline processing for broadcast on the second-to-last axis; shape size must not exceed 2147483647 (i.e., INT32_MAX). |
| other     | input        | Source operand. <br> Supported types: float and Tensor. <br> Tensor supported data types: DT_FP16, DT_BF16, DT_INT16, DT_INT32, DT_FP32. <br> Empty Tensor not supported; shape supports only 2–4 dimensions, supports multi-dimensional broadcast to the same shape; when data type is DT_FP32 or DT_FP16, supports automatic inline processing for broadcast on the second-to-last axis; shape size must not exceed 2147483647 (i.e., INT32_MAX). |

## Return Value

Returns the output Tensor. The data type is the same as input and other, and the shape is the broadcasted size of input and other.

## Constraints

1.  input and other must have the same type.
2.  Implicit type conversion is not supported when other is a scalar.
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
a = pypto.tensor([1, 3], pypto.DT_FP32)
b = pypto.tensor([1, 3], pypto.DT_FP32)
out = pypto.add(a, b)
```

Example result:

```python
Input data a:   [[1.0 2.0 3.0]]
Input data b:   [[2.0 3.0 4.0]]
Output data out: [[3.0 5.0 7.0]]
```

