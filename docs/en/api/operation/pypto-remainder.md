# pypto.remainder

## Supported Products

| Product          | Supported |
|:-----------------|:--------:|
| Atlas A3 Training Series/Atlas A3 Inference Series |    √     |
| Atlas A2 Training Series/Atlas A2 Inference Series |    √     |

## Description

Computes the remainder of dividing each element of input by the corresponding element of other. The formula is as follows:

$$
res_i = input_i - other_i * floor(input_i / other_i)
$$

## Function Prototype

```python
remainder(input: Union[Tensor, int, float], other: Union[Tensor, int, float]) -> Tensor
```

## Parameters


| Parameter | Input/Output | Description                                                                 |
|-----------|--------------|-----------------------------------------------------------------------------|
| input     | input        | Source operand. <br> Supported types: Tensor, int, float. <br> Tensor supported data types: DT_FP32, DT_FP16, DT_BF16, DT_INT32, DT_INT16. <br> Empty Tensor not supported; shape supports 1–5 dimensions, and supports broadcasting along a single dimension to the same shape; Shape Size must not exceed 2147483647 (i.e., INT32_MAX). |
| other     | input        | Source operand. <br> Supported types: Tensor, int, float. <br> Tensor supported data types: DT_FP32, DT_FP16, DT_BF16, DT_INT32, DT_INT16. <br> Empty Tensor not supported; shape supports 1–5 dimensions, and supports broadcasting along a single dimension to the same shape; Shape Size must not exceed 2147483647 (i.e., INT32_MAX). |

## Return Value

Returns the output Tensor. Its shape is the broadcasted shape of input and other, and its data type is the same as input and other.

## Constraints

1. input and other must have the same type;
2. other does not support special values such as 0;
3. For int32, precision is not guaranteed when the data range exceeds \[-2^24, 2^24\].

## Example

### TileShape Configuration Example

Note: Before calling this operation interface, set the TileShape via set_vec_tile_shapes.

The TileShape dimensions should match the output dimensions.

For non-broadcast scenarios: given input shape [m, n], other [m, n], and output [m, n], TileShape is set to [m1, n1], where m1 and n1 are used to tile the m and n axes respectively.

For broadcast scenarios: given input shape [m, n], other [m, 1], and output [m, n], TileShape is set to [m1, n1], where m1 and n1 are used to tile the m and n axes respectively.

```python
pypto.set_vec_tile_shapes(4, 16)
```

### Interface Call Example

```python
a = pypto.tensor([7.0, 8.0, 9.0], pypto.DT_FP32)
b = pypto.tensor([-3.0, -3.0, -3.0], pypto.DT_FP32)
out = pypto.remainder(a, b)
```

Example result:

```python
input data a:    [7.0, 8.0, 9.0]
input data b:    [-3.0, -3.0, -3.0]
output data out: [-2.0, -1.0, 0.0]
```
