# pypto.floor_div

## Supported Products

| Product          | Supported |
|:-----------------|:---------:|
| Atlas A5 Training Series/Atlas A5 Inference Series |    √     |
| Atlas A3 Training Series/Atlas A3 Inference Series |    √     |
| Atlas A2 Training Series/Atlas A2 Inference Series |    √     |

## Description

Divides each element of self by the corresponding element in other and rounds down. The formula is as follows:

$$
res_i = floor(\frac{input_{i}}{other_{i}})
$$

## Function Prototype

```python
def floor_div(input: Tensor, other: Union[Tensor, int]) -> Tensor
```

## Parameters


| Parameter | Input/Output | Description                                                                 |
|-----------|--------------|-----------------------------------------------------------------------------|
| input     | input        | Source operand. <br> Supported type: Tensor. <br> Supported tensor data type: DT_INT32. <br> Empty tensors are not supported; Shape supports only 2–4 dimensions and supports broadcasting along a single dimension to the same shape; Shape Size must not exceed 2147483647 (i.e., INT32_MAX). |
| other     | input        | Source operand. <br> Supported type: Tensor. <br> Supported tensor data type: DT_INT32. <br> Empty tensors are not supported; Shape supports only 2–4 dimensions and supports broadcasting along a single dimension to the same shape; Shape Size must not exceed 2147483647 (i.e., INT32_MAX). |

## Return Value

Returns the output Tensor. The data type is the same as input and other; the Shape is the broadcasted size of input and other.

## Constraints

1. input and other must have the same data type.
2. other must not contain $0$ values.
3. Only single-axis broadcasting is supported.

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
a = pypto.tensor([1, 3], pypto.DT_INT32)
b = pypto.tensor([1, 3], pypto.DT_INT32)
out = pypto.floor_div(a, b)
```

Example output:

```python
input data a:    [[2 4 6]]
input data b:    [[4 2 5]]
output data out:  [[0 2 1]]
```
