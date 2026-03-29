# pypto.ceil

## Supported Products

| Product          | Supported |
|:-----------------|:---------:|
| Atlas A5 Training Series/Atlas A5 Inference Series |    √     |
| Atlas A3 Training Series/Atlas A3 Inference Series |    √     |
| Atlas A2 Training Series/Atlas A2 Inference Series |    √     |

## Description

Computes the ceiling (smallest integer not less than the element) of each element in the input Tensor, element-wise. Integer values are returned as-is; floating-point values are rounded up.

## Function Prototype

```python
ceil(input: Tensor) -> Tensor
```

## Parameters


| Parameter | Input/Output | Description                                                                 |
|-----------|--------------|-----------------------------------------------------------------------------|
| input     | input        | Source operand. <br> Supported type: Tensor. <br> Tensor supported data types: DT_FP32, DT_FP16, DT_BF16, DT_INT32, DT_INT16. Shape supports only 2–4 dimensions. |

## Return Value

Returns a Tensor. Its shape and data type are consistent with the input Tensor, and its elements are the ceiling values of the corresponding elements in the input Tensor.

## Example

### TileShape Configuration Example

Before calling this operation interface, set the TileShape via set_vec_tile_shapes.

The TileShape dimensions should be consistent with the output.

For example, if input shape is [m, n] and output is [m, n], setting TileShape to [m1, n1] means m1 and n1 are used to tile the m and n axes respectively.

```python
pypto.set_vec_tile_shapes(4, 16)
```

### Interface Call Example

```python
x = pypto.tensor([5], pypto.DT_FP32)
y = pypto.ceil(x)
```

Example result:

```python
Input data x: [1.2, 4.7, -1.1, 9.0, 3.9]
Output data y: [2.0, 5.0, -1.0, 9.0, 4.0]
```

