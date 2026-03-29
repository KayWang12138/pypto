# pypto.floor

## Supported Products

| Product          | Supported |
|:-----------------|:---------:|
| Atlas A5 Training Series/Atlas A5 Inference Series |    √     |
| Atlas A3 Training Series/Atlas A3 Inference Series |    √     |
| Atlas A2 Training Series/Atlas A2 Inference Series |    √     |

## Description

Computes the floor (round down) of each element in the input Tensor, element-wise. Integer values are returned as-is; floating-point values are rounded down.

## Function Prototype

```python
floor(input: Tensor) -> Tensor
```

## Parameters


| Parameter | Input/Output | Description                                                                 |
|-----------|--------------|-----------------------------------------------------------------------------|
| input     | input        | Source operand. <br> Supported type: Tensor. <br> Supported tensor data types: DT_FP32, DT_FP16, DT_BF16, DT_INT32, DT_INT16. <br> Empty tensors are not supported; Shape supports only 2–4 dimensions; Shape Size must not exceed 2147483647 (i.e., INT32_MAX). |

## Return Value

Returns a Tensor. Its shape and data type are consistent with the input Tensor; each element is the floor value of the corresponding input element.

## Example

### TileShape Configuration Example

Before calling this operation interface, set the TileShape via set_vec_tile_shapes.

The TileShape dimensions must match the output dimensions.

If the input shape is [m, n] and the output is [m, n], setting TileShape to [m1, n1] means m1 and n1 are used to partition the m and n axes respectively.

```python
pypto.set_vec_tile_shapes(4, 16)
```

### Interface Call Example

```python
x = pypto.tensor([5], pypto.DT_FP32)
y = pypto.floor(x)
```

Example output:

```
input data x: [1.2, 4.3, 9.8, 16.5, 25.4]
output data y: [1.0, 4.0, 9.0, 16.0, 25.0]
```

