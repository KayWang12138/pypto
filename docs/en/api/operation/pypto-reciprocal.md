# pypto.reciprocal

## Supported Products

| Product          | Supported |
|:-----------------|:--------:|
| Atlas A3 Training Series/Atlas A3 Inference Series |    √     |
| Atlas A2 Training Series/Atlas A2 Inference Series |    √     |

## Description

Computes the reciprocal of each element in the input Tensor, element-wise.

## Function Prototype

```python
reciprocal(input: Tensor) -> Tensor
```

## Parameters


| Parameter | Input/Output | Description                                                                 |
|-----------|--------------|-----------------------------------------------------------------------------|
| input     | input        | Source operand. <br> Supported type: Tensor. <br> Tensor supported data types: DT_FP32, DT_BF16, DT_FP16. <br> Empty Tensor not supported; shape supports only 2–4 dimensions; Shape Size must not exceed 2147483647 (i.e., INT32_MAX). |

## Return Value

Returns a Tensor type. Its shape and data type are the same as the input Tensor, and its elements are the reciprocals of the corresponding elements in the input Tensor.

## Example

### TileShape Configuration Example

Note: Before calling this operation interface, set the TileShape via set_vec_tile_shapes.

The TileShape dimensions should match the output dimensions.

Example 1: Given input shape [m, n] and output [m, n], TileShape is set to [m1, n1], where m1 and n1 are used to tile the m and n axes respectively.

```python
pypto.set_vec_tile_shapes(4, 16)
```

### Interface Call Example

```python
x = pypto.tensor([1, 4], pypto.DT_FP32)
y = pypto.reciprocal(x)
```

Example result:

```python
input data x: [1.0, 2.0, 3.0, 4.0, 5.0]
output data y: [1.0, 0.5, 0.33, 0.25, 0.2]
```

