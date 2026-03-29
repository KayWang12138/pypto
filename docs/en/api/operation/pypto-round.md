# pypto.round

## Supported Products

| Product          | Supported |
|:-----------------|:--------:|
| Atlas A5 训练系列产品/Atlas A5 推理系列产品 |    √     |
| Atlas A3 Training Series/Atlas A3 Inference Series |    √     |
| Atlas A2 Training Series/Atlas A2 Inference Series |    √     |

## Description

Rounds the elements of the input Tensor to the specified number of decimal places. If the value is equidistant from two numbers at the specified decimal place, it is rounded to the nearest even number at that decimal place.

## Function Prototype

```python
round(input: Tensor, decimals: int) -> Tensor
```

## Parameters


| Parameter | Input/Output | Description                                                                                                                                                             |
|-----------|--------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| input     | input        | Source operand. <br> Supported type: Tensor. <br> Tensor supported data types: DT_FP32, DT_FP16, DT_BF16, DT_INT32, DT_INT16. <br> Empty Tensor not supported; shape supports only 2–4 dimensions; Shape Size must not exceed 2147483647 (i.e., INT32_MAX). |
| decimals  | input        | Source operand, the number of decimal places to round to. <br> int type.                                                                                               |

## Return Value

Returns a Tensor type. Its shape and data type are the same as the input Tensor, and its elements are the results of rounding the corresponding elements of the input Tensor to the specified number of decimal places.

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
x = pypto.tensor([2, 2], pypto.DT_FP32)
y = pypto.round(x, decimals=1)
```

Example result:

```python
input data x: [[1.21, 2.35], [3.65, 4.76]]
output data y: [[1.2, 2.4], [3.6, 4.8]]
```
