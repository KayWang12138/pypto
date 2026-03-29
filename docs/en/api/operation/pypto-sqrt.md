# pypto.sqrt

## Supported Products

| Product          | Supported |
|:-----------------|:---------:|
| Atlas A3 Training Series/Atlas A3 Inference Series |    √     |
| Atlas A2 Training Series/Atlas A2 Inference Series |    √     |

## Description

Computes the square root of each element in the input tensor element-wise. Returns NaN when the input is negative.

## Function Prototype

```python
sqrt(input: Tensor) -> Tensor
```

## Parameters


| Parameter | Input/Output | Description                                                                 |
|-----------|--------------|-----------------------------------------------------------------------------|
| input     | Input        | Source operand. <br> Supported type: Tensor. <br> Supported tensor data types: DT_FP16, DT_BF16, DT_FP32. <br> Empty tensors are not supported; shape supports only 2–4 dimensions; shape size must not exceed 2147483647 (INT32_MAX). |

## Return Value

Returns a Tensor with the same shape and data type as the input tensor, where each element is the square root of the corresponding element in the input tensor.

## Example

### TileShape Configuration Example

Note: Before calling this operation interface, set the TileShape via `set_vec_tile_shapes`.

The TileShape dimensions must match the output dimensions.

Example 1: If the input `input` shape is `[m, n]` and the output is `[m, n]`, set TileShape to `[m1, n1]`, where `m1` and `n1` tile the `m` and `n` axes respectively.

```python
pypto.set_vec_tile_shapes(4, 16)
```

### Interface Call Example

```python
x = pypto.tensor([5], pypto.DT_FP32)
y = pypto.sqrt(x)
```

Example result:

```
Input x: [1.0, 4.0, 9.0, 16.0, 25.0]
Output y: [1.0, 2.0, 3.0, 4.0,  5.0]
```
