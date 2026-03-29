# pypto.trunc

## Supported Products

| Product          | Supported |
|:-----------------|:---------:|
| Atlas A5 Training Series/Atlas A5 Inference Series |    √     |
| Atlas A3 Training Series/Atlas A3 Inference Series |    √     |
| Atlas A2 Training Series/Atlas A2 Inference Series |    √     |

## Description

Computes the truncated integer value of each element in the input tensor element-wise. Truncation discards the fractional part of a number, retaining only the integer part.

## Function Prototype

```python
trunc(input: Tensor) -> Tensor
```

## Parameters


| Parameter | Input/Output | Description                                                                 |
|-----------|--------------|-----------------------------------------------------------------------------|
| input     | Input        | Source operand. <br> Supported type: Tensor. <br> Supported tensor data types: DT_FP32, DT_FP16, DT_BF16, DT_INT32, DT_INT16. <br> Empty tensors are not supported; shape supports only 2–4 dimensions; shape size must not exceed 2147483647 (INT32_MAX). |

## Return Value

Returns a Tensor with the same shape and data type as the input tensor, where each element is the truncated integer result of the corresponding element in the input tensor.

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
x = pypto.tensor([2, 2], pypto.DT_FP32)
y = pypto.trunc(x)
```

Example result:

```python
Input x: [[3.9  4.1], [-16.8  9.9]]
Output y: [[3.0  4.0], [-16.0  9.0]]
```
