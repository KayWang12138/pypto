# pypto.neg

## Supported Products

| Product          | Supported |
|:-----------------|:---------:|
| Atlas A3 Training Series/Atlas A3 Inference Series |    √     |
| Atlas A2 Training Series/Atlas A2 Inference Series |    √     |

## Description

Computes the negation of each element in the input tensor element-wise, returning a tensor with the same shape as the input.

## Function Prototype

```python
neg(input: Tensor) -> Tensor
```

## Parameters


| Parameter | Input/Output | Description                                                                 |
|-----------|--------------|-----------------------------------------------------------------------------|
| input     | Input        | Source operand. <br> Supported type: Tensor. <br> Supported tensor data types: DT_FP32, DT_FP16, DT_BF16, DT_INT32, DT_INT16. <br> Empty tensors are not supported; shape supports only 2–4 dimensions; shape size must not exceed 2147483647 (INT32_MAX). |

## Return Value

Returns a tensor with the same shape and data type as the input tensor, where each element is the negation of the corresponding element in the input tensor.

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
y = pypto.neg(x)
```

Example result:

```python
Input x: [1.0, 2.0, 3.0, 4.0, 5.0]
Output y: [-1.0, -2.0, -3.0, -4.0, -5.0]
```
