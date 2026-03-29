# pypto.exp2

## Supported Products

| Product          | Supported |
|:-----------------|:---------:|
| Atlas A3 Training Series/Atlas A3 Inference Series |    √     |
| Atlas A2 Training Series/Atlas A2 Inference Series |    √     |

## Description

Computes 2 raised to the power of each element in the input tensor, element-wise. Returns a tensor with the same shape as the input.

## Function Prototype

```python
exp2(input: Tensor) -> Tensor
```

## Parameters


| Parameter | Input/Output | Description                                                                 |
|-----------|--------------|-----------------------------------------------------------------------------|
| input     | Input        | Source operand. <br> Supported type: Tensor. <br> Supported tensor data types: DT_FP32, DT_FP16, DT_BF16, DT_INT32, DT_INT16. <br> Empty tensors are not supported; shape supports 2–4 dimensions only; shape size must not exceed 2147483647 (INT32_MAX). |

## Return Value

Returns the output tensor. When the input data type is DT_FP32, DT_FP16, or DT_BF16, the output data type and shape match the input. When the input data type is DT_INT32 or DT_INT16, the output data type is DT_FP32 and the shape matches the input.


## Example

### TileShape Configuration Example

Note: Before calling this operation interface, set the TileShape via `set_vec_tile_shapes`.

The TileShape dimensions must match the output.

Example 1: If input shape is [m, n] and output is [m, n], and TileShape is set to [m1, n1], then m1 and n1 are used to split the m and n axes respectively.

```python
pypto.set_vec_tile_shapes(4, 16)
```

### Interface Call Example

```python
x = pypto.tensor([3], pypto.DT_FP32)
y = pypto.exp2(x)
```

Example result:

```python
Input x: [0.0    1.0    2.0]
Output y: [1.0    2.0    4.0]
```
