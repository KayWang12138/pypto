# pypto.exp

## Supported Products

| Product          | Supported |
|:-----------------|:---------:|
| Atlas A3 Training Series/Atlas A3 Inference Series |    √     |
| Atlas A2 Training Series/Atlas A2 Inference Series |    √     |

## Description

Computes e raised to the power of each element in the input tensor, element-wise. Returns a tensor with the same shape as the input.

## Function Prototype

```python
exp(input: Tensor) -> Tensor
```

## Parameters


| Parameter | Input/Output | Description                                                                 |
|-----------|--------------|-----------------------------------------------------------------------------|
| input     | Input        | Source operand. <br> Supported type: Tensor. <br> Supported tensor data types: DT_FP16, DT_BF16, DT_FP32. <br> Empty tensors are not supported; shape supports 2–4 dimensions only; shape size must not exceed 2147483647 (INT32_MAX). |

## Return Value

Returns the output tensor. The tensor's data type and shape are the same as `input`.

## Example

### TileShape Configuration Example

Before calling this operation interface, set the TileShape via `set_vec_tile_shapes`.

The TileShape dimensions must match the output.

For example, if input shape is [m, n] and output is [m, n], and TileShape is set to [m1, n1], then m1 and n1 are used to split the m and n axes respectively.

```python
pypto.set_vec_tile_shapes(4, 16)
```

### Interface Call Example

```python
x = pypto.tensor([3], pypto.DT_FP32)
y = pypto.exp(x)
```

Example result:

```python
Input x: [0.0    1.0    2.0]
Output y: [1.0000  2.7183  7.3891]
```
