# pypto.div

## Supported Products

| Product          | Supported |
|:-----------------|:---------:|
| Atlas A3 Training Series/Atlas A3 Inference Series |    √     |
| Atlas A2 Training Series/Atlas A2 Inference Series |    √     |

## Description

Divides each element of `input` by the corresponding element of `other`, element-wise. The formula is:

$$
res_i = input_i \div other_i
$$

## Function Prototype

```python
div(input: Tensor,other: Union[Tensor, float]) -> Tensor
```

## Parameters


| Parameter | Input/Output | Description                                                                 |
|-----------|--------------|-----------------------------------------------------------------------------|
| input     | Input        | Source operand. <br> Supported type: Tensor. <br> Supported tensor data types: DT_FP16, DT_BF16, DT_FP32. <br> Empty tensors are not supported; shape supports 2–4 dimensions only, with multi-dimensional broadcasting to match shapes; when the data type is DT_FP32 or DT_FP16, second-to-last axis broadcasting with automatic inline processing is supported; shape size must not exceed 2147483647 (INT32_MAX). |
| other     | Input        | Source operand. <br> Supported types: float and Tensor. <br> Supported tensor data types: DT_FP16, DT_BF16, DT_FP32. <br> Empty tensors are not supported; shape supports 2–4 dimensions only, with multi-dimensional broadcasting to match shapes; when the data type is DT_FP32 or DT_FP16, second-to-last axis broadcasting with automatic inline processing is supported; shape size must not exceed 2147483647 (INT32_MAX). |

## Return Value

Returns the output tensor. The tensor's data type is the same as `input` and `other`, and the shape is the broadcast shape of `input` and `other`.

## Constraints

1.  `input` and `other` must have the same type.
2.  When `other` is a scalar, implicit type conversion is not supported.
3.  `other` does not support special values such as nan or inf.

## Example

### TileShape Configuration Example

Before calling this operation interface, set the TileShape via `set_vec_tile_shapes`.

The TileShape dimensions must match the output.

For the non-broadcast scenario, if input shape is [m, n] and other is [m, n], output is [m, n], and TileShape is set to [m1, n1], then m1 and n1 are used to split the m and n axes respectively.

For the broadcast scenario, if input shape is [m, n] and other is [m, 1], output is [m, n], and TileShape is set to [m1, n1], then m1 and n1 are used to split the m and n axes respectively.

```python
pypto.set_vec_tile_shapes(4, 16)
```

### Interface Call Example

```python
a = pypto.tensor([1, 3], pypto.DT_FP32)
b = pypto.tensor([1, 3], pypto.DT_FP32)
out = pypto.div(a, b)
```

Example result:

```python
Input a:    [[2.0 4.0 6.0]]
Input b:    [[2.0 2.0 2.0]]
Output out: [[1.0 2.0 3.0]]
```
