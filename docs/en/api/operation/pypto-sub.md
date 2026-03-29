# pypto.sub

## Supported Products

| Product          | Supported |
|:-----------------|:---------:|
| Atlas A3 Training Series/Atlas A3 Inference Series |    √     |
| Atlas A2 Training Series/Atlas A2 Inference Series |    √     |

## Description

Subtracts `other` from `input` element-wise. The formula is:

$$
res_i = input_i - other_i
$$

## Function Prototype

```python
sub(input: Tensor, other: Union[Tensor, float]) -> Tensor
```

## Parameters


| Parameter | Input/Output | Description                                                                 |
|-----------|--------------|-----------------------------------------------------------------------------|
| input     | Input        | Source operand. <br> Supported type: Tensor. <br> Supported tensor data types: DT_FP16, DT_BF16, DT_INT16, DT_INT32, DT_FP32. <br> Empty tensors are not supported; shape supports only 2–4 dimensions; when the data type is DT_FP32 or DT_FP16, second-to-last axis broadcasting with automatic inline handling is supported; shape size must not exceed 2147483647 (INT32_MAX). |
| other     | Input        | Source operand. <br> Supported types: float and Tensor. <br> Supported tensor data types: DT_FP16, DT_BF16, DT_INT16, DT_INT32, DT_FP32. <br> Empty tensors are not supported; shape supports only 2–4 dimensions; when the data type is DT_FP32 or DT_FP16, second-to-last axis broadcasting with automatic inline handling is supported; shape size must not exceed 2147483647 (INT32_MAX). |

## Return Value

Returns an output tensor with the same data type as `input` and `other`, and a shape equal to the broadcasted size of `input` and `other`.

## Constraints

1.  The types of `input` and `other` should be the same.
2.  When `other` is a numeric scalar, implicit type conversion is not supported.
3.  Special values such as `nan` and `inf` are not supported for `other`.

## Example

### TileShape Configuration Example

Note: Before calling this operation interface, set the TileShape via `set_vec_tile_shapes`.

The TileShape dimensions must match the output dimensions.

Example 1: Non-broadcast scenario — input `input` shape is `[m, n]`, `other` is `[m, n]`, output is `[m, n]`. Set TileShape to `[m1, n1]`, where `m1` and `n1` tile the `m` and `n` axes respectively.

```python
pypto.set_vec_tile_shapes(4, 16)
```

Example 2: Broadcast scenario — input `input` shape is `[m, n]`, `other` is `[m, 1]`, output is `[m, n]`. Set TileShape to `[m1, n1]`, where `m1` and `n1` tile the `m` and `n` axes respectively.

```python
pypto.set_vec_tile_shapes(4, 16)
```

### Interface Call Example

```python
x = pypto.tensor([2, 3], pypto.DT_FP32)
y = pypto.tensor([2, 3], pypto.DT_FP32)
z = pypto.sub(x, y)
# Using a scalar and alpha
c = pypto.sub(x, 2.0, alpha=3)
```

Example result:

```python
Input x: : [[9.0 9.0 9.0],
              [9.0 9.0 9.0]]
Input y:   [[1.0 2.0 3.0],
              [1.0 2.0 3.0]]
Output z:   [[8.0 7.0 6.0],
              [8.0 7.0 6.0]]
Output c:   [[3.0 3.0 3.0],
              [3.0 3.0 3.0]]
```
