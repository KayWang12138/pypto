# pypto.eq

## Supported Products

| Product          | Supported |
|:-----------------|:---------:|
| Atlas A3 Training Series/Atlas A3 Inference Series |    √     |
| Atlas A2 Training Series/Atlas A2 Inference Series |    √     |

## Description

Element-wise equality comparison operation.

## Function Prototype

```python
eq(input: Tensor, other: Union[Tensor, float, Element]) -> Tensor
```

## Parameters


| Parameter | Input/Output | Description                                                                 |
|-----------|--------------|-----------------------------------------------------------------------------|
| input     | Input        | Source operand. <br> Supported type: Tensor. <br> Supported tensor data types: DT_FP16, DT_BF16, DT_FP32; the data types of the two source operands must be consistent. <br> Empty tensors are not supported; shape supports 2–4 dimensions only; shape size must not exceed 2147483647 (INT32_MAX). |
| other     | Input        | Source operand. <br> Supported types: Tensor, float, Element. <br> When a float is provided, it is automatically converted to an Element type corresponding to DT_FP32. To use other data types, construct an Element explicitly. <br> Supported data types for Tensor and Element: DT_FP16, DT_BF16, DT_FP32; the data types of the two source operands must be consistent. <br> Empty tensors are not supported; shape supports 2–4 dimensions only; shape size must not exceed 2147483647 (INT32_MAX). |

## Return Value

Returns a tensor with the same shape as the input tensor and with data type DT\_BOOL. If the element value at a given position in `input` equals the element value at the corresponding position in `other`, the return value at that position is True; all other positions are False.

## Constraints

1.  `input` and `other` must have consistent types.
2.  One-dimensional broadcasting is supported.

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
a = pypto.tensor([3], pypto.DT_FP32)
b = pypto.tensor([3], pypto.DT_FP32)
out = pypto.eq(a, b)
```

Example result:

```python
Input a: [1.0 2.0 3.0]
Input b: [2.0 2.0 2.0]
Output out: [False, True, False]
```
