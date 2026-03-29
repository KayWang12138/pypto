# pypto.fillpad

## Supported Products

| Product                                     | Supported |
| :------------------------------------------ | :-------: |
| Atlas A3 Training Series/Atlas A3 Inference Series |    √     |
| Atlas A2 Training Series/Atlas A2 Inference Series |    √     |

## Description

Performs padding on an input Tensor.

Unlike pad, this interface does not change the shape of the tensor. It fills the padding region (i.e., the region beyond the validshape) with the specified value. The current implementation supports only 2-dimensional input tensors and performs constant (Constant) mode right-side (Right) and bottom (Bottom) padding.

## Function Prototype

```python
fillpad(input: Tensor, mode: str = "constant", value: float = 0.0) -> Tensor
```

## Parameters

| Parameter | Input/Output | Description                                                                                                                                                                                                                           |
| --------- | ------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| input     | input        | Source operand to be padded.<br> Supported type: Tensor.<br> Supported tensor data types: DT_FP32, DT_FP16, DT_BF16.<br> Empty tensors are not supported; Shape supports only 2 dimensions; Shape Size must not exceed 2147483647 (i.e., INT32_MAX).                                                    |
| mode      | input        | Padding mode.<br> Supported type: str.<br> Valid values: `'constant'`, `'reflect'`, `'replicate'`, or `'circular'`.<br> Default: `'constant'`.<br> **Note**: Currently only `'constant'` mode is supported.                                             |
| value     | input        | The fill value when padding mode is constant (`'constant'`).<br> Supported type: float.<br> Currently only 3 fixed values are supported: `-inf`, `inf`, `0.0`. Default: `0.0`.                                                                                                                                 |

## Return Value

Returns the output Tensor. The data type is the same as `input`, and the Shape is the size expanded on the corresponding dimensions according to the `pad` parameter.

## Constraints

1. mode currently **supports only `'constant'` (constant padding) mode**; other modes are not supported.
2. value currently **supports only `-inf`, `inf`, `0.0`**.
3. If `input` is not a Tensor type, or `pad` is not an integer sequence, a `TypeError` will be raised.

## Example

### TileShape Configuration Example

Note: Before calling this operation interface, set the TileShape via `set_vec_tile_shapes`.

The TileShape dimensions must match the **output** dimensions.

Example 1: If the input `input` shape is `[m, n]`, the output shape is `[m, n]`. Setting TileShape to `[m1, n1]` means `m1` and `n1` are used to partition the output's `m` and `n` axes respectively.

```python
pypto.set_vec_tile_shapes(4, 16)
```

### Interface Call Example

```python
a = pypto.tensor([4, 4], pypto.DT_FP32)
out = pypto.fillpad(a, "constant", "-inf")
```

Example output:

```python
# input data t4d (logical shape [4, 4]):
[[1.0, 2.0, 0.0, 0.0],
[3.0, 4.0, 0.0, 0.0],
[0.0, 0.0, 0.0, 0.0],
[0.0, 0.0, 0.0, 0.0]]

# output data out (logical shape [4, 4]):
[[1.0, 2.0, -inf, -inf],
[3.0, 4.0, -inf, -inf],
[-inf, -inf, -inf, -inf],
[-inf, -inf, -inf, -inf]]
