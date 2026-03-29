# pypto.pad

## Supported Products

| Product                                             | Supported |
| :-------------------------------------------------- | :-------: |
| Atlas A3 Training Series/Atlas A3 Inference Series  |    √      |
| Atlas A2 Training Series/Atlas A2 Inference Series  |    √      |

## Description

Pads the input tensor.

The padding sizes are described by the `pad` parameter starting from the last dimension of the input tensor, proceeding from back to front. The format of the `pad` parameter is $(pad\_left, pad\_right, pad\_top, pad\_bottom, ...)$. The current implementation only supports right-side (Right) and bottom-side (Bottom) constant (Constant) mode padding on the last two dimensions.

## Function Prototype

```python
pad(input: Tensor, pad: Sequence[int], mode: str = "constant", value: float = 0.0) -> Tensor
```

## Parameters

| Parameter | Input/Output | Description                                                                                                                                                                                                                    |
| --------- | ------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| input     | Input        | The source operand to be padded.<br> Supported type: Tensor.<br> Supported tensor data types: DT_FP32, DT_FP16, DT_BF16.<br> Empty tensors are not supported; shape supports only 1–4 dimensions; shape size must not exceed 2147483647 (INT32_MAX).                                          |
| pad       | Input        | Padding size sequence.<br> Supported types: tuple or list (containing int).<br> The sequence length $m$ must be even, and must satisfy $\frac{m}{2} \leq$ number of dimensions of `input`.<br> Format: `(pad_left, pad_right, pad_top, pad_bottom, ...)`.                            |
| mode      | Input        | Padding mode.<br> Supported type: str.<br> Options: `'constant'`, `'reflect'`, `'replicate'`, or `'circular'`.<br> Default: `'constant'`.<br> **Note**: Currently only `'constant'` mode is supported.                          |
| value     | Input        | The fill value when the padding mode is constant (`'constant'`).<br> Supported type: float.<br> Currently only three fixed values are supported: `-inf`, `inf`, `0.0`. Default: `0.0`.                                                                                                   |

## Return Value

Returns an output tensor with the same data type as `input` and a shape extended according to the `pad` parameter along the corresponding dimensions.

## Constraints

1. The length of `pad` must be 2 or 4.
2. Currently **only right-side (Right) and bottom-side (Bottom) padding is supported in multi-dimensional scenarios, or right-side (Right) padding in 1D scenarios**. That is, the left and top padding amounts in the `pad` sequence must be 0 (e.g., the format must be `(0, pad_right, 0, pad_bottom)` or `(0, pad_right)`).
3. The `mode` parameter currently **only supports `'constant'` (constant padding) mode**; other modes are not yet supported.
4. The `value` parameter currently **only supports `-inf`, `inf`, `0.0`**.
5. If `input` is not a Tensor type, or `pad` is not a sequence of integers, a `TypeError` will be raised.

## Example

### TileShape Configuration Example

Note: Before calling this operation interface, set the TileShape via `set_vec_tile_shapes`.

The TileShape dimensions must match the **output (padded shape)** dimensions.

Example 1: If the input `input` shape is `[m, n]`, and `p` elements are padded to the right on the `n` axis, the output shape is `[m, n+p]`. Set TileShape to `[m1, n1]`, where `m1` and `n1` tile the output `m` and `n+p` axes respectively.

```python
pypto.set_vec_tile_shapes(4, 16)
```

### Interface Call Example

```python
# Example: pad a Tensor with shape [1, 1, 2, 2]
# Pad the last dimension (right side) by 1
# Pad the second-to-last dimension (bottom) by 1
t4d = pypto.tensor([0.0, 1.0, 2.0, 3.0], pypto.DT_FP32)
# Assume the 1D data is reshaped internally to [1, 1, 2, 2]

p1 = (0, 1, 0, 1)  # (pad_left=0, pad_right=1, pad_top=0, pad_bottom=1)
out = pypto.pad(t4d, p1, mode="constant", value=0.0)
```

Example result:

```python
# Input t4d (logical shape [1, 1, 2, 2]):
[[[[0.0, 1.0],
   [2.0, 3.0]]]]

# Output out (logical shape extended to [1, 1, 3, 3]):
[[[[0.0, 1.0, 0.0],
   [2.0, 3.0, 0.0],
   [0.0, 0.0, 0.0]]]]
```
