# pypto.var

## Supported Products

| Product          | Supported |
|:-----------------|:---------:|
| Atlas A3 Training Series/Atlas A3 Inference Series |    √     |
| Atlas A2 Training Series/Atlas A2 Inference Series |    √     |

## Description

Computes the variance of all data along the `dim` dimension of the input tensor. The formula is:
$$
\sigma^2 = \frac{1}{\max(0, ~N - \delta N)}\sum_{i=0}^{N-1}(x_i-\bar{x})^2
$$

## Function Prototype

```python
var(input: Tensor, dim: Union[int, List[int], Tuple[int]] = None, *, correction: float = 1, keepdim: bool = False)
```

## Parameters


| Parameter  | Input/Output | Description                                                                 |
|------------|--------------|-----------------------------------------------------------------------------|
| input      | Input        | Source operand. <br> Supported type: Tensor. <br> Supported tensor data types: DT_FP32, DT_FP16, DT_BF16. <br> Empty tensors are not supported; shape supports only 2–4 dimensions; shape size must not exceed 2147483647 (INT32_MAX). |
| dim        | Input        | Source operand. <br> Supports any single axis or multiple axes. <br> Default: None, meaning all axes. |
| correction | Input        | Source operand. <br> The difference between the sample size and the degrees of freedom. <br> Default uses Bessel's correction, i.e., correction=1. |
| keepdim    | Input        | Source operand. <br> Controls whether to retain the reduced dimension after the reduction. <br> Default: False. |

## Return Value

Returns a Tensor with the same data type as the input tensor.

When `keepdim` is True, the shape of the corresponding `dim` is reduced to 1, and the shapes of all other axes remain unchanged. When `keepdim` is False, the corresponding `dim` is removed.

## Constraints

1. The `dim` axis of `input.shape` cannot be tiled; the dimensions of `viewshape` must match the dimensions of `input`, requiring viewshape\[dim\] \== input.shape\[dim\], while the shape sizes of all other dimensions are unrestricted.
2. `input` does not support empty tensors.
3. Duplicate values are not supported in `dim`, and len(dim) <= input.dim.

## Example

### TileShape Configuration Example

Note: Before calling this operation interface, set the TileShape via `set_vec_tile_shapes`.

The TileShape dimensions must match the input `input` dimensions.

Example 1: If the input `input` shape is `[m, n]` and the output is `[m, 1]`, set TileShape to `[m1, n1]`, where `m1` and `n1` tile the `m` and `n` axes respectively.

```python
pypto.set_vec_tile_shapes(4, 16)
```

### Interface Call Example

```python
x = pypto.tensor([2, 3], pypto.DT_FP32)
y = pypto.var(x, 1, correction=1, keepdim=True)
```

Example result:

```
Input x: [[1., 2., 3.],
            [4., 5., 6.]]
Output y: [[1.],
            [1.]]
```
