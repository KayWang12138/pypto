# pypto.cumsum

## Supported Products

| Product          | Supported |
|:-----------------|:---------:|
| Atlas A3 Training Series/Atlas A3 Inference Series |    √     |
| Atlas A2 Training Series/Atlas A2 Inference Series |    √     |

## Description

Computes the cumulative sum of the input tensor `input` along the specified dimension.

## Function Prototype

```python
cumsum(input: Tensor, dim: int) -> Tensor:
```

## Parameters

| Parameter | Input/Output | Description                                                                 |
|-----------|--------------|-----------------------------------------------------------------------------|
| input     | Input        | Source operand. <br> Supported type: Tensor. <br> Supported tensor data types: DT_FP16, DT_BF16, DT_INT16, DT_INT32, DT_FP32. <br> Empty tensors are not supported; shape supports 1–4 dimensions only; shape size must not exceed 2147483647 (INT32_MAX). |
| dim       | Input        | Source operand specifying the dimension along which to accumulate. <br> Type: int. |

## Return Value

The output tensor shape matches the input `input`.
When `input` is of type DT_FP16, DT_BF16, or DT_FP32, the output data type matches the input. When `input` is of type DT_INT16 or DT_INT32, the output data type is DT_INT64.

## Constraints

1. `dim`: specifies the dimension along which to compute the cumulative sum; must be within the valid dimension range of input tensor `input`, satisfying -input.dim <= dim < input.dim.

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
input = pypto.tensor([2, 3], pypto.DT_INT32)        # shape (2, 3)
dim = 0
y = pypto.cumsum(input, dim)
```

Example result:

```python
Input x:   [[0 1 2],
             [3 4 5]]
Output y:  [[0 1 2],
             [3 5 7]]                             # shape (2, 3)
```
