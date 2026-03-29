# pypto.triu

## Supported Products

| Product                                             | Supported |
| :-------------------------------------------------- | :-------: |
| Atlas A3 Training Series/Atlas A3 Inference Series  |    √     |
| Atlas A2 Training Series/Atlas A2 Inference Series  |    √     |

## Description

Returns the upper triangular part of a 2D tensor or a batch of tensors. All other elements of the result tensor are set to 0.

## Function Prototype

```python
triu(input: Tensor, diagonal: SymInt = 0) -> Tensor:
```

## Parameters

| Parameter | Input/Output | Description                                                                                                                                                                                                                        |
| --------- | ------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| input     | Input        | Source operand. <br> Supported type: Tensor. <br> Supported tensor data types: DT_FP32, DT_FP16, DT_BF16, DT_INT16, DT_INT32, DT_INT8. <br> Empty tensors are not supported; shape supports only 2–5 dimensions; shape size must not exceed 2147483647 (INT32_MAX). |
| diagonal  | Input        | Source operand specifying the diagonal to consider. Default: 0. <br> Type: SymInt.                                                                                                                                              |

## Return Value

Returns a tensor with the same shape and data type as the input `input`.

## Constraints

See Parameters.

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
x = pypto.tensor([3, 3], pypto.data_type.DT_INT32)        # shape (3, 3)
diagonal = 0
out = pypto.triu(x, diagonal)
```

Example result:

```python
Input  x :[[1 2 3],
             [4 5 6],
             [7 8 9]]
Output out:[[1 2 3],
             [0 5 6],
             [0 0 9]]                             # shape (3, 3)
```
