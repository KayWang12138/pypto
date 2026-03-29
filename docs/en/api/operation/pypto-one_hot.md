# pypto.one\_hot

## Supported Products

| Product          | Supported |
|:-----------------|:---------:|
| Atlas A3 Training Series/Atlas A3 Inference Series |    √     |
| Atlas A2 Training Series/Atlas A2 Inference Series |    √     |

## Description

Converts an integer tensor to the corresponding one-hot encoding, where each integer is converted to a vector with a 1 at the corresponding position and 0 elsewhere.

## Function Prototype

```python
one_hot(input: Tensor, num_classes: int) -> Tensor
```

## Parameters


| Parameter   | Input/Output | Description                                                                 |
|-------------|--------------|-----------------------------------------------------------------------------|
| input       | Input        | Source operand. <br> Supported type: Tensor. <br> Supported tensor data types: DT_INT8, DT_INT16, DT_INT32, DT_INT64. <br> Supports 1–3 dimensions. <br> Internal elements must be non-negative. <br> Empty tensors are not supported; shape size must not exceed 2147483647 (INT32_MAX). |
| num_classes | Input        | Length of the one-hot encoding. <br> Must be greater than the maximum element in `input`. |

## Return Value

Returns a tensor with shape \(input, num\_classes\) and data type DT\_INT64.

## Constraints

TileShape tiles the output; the TileShape dimensions must match the output dimensions. The last axis of TileShape must equal `num\_classes`.

## Example

### TileShape Configuration Example

Note: Before calling this operation interface, set the TileShape via `set_vec_tile_shapes`.

The TileShape dimensions must match the output dimensions.

Example 1: If the input `input` shape is `[m, n]` and the output is `[m, n, t]` where `t=num_classes`, set TileShape to `[m1, n1, t1]`. `m1` and `n1` tile the `m` and `n` axes respectively. `t1` must equal `num_classes`; the `t` axis cannot be tiled and must be fully loaded.

```python
pypto.set_vec_tile_shapes(4, 16, 32)
```

### Interface Call Example

```python
x = pypto.tensor([3], pypto.DT_INT32)
y = pypto.one_hot(x, 5)
```

Example result:

```python
Input x: [0, 2, 4]
Output y: [[1, 0, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 0, 1]]
```
