# pypto.argsort

## Supported Products

| Product          | Supported |
|:-----------------|:---------:|
| Atlas A3 Training Series/Atlas A3 Inference Series |    √     |
| Atlas A2 Training Series/Atlas A2 Inference Series |    √     |

## Description

Returns the indices that would sort the input along a specified axis in ascending or descending order.

## Function Prototype

```python
argsort(input: Tensor, dim: Optional[int]=None, descending: bool=True) -> Tensor
```

## Parameters


| Parameter   | Input/Output | Description                                                                 |
|-------------|--------------|-----------------------------------------------------------------------------|
| input       | input        | Source operand. <br> Supported type: Tensor. <br> Tensor supported data types: DT_FP32, DT_FP16. <br> Empty Tensor not supported; shape supports only 1–4 dimensions; shape size must not exceed 2147483647 (i.e., INT32_MAX). |
| dim         | input        | Specifies the dimension along which to sort. <br> Supports 1–4 axes. |
| descending  | input        | If True, returns indices in descending order. If False, returns indices in ascending order. |

## Return Value

Returns a Tensor containing the indices that sort the input along the dim axis according to the descending parameter.

## Constraints

1. ViewShape tiling along the dim axis is not currently supported; i.e., ViewShape[dim] = InputShape[dim] is required.
2. Currently only supports TileShape along the dim axis as a multiple of 32, i.e., TileShape[dim] % 32 = 0.
3. For large shape scenarios `(tileShape Size/tileShape[dim] * ((viewShape[dim] + 31) / 32 * 32) >= 6KB)`, the number of tiles along the sort axis must be less than 128.
4. For 4D inputs, sorting along the 0th axis is not currently supported.
5. When equal values are encountered during sorting, stable sort is used to return the corresponding indices.

## Example

### TileShape Configuration Example

Note: Before calling this operation interface, set the TileShape via set_vec_tile_shapes.

The TileShape dimensions should be consistent with the input.

For example, if input shape is [m, n, p], dim is 2, descending is True, and output is [m, n, p], setting TileShape to [m1, n1, p1] means m1, n1, and p1 are used to tile the m, n, and p axes respectively.

```python
pypto.set_vec_tile_shapes(4, 16, 32)
```

### Interface Call Example

```python
x = pypto.tensor([2, 3], pypto.DT_FP32)
y = pypto.argsort(x, -1, True)
```

Example result:

```python
Input data x: [[1.0 2.0 3.0],
               [1.0 2.0 3.0]]
Output data y: [[2, 1, 0],
                [2, 1, 0]]
```

