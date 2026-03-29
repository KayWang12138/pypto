
# pypto.ones

## Supported Products

| Product          | Supported |
|:-----------------|:---------:|
| Atlas A3 Training Series/Atlas A3 Inference Series |    √     |
| Atlas A2 Training Series/Atlas A2 Inference Series |    √     |

## Description

Creates a tensor of size `size` filled entirely with `1`. The data type is specified by `dtype`, with a default data type of `DT_FP32`.

## Function Prototype

```python
ones(*size: Union[int, Sequence[int]], dtype: Optional[DataType] = None) -> Tensor
```

## Parameters

| Parameter    | Input/Output | Description                                                                 |
|--------------|--------------|-----------------------------------------------------------------------------|
| *size        | Input        | Source operand used to define the shape of the output tensor.<br> Supports variadic arguments (multiple ints) or a single sequence (e.g., List[int] or Tuple[int]). |
| dtype        | Input        | Source operand, optional, used to define the data type of the output tensor.<br> Supported data types: `DT_FP32`, `DT_INT32`, `DT_INT16`, `DT_FP16`, `DT_BF16`.<br> Default value: `pypto.DT_FP32`. |

## Return Value

Returns an output tensor whose data type is determined by `dtype`, shape is `size`, and all values are `1`.

## Constraints

1. The dimensions of `tileshape` must match the dimensions of the output `result`, used to tile `result`.

## Example

### TileShape Configuration Example

Before calling this operation interface, set the TileShape via `set_vec_tile_shapes`. The TileShape dimensions must match the output dimensions.
For example, if the input `size` is `[m, n]` and the output is `[m, n]`, set TileShape to `[m1, n1]`, where `m1` and `n1` tile the `m` and `n` axes respectively.

```python
pypto.set_vec_tile_shapes(2, 3)
```

### Interface Call Example

```python
# Example 1: Pass size as variadic arguments, use default dtype (DT_FP32)
x1 = pypto.ones(2, 3)

# Example 2: Pass size as a list, explicitly specify dtype (DT_INT32)
x2 = pypto.ones([2, 3], dtype=pypto.DT_INT32)
```

Example result:

```python
x1 output: [[1., 1., 1.],
             [1., 1., 1.]]
x2 output: [[1, 1, 1],
             [1, 1, 1]]
```
