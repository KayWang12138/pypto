# pypto.clip

## Supported Products

| Product          | Supported |
|:-----------------|:---------:|
| Atlas A3 Training Series/Atlas A3 Inference Series |    √     |
| Atlas A2 Training Series/Atlas A2 Inference Series |    √     |

## Description

Clips the data in the input Tensor to the specified range between a minimum and maximum value. Values below the minimum are replaced by the minimum, values above the maximum are replaced by the maximum, and all other values remain unchanged. This interface is a non-in-place operation that does not modify the input Tensor; instead it returns a new Tensor as output.

## Function Prototype

```python
clip(
    input: Tensor,
    min: Optional[Union[Tensor, Element, float, int]] = None,
    max: Optional[Union[Tensor, Element, float, int]] = None
)-> Tensor
```

## Parameters


| Parameter | Input/Output | Description                                                                 |
|-----------|--------------|-----------------------------------------------------------------------------|
| input     | input        | Source operand. <br> Supported type: Tensor. <br> Tensor supported data types: DT_FP32, DT_FP16, DT_BF16, DT_INT32, DT_INT16. <br> Empty Tensor not supported; data dimensions support only 2–4; element count must not exceed UINT32_MAX. |
| min       | input        | Source operand. <br> Supported types: int\float\Element and Tensor. <br> When int or float, automatically converted to Element type DT_INT_32\DT_FP32. For other data types, construct via Element. <br> Tensor and Element supported data types: DT_FP32, DT_FP16, DT_INT32, DT_INT16. <br> Empty Tensor not supported; data dimensions support only 2–4; element count must not exceed UINT32_MAX. <br> Optional; default value is -INF. <br> NaN, INF, -INF are only defined for floating-point operations, i.e., only take effect when data type is DT_FP16/DT_FP32; when data type is DT_INT16 or DT_INT32, the default value comparison logic is skipped. |
| max       | input        | Source operand. <br> Supported types: int\float\Element and Tensor. <br> When int or float, automatically converted to Element type DT_INT_32\DT_FP32. For other data types, construct via Element. <br> Tensor and Element supported data types: DT_FP32, DT_FP16, DT_INT32, DT_INT16. <br> Empty Tensor not supported; data dimensions support only 2–4; element count must not exceed UINT32_MAX. <br> Optional; default value is INF. <br> NaN, INF, -INF are only defined for floating-point operations, i.e., only take effect when data type is DT_FP16/DT_FP32; when data type is DT_INT16 or DT_INT32, the default value comparison logic is skipped. |


## Return Value

When the input is a scalar, the output is:

$$
Y_{i} = \text{MIN}\left( \text{MAX}\left(X_{i}, \text{min\_value}\right), \text{max\_value} \right)
$$

When the input is a Tensor, the output is:

$$
Y_{i} = MIN\left( MAX(X_{i}, min\_value_{i}), max\_value_{i} \right)
$$

The output Tensor data type is the same as the input.

When either min or max is NAN, the output is NAN.

When min \> max, the output at the corresponding positions is the value of max.

## Constraints

1.  min / max must have the same type: either both Element or both Tensor.
2.  When min / max is a Tensor type, its shape must be broadcastable to the input shape.
3.  min and max can both be omitted simultaneously; the original value is returned.

## Example

### TileShape Configuration Example

Before calling this operation interface, set the TileShape via set_vec_tile_shapes.

The TileShape dimensions should be consistent with the output.

For non-broadcast scenarios, if input shape is [m, n], max and min are [m, n], and output is [m, n], setting TileShape to [m1, n1] means m1 and n1 are used to tile the m and n axes respectively.

For broadcast scenarios, if input shape is [m, n], max and min are [m, 1], and output is [m, n], setting TileShape to [m1, n1] means m1 and n1 are used to tile the m and n axes respectively.

```python
pypto.set_vec_tile_shapes(4, 16)
```

### Interface Call Example

```python
x = pypto.tensor([2,3], pypto.DT_INT32)
min = pypto.tensor([2,3], pypto.DT_INT32)
max = pypto.tensor([2,3], pypto.DT_INT32)
out = pypto.clip(x,min,max)
```

Example result:

```python
Input data self: [[-2 1 2], [3 4 5]]
Input data min: [[-1 0 2], [0 3 5]]
Input data max: [[1 2  1], [4 4 4]]
Output data out: [[-1 1 1], [3 4 4]]
```

Example 2:

```python
x = pypto.tensor([2,3], pypto.DT_INT32)
min = 1
max = 3
out = pypto.clip(x,min,max)
```

Example result:

```python
Input data x: [[0 2 4], [3 4 6]]
Output data out: [[1 2 3], [3 3 3]]
```

