# pypto.arange

## Supported Products

| Product          | Supported |
|:-----------------|:---------:|
| Atlas A3 Training Series/Atlas A3 Inference Series |    √     |
| Atlas A2 Training Series/Atlas A2 Inference Series |    √     |

## Description

Creates a 1D Tensor of length $\left\lceil \frac{\text{end} - \text{start}}{\text{step}} \right\rceil$, containing an arithmetic sequence in the interval \[start, end\) with step size step.

## Function Prototype

```
arange(start: Union[int, float] = 0, end: Union[int, float], step: Union[int, float] = 1) -> Tensor
```

## Parameters


| Parameter | Input/Output | Description                                                                 |
|-----------|--------------|-----------------------------------------------------------------------------|
| start     | input        | Source operand. <br> Supported data types: DT_FP16, DT_BF16, DT_INT16, DT_INT32, DT_FP32. <br> Default value: 0. |
| end       | input        | Source operand. <br> Supported data types: DT_FP16, DT_BF16, DT_INT16, DT_INT32, DT_FP32. <br> This parameter cannot be omitted. |
| step      | input        | Source operand. <br> Supported data types: DT_FP16, DT_BF16, DT_INT16, DT_INT32, DT_FP32. <br> Default value: 1. |

## Return Value

Returns a 1D output Tensor. If any input value is of floating-point data type, the output Tensor data type is float; otherwise it is int.

## Constraints

1. step cannot be 0; as a floating-point number, abs\(step\)>1e-8;

2. \(end-start\)/step must be greater than 0;

3. If start, end, and step are all integer inputs, none of them can exceed the int32 range.

## Example

### TileShape Configuration Example

Before calling this operation interface, set the TileShape via set_vec_tile_shapes.

The TileShape and the output have the same dimension, which is 1D.

For example, if start is m, end is n, step is p, and the output shape is [q], setting TileShape to [q1] means q1 is used to tile the q axis.

```python
pypto.set_vec_tile_shapes(16)
```

### Interface Call Example

```python
y1 = pypto.arange(1.0, 4.0, 0.5)
y2 = pypto.arange(1.0, 4.0)
y3 = pypto.arange(4)
```

Example result:

```python
Output data y1: [1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
Output data y2: [1.0, 2.0, 3.0]
Output data y3: [0, 1, 2, 3]
```

