# pypto.signbit

## Supported Products

| Product          | Supported |
|:-----------------|:--------:|
| Atlas A3 Training Series/Atlas A3 Inference Series |    √     |
| Atlas A2 Training Series/Atlas A2 Inference Series |    √     |

## Description

Checks whether the sign bit of each element in the input tensor is set (i.e., whether the element is negative). Element-wise operation.

Logic:
- If the element is negative (including −∞ and −0.0), returns True.
- If the element is positive (including +∞ and +0.0) or NaN, returns False.

## Function Prototype

```python
signbit(input: Tensor) -> Tensor
```

## Parameters


| Parameter | Input/Output | Description                                                                 |
|-----------|--------------|-----------------------------------------------------------------------------|
| input     | input        | Source operand. <br> Supported type: Tensor. <br> Tensor supported data types: DT_FP16, DT_BF16, DT_FP32, DT_INT8, DT_INT16, DT_INT32. <br> Empty Tensor not supported; shape supports only 2–4 dimensions; Shape Size must not exceed 2147483647 (i.e., INT32_MAX). |

## Return Value

Returns a Tensor type. Its shape is the same as the input Tensor, its data type is DT_BOOL, and its elements indicate whether the sign bit of the corresponding element in the input Tensor is set (True indicates negative, False indicates non-negative).

## Constraints

1.  TileShape dimensions must match the input dimensions;
2.  Due to temporary memory usage, when the input data type is DT_FP32, the TileShape size has an additional constraint: assuming TileShape is [a,b,c,d], then a*b*c*d*sizeof(self) + a*b*c*d*sizeof(FP16) + a*b*c*d*sizeof(UINT8) < UB.

## Example

### TileShape Configuration Example

Note: Before calling this operation interface, set the TileShape via set_vec_tile_shapes.

The TileShape dimensions should match the output dimensions.

Example 1: Given input shape [m, n] and output [m, n], TileShape is set to [m1, n1], where m1 and n1 are used to tile the m and n axes respectively.

```python
pypto.set_vec_tile_shapes(4, 16)
```

### Interface Call Example

```python
x = pypto.tensor([-5, 0, 5, 10, -2], pypto.DT_FP32)
y = pypto.signbit(x)
```

Example result:

```python
input data x: [-5.0, 0.0, 5.0, 10.0, -2.0]
output data y: [True, False, False, False, True]
```
