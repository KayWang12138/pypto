# pypto.logical\_not

## Supported Products

| Product          | Supported |
|:-----------------|:---------:|
| Atlas A3 Training Series/Atlas A3 Inference Series |    √     |
| Atlas A2 Training Series/Atlas A2 Inference Series |    √     |

## Description

Converts 0 values in the input Tensor to True, and non-zero values to False.

## Function Prototype

```python
logical_not(input: Tensor) -> Tensor
```

## Parameters


| Parameter | Input/Output | Description                                                                 |
|-----------|--------------|-----------------------------------------------------------------------------|
| input     | input        | Source operand. <br> Supported type: Tensor. <br> Supported tensor data types: DT_FP32, DT_FP16, DT_BF16, DT_BOOL, DT_INT8, DT_UINT8. <br> Empty tensors are not supported; Shape supports only 2–4 dimensions; Shape Size must not exceed 2147483647 (i.e., INT32_MAX). |

## Return Value

Returns the output Tensor. The data type is DT\_BOOL; the Shape is the same as the source operand input.

## Constraints

1.  TileShape dimensions must match those of input;
2.  Due to temporary memory usage, when the input data type is DT\_FP32, there is an additional constraint on TileShape size. Assuming TileShape is \[a,b,c,d\], then a\*b\*c\*d\*sizeof\(self\) + a\*b\*c\*d\*sizeof\(BOOL\) + 20.25KB < UB. For other input data types: a\*b\*c\*d\*sizeof\(self\) + a\*b\*c\*d\*sizeof\(BOOL\) + 12.54KB < UB.

## Example

### TileShape Configuration Example

Note: Before calling this operation interface, set the TileShape via set_vec_tile_shapes.

The TileShape dimensions must match the output dimensions.

Example 1: If input shape is [m, n] and output is [m, n], setting TileShape to [m1, n1] means m1 and n1 are used to partition the m and n axes respectively.

```python
pypto.set_vec_tile_shapes(4, 16)
```

### Interface Call Example

```python
a = pypto.tensor([5], pypto.DT_INT32)
out = pypto.logical_not(a)
```

Example output:

```python
input data x: [0 1 2 3 4]
output data y: [True False False False False]
```

