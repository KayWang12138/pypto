# pypto.logical\_and

## Supported Products

| Product          | Supported |
|:-----------------|:---------:|
| Atlas A3 Training Series/Atlas A3 Inference Series |    √     |
| Atlas A2 Training Series/Atlas A2 Inference Series |    √     |

## Description

Performs element-wise logical AND operation on two input Tensors. Operation rules:

-   If the input Tensors are of type bool, then True and True -\> True; all other cases yield False.
-   If the input Tensor contains numeric values, they are automatically converted to True/False: 0 becomes False, non-zero becomes True.

## Function Prototype

```python
logical_and(input: Tensor, other: Tensor) -> Tensor
```

## Parameters


| Parameter | Input/Output | Description                                                                 |
|-----------|--------------|-----------------------------------------------------------------------------|
| input     | input        | Source operand. <br> Supported type: Tensor. <br> Supported tensor data types: DT_FP32, DT_FP16, DT_BF16, DT_INT8, DT_UINT8, DT_BOOL, DT_INT16, DT_INT32. <br> Empty tensors are not supported; Shape supports only 2–4 dimensions; input Tensors may have different data types; broadcasting is supported; Shape Size must not exceed 2147483647 (i.e., INT32_MAX). |
| other     | input        | Source operand. <br> Supported type: Tensor. <br> Supported tensor data types: DT_FP32, DT_FP16, DT_BF16, DT_INT8, DT_UINT8, DT_BOOL, DT_INT16, DT_INT32. <br> Empty tensors are not supported; Shape supports only 2–4 dimensions; input Tensors may have different data types; broadcasting is supported; Shape Size must not exceed 2147483647 (i.e., INT32_MAX). |

## Return Value

Returns the output Tensor. The data type is DT\_BOOL; the shape is the broadcasted shape.

## Constraints

1.  TileShape dimensions must match those of input and other;
2.  Due to temporary memory usage, there is an additional constraint on TileShape size. Assuming TileShape is \[a,b,c,d\], then a\*b\*c\*d\*sizeof\(self\) + a\*b\*c\*d\*sizeof\(other\) + a\*b\*c\*d\*sizeof\(BOOL\) + 1.1875KB < UB.

## Example

### TileShape Configuration Example

Note: Before calling this operation interface, set the TileShape via set_vec_tile_shapes.

The TileShape dimensions must match the output dimensions.

Example 1: Non-broadcast scenario, input shape is [m, n], other is [m, n], output is [m, n]. Setting TileShape to [m1, n1] means m1 and n1 are used to partition the m and n axes respectively.

```python
pypto.set_vec_tile_shapes(4, 16)
```

Example 2: Broadcast scenario, input shape is [m, n], other is [m, 1], output is [m, n]. Setting TileShape to [m1, n1] means m1 and n1 are used to partition the m and n axes respectively.

```python
pypto.set_vec_tile_shapes(4, 16)
```

### Interface Call Example

```python
x = pypto.tensor([2], pypto.DT_BOOL)
y1 = pypto.tensor([2], pypto.DT_BOOL)
z1 = pypto.logical_and(x, y1)
# Broadcasting is supported
y2 = pypto.tensor([2,2], pypto.DT_BOOL)
z2 = pypto.logical_and(x, y2)
```

Example output:

```python
input data x:  [True, False]
input data y1: [True, True]
input data y2: [[True, False], [False, True]]
output data z1: [True, False]
output data z2: [[True, False], [False, False]]
```

