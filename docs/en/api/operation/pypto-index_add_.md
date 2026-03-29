# pypto.index\_add\_

## Supported Products

| Product          | Supported |
|:-----------------|:---------:|
| Atlas A3 Training Series/Atlas A3 Inference Series |    √     |
| Atlas A2 Training Series/Atlas A2 Inference Series |    √     |

## Description

Multiplies each block of source by the scaling factor alpha (default 1) and adds it to the corresponding data block in input, where the index and data block direction are specified by index and dim.

## Function Prototype

```python
index_add_(input: Tensor, dim: int, index: Tensor, source: Tensor, *, alpha: Union[int, float] = 1) -> Tensor
```

## Parameters


| Parameter | Input/Output | Description                                                                 |
|-----------|--------------|-----------------------------------------------------------------------------|
| input     | input        | Source operand. <br> Supported type: Tensor. <br> Supported tensor data types: DT_FP32, DT_FP16, DT_BF16, DT_INT16, DT_INT32. <br> Empty tensors are not supported; Shape supports only 2–4 dimensions; Shape Size must not exceed 2147483647 (i.e., INT32_MAX). |
| dim       | input        | int type; the dimension along which the addition is applied to input; <br> Supports any value not exceeding the number of dimensions of input. See Constraints for details. |
| index     | input        | Source operand; values represent the index along the dim axis of input; <br> Supported type: Tensor. <br> Supported tensor data types: DT_INT32, DT_INT64; <br> Empty tensors are not supported; Shape supports only 1 dimension; indices correspond one-to-one with the dim axis indices of source; shape size equals the shape size of source along the dim axis. |
| source    | input        | Source operand to be added to input; <br> Supported type: Tensor. <br> The data type of Tensor is the same as input. <br> Shape supports 2–4 dimensions; the shape size along the dim axis equals the size of index; other dimension sizes equal the corresponding sizes in input. |
| alpha     | input        | Scalar keyword argument; <br> Represents the scaling factor during accumulation. Default is 1. |

## Return Value

In-place operation returns input.

## Constraints

1. index must be an integer type (DT\_INT32 or DT\_INT64); values must not exceed the shape size of input along the dim dimension; must be 1-dimensional; shape size must equal the shape size of source along the dim axis;

2. dim is of int type; valid range: -input.dim <= dim < input.dim;

3. input and source must have the same data type and number of dimensions;

4. The dim axis viewshape of input.shape and source.shape cannot be partitioned; viewshape\[dim\]\>=max\(input.shape\[dim\], source.shape\[dim\]\) is required; no restrictions on other dimension sizes;

5. The TileShape dimensions match result and are used to partition input and source. TileShape\[dim\] = viewshape\[dim\]; the total size of all input and output TileShapes must not exceed UB memory size.

## Example

### TileShape Configuration Example

Before calling this operation interface, set the TileShape via set_vec_tile_shapes.

The TileShape dimensions must match the output dimensions.

If input is [m, n, p], dim is 1, source is [m, t, p], and index is [t], output is [m, n, p]. Setting TileShape to [m1, t1, p1] means m1 and p1 are used to partition the m and p axes respectively. The n and t axes cannot be partitioned; they must be fully loaded.

```python
pypto.set_vec_tile_shapes(4, 16, 32)
```

### Interface Call Example

```python
x = pypto.tensor([2, 3], pypto.DT_INT32)        # shape (2, 3)
source = pypto.tensor([3, 3], pypto.DT_INT32)   # shape (3, 3)
index = pypto.tensor([3], pypto.DT_INT32)   # shape (3,)
dim = 0
# use alpha
y = pypto.index_add_(x, dim, index, source, alpha=1)
# not use alpha
y = pypto.index_add_(x, dim, index, source)
```

Example output:

```python
input data x:   [[0 0 0],
               [0 0 0]]
      source: [[1 1 1],
               [1 1 1],
               [1 1 1]]
      index:   [0 1 0]
output data y:   [[2 2 2],
               [1 1 1]]               # shape (2, 3)
```

