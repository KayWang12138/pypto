# pypto.index\_put\_

## Supported Products

| Product          | Supported |
|:-----------------|:---------:|
| Atlas A3 Training Series/Atlas A3 Inference Series |    √     |
| Atlas A2 Training Series/Atlas A2 Inference Series |    √     |

## Description

Updates multiple or multiple blocks of data from values into self based on the indices. If the accumulate parameter is True, the values are accumulated with the values already stored at the corresponding positions; if accumulate is False, the values directly overwrite the existing values.

## Function Prototype

```python
index_put_(input: Tensor, indices: tuple, values: Tensor, accumulate: bool = False) -> None
```

## Parameters


|  Parameter   | Input/Output | Description                                                                  |
|--------------|--------------|----------------------------------------------------------------------|
|   input      |    input     | Source operand. <br> Supported type: Tensor. <br> Supported tensor data types: DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, DT_INT32, DT_UINT32, DT_INT64, DT_UINT64, DT_BF16, DT_FP16, DT_FP32. <br> Empty tensors are not supported; Shape supports only 1–4 dimensions; Shape Size must not exceed 2147483647 (i.e., INT32_MAX). |
|  indices     |   input      | A tuple of Tensors, where each Tensor represents the index for one dimension. <br> Supported type: tuple\[Tensor\]; each Tensor is one-dimensional and all have the same size. <br> Supported tensor data types: DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, DT_INT32, DT_UINT32, DT_INT64, DT_UINT64. <br> Empty tensors are not supported; the number of Tensors in the tuple must not exceed the number of dimensions of input. |
|   values     |   input      | The values to be updated into input. <br> Supported type: Tensor. <br> Supported tensor data types: DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, DT_INT32, DT_UINT32, DT_INT64, DT_UINT64, DT_BF16, DT_FP16, DT_FP32. <br> Empty tensors are not supported; number of dimensions must not exceed that of input. |
| accumulate   |   input (optional)    | Accumulation parameter; default is False. <br> Supported type: bool. |

## Return Value

In-place operation on input; no return value.

## Constraints

1. The one-dimensional Tensors in indices have the same size; broadcasting is not supported. The value of the i-th Tensor in indices must be less than the shape size of the (i-1)-th dimension of input. When the selection in indices causes repeated updates to the same position, the result is undefined.

2. values does not support broadcasting; its 0th dimension shape must match the shape of the one-dimensional Tensors in indices. If values has 2 or more dimensions, the last i dimensions (i>0) excluding the 0th dimension must be exactly the same as the last i dimensions of input.

3. The number of dimensions of input, the number of Tensors in indices, and the number of dimensions of values must satisfy: (input.shape.size) + 1 = (indices.size) + (values.shape.size).

4. input and values must have the same data type.

5. The viewshape is one-dimensional, partitioning each one-dimensional Tensor in indices and the 0th dimension of values; other dimensions of values are not partitioned.

6. The TileShape dimensions must not exceed the dimensions of values, and are used to partition each one-dimensional Tensor in indices and values. The total size of TileShapes for indices and values must not exceed UB memory size.

## Example

### TileShape Configuration Example

Before calling this operation interface, set the TileShape via set_vec_tile_shapes.

The TileShape dimensions must not exceed the dimensions of values. If the TileShape dimensions are smaller than those of values, the TileShape will automatically be padded with the remaining dimensions matching the shape of values.

If input is [m, n, p], indices is ([t]), and values is [t, n, p], setting TileShape to [t1, n1, p1] means t1 partitions the t axis, n1 partitions the n axis, p1 partitions the p axis, and the m axis is not partitioned.

If input is [m, n, p], indices is ([t]), and values is [t, n, p], setting TileShape to [t1, n1] means TileShape will be automatically padded to [t1, n1, p], where t1 partitions the t axis, n1 partitions the n axis, and the m and p axes are not partitioned.

If input is [m, n, p], indices is ([t], [t]), and values is [t, p], setting TileShape to [t1, p1] means t1 partitions the t axis, p1 partitions the p axis, and the m and n axes are not partitioned.

```python
pypto.set_vec_tile_shapes(4, 16)
```

### Interface Call Example

```python
x = pypto.tensor([3, 3], pypto.DT_INT32)
indices0 = pypto.tensor([2], pypto.DT_INT32)
indices = (indices0, )
values = pypto.tensor([2, 3], pypto.DT_INT32)
accumulate = True
# accumulate is True
pypto.index_put_(x, indices, values, accumulate)
# accumulate is False(default)
pypto.index_put_(x, indices, values)
```

Example output:

```python
input data x:      [[1 1 1],
                 [1 1 1],
                 [0 0 0]]
      indices:   ([1 2], )
      values:    [[0 1 0],
                  [0 2 0]]
x after in-place update:   [[1 1 1],
                 [1 2 1],
                 [0 2 0]]               # accumulate is True
                 [[1 1 1],
                 [0 1 0],
                 [0 2 0]]               # accumulate is False
```

