# pypto.index\_select

## Supported Products

| Product          | Supported |
|:-----------------|:---------:|
| Atlas A3 Training Series/Atlas A3 Inference Series |    √     |
| Atlas A2 Training Series/Atlas A2 Inference Series |    √     |

## Description

Returns a new tensor that indexes the input tensor along dimension `dim` using the elements in the index tensor `index`.

The returned tensor has the same number of dimensions as the original tensor (input). The size of dimension `dim` equals the length of `index`; the sizes of other dimensions are the same as the original tensor.

$$
\begin{array}{l}
\text{shape}(\mathbf{input}) = (S_0, S_1, \ldots, S_{n-1}) \\
dim = d \\
\text{shape}(\mathbf{index}) = (I_0,) \\
\text{shape}(\mathbf{result}) = (S_0, \ldots, S_{d-1}, I_0, S_{d+1}, \ldots, S_{n-1}) \\
\mathbf{result}[s_0, \ldots, s_{d-1}, i, s_{d+1}, \ldots, s_{n-1}] = \mathbf{input}[s_0, \ldots, s_{d-1}, \mathbf{index}[i], s_{d+1}, \ldots, s_{n-1}]
\end{array}
$$
## Function Prototype

```python
index_select(input: Tensor, dim: int, index: Tensor) -> Tensor:
```

## Parameters


| Parameter | Input/Output | Description                                                                 |
|-----------|--------------|-----------------------------------------------------------------------------|
| input     | input        | Source operand. <br> Supported type: Tensor. <br> Supported tensor data types: DT_FP32, DT_FP16, DT_BF16, DT_INT16, DT_INT8, DT_INT32. <br> Empty tensors are not supported; Shape supports only 2–4 dimensions; Shape Size must not exceed 2147483647 (i.e., INT32_MAX). |
| dim       | input        | int type; the dimension to index; <br> Supports any value not exceeding the number of dimensions of input. See Constraints for details. |
| index     | input        | Source operand; <br> Supported type: Tensor. <br> Supported tensor data types: DT_INT32, DT_INT64; <br> Empty tensors are not supported; Shape supports only 1–2 dimensions; Shape Size must not exceed 2147483647 (i.e., INT32_MAX); values must be valid indices, i.e., must not exceed the shape size of input along the dim axis.

## Return Value

Returns the output Tensor. The data type of the output Tensor matches the data type of input; the shape of the output Tensor is determined jointly by input, dim, and index. See Description for details.

## Constraints

1. index must be an integer type (DT\_INT32 or DT\_INT64); values must be valid indices, i.e., must not exceed input.shape[dim];

2. dim is of int type; valid range: -input.dim <= dim < input.dim. Negative values are supported and interpreted as dim + input.dim;

3. The dim axis viewshape of input.shape cannot be partitioned; viewshape\[dim\]\>=input.shape\[dim\] is required; no restrictions on other dimension sizes;

4. The TileShape dimensions match result and are used to partition result. TileShape settings must ensure result does not exceed UB size. See [TileShape Configuration Example]() for details.

## Example

### TileShape Configuration Example

Before calling this operation interface, set the TileShape via set_vec_tile_shapes.

The TileShape dimensions must be consistent with the output tensor and are used to control the size of the output Tile block.

For example, with input$ input[B,S,D]$, index $index[T]$, axis $	ext{axis}=-2$, and output $output[B,T,D]$: set TileShape to $[b_1, t_1, d_1]$. This configuration acts directly on the output dimensions and maps to input and index. $b_1$ partitions the batch dimension B of input, $d_1$ partitions the feature dimension D of input, while the sequence dimension S of input (i.e., axis −2) is not partitioned and serves only as the index source; $t_1$ acts on the length dimension T of index. Tile memory usage must satisfy the constraint $b_1 \cdot t_1 \cdot d_1 \cdot \text{sizeof}(\mathbf{output}) < \text{UBSize}$.

### Interface Call Example

```python
x = pypto.tensor([3, 4], pypto.DT_FP32)
indices = pypto.tensor([2,], pypto.DT_INT32)
out1 = pypto.index_select(x, 0, indices)
out2 = pypto.index_select(x, 1, indices)
```

Example output:

```python
input x:        [[ 0.1427,  0.0231, -0.5414, -1.0009],
                [-0.4664,  0.2647, -0.1228, -1.1068],
                [-1.1734, -0.6571,  0.7230, -0.6004]]
input index:    [0, 2]
output out1 :    [[ 0.1427,  0.0231, -0.5414, -1.0009],
                [-1.1734, -0.6571,  0.7230, -0.6004]]
output out2 :    [[ 0.1427, -0.5414],
                [-0.4664, -0.1228],
                [-1.1734,  0.7230]]
```
