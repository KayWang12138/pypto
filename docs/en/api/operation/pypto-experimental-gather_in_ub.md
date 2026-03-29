# pypto.experimental.gather\_in\_ub

## Supported Products

| Product | Supported |
|:--------|:---------:|
| Atlas A3 Training Series / Atlas A3 Inference Series | √ |
| Atlas A2 Training Series / Atlas A2 Inference Series | √ |

## Description

This is a custom interface with many constraints. Stability is not guaranteed.

This operator supports sparse attention mechanisms. It loads the KV cache of selected tokens from GM (Global Memory) into the UB (Unified Buffer), with support for Paged Attention.

## Function Prototype

```python
gather_in_ub(param: Tensor, indices: Tensor, block_table: Tensor,
                 block_size: int, axis: int) -> Tensor
```

## Parameters

| Parameter | Input/Output | Description |
|-----------|--------------|-------------|
| param | Input | Source operand. <br> Supported data types: DT_FP32, DT_FP16. <br> Empty Tensors are not supported; only 2D shapes are supported. <br> In practice, represents the KV cache with shape [token_size, hidden_dim]. |
| indices | Input | Source operand. <br> Supported data type: DT_INT32. <br> Empty Tensors are not supported; only 2D shapes are supported. <br> In practice, represents the Top-K output with shape [1, k]. |
| block_table | Input | Source operand. <br> Supported data type: DT_INT32. <br> Empty Tensors are not supported; only 2D shapes are supported. <br> In practice, represents the page table in Paged Attention with shape [1, block_table_size], where block_table_size is the length of the page table. |
| block_size | Input | Source operand. <br> Type: int. <br> Represents the number of tokens that can fit in one block in Paged Attention. |
| axis | Input | Source operand. <br> Type: int. <br> Only axis -2 is supported. |

## Return Value

Returns the output Tensor. The Tensor's data type is the same as `param`, and the shape is \[k, hidden\_dim\], representing the gathered KV cache of the selected tokens.

## Example

### TileShape Configuration Example

Before calling this operation interface, set the TileShape via `set_vec_tile_shapes`.

The TileShape dimensions must match the output tensor and are used to control the size of the output Tile block.

Taking input $param[token\_size, hidden\_dim]$, index $indices[1, k]$, axis $\text{axis}=-2$, and output $output[k, hidden\_dim]$ as an example:

Set TileShape to $[k_1, hidden\_dim_1]$. This configuration applies directly to each dimension of the output, and is also mapped to the input and indices. $k_1$ splits the k dimension of `indices`, and $hidden\_dim_1$ splits the feature dimension $hidden\_dim$ of `param`. The Tile memory usage must satisfy the constraint $b_1 \cdot k_1 \cdot hidden\_dim_1 \cdot \text{sizeof}(\mathbf{output}) < \text{UB\_Size}$.

### Interface Call Example

![](../figures/zh-cn_image_0000002524825989.png)

In the scenario above, `indices` is the Top-K result, `block_table` is the page table for Paged Attention, `param` is the KV cache, and `block_size` is 2. The final result gathers the KV cache for the selected tokens.

Taking token id 4 as an example (highlighted in red in the figure), the actual offset is computed based on `blockSize`:

```
blockIdx = 4 / 2;         // Compute the logical block index: the 2nd logical block
tail = 4 % 2;             // Compute the in-block offset: offset is 0
slcBlockIdx = blockTable[0, blockIdxInBatch];  // Look up the table to get the physical block: the 1st physical block
offsets = slcBlockIdx * blockSize + tail;       // Compute the actual offset: 2
```

The data is then transferred.

```python
param = pypto.tensor([6, 4], pypto.DT_FP32)
indices = pypto.tensor([1, 3], pypto.DT_INT32)
blockTable = pypto.tensor([1, 3], pypto.DT_INT32)
blockSize = 2
axis = -2
result = pypto.experimental.gather_in_ub(param , indices , blockTable, blockSize , axis)
```

Example result:

```python
Input data param:
[
  # token 0
  [  0,  1,  2,  3],
  # token 1
  [ 10, 11, 12, 13],
  # token 2
  [ 20, 21, 22, 23],
  # token 3
  [ 30, 31, 32, 33],
  # token 4
  [ 40, 41, 42, 43],
  # token 5
  [ 50, 51, 52, 53],
]
Input data indices: [0, 4, 3]
Input data blockTable: [0, 2, 1]
Output data out:
[
   [  0,  1,  2,  3],
   [ 20, 21, 22, 23],
   [ 50, 51, 52, 53],
]
```
