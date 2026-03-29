# pypto.scatter\_update

## Supported Products

| Product          | Supported |
|:-----------------|:--------:|
| Atlas A3 Training Series/Atlas A3 Inference Series |    √     |
| Atlas A2 Training Series/Atlas A2 Inference Series |    √     |

## Description

Feature 1: Inplace operation. Updates a 4D input with a 4D src according to a 2D index. The formula is as follows:

$$
input\left[\frac{\text{index}[i][j]}{\text{blockSize}}\right]\left[\text{index}[i][j] \% \text{blockSize}\right][0][\dots] = src[i][j][0][\dots]
$$

Feature 2: Inplace operation. Updates a 2D input with a 2D src according to a 2D index. The formula is as follows (where s is the size of the second dimension of index, i.e., index.shape[1]):

$$
input[[\text{index}[i][j]][\dots]] = src[i*s + j][\dots]
$$

## Function Prototype

```python
scatter_update(input: Tensor, dim: int, index: Tensor, src: Tensor) -> Tensor
```

## Parameters


| Parameter | Input/Output | Description                                                                 |
|-----------|--------------|-----------------------------------------------------------------------------|
| input     | input        | Supported type: Tensor. <br> Tensor supported data types: DT_FP32, DT_FP16, DT_BF16, DT_INT32, DT_INT16. <br> Supported dimensions: 2D, 4D. <br> 2D shape: [blockNum * blockSize, d]; 4D shape: [blockNum, blockSize, 1, d]. <br> Empty Tensor not supported; Shape Size must not exceed 2147483647 (i.e., INT32_MAX). |
| dim       | input        | Please keep the default value -2. |
| index     | input        | A set of indices into input. <br> Supported type: Tensor. <br> Tensor supported data types: DT_INT64, DT_INT32, DT_INT16. <br> Supported dimensions: 2D. <br> Shape: [b, s]. |
| src       | input        | src is a set of update values. <br> Supported type: Tensor. <br> Tensor supported data types: DT_FP32, DT_FP16, DT_BF16, DT_INT32, DT_INT16; data type must match input. <br> Supported dimensions: 2D, 4D. <br> 2D shape: [b * s, d]; 4D shape: [b, s, 1, d]. <br> Empty Tensor not supported; Shape Size must not exceed 2147483647 (i.e., INT32_MAX). |

## Return Value

Returns the updated input; this is an inplace operation.

## Constraints

Broadcast constraint: broadcast is not supported.

ViewShape constraint: In the 2D scenario, ViewShape is \[viewB \* s, d\]; in the 4D scenario, ViewShape is \[viewB, viewS, 1, d\]. The last axis d cannot be tiled. In the 2D scenario, \[viewB \* s, d\] is used for tiling src, where the 0th dimension is a multiple of the index's 1st dimension s, and \[viewB, S\] is used for tiling index. In the 4D scenario, \[viewB, viewS, 1, d\] is used for tiling src, and \[viewB, viewS\] is used for tiling index.

TileShape constraint: In the 2D scenario, TileShape is \[tileS, d\]; in the 4D scenario, TileShape is \[tileB, tileS, 1, d\]. The last axis d cannot be tiled. In the 2D scenario, TileShape is used for tiling src, and \[1, tileS\] is used for tiling index; tileBS must be a divisor of the index's 1st dimension s. For example, if src is \[12, 64\] and index is \[3, 4\], TileShape is \[TileS, 64\], where TileS can be 1, 2, or 4. In the 4D scenario, TileShape is used for tiling src, and \[tileB, tileS\] is used for tiling index. Since TileShape tiling applies to both src and index, the sum of tile block sizes must be within the UB limit.

2D example:
input: [15, 8], index: [5, 2], src: [10, 8], viewShape: [viewB \* s, 8], viewB must be an integer (i.e., the 0th dimension must be a multiple of s), tileShape: [tileS, 8], tileS must be a divisor of s, i.e., 1 or 2.

## Example

### TileShape Configuration Example

Before calling this operation interface, set the TileShape via set_vec_tile_shapes.

The TileShape dimensions should match the input src dimensions.

The input and output are both on GM and do not involve tile tiling. The input index and input src need to be loaded into UB, involving tile tiling.

For example, given input [t, d], dim equal to -2, index [b, s], and src [bs, d] (where bs=b*s), the output is [t, d] and TileShape is set to [bs1, d1]. bs1 is used to tile the bs axis; the d axis cannot be tiled, so d1 must equal d.

```python
pypto.set_vec_tile_shapes(16, 64)
```

### Interface Call Example

-   Update 2D input with 2D src according to 2D index. Note the inplace write pattern: the output on the left side of the assignment must be the same as the input:

    ```python
    x = pypto.tensor([8, 3], pypto.DT_INT32)
    y = pypto.tensor([2, 2], pypto.DT_INT64)
    z = pypto.tensor([4, 3], pypto.DT_INT32)
    x = pypto.scatter_update(x, -2, y, z)
    ```

    Example result:

    ```python
    input data x:[[0 0 0],
                  [0 0 0],
                  [0 0 0],
                  [0 0 0],
                  [0 0 0],
                  [0 0 0],
                  [0 0 0],
                  [0 0 0]]
    input data y:[[1 2],
                  [4 5]]
    input data z:[[1 2 3],
                  [4 5 6],
                  [7 8 9],
                  [10 11 12]]
    output data x:[[0 0 0],
                   [1 2 3],
                   [4 5 6],
                   [0 0 0],
                   [7 8 9],
                   [10 11 12],
                   [0 0 0],
                   [0 0 0]]
    ```

-   Update 4D input with 4D src according to 2D index. Note the inplace write pattern: the output on the left side of the assignment must be the same as the input:

    ```python
    x = pypto.tensor([2, 6, 1, 3], pypto.DT_INT32)
    y = pypto.tensor([2, 2], pypto.DT_INT64)
    z = pypto.tensor([2, 2, 1, 3], pypto.DT_INT32)
    x = pypto.scatter_update(x, -2, y, z)
    ```

    Example result:

    ```python
    input data x:[[
                    [[0 0 0]],
                    [[0 0 0]],
                    [[0 0 0]],
                    [[0 0 0]],
                    [[0 0 0]],
                    [[0 0 0]],
                  ],
                  [
                    [[0 0 0]],
                    [[0 0 0]],
                    [[0 0 0]],
                    [[0 0 0]],
                    [[0 0 0]],
                    [[0 0 0]],
                  ]]
    input data y:[[1 8],
                  [4 10]]
    input data z:[[
                    [[1 2 3]],
                    [[4 5 6]],
                  ],
                  [
                    [[7 8 9]],
                    [[10 11 12]],
                  ]]
    output data x:[[
                    [[0 0 0]],
                    [[1 2 3]],
                    [[0 0 0]],
                    [[0 0 0]],
                    [[7 8 9]],
                    [[0 0 0]],
                  ],
                  [
                    [[0 0 0]],
                    [[0 0 0]],
                    [[4 5 6]],
                    [[0 0 0]],
                    [[10 11 12]],
                    [[0 0 0]],
                  ]]
    ```

