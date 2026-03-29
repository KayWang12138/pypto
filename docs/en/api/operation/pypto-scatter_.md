# pypto.scatter\_

## Supported Products

| Product          | Supported |
|:-----------------|:--------:|
| Atlas A3 Training Series/Atlas A3 Inference Series |    √     |
| Atlas A2 Training Series/Atlas A2 Inference Series |    √     |

## Description

Writes values from src into input at positions specified by index. The 3D formula is as follows; other dimensions follow the same pattern:
 <br> When src is a fixed scalar:
$$
\begin{cases}
input\left[ index\left[i\right]\left[j\right]\left[k\right] \right]\left[j\right]\left[k\right] = src & \text{if } dim = 0 \\
input\left[i\right]\left[ index\left[i\right]\left[j\right]\left[k\right] \right]\left[k\right] = src & \text{if } dim = 1 \\
input\left[i\right]\left[j\right]\left[ index\left[i\right]\left[j\right]\left[k\right] \right] = src & \text{if } dim = 2
\end{cases}
$$
 <br> When src is a Tensor:
$$
\begin{cases}
input\left[ index\left[i\right]\left[j\right]\left[k\right] \right]\left[j\right]\left[k\right] = src\left[i\right]\left[j\right]\left[k\right] & \text{if } dim = 0 \\
input\left[i\right]\left[ index\left[i\right]\left[j\right]\left[k\right] \right]\left[k\right] = src\left[i\right]\left[j\right]\left[k\right] & \text{if } dim = 1 \\
input\left[i\right]\left[j\right]\left[ index\left[i\right]\left[j\right]\left[k\right] \right] = src\left[i\right]\left[j\right]\left[k\right] & \text{if } dim = 2
\end{cases}
$$

## Function Prototype

```python
scatter_(input: Tensor, dim: int, index: Tensor, src: Union[float, Element, Tensor], *, reduce: str = None) -> Tensor
```

## Parameters


| Parameter | Input/Output | Description                                                                 |
|-----------|--------------|-----------------------------------------------------------------------------|
| input     | input        | Supported type: Tensor. <br> Tensor supported data types: DT_FP32, DT_FP16, DT_BF16. <br> Empty Tensor not supported; shape supports only 2–4 dimensions; Shape Size must not exceed 2147483647 (i.e., INT32_MAX). |
| dim       | input        | Specifies the dimension used for indexing; supports any dimension within the range of input. <br> Valid dimension index range: -input.dim to input.dim - 1. |
| index     | input        | A set of indices into input. <br> Supported type: Tensor. <br> Tensor supported data types: INT64, INT32. <br> Supported dimensions: same as input. <br> For all dimensions d != dim: index.size(d) <= input.size(d). <br> When src is a Tensor, all dimensions must satisfy: index.size(d) <= src.size(d). <br> Empty Tensor not supported; Shape Size must not exceed 2147483647 (i.e., INT32_MAX). |
| src       | input        | src is the scalar or Tensor used for updates. <br> When src is an Element, supported data types: DT_FP32, DT_FP16, DT_BF16; INF/NAN not supported. <br> When src is a Tensor, supported data types: DT_FP32, DT_FP16; data type must match input. <br> |
| reduce    | input        | The reduction operation to apply; supports 'add' or 'multiply'. When not provided, defaults to direct replacement. |

## Return Value

Returns the updated input; this is an inplace operation.

## Constraints

1. Broadcast constraint: input and index do not support broadcast;

2. The dim axis of input.shape cannot be tiled; the viewshape dimensions match the input dimensions, requiring viewshape\[dim\] \>= max\( input.shape\[dim\], index.shape\[dim\] \); there are no restrictions on the shape sizes of other dimensions;

3. The dim axis of input.shape cannot be tiled; the tileshape dimensions match the input dimensions, with tileshape\[dim\] \>= viewshape\[dim\]; there are no restrictions on the shape sizes of other dimensions. input, index, and result will all be in UB, so the total sum of all input and output tileshape sizes must not exceed the UB memory size.

4. For tiling the non-dim axes of input.shape and index.shape, after tiling viewshape[non dim], the number of tiling blocks for the non-dim axes of input and index must be the same. The same requirement applies when tiling tileshape.

5. When src is a Tensor, dim is the last axis, reduce is None, and there are non-unique indices within a row of index, the behavior is undefined and a value will be arbitrarily selected from src.

## Example

### TileShape Configuration Example

Before calling this operation interface, set the TileShape via set_vec_tile_shapes.

The TileShape dimensions should match the output dimensions.

For example, given input shape [a, b, c], dim equal to 1, index [m, t, p] (where m<=a, p<=c), and src [x, y, z] (where x>=m, y>=t, z>=p), the output is [a, b, c] and TileShape is set to [m1, t1, p1]. m1 and p1 are used to tile the m and p axes respectively. t1 must be greater than or equal to both b and t; the dim axis cannot be tiled, and the full b and t axes must be loaded.

```python
pypto.set_vec_tile_shapes(4, 16, 32)
```

### Interface Call Example

-   Update values in 2D input at positions specified by 2D index

    ```python
    x = pypto.tensor([3, 5], pypto.DT_FP32)
    y = pypto.tensor([2, 2], pypto.DT_INT64)
    o = pypto.scatter_(x, 0, y, 2.0)
    ```

    Example result:

    ```
    input data x:[[0 0 0 0 0],
                  [0 0 0 0 0],
                  [0 0 0 0 0]]
    input data y:[[1 2],
                  [0 1]]
    output data o:[[2.0 0   0 0 0],
                   [2.0 2.0 0 0 0],
                   [0   2.0 0 0 0]]
    ```

