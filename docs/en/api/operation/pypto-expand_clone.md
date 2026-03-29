# pypto.expand\_clone

## Supported Products

| Product          | Supported |
|:-----------------|:---------:|
| Atlas A3 Training Series/Atlas A3 Inference Series |    √     |
| Atlas A2 Training Series/Atlas A2 Inference Series |    √     |

## Description

Broadcasts the input tensor along the single axis whose size equals 1 to match the target shape, and returns a new tensor that occupies real memory.

## Function Prototype

```python
expand_clone(
    input: Tensor,
    shape: List[int],
    *,
    valid_shape: Optional[List[Union[int, SymbolicScalar]]] = None
) -> Tensor
```

## Parameters


| Parameter   | Input/Output | Description                                                                 |
|-------------|--------------|-----------------------------------------------------------------------------|
| input       | Input        | Source operand. <br> Supported type: Tensor. <br> Supported tensor data types: DT_BF16, DT_FP32, DT_FP16, DT_INT8, DT_INT16, DT_INT32, DT_UINT8, DT_UINT16, DT_UINT32, DT_BOOL. <br> Empty tensors are not supported; shape supports 2–4 dimensions only; the size of the broadcast axis must be 1; shape size must not exceed 2147483647 (INT32_MAX). |
| shape       | Input        | Source operand, the target shape. <br> Supported data type: List[int]. <br> Shape size must not exceed INT32_MAX; the number of dimensions must match the input; all axes except the broadcast axis must have sizes equal to the corresponding axes of `input`. |
| valid_shape | Input        | Keyword argument. <br> Source operand used to define the dynamic shape of the output tensor; used in dynamic graph mode and may be omitted in static graph mode. <br> Supported types: List[SymbolicScalar], List[int]. |

## Return Value

Returns the output tensor. The data type matches `input` and the shape is `shape`.

## Constraints

1.  Only one-dimensional broadcasting is supported; the broadcast axis of the input tensor must have size 1.
2.  The viewshape of `input` has the same number of dimensions as `input`, with viewshape\[dim\]=1 and input\[dim\]=1, where dim is the expanded axis; no restriction on the remaining dimensions. Examples:
    1.  \[a,1\] expanded to \[a,5\]: dim=1, meaning expansion is performed along dim 1.
    2.  len\(viewshape\)=2 and viewshape\[dim\]=1.

3.  Notes on `valid_shape`:

    In dynamic graph mode, suppose tensor input \[a,1\] is expanded to \[a,5\] with ViewShape set to \[a,2\]. The framework generates tiles of \[a,2\] via pypto.loop and assembles them at offsets. Without `valid_shape`, the code defaults to generating full \[a,2\] tensors (as in pypto.expand\_clone\(input, \[a,2\]\)).

    However, when the total size \[a,5\] is not evenly divisible by the tile size \[a,2\], the effective shape of the tail tile (e.g., \[a,1\]) cannot be inferred automatically by the framework. For example, the last column may contain only 1 element rather than a full \[a,2\] tile. In this case, `valid_shape` must be explicitly specified to indicate the actual effective shape of the tail tile, as follows:

    pypto.expand\_clone\(input, \[a,2\], valid\_shape = \[a, pypto.min\(2, 5 - 2 \* b\_idx\),\)

    where b\_idx is the loop index.

4.  The tileshape dimensions match the result dimensions and are used to split the result.
5.  There are no additional constraints on the size of tileshape, except that it must not exceed the UB size.

## Example

### TileShape Configuration Example

Before calling this operation interface, set the TileShape via `set_vec_tile_shapes`.

The TileShape dimensions must match the output.

For example, if input shape is [m, 1] and output is [m, n], and TileShape is set to [m1, n1], then m1 and n1 are used to split the m and n axes respectively.

```python
pypto.set_vec_tile_shapes(4, 16)
```

### Interface Call Example

```python
# static graph
a = pypto.tensor([1,8], pypto.DT_INT32)
out1 = pypto.expand_clone(a, [4,8])
# dynamic graph
out2 = pypto.expand_clone(a, [4,8], valid_shape = [pypto.symbolic_scalar(4), pypto.symbolic_scalar(8)])
```

Example result:

```python
Input a:     [[1, 2, 3, 4, 5, 6, 7, 8]]
Output out1: [[1, 2, 3, 4, 5, 6, 7, 8],
               [1, 2, 3, 4, 5, 6, 7, 8],
               [1, 2, 3, 4, 5, 6, 7, 8],
               [1, 2, 3, 4, 5, 6, 7, 8]]
Output out2: [[1, 2, 3, 4, 5, 6, 7, 8],
               [1, 2, 3, 4, 5, 6, 7, 8],
               [1, 2, 3, 4, 5, 6, 7, 8],
               [1, 2, 3, 4, 5, 6, 7, 8]]
```
