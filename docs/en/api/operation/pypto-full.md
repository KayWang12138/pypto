# pypto.full

## Supported Products

| Product          | Supported |
|:-----------------|:---------:|
| Atlas A3 Training Series/Atlas A3 Inference Series |    √     |
| Atlas A2 Training Series/Atlas A2 Inference Series |    √     |

## Description

Creates a Tensor of the given size filled with fill\_value. The data type is specified by dtype.

## Function Prototype

```python
full(size: List[int], fill_value: Union[int, float, Element], dtype: DataType, *, valid_shape: Optional[Union[List[int], List[SymbolicScalar]]] = None ) -> Tensor
```

## Parameters


| Parameter    | Input/Output | Description                                                                 |
|--------------|--------------|-----------------------------------------------------------------------------|
| size         | input        | Source operand that defines the output Tensor shape. <br> Supported data type: List[int]. |
| fill_value   | input        | Source operand used to fill the output Tensor. <br> Supported data types: int, float, Element. <br> When of type int or float, it will be automatically converted to Element type, where int corresponds to DT_INT_32 and float corresponds to DT_FP32. For other data types, use Element to construct. <br> Element supported data types: DT_FP32, DT_INT32, DT_INT16, DT_FP16, DT_BF16. <br> Must be the same type as dtype; implicit conversion is not supported. |
| dtype        | input        | Source operand that defines the output Tensor type. <br> Supported data types: DT_FP32, DT_INT32, DT_INT16, DT_FP16, DT_BF16. <br> Must be the same type as fill_value; implicit conversion is not supported. |
| valid_shape  | input        | Source operand that defines the dynamic shape of the output Tensor; a keyword argument used for dynamic graphs. Can be omitted for static graphs. <br> Supported types: List[SymbolicScalar], List[int]. |

## Return Value

Returns the output Tensor. The data type is the same as dtype; the Shape is of size; all values are fill\_value.

## Constraints

1.  valid\_shape is used in dynamic graph scenarios.

    In a dynamic graph scenario, if you need to generate a \[5,5\] Tensor and set the ViewShape to \[2,2\], the framework will loop through pypto.loop to generate \[2,2\] tiles and concatenate them by offset. If valid\_shape is not provided, the code will default to generating a fully \[2,2\] Tensor (e.g., pypto.full\(\[2,2\], 1, pypto.DT\_INT32\)).

    However, when the total size \[5,5\] is not evenly divisible by the tile size \[2,2\], the effective shape of the tail tile (e.g., \[1,1\]) cannot be automatically inferred by the framework. For example, the last row/column may contain only 1 element rather than a complete \[2,2\] tile. In this case, valid\_shape must be explicitly specified to indicate the actual effective shape of the tail tile, as follows:

    pypto.full\(\[2, 2\], 1, pypto.DT\_INT32, valid\_shape=\[pypto.min\(2, 5 - 2 \* b\_idx\), pypto.min\(2, 5 - 2 \* s\_idx\)\]\), where b\_idx and s\_idx are loop indices.

2.  The tileshape dimensions must match the result dimensions, and are used to partition result.

## Example

### TileShape Configuration Example

Before calling this operation interface, set the TileShape via set_vec_tile_shapes.

The TileShape dimensions must match the output dimensions.

If size is [m, n] and the output is [m, n], setting TileShape to [m1, n1] means m1 and n1 are used to partition the m and n axes respectively.

```python
pypto.set_vec_tile_shapes(4, 16)
```

### Interface Call Example

```python
# Valid shapes use keyword argument
x1 = 1.0 # must be 1.0; implicit conversion is not supported
y1 = pypto.full([2,2], x1, pypto.DT_FP32, valid_shape = [pypto.symbolic_scalar(2), pypto.symbolic_scalar(2)])

x2 = pypto.Element(pypto.DT_INT32,1)
y2 = pypto.full([2,2], x2, pypto.DT_INT32, valid_shape = [pypto.symbolic_scalar(2), pypto.symbolic_scalar(2)])

# In static graphs, validshape can be ignored
x3 = pypto.Element(pypto.DT_INT32,1)
y3 = pypto.full([2,2], x3, pypto.DT_INT32)
```

Example output:

```python
y1 output data: [[1.0,1.0], [1.0,1.0]]
y2 output data: [[1,1],[1,1]]
y3 output data: [[1,1], [1,1]
```

