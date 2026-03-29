# pypto.bitwise\_not

## Supported Products

| Product          | Supported |
|:-----------------|:---------:|
| Atlas A3 Training Series/Atlas A3 Inference Series |    √     |
| Atlas A2 Training Series/Atlas A2 Inference Series |    √     |

## Description

Performs a bitwise NOT operation element-wise on input. The formula is:

$$
res_i = \sim input_i
$$

## Function Prototype

```python
bitwise_not(input: Tensor) -> Tensor
```

## Parameters


| Parameter | Input/Output | Description                                                                 |
|-----------|--------------|-----------------------------------------------------------------------------|
| input     | input        | Source operand. <br> Supported type: Tensor. <br> Tensor supported data types: DT_INT16, DT_UINT16, DT_BOOL. <br> Empty Tensor not supported; shape supports only 2–4 dimensions; shape size must not exceed 2147483647 (i.e., INT32_MAX). |

## Return Value

Returns the output Tensor. The data type and shape are the same as input.

## Example

### TileShape Configuration Example

Before calling this operation interface, set the TileShape via set_vec_tile_shapes.

The TileShape dimensions should be consistent with the output.

For example, if input shape is [m, n] and output is [m, n], setting TileShape to [m1, n1] means m1 and n1 are used to tile the m and n axes respectively.

```python
pypto.set_vec_tile_shapes(4, 16)
```

### Interface Call Example

```python
a = pypto.tensor([2], pypto.DT_INT16)
out = pypto.bitwise_not(a)
```

Example result:

```python
Input data a:   [2, 5]
Output data out: [-3, -6]
```

