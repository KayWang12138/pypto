# pypto.abs

## Supported Products

| Product          | Supported |
|:-----------------|:---------:|
| Atlas A3 Training Series/Atlas A3 Inference Series |    √     |
| Atlas A2 Training Series/Atlas A2 Inference Series |    √     |

## Description

Computes the absolute value of each element in the input Tensor, element-wise.

## Function Prototype

```python
abs(input: Tensor) -> Tensor
```

## Parameters


| Parameter | Input/Output | Description                                                                 |
|-----------|--------------|-----------------------------------------------------------------------------|
| input     | input        | Source operand. <br> Supported type: Tensor. <br> Tensor supported data types: DT_FP16, DT_BF16, DT_FP32. <br> Empty Tensor not supported; shape supports only 2–4 dimensions; shape size must not exceed 2147483647 (i.e., INT32_MAX). |

## Return Value

Returns a Tensor. Its shape and data type are consistent with the input Tensor, and its elements are the absolute values of the corresponding elements in the input Tensor.

## Example

### TileShape Configuration Example

Note: Before calling this operation interface, set the TileShape via set_vec_tile_shapes.

The TileShape dimensions should be consistent with the output.

Example 1: If the input shape is [m, n] and the output is [m, n], setting TileShape to [m1, n1] means m1 and n1 are used to tile the m and n axes respectively.

```python
pypto.set_vec_tile_shapes(4, 16)
```

### Interface Call Example

```python
x = pypto.tensor([5], pypto.DT_FP32)
y = pypto.abs(x)
```

Example result:

```python
Input data x: [-1.0, 2.0, -3.0, 4.0, 5.0]
Output data y: [1.0,  2.0,  3.0, 4.0, 5.0]
```

