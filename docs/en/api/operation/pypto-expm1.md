# pypto.expm1

## Supported Products

| Product          | Supported |
|:-----------------|:---------:|
| Atlas A3 Training Series/Atlas A3 Inference Series |    √     |
| Atlas A2 Training Series/Atlas A2 Inference Series |    √     |

## Description

Computes the natural exponential of each element in the input Tensor minus 1:
$$
y_i = e^{x_i} - 1
$$

## Function Prototype

```python
expm1(input: Tensor) -> Tensor
```

## Parameters


| Parameter | Input/Output | Description                                                                                                                                                             |
|----------|--------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| input    | input        | Source operand. <br> Supported type: Tensor. <br> Supported tensor data types: DT_FP32, DT_FP16, DT_BF16, DT_INT32, DT_INT16. <br> Empty tensors are not supported; Shape supports only 2–4 dimensions; Shape Size must not exceed 2147483647 (i.e., INT32_MAX). |

## Return Value

Returns a Tensor. Its shape is consistent with the input Tensor. When the input data type is DT_FP32, DT_FP16, or DT_BF16, the output data type matches the input; when the input data type is DT_INT32 or DT_INT16, the output data type is DT_FP32. Each element is the result of computing the natural exponential of the corresponding input element minus 1.

## Example

### TileShape Configuration Example

Note: Before calling this operation interface, set the TileShape via set_vec_tile_shapes.

The TileShape dimensions must match the output dimensions.

Example 1: If the input shape is [m, n] and the output is [m, n], setting TileShape to [m1, n1] means m1 and n1 are used to partition the m and n axes respectively.

```python
pypto.set_vec_tile_shapes(4, 16)
```

### Interface Call Example

```python
x = pypto.tensor([2, 2], pypto.DT_FP32)
y = pypto.expm1(x)
```

Example output:

```python
input data x: [[1., 2.], [3., 4.]]
output data y: [[1.7183, 6.3891], [19.0855, 53.5981]]
```
