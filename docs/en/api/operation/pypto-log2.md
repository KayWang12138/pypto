# pypto.log2

## Supported Products

| Product          | Supported |
|:-----------------|:---------:|
| Atlas A3 Training Series/Atlas A3 Inference Series |    √     |
| Atlas A2 Training Series/Atlas A2 Inference Series |    √     |

## Description

Computes the base-2 logarithm of input.

## Function Prototype

```python
log2(input: Tensor) -> Tensor:
```

## Parameters


| Parameter | Input/Output | Description                                                                 |
|-----------|--------------|-----------------------------------------------------------------------------|
| input     | input        | Source operand. <br> Supported type: Tensor. <br> Supported tensor data types: DT_FP32, DT_FP16, DT_BF16. <br> Supported dimensions: 1–4 <br> Empty tensors are not supported; Shape Size must not exceed 2147483647 (i.e., INT32_MAX). |

## Return Value

Returns the output Tensor. The data type is the same as input; the Shape matches input.

## TileShape Configuration Example

The TileShape dimensions must match the output dimensions.

If input shape is [m, n] and output is [m, n], setting TileShape to [m1, n1] means m1 and n1 are used to partition the m and n axes respectively.

```python
pypto.set_vec_tile_shapes(m1, n1)
```

## Example

```python
x = pypto.tensor([3], pypto.DT_FP32)
y = pypto.log2(x)
```

Example output:

```python
input data x: [1.0     2.0    3.0]
output data y: [0.0000 1.0000 1.5849]
```

