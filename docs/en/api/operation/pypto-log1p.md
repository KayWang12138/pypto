# pypto.log1p

## Supported Products

| Product          | Supported |
|:-----------------|:---------:|
| Atlas A3 Training Series/Atlas A3 Inference Series |    √     |
| Atlas A2 Training Series/Atlas A2 Inference Series |    √     |

## Description

Computes the natural logarithm (base e) of 1+input.

## Function Prototype

```python
log1p(input: Tensor) -> Tensor:
```

## Parameters


| Parameter | Input/Output | Description                                                                 |
|-----------|--------------|-----------------------------------------------------------------------------|
| input     | input        | Source operand. <br> Supported type: Tensor. <br> Supported tensor data types: DT_FP32, DT_FP16, DT_BF16. <br> Supported dimensions: 2–4 <br> Empty tensors are not supported; Shape Size must not exceed 2147483647 (i.e., INT32_MAX). |

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
y = pypto.log1p(x)
```

Example output:

```python
input data x: [1e-99]
output data y: [1e-99]
```

