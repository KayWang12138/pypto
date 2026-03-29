# pypto.rms\_norm

## Supported Products

| Product          | Supported |
|:-----------------|:--------:|
| Atlas A3 Training Series/Atlas A3 Inference Series |    √     |
| Atlas A2 Training Series/Atlas A2 Inference Series |    √     |

## Description

Applies Root Mean Square Layer Normalization (RMSNorm) along the last dimension. If gamma is provided, an element-wise scaling is applied along the last dimension.

## Function Prototype

```python
rms_norm(input: Tensor, gamma: Tensor = None, epsilon: float = 1e-6) -> Tensor
```

## Parameters


| Parameter | Input/Output | Description                                                                 |
|-----------|--------------|-----------------------------------------------------------------------------|
| input     | input        | Source operand. <br> Supported data types: data types supported by PyPto. <br> Can be a Tensor of any shape [..., C], where the last dimension C typically represents the number of channels or features. |
| gamma     | input        | Optional scaling parameter; shape should be [C]. |
| epsilon   | input        | Numerical stability constant; default value is 1e-6. |

## Return Value

Returns the normalized Tensor with the same shape as the input Tensor input. The output Tensor is converted back to the original data type of the input Tensor.

## Example

```python
x = pypto.tensor([2, 4], pypto.DT_FP32)
gamma = pypto.tensor([4], pypto.DT_FP32)
y = pypto.rms_norm(x, gamma)
```

Example result:

```python
input data x: [[1, 2, 3, 4],
               [5, 6, 7, 8]]
input data gamma: [1, 1, 1, 1]
output data y: [[0.3651, 0.7302, 1.0954, 1.4605],
                [0.7580, 0.9097, 1.0613, 1.2129]]
```

