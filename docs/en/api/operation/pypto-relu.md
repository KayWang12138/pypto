# pypto.relu

## Supported Products

| Product          | Supported |
|:-----------------|:--------:|
| Atlas A3 Training Series/Atlas A3 Inference Series |    √     |
| Atlas A2 Training Series/Atlas A2 Inference Series |    √     |

## Description

Applies the Rectified Linear Unit (ReLU) operation to each element of input, retaining only the positive part and setting negative values to 0. The formula is as follows:

$$
res_i = \max(0, input_i)
$$

## Function Prototype

```python
relu(input: Tensor) -> Tensor
```

## Parameters

| Parameter | Input/Output | Description                                                                 |
|-----------|--------------|-----------------------------------------------------------------------------|
| input     | input        | Source operand. <br> Supported type: Tensor. <br> Tensor supported data types: DT_FP16, DT_FP32, DT_BF16. <br> Empty Tensor not supported; shape supports only 2–4 dimensions; Shape Size must not exceed 2147483647 (i.e., INT32_MAX). |

## Return Value

Returns the output Tensor. The data type and shape of the Tensor are the same as input.

## Constraints

1.  input does not support special values such as nan or inf.

## Example

### TileShape Configuration Example

Note: Before calling this operation interface, set the TileShape via set_vec_tile_shapes.

The TileShape dimensions should match the output dimensions.

Example 1: Given input shape [m, n] and output [m, n], TileShape is set to [m1, n1], where m1 and n1 are used to tile the m and n axes respectively.

```python
pypto.set_vec_tile_shapes(4, 16)
```

### Interface Call Example

```python
input_tensor = pypto.tensor([-2.0, 0.0, 3.0], pypto.DT_FP32)
out = pypto.relu(input_tensor)
```

Example result:

```python
input data input: [[-2.0 0.0 3.0]]
output data out:  [[ 0.0 0.0 3.0]]
```
