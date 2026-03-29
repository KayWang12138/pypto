# pypto.prelu

## Supported Products

| Product          | Supported |
|:-----------------|:--------:|
| Atlas A3 Training Series/Atlas A3 Inference Series |    √     |
| Atlas A2 Training Series/Atlas A2 Inference Series |    √     |

## Description

Applies a Parametric Rectified Linear Unit (PReLU) operation to each element of input. Elements greater than or equal to 0 are kept unchanged, while elements less than 0 are multiplied by a weight coefficient. The formula is as follows:

$$
res_i = \begin{cases}
input_i & \text{if } input_i \geq 0 \\
weight_i \times input_i & \text{if } input_i < 0
\end{cases}
$$

where weight is a one-dimensional tensor whose length equals the size of the second dimension (channel dimension) of input, with weights shared per channel.

## Function Prototype

```python
prelu(input: Tensor, weight: Tensor) -> Tensor
```

## Parameters

| Parameter | Input/Output | Description                                                                 |
|-----------|--------------|-----------------------------------------------------------------------------|
| input     | input        | Source operand. <br> Supported type: Tensor. <br> Tensor supported data types: DT_FP16, DT_FP32, DT_BF16. <br> Empty Tensor not supported; shape supports only 2–4 dimensions; Shape Size must not exceed 2147483647 (i.e., INT32_MAX). |
| weight    | input        | Weight parameter. <br> Supported type: Tensor. <br> Tensor supported data types: DT_FP16, DT_FP32, DT_BF16, must match the type of input. <br> Shape must be one-dimensional, with length equal to the size of the second dimension of input. |

## Return Value

Returns the output Tensor. The data type and shape of the Tensor are the same as input.

## Constraints

1.  input and weight must have the same type.
2.  The shape of weight must be one-dimensional and its length must equal the size of the second dimension of input.
3.  input and weight do not support special values such as nan or inf.
4.  Due to temporary memory usage, when the input dimension is two-dimensional, the TileShape size has an additional constraint: assuming TileShape is \[a,b\], then a\*b\*sizeof\(self\) + b/8 + 8KB < UB.

## Example

### TileShape Configuration Example

Note: Before calling this operation interface, set the TileShape via set_vec_tile_shapes.

The TileShape dimensions should match the output dimensions.

Example 1: Given input and weight with shapes [m, n] and [n\] respectively, the output is [m, n]. TileShape is set to [m1, n1], where m1 and n1 are used to tile the m and n axes respectively.

```python
pypto.set_vec_tile_shapes(4, 16)
```

### Interface Call Example

```python
# Example: PReLU operation
# input shape is [2, 3], weight shape is [3]
# For negative elements, multiply by the corresponding weight per channel
input_tensor = pypto.tensor([[-2.0, 1.0, -3.0], [0.5, -1.0, 2.0]], pypto.DT_FP32)
weight_tensor = pypto.tensor([0.25, 0.5, 0.1], pypto.DT_FP32)
out = pypto.prelu(input_tensor, weight_tensor)
```

Example result:

```python
input data input:  [[-2.0,  1.0, -3.0], [ 0.5, -1.0,  2.0]]
input data weight: [ 0.25, 0.5,  0.1]
output data out:   [[-0.5,  1.0, -0.3], [ 0.5, -0.5,  2.0]]
```

Calculation description:
- Channel 0: -2.0 < 0, result = 0.25 × (-2.0) = -0.5; 0.5 ≥ 0, result = 0.5
- Channel 1: 1.0 ≥ 0, result = 1.0; -1.0 < 0, result = 0.5 × (-1.0) = -0.5
- Channel 2: -3.0 < 0, result = 0.1 × (-3.0) = -0.3; 2.0 ≥ 0, result = 2.0
