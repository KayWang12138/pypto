# pypto.gather

## Supported Products

| Product          | Supported |
|:-----------------|:---------:|
| Atlas A3 Training Series/Atlas A3 Inference Series |    √     |
| Atlas A2 Training Series/Atlas A2 Inference Series |    √     |

## Description

Selects elements from the input Tensor where the corresponding bit position (specified by the built-in Mask selected by PatternMode) is 1, forming the output Tensor; positions where the bit is 0 are discarded. PatternMode has 7 modes:
- PatternMode=1: Take the first element of every two elements along the last axis
- PatternMode=2: Take the second element of every two elements along the last axis
- PatternMode=3: Take the first element of every four elements along the last axis
- PatternMode=4: Take the second element of every four elements along the last axis
- PatternMode=5: Take the third element of every four elements along the last axis
- PatternMode=6: Take the fourth element of every four elements along the last axis
- PatternMode=7: Take all elements along the last axis

## Function Prototype

```python
gathermask(self: Tensor, pattern_mode: int) -> Tensor
```

## Parameters


| Parameter     | Input/Output | Description                                                                 |
|---------------|--------------|-----------------------------------------------------------------------------|
| self          | input        | Source operand. <br> Supported type: Tensor. <br> Supported tensor data types: DT_INT16, DT_UINT16, DT_INT32, DT_UINT32, DT_FP16, DT_BF16, DT_FP32. <br> Empty tensors are not supported; Shape supports 1–4 dimensions; shape size must not exceed 2147483647 (i.e., INT32_MAX). |
| pattern_mode  | input        | Source operand. <br> int type; valid range: 1–7. |


## Return Value

Returns the output Tensor. The data type of the output Tensor matches self. The output Tensor shape is as follows:
- pattern_mode <= 2: output last axis = self last axis / 2; other axes are the same as the input shape;
- 2 < pattern_mode < 7: output last axis = self last axis / 4; other axes are the same as the input shape;
- pattern_mode = 7: output shape = input shape

## Constraints

1. When 1 <= pattern_mode <= 2:
   - The last axis of self.shape must be divisible by 2
   - The last axis of tileshape must be an integer multiple of 2
   - The last axis of viewshape must be an integer multiple of 2
   - The last axis of self.shape is not view-partitioned
2. When 3 <= pattern_mode <= 6:
   - The last axis of self.shape must be an integer multiple of 4
   - The last axis of tileshape must be an integer multiple of 4
   - The last axis of viewshape must be an integer multiple of 4
   - The last axis of self.shape is not view-partitioned

## Example

### TileShape Configuration Example

Note: Before calling this operation interface, set the TileShape via set_vec_tile_shapes.

The TileShape dimensions must match the output dimensions.

If input self is [x, y, z], pattern_mode is 1, and output is [x, y, z/2], setting TileShape to [x1, y1, 2*z1] means x1, y1, and 2\*z1 are used to partition the x, y, and z axes respectively.

```python
pypto.set_vec_tile_shapes(4, 16, 32)
```

### Interface Call Example

```python
x = pypto.tensor([3, 6], pypto.DT_INT32)        # shape (3, 6)
pattern_mode = 1
y = pypto.gathermask(x, pattern_mode)
```

Example output:

```python
input data x: [[0,  1,  2,  3,  4,  5],
             [6,  7,  8,  9,  10,  11],
             [12,  13,  14,  15,  16,  17]]
     pattern_mode: 1
output data y: [[0,  2,  4],
             [6,  8,  10],
             [12,  14,  16]]
```

