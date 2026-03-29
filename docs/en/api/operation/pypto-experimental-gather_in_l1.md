# pypto.experimental.gather\_in\_l1

## Supported Products

| Product | Supported |
|:--------|:---------:|
| Atlas A3 Training Series / Atlas A3 Inference Series | √ |
| Atlas A2 Training Series / Atlas A2 Inference Series | √ |

## Description

This is a custom interface with many constraints. Stability is not guaranteed.

Transfers data from the specified rows of a Tensor on GM (Global Memory) in a scatter fashion, moving the first `size` elements of each row to L1.

## Function Prototype

```python
gather_in_l1(src: Tensor, indices: Tensor, block_table: Tensor, block_size: int,
                 size: int, is_b_matrix: bool, is_trans: bool) -> Tensor
```

## Parameters

| Parameter | Input/Output | Description |
|-----------|--------------|-------------|
| src | Input | Source operand. <br> Supported data types: DT_FP32, DT_FP16, DT_BF16, DT_INT8. <br> Empty Tensors are not supported; only 2D shapes are supported. |
| indices | Input | Row offset for the source operand. <br> Supported data types: DT_INT32, DT_INT64. <br> Empty Tensors are not supported; only 2D shapes are supported. <br> Shape must be [1, n]. |
| block_table | Input | Source operand. <br> Supported data type: DT_INT32. <br> Empty Tensors are not supported; only 2D shapes are supported. <br> In practice, represents the page table in Paged Attention, with shape [1, block_table_size], where block_table_size is the length of the page table. |
| block_size | Input | Source operand. <br> Type: int. <br> Represents the number of tokens that can fit in one block in Paged Attention. |
| size | Input | Number of elements to transfer per row. <br> Must be less than the number of columns in the source operand. |
| is_b_matrix | Input | Whether the transferred result (i.e., the output Tensor) is used as the B matrix in a matmul operation. |
| is_trans | Input | Whether the transferred result (i.e., the output Tensor) is transposed. |

## Return Value

Returns the output Tensor.

## Example

```python
src = pypto.tensor([16, 32], pypto.DT_FP32, "tensor_src")
offset = pypto.tensor([1, 32], pypto.DT_INT32, "tensor_offset")
out = pypto.experimental.gather_in_l1(src , offset, 20, false, false)
```
