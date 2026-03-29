# pypto.from\_torch

## Supported Products

| Product          | Supported |
|:-----------------|:---------:|
| Atlas A3 Training Series/Atlas A3 Inference Series |    √     |
| Atlas A2 Training Series/Atlas A2 Inference Series |    √     |

## Description

Converts a torch.Tensor to a pypto.Tensor. The name of the resulting pypto.Tensor can be explicitly specified. Specified dimensions of the converted pypto.Tensor can be marked as dynamic dimensions, indicating that the dimension may vary during subsequent compilation or execution stages.

## Function Prototype

```python
from_torch(tensor: torch.Tensor, name: str="", *, dynamic_axis: Optional[List[int]] = None,
           tensor_format: Optional[TileOpFormat] = None, dtype: Optional[DataType] = None) -> pypto.Tensor
```

## Parameters


| Parameter      | Input/Output | Description                                                                 |
|----------------|--------------|-----------------------------------------------------------------------------|
| tensor         | Input        | The torch.Tensor object to be converted to a pypto.Tensor. |
| name           | Input        | The name of the pypto.Tensor. Defaults to an empty string, which means from_torch will assign a name automatically. |
| dynamic_axis   | Input        | A list of dimension indices to mark as dynamic. Defaults to None, meaning no dimensions are marked. |
| tensor_format  | Input        | The pypto.TileOpFormat format to specify. When None, it is automatically inferred from the Tensor NPU Format. |
| dtype          | Input        | The pypto.DataType type to specify. When None, it is automatically inferred from the torch.Tensor's dtype. |

## Return Value

Returns the converted pypto.Tensor.

## Constraints

-   The input tensor must be of type torch.Tensor or a subclass thereof.
-   The input tensor must be contiguous in the specified memory format (tensor.is\_contiguous\(\) == True).
-   The input tensor supports the following data types (dtype):
    -   torch.float16
    -   torch.bfloat16
    -   torch.float32
    -   torch.float64
    -   torch.int8
    -   torch.uint8
    -   torch.int16
    -   torch.uint16
    -   torch.int32
    -   torch.uint32
    -   torch.int64
    -   torch.uint64
    -   torch.bool

## Example

```python
x= torch.randn(2, 3)
x_pto = pypto.from_torch(x)
print(x_pto.shape)
y = torch.randn(2, 3)
y_pto = pypto.from_torch(y, "y", dynamic_axis=[0])
print(y_pto.shape)
z = torch.randn(2, 3)
z_pto = pypto.from_torch(z, "z", tensor_format=pypto.TileOpFormat.TILEOP_NZ)
print(z_pto.format)
k = torch.randn(2, 3)
k_pto = pypto.from_torch(k, "k", dtype=pypto.DataType.DT_HF8)
print(k_pto.dtype)
```

Example output:

```python
[2, 3]
[SymbolicScalar(RUNTIME_GetInputShapeDim(ARG_input_tensor,0)), 3]
TileOpFormat.TILEOP_NZ
DataType.DT_HF8
```

