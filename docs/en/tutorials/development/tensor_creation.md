# Tensor Creation

A tensor is the fundamental data structure in PyPTO, used to represent multi-dimensional arrays that will be used in a computation graph and executed on the NPU.

In PyPTO, a tensor represents the structure and attributes of its data, enabling PyPTO to build a computation graph and optimize it before execution. Tensors contain actual values only at execution time; values in uninitialized tensors are random and must be initialized on demand at runtime.

## Creating Tensors

-   Creating a basic tensor

    ```python
    #Create a tensor with shape [2, 3] and data type FP16
    tensor = pypto.tensor([2, 3], pypto.DT_FP16, "my_tensor")
    ```

    Parameters:

    -   shape: Dimensions, supports a list of integers.
    -   dtype: Represents the data type stored in the tensor; supports the DataType type. For example, DT\_FP16 represents a 16-bit half-precision floating-point number.
    -   name: Name, supports string type, optional. However, it is recommended to provide meaningful names for tensors to aid debugging and understanding of the computation graph structure.
    -   format: Data layout format, supports TileOpFormat type, optional, default: TILEOP\_ND.
                When format is explicitly specified, performance is better; the torch tensor passed in must be consistent with the format declared for pypto.Tensor.

-   Creating a tensor with a format

    ```python
    #Create a tensor using NZ format
    tensor = pypto.tensor([-1, 32], pypto.DT_FP16, "nz_tensor", pypto.TileOpFormat.TILEOP_NZ)
    ```

    Supported formats:

    -   TILEOP\_ND: ND format, N-dimensional array, using row-major order in PyPTO.
    -   TILEOP\_NZ: NZ format, a special format for matrix multiplication. A 2D matrix is divided into several fractals (fractal size is better suited to a single Cube computation); fractals are arranged in column-major order (N-shape); each fractal is arranged in row-major order (Z-shape). For details, see [Data Layout Formats](https://www.hiascend.com/document/detail/zh/canncommercial/83RC1/opdevg/Ascendcopdevg/atlas_ascendc_10_0099.html).

-   Creating a tensor in a sub-function and returning it to the main function

    ```python
    def sub_function():
        #Create a tensor with shape [2, 3] and data type FP16
        tensor = pypto.tensor([2, 3], pypto.DT_FP16, "my_tensor")
        return tensor

    def main_function():
         sub_tensor = sub_function()
    ```

-   Converting a PyTorch tensor to a PyPTO tensor

    ```python
    # prepare data
    input_data = torch.rand(shape, dtype=torch.float, device='npu')
    output_data = torch.zeros(shape, dtype=torch.float, device='npu')

    #convert from torch tensor to pypto tensor
    pto_input = pypto.from_torch(input_data, "in_0")
    pto_output = pypto.from_torch(output_data, "out_0")
    ```

## Viewing Tensor Attributes

A tensor has basic attributes including shape, dtype, format, dim, and name. These attribute values can be queried through the pypto.tensor-related operation interfaces.

```python
tensor = pypto.tensor([2,3, 4], pypto.DT_FP16, "example")

#shape
print(tensor.shape)  #[2, 3, 4]

#data type
print(tensor.dtype)  #DataType.DT_FP16

#number of dimensions
print(tensor.dim)    # 3

#format
print(tensor.format)  #TILEOP_ND

#name
print(tensor.name)    # "example"
tensor.name = "new_name"  #can be changed
```

## Handling Tensors with Dynamic Dimensions

In real-world scenarios, tensors are typically variable-length data. You can define a tensor with a dynamic shape using the method below, marking dynamic dimensions with -1:

```python
tensor = pypto.tensor([-1, 32], pypto.DT_FP16, "dynamic")

#Print the tensor dimensions; SymbolicScalar indicates the current shape is a symbolic scalar
print(tensor.shape)
>>> [SymbolicScalar(RUNTIME_GetInputShapeDim(ARG_dynamic,0)), 32]
```

You can obtain a symbolic scalar for a dynamic dimension and retrieve its concrete value at runtime using the following method:

```python
b = pypto.symbolic_scalar(tensor_shape[0])
```

If the tensor is derived from a PyTorch tensor, you can use the `dynamic_axis=[int]` parameter of the `pypto.from_torch` interface to define a tensor with dynamic dimensions.

```python
# prepare data
input_data = torch.rand(shape, dtype=torch.float, device='npu')
output_data = torch.zeros(shape, dtype=torch.float, device='npu')

#convert from torch tensor to pypto tensor with dynamic axis
pto_input = pypto.from_torch(input_data, "in_0", dynamic_axis=[0])
pto_output = pypto.from_torch(output_data, "out_0", dynamic_axis=[0])
```

