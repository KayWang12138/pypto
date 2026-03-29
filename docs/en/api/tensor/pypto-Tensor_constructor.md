# pypto.Tensor Constructor

## Supported Products

| Product          | Supported |
|:-----------------|:---------:|
| Atlas A3 Training Series/Atlas A3 Inference Series |    √     |
| Atlas A2 Training Series/Atlas A2 Inference Series |    √     |

## Description

Creates a Tensor object. The Tensor is initialized with uninitialized random values upon creation.

## Function Prototype

```python
__init__(self,
         shape=None,
         dtype: Union[DataType, None] = None,
         name: str = "",
         format: TileOpFormat = TileOpFormat.TILEOP_ND,
         data_ptr: Optional[int] = None,
         device=None,
         ori_shape=None
)
```

## Parameters


| Parameter  | Input/Output | Description                                                                 |
|------------|--------------|-----------------------------------------------------------------------------|
| shape      | input        | The shape of the Tensor. Can be one of the following types:<br> - None: creates an empty Tensor<br> - List[int]: a list of integers specifying the size of each dimension<br> - List[Union[int, SymbolicScalar]]: a list containing integers or symbolic scalars, used for dynamic shapes |
| dtype      | input        | The data type of the Tensor. |
| name       | input        | The name of the Tensor. |
| format     | input        | The format of the Tensor. Optional values include:<br> - TileOpFormat.TILEOP_ND (default)<br> - TileOpFormat.TILEOP_NZ |
| data_ptr   | input        | Data pointer, defaults to None. Currently for internal use by the frontend framework only; operator developers can ignore this. |
| device     | input        | Device information, defaults to None. |
| ori_shape  | input        | Original shape, used to store the original shape information of the Tensor, defaults to None. |

## Return Value

A Tensor object.

## Constraints

None.

## Example

```python
# Create an empty Tensor
empty_tensor = pypto.Tensor()

# Create a Tensor with specified shape and data type
tensor1 = pypto.Tensor(shape=(4, 4), dtype=pypto.DT_FP32)
tensor2 = pypto.Tensor(shape=[8, 16, 32], dtype=pypto.DT_INT32)

# Create a named Tensor
named_tensor = pypto.Tensor(shape=(4, 4),
                            dtype=pypto.DT_FP32,
                            name="input_tensor" )

# Create a Tensor with a specified format
sparse_tensor = pypto.Tensor(shape=(4, 32),
                             dtype=pypto.DT_FP32,
                             format=pypto.TileOpFormat.TILEOP_NZ )

# Create a dynamic shape Tensor (using symbolic scalars)
dynamic_shape = [pypto.SymbolicScalar("N"), 4, 8]
dynamic_tensor = pypto.Tensor(shape=dynamic_shape,
                              dtype=pypto.DT_FP32 )

# Create using the pypto.tensor convenience function (recommended)
tensor3 = pypto.tensor((4, 4), pypto.DT_FP32)
tensor4 = pypto.tensor((4, 4), pypto.DT_FP32, name="my_tensor")
```

