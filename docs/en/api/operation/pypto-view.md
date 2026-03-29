# pypto.view

## Supported Products

| Product          | Supported |
|:-----------------|:---------:|
| Atlas A3 Training Series/Atlas A3 Inference Series |    √     |
| Atlas A2 Training Series/Atlas A2 Inference Series |    √     |

## Description

Extracts a partial view from the input tensor for use in subsequent computations.

## Function Prototype

```python
view(input: Tensor, shape: List[int] = None, offsets: List[Union[int, SymbolicScalar]] = None, *, valid_shape: Optional[List[Union[int, SymbolicScalar]]] = None, dtype: DataType = None,
) -> Tensor:
```

## Parameters


| Parameter   | Input/Output | Description                                                                 |
|-------------|--------------|-----------------------------------------------------------------------------|
| input       | Input        | Source operand.<br> Supported data types: all data types supported by PyPto.<br> Empty tensors are not supported; shape size must not exceed 2147483647 (INT32_MAX). |
| shape       | Input        | The size of the extracted view.<br> Shape size must not exceed 2147483647 (INT32_MAX). |
| offsets     | Input        | The offset of each dimension relative to `input` when extracting the view.<br> Must ensure offsets are less than the shape of `input`. |
| valid_shape | Input        | The size of the valid data in the extracted tile block.<br> Must ensure valid_shape is less than the shape of `input`. In scenarios such as page_attention where the input kv_cache and similar tensors contain invalid data, the output valid_shape cannot be correctly inferred and must be passed manually. |
| dtype       | Input        | The data type of the return value, allowing the input data to be interpreted as a different data type. |

## Return Value

Returns an output tensor with the same data type as `input` and a shape specified by the `shape` parameter. If `valid\_shape` is specified, the actual data size is `valid\_shape`. If `dtype` is specified, the input is read according to `dtype`.


## Example

-   Basic usage

    ```python
    x = pypto.tensor([4, 8], pypto.DT_FP32)
    shape = [4, 4]
    offsets = [0, 4]
    y = pypto.view(x, shape, offsets)
    ```

    Example result:

    ```python
    Input x: [[1 1 2 2 3 3 4 4],
                [1 1 2 2 3 3 4 4],
                [1 1 2 2 3 3 4 4],
                [1 1 2 2 3 3 4 4]]
    Output y: [[3 3 4 4],
                [3 3 4 4],
                [3 3 4 4],
                [3 3 4 4]]
    ```

-   Adding valid\_shape

    ```python
    x = pypto.tensor([4, 8], pypto.DT_FP32)
    shape = [4, 4]
    offsets = [2, 4]
    valid_shape = [2, 4]
    y = pypto.view(x, shape, offsets, valid_shape)
    ```

    Example result:

    ```python
    Input x: [[1 1 2 2 3 3 4 4],
                [1 1 2 2 3 3 4 4],
                [1 1 2 2 5 5 6 6],
                [1 1 2 2 5 5 6 6]]
    Output y: [[5 5 6 6],
                [5 5 6 6],
                [0 0 0 0],
                [0 0 0 0]]
    ```

-   Specifying dtype

    ```python
    x = pypto.tensor([2, 2], pypto.DT_FP32)
    y = pypto.view(x, dtype=pypto.DT_INT8)
    ```

    Result:

    ```python
    Input x:
    [[0.9405094  0.20237109],
     [0.99819463 0.13246714]]

    Output y:
    [[  57  -59  112   63   94   58   79   62],
     [ -81 -119  127   63  119  -91    7   62]]

    ```
