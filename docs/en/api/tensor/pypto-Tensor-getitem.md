# pypto.Tensor Indexing

Tensor indexing is one of the core operations of a Tensor, used to filter, extract, or modify elements at specific positions in a Tensor. Through indexing operations, developers can precisely retrieve portions of data from a Tensor (such as individual elements, sub-tensors, or data along specific dimensions), or assign values to elements at specified positions.

## I. \_\_getitem\_\_

## Description

Retrieves a sub-tensor or individual element from a Tensor using an index or slice. This method supports multiple indexing modes, providing a flexible and intuitive way to access data.

## Function Prototype

```python
def __getitem__(self, key, *, valid_shape: Optional[List[Union[int, SymbolicScalar]]] = None)
```

## Parameters


| Parameter   | Input/Output | Description                                                                 |
|-------------|--------------|-----------------------------------------------------------------------------|
| key         | input        | Tensor index, used to retrieve data at the corresponding position in the Tensor.<br> Supported types:<br> - int or SymbolicScalar (symbolic scalar): a single integer index.<br> - slice: a slice object.<br> - tuple: a combination of multi-dimensional indices, including: int or SymbolicScalar, slice, Ellipsis(...). |
| valid_shape | input        | Represents the size of valid data in the output Tensor. |

## Return Value

Returns the Tensor data at the corresponding index position.

## Constraints

1. For slice objects (format: start:end:step), the step parameter is not currently supported and defaults to 1.

Unsupported example: a\[1:2:2, :\].

2. Boolean type indexing is not currently supported.

Unsupported example: a\[True, False, True, False\].

3. Tensor type indexing is not currently supported.

Unsupported example: a\[b\], \(b = pypto.Tensor\(\[2\], pypto.DT\_INT32\).

## Usage Examples

1. Full slice

   Use a slice to retrieve a sub-region of the Tensor.

   ```python
   a = pypto.tensor([4, 4], pypto.DT_FP32)
   b = a[:2, :2] #equivalent to view(a, [2, 2], [0, 0])
   ```

   Example result:

   ```python
   input data a: [[1, 2, 3, 4],
               [5, 6, 7, 8],
               [9, 10, 11, 12],
               [13, 14, 15, 16]]
   output data b: [[1, 2],
               [5, 6]]
   ```

2. Mixed indexing and slicing

   Combine integer indexing and slices to reduce dimensions and extract specific rows or columns.

   ```python
   a = pypto.tensor([4, 4], pypto.DT_FP32)
   b = a[1, 1:3] #equivalent to first view(a, [1, 2], [1, 1]), then reshape to [2]
   ```

   Example result:

   ```python
   input data a: [[1, 2, 3, 4],
               [5, 6, 7, 8],
               [9, 10, 11, 12],
               [13, 14, 15, 16]]
   output data b: [6, 7]
   ```

3. Negative indexing

    Supports Python-style negative indexing, counting from the end.

    ```python
    a = pypto.tensor([4, 4], pypto.DT_FP32)
    b = a[-1, -3:-1] #equivalent to s[3, 1:3]
    ```

    Example result:

    ```python
    input data a: [[1, 2, 3, 4],
                [5, 6, 7, 8],
                [9, 10, 11, 12],
                [13, 14, 15, 16]]
    output data b: [14, 15]
    ```

4. Ellipsis (...)

   Use \`...\` to automatically fill in all intermediate dimensions, simplifying multi-dimensional indexing.

   ```python
   a = pypto.tensor([4, 4], pypto.DT_FP32)
   b = a[..., 1:3] #equivalent to s[:, 1:3]
   ```

   Example result:

   ```python
   input data a: [[1, 2, 3, 4],
               [5, 6, 7, 8],
               [9, 10, 11, 12],
               [13, 14, 15, 16]]
   output data b: [[2, 3],
               [6, 7],
               [10, 11],
               [14, 15]]
   ```

5. Single element access

   Integer indexing to retrieve a single element from the Tensor (only DT\_INT32 type is supported).

   ```python
   a = pypto.tensor([4, 4], pypto.DT_INT32)
   b = a[0, 0] #returns SymbolicScalar
   ```

   Example result:

   ```python
   input data a: [[1, 2, 3, 4],
               [5, 6, 7, 8],
               [9, 10, 11, 12],
               [13, 14, 15, 16]]
   output data b: 1
   ```

6. Gather operation

   When the index is in the form \[int:Tensor\], a gather operation is executed where the int type corresponds to dim and the Tensor type corresponds to index. This slice syntax is equivalent to Tensor.gather\(dim, index\).

   ```python
   a = pypto.tensor([4, 4], pypto.DT_FP32)
   index = pypto.tensor([1, 4], pypto.DT_INT32)
   b = a[0:index] #calls gather(a, 0, index)
   ```

   Example result:

   ```python
   input data a: [[1, 2, 3, 4],
               [5, 6, 7, 8],
               [9, 10, 11, 12],
               [13, 14, 15, 16]]
   input data index: [[0, 1, 2, 3]]
   output data b: [[1, 6, 11, 16]]
   ```

## II. \_\_setitem\_\_

## Description

Assigns a value to a specified position in a Tensor using an index or slice.

## Function Prototype

```python
def __setitem__(self, key, value)
```

## Parameters


| Parameter | Input/Output | Description                                                                 |
|-----------|--------------|-----------------------------------------------------------------------------|
| key       | input        | Tensor index, used to retrieve data at the corresponding position in the Tensor.<br> Supported types:<br> - int or SymbolicScalar (symbolic scalar): a single integer index.<br> - slice: a slice object.<br> - tuple: a combination of multi-dimensional indices, including: int or SymbolicScalar, slice, Ellipsis(...). |
| value     | input        | The value to set, supports Tensor or scalar (float/int). |

## Return Value

Returns the Tensor with the assigned value at the corresponding position.

## Constraints

1. For slice objects (format: start:end:step), the step parameter is not currently supported and defaults to 1.

Unsupported example: a\[1:2:2, :\].

2. Boolean type indexing is not currently supported.

Unsupported example: a\[True, False, True, False\].

3. Tensor type indexing is not currently supported.

Unsupported example: a\[b\], \(b = pypto.Tensor\(\[2\], pypto.DT\_INT32\).

## Usage Examples

1. Full slice

   Use a slice to assemble a small Tensor into a specified position in a large Tensor.

   ```python
   a = pypto.Tensor([4, 4], pypto.DT_FP32)
   b = pypto.Tensor([2, 2], pypto.DT_FP32)
   a[0:, 0:] = b #equivalent to assemble(b, (0, 0), a)
   ```

   Example result:

   ```python
   input data a: [[0, 0, 0, 0],
               [0, 0, 0, 0],
               [0, 0, 0, 0],
               [0, 0, 0, 0]]
   input data b: [[10, 10]
               [10, 10]]
   output data a: [[10, 10, 0, 0],
               [10, 10, 0, 0],
               [0, 0, 0, 0],
               [0, 0, 0, 0]]
   ```

2. Mixed indexing and slicing

   Combine integer indexing and slices to operate on specific rows or columns.

   ```python
   a = pypto.Tensor([4, 4], pypto.DT_FP32)
   b = pypto.Tensor([2], pypto.DT_FP32)
   a[0, 1:3] = b #b is reshaped to (1, 2), equivalent to pypto.assemble(b, (0, 1), a)
   ```

   Example result:

   ```python
   input data a: [[0, 0, 0, 0],
               [0, 0, 0, 0],
               [0, 0, 0, 0],
               [0, 0, 0, 0]]
   input data b: [10, 10]
   output data a: [[0, 10, 10, 0],
               [0, 0, 0, 0],
               [0, 0, 0, 0],
               [0, 0, 0, 0]]
   ```

3. Negative indexing

   Supports Python-style negative indexing, counting from the end.

   ```python
   a = pypto.Tensor([4, 4], pypto.DT_FP32)
   b = pypto.Tensor([2], pypto.DT_FP32)
   a[-1, -3:-1] = b #equivalent to a[3, 1:3]
   ```

   Example result:

   ```python
   input data a: [[0, 0, 0, 0],
               [0, 0, 0, 0],
               [0, 0, 0, 0],
               [0, 0, 0, 0]]
   input data b: [10, 10]
   output data a: [[0, 0, 0, 0],
               [0, 0, 0, 0],
               [0, 0, 0, 0],
               [0, 10, 10, 0]]
   ```

4. Ellipsis (...)

   Use ... to automatically fill in intermediate dimensions.

   ```python
   a = pypto.Tensor([4, 4], pypto.DT_FP32)
   b = pypto.Tensor([2, 2], pypto.DT_FP32)
   a[..., 2:4] = b #equivalent to a[0:2, 2:4]
   ```

   Example result:

   ```python
   input data a: [[0, 0, 0, 0],
               [0, 0, 0, 0],
               [0, 0, 0, 0],
               [0, 0, 0, 0]]
   input data b: [[10, 10]
               [10, 10]]
   output data a: [[0, 0, 10, 10],
               [0, 0, 10, 10],
               [0, 0, 0, 0],
               [0, 0, 0, 0]]
   ```

5. Single element assignment

   Integer indexing to assign a value to a single element (only DT\_INT32 type is supported).

   ```python
   a = pypto.Tensor([4, 4], pypto.DT_INT32)
   a[2, 3] = 5 #calls SetTensorData
   ```

   Example result:

   ```python
   input data a: [[0, 0, 0, 0],
               [0, 0, 0, 0],
               [0, 0, 0, 0],
               [0, 0, 0, 0]]
   output data a: [[0, 0, 0, 0],
               [0, 0, 0, 0],
               [0, 0, 0, 5],
               [0, 0, 0, 0]]
   ```

6. Scatter operation

   When key is a slice, key.start is an int, and key.stop is a Tensor \(a\[start:stop\]\), a scatter operation is executed.

   ```python
   a = pypto.Tensor([4, 4], pypto.DT_FP32)
   indices = pypto.Tensor([1, 4], pypto.DT_INT32) # index Tensor
   values = pypto.Tensor([1, 4], pypto.DT_FP32)
   # scatter along dimension 0
   a[0:indices] = values #calls pypto.scatter(a, 0, indices, values)
   ```

   Example result:

   ```python
   input data a: [[0, 0, 0, 0],
               [0, 0, 0, 0],
               [0, 0, 0, 0],
               [0, 0, 0, 0]]
   input data indices: [[0, 1, 2, 3]]
   input data values: [[10, 10, 10, 10]]
   output data a: [[10, 0, 0, 0],
               [0, 10, 0, 0],
               [0, 0, 10, 0],
               [0, 0, 0, 10]]
   ```

