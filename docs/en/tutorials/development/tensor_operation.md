# Tensor Operations

## Supported Mathematical Operations

PyPTO provides a comprehensive set of operations for tensor computation, designed to offer users efficient and flexible computational capabilities. These operations are divided into vector operations and matrix operations, and can be combined to implement more complex computation logic.

### Vector Operations

Vector operations perform element-wise computations on tensors and are suitable for a variety of basic mathematical operations.

-   Arithmetic operations

    ```python
    # Addition
    result = pypto.add(a, b)
    result = pypto.add(a, scalar)  # Add a scalar to a tensor
    result = pypto.add(a, b, alpha=2.0)  # a + 2.0 * b
    result = a + b

    # Subtraction
    result = pypto.sub(a, b)
    result = pypto.sub(a, scalar)
    result = pypto.sub(a, b, alpha=2.0)  # a - 2.0 * b
    result = a - b

    # Multiplication
    result = pypto.mul(a, b)
    result = pypto.mul(a, scalar)
    result = a * b

    # Division
    result = pypto.div(a, b)
    result = pypto.div(a, scalar)
    result = a / b

    # Exponentiation
    result = pypto.pow(a, scalar)  # a ** scalar
    ```

-   Mathematical functions

    ```python
    # Exponential and logarithm
    result = pypto.exp(x)      # e ** x
    result = pypto.log(x)      # ln(x)
    result = x.exp()
    result = x.log()

    # Square root
    result = pypto.sqrt(x)     # √x
    result = pypto.rsqrt(x)    # 1/√x
    result = x.sqrt()
    result = x.rsqrt()

    # Trigonometric functions
    result = pypto.sin(x)
    result = pypto.cos(x)
    result = x.sin()
    result = x.cos()

    # Absolute value
    result = pypto.abs(x)
    result = x.abs()

    # Negation
    result = pypto.neg(x)      # -x
    result = x.neg()
    ```

-   Activation functions

    ```python
    # Sigmoid
    result = pypto.sigmoid(x)  # 1 / (1 + exp(-x))
    result = x.sigmoid()

    # ReLU variants
    result = pypto.relu(x)     # max(0, x)
    result = pypto.gelu(x)     # GELU activation
    result = x.relu()
    result = x.gelu()

    # Softmax
    result = pypto.softmax(x, dim=-1)  # Softmax along the dim dimension
    result = x.softmax(x, dim=-1)
    ```

-   Comparison operations

    ```python
    # Maximum and minimum
    result = pypto.maximum(a, b)  # Element-wise maximum
    result = pypto.minimum(a, b)  # Element-wise minimum

    # Clipping
    result = pypto.clip(x, min_val, max_val)  # Clip the value range of x
    ```

-   Reduction operations

    ```python
    # Sum
    result = pypto.sum(x, dim=-1, keepdim=False)
    result = x.sum(dim=-1, keepdim=False)

    # Maximum
    result = pypto.amax(x, dim=-1, keepdim=False)
    result = x.amax(dim=-1, keepdim=False)

    # Minimum
    result = pypto.amin(x, dim=-1, keepdim=False)
    result = x.amin(dim=-1, keepdim=False)
    ```

-   In-place modification \(inplace\)

    ```python
    # In-place operation using move()
    output.move(pypto.add(a, b))  # Efficient, no copy

    # Or using assignment
    output[:] = pypto.add(a, b)   # Also efficient
    ```

-   Broadcast pattern

    Many operations support broadcasting, making operations more flexible and efficient.

    ```python
    # Tensor + scalar (scalar broadcast)
    result = pypto.add(tensor, 2.0)

    # Tensor + 1D tensor (1D tensor broadcast)
    bias = pypto.tensor([features], pypto.DT_BF16, "bias")
    result = pypto.add(tensor, bias)
    ```

### Matrix Operations

Matrix operations are optimized for the Cube cores of the NPU and are suitable for large-scale matrix computations.

```python
# Basic matrix multiplication
# C = A @ B, where A: [M, K], B: [K, N], C: [M, N]
result = pypto.matmul(A, B, out_dtype=pypto.DT_BF16)

# With transpose, where A: [M, K], B: [N, K]
result = pypto.matmul(A, B, out_dtype=pypto.DT_BF16, a_trans=False, b_trans=True)

# Batched matrix multiplication
# A: [B, M, K], B: [B, K, N], result: [B, M, N]
result = pypto.matmul(A, B, out_dtype=pypto.DT_BF16)

# With bias
bias = pypto.tensor([1, N], pypto.DT_BF16, "bias")
result = pypto.matmul(A, B, out_dtype=pypto.DT_BF16, extend_params={'bias_tensor': bias})

# With transpose, where A: [M, K], B: [N, K], output in NZ format
result = pypto.matmul(A, B, out_dtype=pypto.DT_BF16, a_trans=False, b_trans=True， c_matrix_nz=True)
```

Matrix multiplication parameters:

-   `input`: Left matrix \[M, K\] or \[B, M, K\]
-   `mat2`: Right matrix \[K, N\] or \[B, K, N\]
-   `out_dtype`: Output data type
-   `a_trans`: Transpose the left matrix (default: False)
-   `b_trans`: Transpose the right matrix (default: False)
-   `c_matrix_nz`: Output in NZ format (default: False)
-   `extend_params`: Extended features (bias, dequantization, etc.).

### Composite Operations

The basic operations above can be combined to implement more complex computation logic.

```python
def softmax_core(x: pypto.Tensor) -> pypto.Tensor:
    row_max = pypto.amax(x, dim=-1, keepdim=True)  # Compute row maximum
    sub = x - row_max                              # Value normalization
    exp = pypto.exp(sub)                           # Exponential operation
    esum = pypto.sum(exp, dim=-1, keepdim=True)    # Summation
    return exp / esum                              # Probability normalization

def softmax_kernel(x: pypto.Tensor, y: pypto.Tensor) -> None:
    ...
    for idx in pypto.loop(b_loop):
        ...
        softmax_out = softmax_core(x_view)
        ...
```

## Supported Logical Structure Transformations

Logical structure transformation operations allow users to perform shape, dimension, and type transformations on tensors to accommodate different computational requirements.

### Views and Assembly

-   View: A view operation creates a new tensor reference to the same underlying data, suitable for tile-based processing and local computations.

    ```python
    # View
    view = pypto.view(tensor, view_shape, offset, valid_shape)
    ```

    Example:

    ```python
    # Create a view with a specific shape and offset
    view = pypto.view(
        tensor,
        view_shape=[32, 32],
        offset=[10, 20],
        valid_shape=[actual_h, actual_w]  # optional
    )
    ```

    Parameters:

    -   tensor: Source tensor
    -   view\_shape: Shape of the view
    -   offset: Starting position in the source tensor
    -   valid\_shape: Actual valid size (used for boundary handling)

    Tensors support Python-style indexing and slicing for flexible data access.

    ```python
    tensor = pypto.tensor([10, 20], pypto.DT_FP16, "tensor")

    # Single element (creates a view)
    element = tensor[0, 0]  # Only supports INT32

    # Slicing (creates a view)
    slice_tensor = tensor[0:5, 10:20]

    # Ellipsis
    ellipsis_slice = tensor[..., 0:10]
    ```

-   Assembly: The assemble function places a smaller tensor into a larger tensor at a specified offset, suitable for merging results after tile-based processing.

    ```python
    # Assemble a small tensor into a large tensor
    pypto.assemble(
        small_tensor,      # Source tensor
        offsets=[10, 20],  # Target position
        large_tensor       # Destination tensor
    )
    ```

    Example:

    ```python
    # Small tile result tensor
    tile_result = pypto.tensor([32, 32], pypto.DT_FP16, "tile")

    # Large output tensor
    output = pypto.tensor([100, 200], pypto.DT_FP16, "output")

    # Assemble into the output tensor at position [10, 20]
    pypto.assemble(tile_result, [10, 20], output)
    ```

### Reshaping / Dimension Manipulation

```python
# Reshape a tensor
reshaped = pypto.reshape(tensor, [new_shape])

# Transpose
transposed = pypto.transpose(tensor, dim0=0, dim1=1)
```

Reshaping does not change the data, only the view of the dimensions.

### Type Conversion

```python
# Convert to a different data type
result = pypto.cast(tensor, pypto.DT_FP32, mode=pypto.CastMode.CAST_NONE)
```




