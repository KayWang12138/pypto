# Tiling Configuration

Properly setting the TileShape is critical for optimizing operator performance. TileShape defines how data is partitioned across different compute units in the hardware, affecting data transfer efficiency and compute efficiency. By configuring TileShape appropriately, you can significantly improve compute performance, reduce data transfer overhead, and achieve efficient computation.

## Principles

The core of TileShape configuration is to partition data blocks reasonably according to hardware resources and computational requirements, in order to maximize hardware utilization and minimize data transfer overhead, thereby improving compute performance.

-   Vector computation: In vector computation, `set_vec_tile_shapes` is used to set the tile size along each dimension of the vector data. Appropriate tiling allows data to fully utilize the Unified Buffer (UB) and be processed efficiently on the vector compute unit.
-   Matrix computation: In matrix computation, denoting the matrix multiplication shape change as \(m, k\) x \(k, n\) = \(m, n\), `set_cube_tile_shapes` is used to set the tile size along the m, k, and n dimensions of the matrices in order. Appropriate tiling can fully utilize the L0 and L1 buffers and reduce data transfer overhead.

## Tiling Configuration for Vector Computation

`set_vec_tile_shapes` is used to set the TileShape for each dimension in vector computation.

```python
# Set TileShape for vector computation
pypto.set_vec_tile_shapes(1, 1, 8, 8)
# Get and print the set TileShape
print(pypto.get_vec_tile_shapes())  # Output: [1, 1, 8, 8]
```

`pypto.set_vec_tile_shapes(1, 1, 8, 8)` indicates that the vector has four dimensions, and each dimension is tiled according to sizes 1, 1, 8, 8 respectively. The original vector is transferred to the UB in tiles of size \(1, 1, 8, 8\) for computation.

An example use case is as follows:

```python
@pypto.frontend.jit
def compute_with_vec_tile_shapes_kernel(
    a: pypto.Tensor((32, 32), pypto.DT_FP32),
    b: pypto.Tensor((32, 32), pypto.DT_FP32),
    out: pypto.Tensor((32, 32), pypto.DT_FP32),
    set_shapes: tuple
):
    pypto.set_vec_tile_shapes(*set_shapes)
    out[:] = pypto.add(a, b)

def compute_with_vec_tile_shapes_op(a: torch.Tensor, b: torch.Tensor, set_shapes: tuple, dynamic: bool = False) -> torch.Tensor:
    # Invoke directly by passing torch tensors
    out = torch.empty_like(a)
    compute_with_vec_tile_shapes_kernel(a, b, out, set_shapes)
    return out

def test_set_vec_tile_shapes_basic():
    ...
    a = torch.tensor([[[1, 2, 3],
                       [1, 2, 3]]], dtype=dtype, device=f'npu:{device_id}')
    b = torch.tensor([[[4, 5, 6],
                       [4, 5, 6]]], dtype=dtype, device=f'npu:{device_id}')
    expected = torch.tensor([[[5, 7, 9],
                            [5, 7, 9]]], dtype=dtype, device=f'npu:{device_id}')
    set_shapes = (1, 2, 8)
    out = compute_with_vec_tile_shapes_op(a, b, set_shapes)
    assert_allclose(out.cpu().numpy(), expected.cpu().numpy(), rtol=1e-3, atol=1e-3)
```

The example above demonstrates the use of `set_vec_tile_shapes` in a simple vector addition scenario.

Note that different TileShape settings generally do not affect the computation result of the vector, but they do affect the runtime, as shown in the following example:

```python
@pypto.frontend.jit
def compute_with_vec_specific_tile_shapes_kernel(
    a: pypto.Tensor((4, 32, 64, 256), pypto.DT_FP32),
    b: pypto.Tensor((4, 32, 64, 256), pypto.DT_FP32),
    out: pypto.Tensor((4, 32, 64, 256), pypto.DT_FP32),
):
    pypto.set_vec_tile_shapes(1, 2, 4, 128)
    out[:] = pypto.add(a, b)

@pypto.frontend.jit
def compute_with_vec_another_tile_shapes_kernel(
    a: pypto.Tensor((4, 32, 64, 256), pypto.DT_FP32),
    b: pypto.Tensor((4, 32, 64, 256), pypto.DT_FP32),
    out: pypto.Tensor((4, 32, 64, 256), pypto.DT_FP32),
):
    pypto.set_vec_tile_shapes(2, 4, 8, 256)
    out[:] = pypto.add(a, b)

def compute_with_vec_specific_tile_shapes_op(a: torch.Tensor, b: torch.Tensor, dynamic: bool = False) -> torch.Tensor:
    out = torch.empty_like(a)
    compute_with_vec_specific_tile_shapes_kernel(a, b, out)
    return out

def compute_with_vec_another_tile_shapes_op(a: torch.Tensor, b: torch.Tensor, dynamic: bool = False) -> torch.Tensor:
    out = torch.empty_like(a)
    compute_with_vec_another_tile_shapes_kernel(a, b, out)
    return out

def test_set_vec_different_tile_shapes_runtime():
    ...
    a = torch.randn((4, 32, 64, 256), dtype=dtype, device=f'npu:{device_id}')
    b = torch.randn((4, 32, 64, 256), dtype=dtype, device=f'npu:{device_id}')
    TEST_TIME = 1
    start = time.perf_counter()
    for _ in range(TEST_TIME):
        out1 = compute_with_vec_specific_tile_shapes_op(a, b)
    runtime_1 = time.perf_counter() - start
    start = time.perf_counter()
    for _ in range(TEST_TIME):
        out2 = compute_with_vec_another_tile_shapes_op(a, b)
    runtime_2 = time.perf_counter() - start
    print(f"runtime_1(pypto.set_vec_tile_shapes(1, 2, 4, 128)): {runtime_1}")
    print(f"runtime_2(pypto.set_vec_tile_shapes(2, 4, 8, 256)): {runtime_2}")
```

In this example, for the addition of two vectors with shape \(4, 32, 64, 256\), the runtime of `set_vec_tile_shapes(1, 2, 4, 128)` is noticeably longer than that of `set_vec_tile_shapes(2, 4, 8, 256)`.

For a complete example, refer to: [tiling_config.py](../../../examples/01_beginner/tiling/tiling_config.py).

## Tiling Configuration for Cube Computation

`set_cube_tile_shapes` is used to set the TileShape along the m, k, and n dimensions for each matrix in matrix computation.

```python
# Set TileShape for Cube computation
pypto.set_cube_tile_shapes([16, 16], [256, 512], [128, 128])
# Get and print the set TileShape
print(pypto.get_cube_tile_shapes())  # Output: [[16, 16], [256, 512, 512], [128, 128]]
```

`pypto.set_cube_tile_shapes([16, 16], [256, 512], [128, 128])`: Denoting the matrix multiplication shape change as \(m, k\) x \(k, n\) = \(m, n\), the three lists set the tile sizes for the m, k, and n dimensions of the matrices respectively. For each list, the first element sets the tile size for L0 and the second sets the tile size for L1. When the last parameter is set to False, the L0 and L1 tile sizes must be equal; when the last parameter is set to True, the L0 tile size can be smaller than L1 but must be evenly divisible by the L1 tile size.

An example use case is as follows:

```python
@pypto.frontend.jit
def compute_with_cube_tile_shapes_kernel(
    a: pypto.Tensor((64, 64), pypto.DT_FP32),
    b: pypto.Tensor((64, 64), pypto.DT_FP32),
    out: pypto.Tensor((64, 64), pypto.DT_FP32),
    set_shapes: list
):
    pypto.set_cube_tile_shapes(*set_shapes)
    out[:] = pypto.matmul(a, b, a.dtype)

def compute_with_cube_tile_shapes_op(a: torch.Tensor, b: torch.Tensor, set_shapes: list, dynamic: bool = False) -> torch.Tensor:
    # Invoke directly by passing torch tensors
    out = torch.empty((64, 64), dtype=a.dtype, device=a.device)
    compute_with_cube_tile_shapes_kernel(a, b, out, set_shapes)
    return out

def test_set_cube_tile_shapes_basic():
    ...
    a = torch.tensor([[1, 2], [3, 4]], dtype=dtype, device=f'npu:{device_id}')
    b = torch.tensor([[5, 6], [7, 8]], dtype=dtype, device=f'npu:{device_id}')
    expected = torch.tensor([[19, 22], [43, 50]], dtype=dtype, device=f'npu:{device_id}')
    set_shapes = [[32, 32], [64, 64], [64, 64]]
    out = compute_with_cube_tile_shapes_op(a, b, set_shapes)
    assert_allclose(out.cpu().numpy(), expected.cpu().numpy(), rtol=1e-3, atol=1e-3)
```

The example above demonstrates the use of `set_cube_tile_shapes` in a simple matrix multiplication scenario.

Note that different TileShape settings generally do not affect the computation result of the matrix, but they do affect the runtime, as shown in the following example:

```python
import pypto
import torch
import time

@pypto.frontend.jit
def compute_with_cube_specific_tile_shapes_kernel(
    a: pypto.Tensor((4, 64, 512), pypto.DT_FP32),
    b: pypto.Tensor((4, 128, 512), pypto.DT_FP32),
    out: pypto.Tensor((4, 64, 128), pypto.DT_FP32),
):
    pypto.set_cube_tile_shapes([32, 32], [32, 32], [32, 32])
    out[:] = pypto.matmul(a, b, a.dtype, b_trans=True)

@pypto.frontend.jit
def compute_with_cube_another_tile_shapes_kernel(
    a: pypto.Tensor((4, 64, 512), pypto.DT_FP32),
    b: pypto.Tensor((4, 128, 512), pypto.DT_FP32),
    out: pypto.Tensor((4, 64, 128), pypto.DT_FP32),
):
    pypto.set_cube_tile_shapes([64, 64], [128, 128], [128, 128])
    out[:] = pypto.matmul(a, b, a.dtype, b_trans=True)

def compute_with_cube_specific_tile_shapes_op(a: torch.Tensor, b: torch.Tensor, dynamic: bool = False) -> torch.Tensor:
    out = torch.empty((4, 64, 128), dtype=a.dtype, device=a.device)
    compute_with_cube_specific_tile_shapes_kernel(a, b, out)
    return out

def compute_with_cube_another_tile_shapes_op(a: torch.Tensor, b: torch.Tensor, dynamic: bool = False) -> torch.Tensor:
    out = torch.empty((4, 64, 128), dtype=a.dtype, device=a.device)
    compute_with_cube_another_tile_shapes_kernel(a, b, out)
    return out

def test_set_cube_different_tile_shapes_runtime():
    ...
    a = torch.randn((4, 64, 512), dtype=dtype, device=f'npu:{device_id}')
    b = torch.randn((4, 128, 512), dtype=dtype, device=f'npu:{device_id}')
    TEST_TIME = 1
    start = time.perf_counter()
    for _ in range(TEST_TIME):
        out1 = compute_with_cube_specific_tile_shapes_op(a, b)
    runtime_1 = time.perf_counter() - start
    start = time.perf_counter()
    for _ in range(TEST_TIME):
        out2 = compute_with_cube_another_tile_shapes_op(a, b)
    runtime_2 = time.perf_counter() - start
    print(f"runtime_1(pypto.set_cube_tile_shapes([32, 32], [32, 32], [32, 32])): {runtime_1}")
    print(f"runtime_2(pypto.set_cube_tile_shapes([64, 64], [128, 128], [128, 128])): {runtime_2}")
```

In this example, for the multiplication of two matrices with shapes \(4, 64, 512\) and \(4, 512, 128\), the runtime of `set_cube_tile_shapes([32, 32], [32, 32], [32, 32])` is noticeably longer than that of `set_cube_tile_shapes([64, 64], [128, 128], [128, 128])`.

For a complete example, refer to: examples/01\_beginner/tiling/tiling\_config.py

## Constraints

-   When setting TileShape parameters, constraints must be satisfied: the values must match the number and size of dimensions of the tensor to be processed, and must not be too small or too large.

    TileShape values must not be too small. A TileShape that is too small will cause the number of tile iterations TensorShape/TileShape (i.e., the product of the per-dimension ratios of TensorShape to TileShape) to be too large, resulting in an excessively large number of online loop unrollings in the expression table. This may cause compilation of the expression table to fail and will increase runtime overhead. The size of the expression table is related to the number of online loop unrollings and the number of operator inputs. It is recommended to keep the value of \(TensorShape/TileShape\)\*\(1 + number\_of\_operator\_inputs\) below 18000.

    TileShape values must not be too large. A TileShape that is too large will exceed the storage capacity of the corresponding hardware (buffer). The size of the tiled data (product of the data type size and the sizes of each tiled dimension) must not exceed the storage capacity of the corresponding hardware unit.

    In addition, the number of dimensions for `set_vec_tile_shapes` must not exceed 4, and outer-axis tile sizes must satisfy 32-byte alignment. For `set_cube_tile_shapes`, kL0, kL1, nL0, and nL1 must all satisfy 32-byte alignment. For detailed configuration requirements, refer to the relevant interface documentation.

-   Setting TileShape affects runtime. In general, the more fully hardware unit capacity is utilized — i.e., the larger the amount of data computed in a single pass — the shorter the runtime. However, setting larger TileShape parameters does not necessarily mean faster on-board execution; the overhead of data transfer and other stages must also be considered.

## Other Operations

-   Performance observation: You can use performance analysis tools (such as a swimlane graph) to observe performance under different TileShape settings, in order to evaluate the reasonableness of TileShape configurations and find the optimal TileShape for the current scenario.
-   Accuracy impact: Extreme values aside, TileShape settings generally do not affect accuracy (behavior the framework does not intend to produce).

