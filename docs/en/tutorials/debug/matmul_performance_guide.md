# Matmul High-Performance Programming
PyPTO provides users with an efficient and convenient operator development framework. For performance optimization, PyPTO offers a rich set of optional configurations, including operator tile configurations, algorithm configurations, and graph optimization configurations. These configuration options provide users with great development flexibility while also significantly raising the barrier to use.

This tutorial aims to provide users with a method for analyzing and optimizing matmul operators, helping users achieve better overall performance when developing operators with PyPTO.

## Performance Optimization Goal
Analyzing the theoretical performance upper bound is the first step in operator performance optimization. For any operator or algorithm, completing a certain computation generally involves two main parts: data transfer and data computation. The industry commonly uses arithmetic intensity to measure the ratio of these two parts, which is also an important basis for understanding the theoretical performance upper bound of an operator.

Arithmetic Intensity reflects the ratio of the amount of computation to the amount of data access for an operator or algorithm at the objective theoretical level. It is defined as total floating-point operations (FLOPs) divided by total memory access bytes, in units of `FLOPs/Byte`. For a specified hardware platform, defining the ratio of peak compute power to peak bandwidth as the compute-to-bandwidth ratio (with the same units as arithmetic intensity), when arithmetic intensity is greater than the compute-to-bandwidth ratio, the current algorithm's performance on that hardware platform can be considered compute-bound; otherwise, it is memory-bound.

The following further analyzes the performance upper bound of the matmul operator on a specified hardware platform.

Consider the data load volume (commonly called MTE2 transfer volume) and computation volume for a standard matrix multiplication. The condition for being compute-bound is:
$$
\frac{CP}{BW} \leq \frac{M \cdot N \cdot K \cdot 2}{M \cdot K \cdot \frac{N}{nL1} \cdot aByte + K \cdot N \cdot \frac{M}{mL1} \cdot bByte}
$$
In the above formula, $CP$ and $BW$ represent the peak compute power and bandwidth of the specified hardware platform; $mL1$ and $nL1$ represent the tile sizes in the M and N axes, which are the tile sizes the user needs to specify when calling `pypto.set_cube_tile_shapes`.

*Note: The theoretical formula above considers only the most basic pure inner product algorithm and temporarily ignores the data write-out overhead to simplify the analysis.*

It can be seen that the tile size directly determines the number of tasks after matmul tiling, which in turn determines the number of cores used and the number of computation rounds during actual execution. At the same time, the tile size theoretically determines the arithmetic intensity of the matmul operator.

Therefore, the first step in optimizing matmul performance is to optimize the tile configuration.

## Tile Configuration Optimization
#### Increasing Arithmetic Intensity
When M and N are sufficiently large (such as in training scenarios), the arithmetic intensity of matmul before tiling is large enough and the number of tasks after tiling is sufficient to fill all cores. In this case, we tend to choose larger tile configurations in both the M and N dimensions to achieve the highest possible arithmetic intensity after tiling, in hopes of reaching compute-bound.

Intuitively, a larger tile leads to higher arithmetic intensity, making it easier for matmul to reach compute-bound. This is because tiling inevitably introduces redundant transfers — more tiling means more redundant transfers and lower arithmetic intensity. On the other hand, tile size is constrained by on-chip multi-level cache space (L1, L0) and cannot be increased indefinitely.

Denote the tile size as:
```
pypto.set_cube_tile_shapes([mL0, mL1], [kL0, kL1], [nL0, nL1], enable_split_k=False)
```
Here, mL0, kL0, and nL0 represent the tile sizes in the L0 Buffer; mL1, kL1, and nL1 represent the tile sizes in the L1 Buffer.

Since the NPU's CUBE computation uses fractal data blocks as the minimum computation granularity, tile sizes must also satisfy fractal format requirements (outer axis aligned to 16 elements, inner axis aligned to 32 bytes). Additionally, tile sizes must satisfy buffer space constraints. For specific computation constraints, refer to: pypto\docs\api\config\pypto-set_cube_tile_shapes.md.

Taking the A2/A3 platforms as an example, for scenarios where both A and B matrices are of type FP16, the recommended tile configurations that satisfy buffer space constraints are:
```
pypto.set_cube_tile_shapes([128, 128], [64, 256], [256, 256], enable_split_k=False)
pypto.set_cube_tile_shapes([256, 256], [64, 256], [128, 128], enable_split_k=False)
pypto.set_cube_tile_shapes([128, 128], [128, 512], [128, 128], enable_split_k=False)
```

Advantages of the above tile configurations:
 - Can achieve high arithmetic intensity while satisfying L0 Buffer constraints;
$$
AI = \frac{M \cdot N \cdot K \cdot 2}{M \cdot K \cdot \frac{N}{nL1} \cdot aByte + K \cdot N \cdot \frac{M}{mL1} \cdot bByte}
$$
According to the formula above, AI is maximized when $mL1 = nL1$. Since $mL1 * nL1 * sizeof(float) <= L0C\_SIZE = 131072$, AI is maximized when $mL1 = nL1 = sqrt(L0C\_SIZE / sizeof(float)) = 181$. However, since tile sizes must satisfy fractal format alignment requirements, and the effect of tile size on write and read bandwidth must also be considered, combinations of 128-256 are generally chosen.
 - Both MTE2 and MTE1 transfers can enable double buffer, allowing pipeline parallelism;
 Under the above configuration, the L0A and L0B space occupancy is 32 KB, which is just enough to enable MTE1 double buffer. Similarly, under the above tile configurations, MTE2 can also enable double buffer. Additionally, since kL1 > kL0 enables large-block transfers, the amount of data per MTE2 transfer can be further increased, which helps improve MTE2 bandwidth utilization.
 However, note that when mL1 and nL1 use 128-256 combinations, nbuffer cannot be enabled on L0C. Therefore, this tile configuration is suitable for scenarios with a large K axis (i.e., relatively fewer write-out operations). When frequent write-outs are required, consider using a 128-128 combination.

It should be noted that the above tile configurations are not fixed; users need to make comprehensive decisions based on the computation scenario (considering input shape, dtype, format, etc.) and the hardware platform. In practice, if matmul is viewed as an algorithm, then tile configuration (tiling) is the most critical part of that algorithm.

At the same time, reaching compute-bound requires not only increasing arithmetic intensity but also maximizing memory access bandwidth. This is covered in the subsequent sections.

#### Reducing Redundant Loads
When one of M or N is relatively small (such as in inference scenarios), the arithmetic intensity of matmul before tiling is relatively low, and generally only memory-bound can be reached. In this case, the main optimization approach is to minimize redundant data loads (mainly MTE2 redundant loads) while ensuring all cores are fully utilized.

In general, the number of cores used should be at least 0.8 times the total number of cores (for a 24-core platform, at least 20 cores must be used). This ensures a relatively high utilization rate of MTE2 bandwidth. Expressed as a formula:
$$
\frac{M}{mL1} \cdot \frac{N}{nL1} >= 0.8 \cdot coreNum
$$
On this basis, reduce redundant loads as much as possible. When there is L1 tiling on the K axis, the total MTE2 load size is:
$$
MTE2\_LOAD\_SIZE = M \cdot K \cdot \frac{N}{nL1} \cdot aByte + K \cdot N \cdot \frac{M}{mL1} \cdot bByte
$$
In summary, the tile configuration problem becomes an extremum problem under constraints.

As an example, for an input specification of M = 96, K = 1536, N = 3072, data type FP16:
To avoid redundant MTE2 loads of the B matrix, a good tiling approach is to set mL1 = M = 96. Also, to fully utilize all cores, it is recommended to set nL1 = N / coreNum = 128. On this basis, further considering MTE2 and MTE1 pipeline parallelism and MTE2 bandwidth utilization, set kL1 = 4kL0.
In summary, a good tile configuration is:
```
pypto.set_cube_tile_shapes([96, 96], [64, 256], [128, 128], enable_split_k=False)
```

For further optimization, the A matrix can be loaded into L1 all at once and kept resident for repeated use. This reduces the total MTE2 load size to:
$$
MTE2\_LOAD\_SIZE = M \cdot K \cdot aByte + K \cdot N \cdot bByte
$$
This completely eliminates redundant MTE2 loads and further improves overall performance. The tile configuration in this case is:
```
pypto.set_cube_tile_shapes([96, 96], [64, 1536, 256], [128, 128], enable_split_k=False)
```
This uses a relatively advanced tile configuration approach: independently setting kAL1 and kBL1, in the form:
```
pypto.set_cube_tile_shapes([mL0, mL1], [kL0, kAL1, kBL1], [nL0, nL1], enable_split_k=False)
```
Here, kAL1 and kBL1 represent the tile sizes for the K axis of the A matrix and B matrix in L1, respectively. When only two K-axis tile values are specified, it means kAL1 = kBL1 = kL1.

#### K-Axis Core Splitting
For scenarios where M and N are small while the K axis is large, splitting cores only along the M and N axes may not fully utilize all cores, leading to poor overall performance. In such cases, K-axis core splitting can be used for optimization.

As an example, for an input specification of M = 128, K = 8192, N = 256, data type FP16, use the following tile configuration to implement K-axis core splitting. Set `enable_split_k` to True to enable K-axis core splitting, while retaining the large-block transfer configuration on the K axis to achieve pipeline parallelism:
```
pypto.set_cube_tile_shapes([128, 128], [64, 256], [128, 128], enable_split_k=True)
```

The pseudocode for K-axis splitting is as follows:
```
c = 0.0

for kIdx in range(K / kL1):
    for mIdx in range(M / mL1):
        for nIdx in range(N / nL1):
            c_partial = Matmul(A[mIdx * mL1 : mIdx * mL1 + mL1, kIdx * kL1 : kIdx * kL1 + kL1], B[kIdx * kL1 : kIdx * kL1 + kL1, nIdx * nL1 : nIdx * nL1 + nL1])

    c += c_partial
```
Use kL1 to tile the K axis; within a single core, only compute the partial sum for a length of kL1 and then write it out. Finally, accumulate all partial sums.

## Memory Access Bandwidth Optimization
#### Increasing L2 Cache Hit Rate
Returning to the compute-bound criterion formula:
$$
\frac{CP}{BW} \leq \frac{M \cdot N \cdot K \cdot 2}{M \cdot K \cdot \frac{N}{nL1} \cdot aByte + K \cdot N \cdot \frac{M}{mL1} \cdot bByte}
$$
The previous optimization measures focused primarily on increasing arithmetic intensity. This section focuses on how to improve bandwidth utilization to reduce the compute-to-bandwidth ratio. Considering the effect of L2 cache hit rate on comprehensive bandwidth, for a pure inner product algorithm, the total MTE2 load size is:
$$
MTE2\_TOTAL\_LOAD\_SIZE = M \cdot K \cdot \frac{N}{nL1} \cdot aByte + K \cdot N \cdot \frac{M}{mL1} \cdot bByte
$$
Of which the amounts loaded from HBM and L2 are:
$$
HBM\_LOAD\_SIZE = M \cdot K \cdot aByte + K \cdot N \cdot bByte
$$
$$
L2\_LOAD\_SIZE = MTE2\_TOTAL\_LOAD\_SIZE - HBM\_LOAD\_SIZE
$$
Define the L2 cache hit rate as the proportion of L2 load volume in the total load volume:
$$
l2\_hit\_ratio = \frac{L2\_LOAD\_SIZE}{MTE2\_TOTAL\_LOAD\_SIZE}=1-\frac{\frac{1}{N} \cdot aByte+\frac{1}{M} \cdot bByte}{\frac{1}{nL1} \cdot aByte+\frac{1}{mL1} \cdot bByte}
$$
For the A2/A3 platforms, the L2 bandwidth is more than 3 times the HBM bandwidth. Therefore, improving the L2 cache hit rate should be a priority.

Further considering the scenario where both A and B matrix data types are FP16, and considering the L2 cache hit rate for a single round:
$$
l2\_hit\_ratio = 1-\frac{\frac{1}{nDim \cdot nL1} +\frac{1}{mDim \cdot mL1} }{\frac{1}{nL1} +\frac{1}{mL1} }
$$
In the above formula, $mDim$ and $nDim$ are the number of cores in the M and N axes during a single round of computation. For a given tile configuration, the single-round L2 cache hit rate is maximized when $nDim \cdot nL1 = mDim \cdot mL1$.

Combining considerations of arithmetic intensity and L2 cache hit rate, and taking the A2/A3 platforms as an example: to maximize arithmetic intensity, we generally use a 128-256 tile configuration. Taking mL1 = 128 and nL1 = 256, the L2 cache hit rate is highest when $nDim \cdot 256 = mDim \cdot 128$. For a 24-core platform, mDim = 6, nDim = 4 or mDim = 8, nDim = 3 can be used.

As an example, for M = N = K = 6144 with FP16 data type, using only the 128-256 tile configuration, the overall latency is approximately 2.1 ms, equivalent to approximately 220 TFLOPS. The test code is as follows:
```
import torch
import torch_npu
import pypto

def create_mm_kernel(M, K, N):
    @pypto.frontend.jit(
        debug_options={"runtime_debug_mode": 1, "compile_debug_mode": 0}
    )
    def matmul_pto(
        a: pypto.Tensor([M, K], pypto.DT_FP16),
        b: pypto.Tensor([K, N], pypto.DT_FP16),
    ) -> pypto.Tensor([M, N], pypto.DT_FP16):
        pypto.set_cube_tile_shapes([128, 128], [64, 256], [256, 256], enable_split_k=False)
        tensorC = pypto.matmul(a, b, out_dtype=pypto.DT_FP16)
        return tensorC

    return matmul_pto

def test_native_mm():
    M = 6144
    K = 6144
    N = 6144

    a1 = torch.rand([M, K],  dtype=torch.float16)
    b1 = torch.rand([K, N],  dtype=torch.float16)
    c1 = create_mm_kernel(M, K, N)(a1.npu(), b1.npu())

if __name__ == "__main__":
    test_native_mm()
```

Further optimizing the L2 cache hit rate on top of the above tile configuration, the overall latency is approximately 1.6 ms, equivalent to approximately 290 TFLOPS. The test code is as follows:
```
import torch
import torch_npu
import pypto

def create_mm_kernel_with_l2_split(M, K, N, m_view, n_view):
    @pypto.frontend.jit(
        debug_options={"runtime_debug_mode": 1, "compile_debug_mode": 0}
    )
    def matmul_pto(
        a: pypto.Tensor([M, K], pypto.DT_FP16, format=pypto.TileOpFormat.TILEOP_ND),
        b: pypto.Tensor([K, N], pypto.DT_FP16, format=pypto.TileOpFormat.TILEOP_ND),
    ) -> pypto.Tensor([M, N], pypto.DT_FP16):
        pypto.set_cube_tile_shapes([128, 128], [64, 256], [256, 256], enable_split_k=False)

        m_loop = (M + m_view - 1) // m_view
        n_loop = (N + n_view - 1) // n_view
        outTensor = pypto.Tensor([M, N], pypto.DT_FP16)
        for m_idx in pypto.loop(0, m_loop, 1, name="LOOP_LO_mIdx", idx_name="m_idx"):
            for n_idx in pypto.loop(0, n_loop, 1, name="LOOP_LO_nIdx", idx_name="n_idx"):
                a_view = a[m_idx * m_view : m_idx * m_view + m_view, :]
                b_view = b[:, n_idx * n_view : n_idx * n_view + n_view]
                out_view = pypto.matmul(a_view, b_view, out_dtype=pypto.DT_FP16)
                outTensor[m_idx * m_view : m_idx * m_view + m_view, n_idx * n_view : n_idx * n_view + n_view] = out_view
        return outTensor

    return matmul_pto

def test_mm_with_l2_split():
    M = 6144
    K = 6144
    N = 6144

    mL1 = 128
    nL1 = 256
    mDim = 6
    nDim = 4

    m_view = mL1 * mDim
    n_view = nL1 * nDim

    a1 = torch.rand([M, K],  dtype=torch.float16)
    b1 = torch.rand([K, N],  dtype=torch.float16)
    c1 = create_mm_kernel_with_l2_split(M, K, N, m_view, n_view)(a1.npu(), b1.npu())

if __name__ == "__main__":
    test_mm_with_l2_split()
```
