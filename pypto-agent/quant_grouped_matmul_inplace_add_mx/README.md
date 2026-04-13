# QuantGroupedMatmulInplaceAddAdd with MX Quantization

本算子实现了基于 MX 量化的分组矩阵乘法就地加法操作，专为 micro-batch 训练场景设计，用于高效的梯度累积。

---

## 功能说明

`quant_grouped_matmul_inplace_add_mx` 算子将 GroupedMatMul 和 InplaceAdd 融合，使用 MX 量化方式，用于 micro-batch 训练场景的梯度累计，提高网络性能。

该算子的核心特性：

1. **MX 量化**：使用 MXFP8 格式进行量化，数据使用 FP8E5M2/FP8E4M3FN 格式，缩放因子使用 FP8E8M0 格式
2. **分组计算**：支持将输入矩阵按不同分组进行计算
3. **就地加法**：输出张量同时作为输入，执行就地加法操作，减少内存拷贝
4. **高效融合**：通过 `scaled_mm` 算子实现缩放和矩阵乘法的融合计算
5. **新前端写法**：使用 PyPTO 新前端 API，支持类型注解和自动类型推导

**说明：**
<blockquote>MX 量化（Microscaling Quantization）是一种基于块缩放的量化格式。每 32 个元素（K 轴）共享一个缩放因子，缩放因子仅包含指数部分（FP8E8M0 格式），数据部分为 8 位浮点数（FP8E5M2/FP8E4M3FN 格式），适用于训练场景的高效量化计算。</blockquote>

## 计算公式

MX 量化场景下的计算公式：

$$
y_i[m,n] = \sum_{j=0}^{kLoops-1} ((\sum_{k=0}^{gsK-1} (x1Slice_i * x2Slice_i)) * (scale1_i[m, j] * scale2_i[j, n])) + y_i[m,n]
$$

其中：
- `gsK` = 32：K 轴的量化 block size
- `x1Slice_i`：`x1_i` 第 m 行长度为 gsK 的向量
- `x2Slice_i`：`x2_i` 第 n 列长度为 gsK 的向量
- K 轴从 `j*gsK` 起始切片，j 的取值范围 [0, kLoops)
- `kLoops` = ceil(`K_i` / gsK)，支持最后的切片长度不足 gsK

简化表示为：
$$
y = y + \text{ScaledMatmul}(x1, x2, \text{scale1}, \text{scale2})
$$

## 函数原型

```python
def quant_grouped_matmul_inplace_add_pypto(
    config: QuantGroupedMatmulInplaceAddConfig
) -> Callable
```

返回的 kernel 函数原型：

```python
def quant_grouped_matmul_inplace_add_impl(
    x1: pypto.Tensor,
    x2: pypto.Tensor,
    scale1: pypto.Tensor,
    scale2: pypto.Tensor,
    y: pypto.Tensor
) -> pypto.Tensor
```

## 参数说明

### 配置类 QuantGroupedMatmulInplaceAddConfig

| 参数名 | 类型 | 描述 | 默认值 |
|--------|------|------|--------|
| ori_shape | list | 原始形状 [M, K, N] | - |
| num_groups | int | 分组数量 | - |
| m_tile_shape | list | M 维度的 tile shape | - |
| k_tile_shape | list | K 维度的 tile shape | - |
| n_tile_shape | list | N 维度的 tile shape | - |
| vec_tile_shape | list | 向量操作的 tile shape | - |
| x1_dtype | pypto.DataType | x1 的数据类型 | DT_FP8E5M2 |
| x2_dtype | pypto.DataType | x2 的数据类型 | DT_FP8E5M2 |
| scale_dtype | pypto.DataType | scale 的数据类型 | DT_FP8E8M0 |
| out_dtype | pypto.DataType | 输出的数据类型 | DT_FP32 |
| description | str | 测试用例描述 | "" |

### Kernel 输入输出参数

| 参数名 | 输入/输出 | 描述 | 数据类型 | 数据格式 | 维度(shape) |
|--------|-----------|------|----------|----------|-------------|
| x1 | 输入 | 输入矩阵 1 | fp8e5m2/fp8e4m3fn | ND | [K, M] |
| x2 | 输入 | 输入矩阵 2 | fp8e5m2/fp8e4m3fn | ND | [K, N] |
| scale1 | 输入 | x1 的量化缩放因子 | fp8e8m0 | ND | [(K/64) + num_groups, M, 2] |
| scale2 | 输入 | x2 的量化缩放因子 | fp8e8m0 | ND | [(K/64) + num_groups, N, 2] |
| y | 输入输出 | 输入输出矩阵 | float32 | ND | [num_groups, M, N] |

## 约束说明

- **K 轴对齐**：K 轴必须 64 对齐（MX 量化要求）
- **分组限制**：最多支持 1024 个分组
- **维度限制**：x1 和 x2 的每一维大小在 32 字节对齐后应小于 int32 的最大值 (2147483647)
- **内轴限制**：内轴大小需小于 2097152
- **确定性**：默认确定性实现

## 调用示例

### 基本用法

```python
import pypto
import torch

# 定义配置
config = QuantGroupedMatmulInplaceAddConfig(
    ori_shape=[16, 64, 16],  # [M, K, N]
    num_groups=2,
    m_tile_shape=[16, 16],
    k_tile_shape=[64, 64],
    n_tile_shape=[16, 16],
    vec_tile_shape=[1, 8, 256, 32],
)

# 创建 kernel
kernel = quant_grouped_matmul_inplace_add_pypto(config)

# 创建输入张量
x1 = torch.randn(64, 16, dtype=torch.float8_e5m2, device='npu:0')
x2 = torch.randn(64, 16, dtype=torch.float8_e5m2, device='npu:0')
y = torch.randn(2, 16, 16, dtype=torch.float32, device='npu:0')

# 创建量化参数张量
scale1 = torch.ones((64//64 + 2, 16, 2), dtype=torch.float8_e8m0fnu, device='npu:0')
scale2 = torch.ones((64//64 + 2, 16, 2), dtype=torch.float8_e8m0fnu, device='npu:0')

# 调用 kernel
result = kernel(x1, x2, scale1, scale2, y)
```

### 运行测试

```python
# 运行基本测试
run_quant_grouped_matmul_inplace_add_case(
    QuantGroupedMatmulInplaceAddConfig(
        ori_shape=[16, 64, 16],
        num_groups=2,
        m_tile_shape=[16, 16],
        k_tile_shape=[64, 64],
        n_tile_shape=[16, 16],
        vec_tile_shape=[1, 8, 256, 32],
        description="Basic test with 2 groups"
    )
)
```

### 新前端写法特点

1. **类型注解**：在函数签名中使用 `pypto.Tensor(shape, dtype)` 进行类型注解
2. **自动编译**：使用 `@pypto.frontend.jit()` 装饰器自动编译
3. **返回值注解**：使用 `-> pypto.Tensor(shape, dtype)` 注解返回值类型
4. **配置驱动**：通过配置类管理所有参数，便于扩展和维护

```python
@pypto.frontend.jit()
def quant_grouped_matmul_inplace_add_impl(
    x1: pypto.Tensor(x1_shape, config.x1_dtype),
    x2: pypto.Tensor(x2_shape, config.x2_dtype),
    scale1: pypto.Tensor(scale1_shape, config.scale_dtype),
    scale2: pypto.Tensor(scale2_shape, config.scale_dtype),
    y: pypto.Tensor(y_shape, config.out_dtype)
) -> pypto.Tensor(y_shape, config.out_dtype):
    # 实现代码
    ...
    return y
```

## 性能优化建议

1. **Tile Shape 配置**：根据实际硬件和输入形状调整 tile shape
   - `m_tile_shape`：M 维度的 tile 大小，建议 [16, 16] 或 [32, 32]
   - `k_tile_shape`：K 维度的 tile 大小，建议 [64, 64]
   - `n_tile_shape`：N 维度的 tile 大小，建议 [16, 16] 或 [32, 32]

2. **Vector Tile Shape 配置**：用于向量操作的 tile 配置
   - `vec_tile_shape`：建议 [1, 8, 256, 32]

3. **内存对齐**：确保 K 轴 64 对齐，M 和 N 轴 32 字节对齐

4. **分组策略**：合理选择分组数，避免过多分组导致性能下降

## 与 CANN API 的对应关系

本 PyPTO 算子对应 CANN 的 `aclnnQuantGroupedMatmulInplaceAdd` 接口：

- **CANN 接口**：`aclnnQuantGroupedMatmulInplaceAdd`
- **量化方式**：MX 量化
- **产品支持**：Ascend 950PR/Ascend 950DT

## 常见问题

### Q: 为什么 K 必须被 64 整除？

A: 这是 MX 量化的要求。MX 量化使用 64 作为 K 轴的量化单位，scale 张量的形状也基于此设计。

### Q: scale 张量的形状是如何计算的？

A: scale1 的形状为 `[(K/64) + num_groups, M, 2]`，scale2 的形状为 `[(K/64) + num_groups, N, 2]`。其中：
- `K/64`：K 轴的量化块数
- `num_groups`：分组数
- `M` 或 `N`：对应的输出维度
- `2`：每个量化点需要 2 个值（scale 和可能的偏移）

### Q: 如何验证结果的正确性？

A: 可以使用提供的 `gen_golden_output` 函数进行对比验证，或使用测试用例中的 `run_quant_grouped_matmul_inplace_add_case` 函数。

### Q: 新前端写法有什么优势？

A: 新前端写法具有以下优势：
1. **类型安全**：通过类型注解提供编译时类型检查
2. **代码简洁**：减少样板代码，提高可读性
3. **易于维护**：配置类集中管理参数，便于扩展
4. **自动推导**：支持自动类型推导和形状推断

## 参考资源

- [CANN aclnnQuantGroupedMatmulInplaceAdd 文档](D:\cann\ops-transformer\gmm\quant_grouped_matmul_inplace_add\docs\aclnnQuantGroupedMatmulInplaceAdd.md)
- [PyPTO scaled_mm API 文档](D:\cann\pypto\docs\api\operation\pypto-scaled_mm.md)
- [PyPTO quant_matmul_reduce_sum 示例](D:\cann\pypto\models\experimental\matmul\quant_matmul_reduce_sum.py)
