# matmul 算子设计文档

> **算子名称**: matmul
> **算子分类**: matmul
> **生成时间**: 2026-03-28T23:35:00Z
> **基于**: spec.md, api_report.md

---

## 1. 概述

### 1.1 功能描述

矩阵乘法算子，支持 batch 维度广播的批量矩阵乘法。计算两个张量的矩阵乘积，其中最后两个维度执行标准矩阵乘法，前面的维度作为 batch 维度并支持广播。

**核心特性**：
- 2D 矩阵乘法：[M, K] @ [K, N] → [M, N]
- 3D/4D batch matmul：[B, M, K] @ [B, K, N] → [B, M, N]
- Batch 维度广播：[B1, 1, M, K] @ [1, B2, K, N] → [B1, B2, M, N]
- 动态轴支持：batch, M, N, K 可动态变化

### 1.2 数学公式

$$C = A @ B, \quad C_{ij} = \sum_k A_{ik} \cdot B_{kj}$$

### 1.3 算法描述

```
Algorithm: Batched Matrix Multiplication
────────────────────────────────────────
输入: A [..., M, K], B [..., K, N]
输出: C [..., M, N]

1. 解析输入 shape:
   1.1 提取 A 的最后两维: M, K
   1.2 提取 B 的最后两维: K, N
   1.3 对齐 batch 维度并计算广播后的 batch shape

2. 广播 batch 维度:
   2.1 比较 A 和 B 的 batch 维度
   2.2 将维度为 1 的轴扩展到对应大小
   2.3 确定输出 batch shape

3. 执行矩阵乘法:
   3.1 for each batch index (b1, b2, ...):
       3.1.1 C[b1,b2,...,:,:] = A[b1,b2,...,:,:] @ B[b1,b2,...,:,:]
       3.1.2 即 C[...,i,j] = sum_k(A[...,i,k] * B[...,k,j])

4. return C
```

### 1.4 数据流图

```
    输入 A                    输入 B
+-----------------+      +-----------------+
| [..., M, K]     |      | [..., K, N]     |
| float32/float16 |      | float32/float16 |
+--------+--------+      +--------+--------+
         |                        |
         |   +--------------------+
         v   v
    +----------------+
    |  C = A @ B     |
    | (batch matmul) |
    +-------+--------+
            v
      +-----------------+
      | 输出 C          |
      | [..., M, N]     |
      | float32/float16 |
      +-----------------+

动态轴: batch (可广播), M, N, K
约束: A 的 K 维度必须等于 B 的 K 维度
```

---

## 2. API 映射设计

### 2.1 数学公式分解

将公式拆解为基本操作步骤：

| 步骤 | 数学表达 | 说明 |
|------|----------|------|
| 1 | C = A @ B | 批量矩阵乘法，支持 batch 维度广播 |

### 2.2 PyPTO API 映射表

| 步骤 | 数学表达 | PyPTO API | 参数 | 文档路径 |
|------|----------|-----------|------|----------|
| 1 | C = A @ B | `pypto.matmul(input, mat2, out_dtype, *, a_trans=False, b_trans=False, c_matrix_nz=False, extend_params=None)` | input=A, mat2=B, out_dtype=输入dtype | `docs/api/operation/pypto-matmul.md` |

### 2.3 计算步骤序列

```python
# 伪代码展示计算流程
# 1. 设置 Cube Tiling（必须）
pypto.set_cube_tile_shapes([128, 128], [128, 128], [128, 128])

# 2. 3D/4D 场景需额外设置 Vector Tiling
if input_dims >= 3:
    pypto.set_vec_tile_shapes(128, 128)

# 3. 执行矩阵乘法
out[:] = pypto.matmul(a, b, out_dtype)
```

### 2.4 设计依据

- **来源**：spec.md（数学公式）、api_report.md（API 映射）、docs/api/operation/pypto-matmul.md（API 规格）
- **说明**：PyPTO 提供了完整的 `pypto.matmul` API，支持 2D/3D/4D 矩阵乘法、batch 广播、动态轴等所有需求特性。直接调用该 API 即可满足算子需求，无需额外组合操作。

---

## 3. 数据规格设计

### 3.1 OperatorInput dataclass

```python
@dataclass
class MatmulInput:
    a: Tensor  # 左操作数，shape: [..., M, K], dtype: float32/float16
    b: Tensor  # 右操作数，shape: [..., K, N], dtype: float32/float16
```

### 3.2 OperatorOutput dataclass

```python
@dataclass
class MatmulOutput:
    c: Tensor  # 输出矩阵，shape: [..., M, N], dtype: float32/float16
```

### 3.3 中间 Tensor 定义

无中间 Tensor。`pypto.matmul` 直接从输入计算到输出。

### 3.4 数据格式选择

| Tensor | 格式 | 说明 |
|--------|------|------|
| A, B, C | ND (TILEOP_ND) | 默认格式，适用于大多数场景。NZ 格式需要额外配置且 FP32 不支持。 |

### 3.5 动态轴定义

| 轴名称 | 含义 | 取值范围 | 标记方式 |
|--------|------|----------|----------|
| batch | 批量大小（可多维度） | [1, 65535] 每个维度 | `dynamic_axis=[0, 1, ...]` |
| M | 左矩阵行数 | [1, 65535] | `dynamic_axis=[-2]`（A） |
| N | 右矩阵列数 | [1, 65535] | `dynamic_axis=[-1]`（B） |
| K | 收缩维度 | [1, 65535] | `dynamic_axis=[-1]`（A）/ `[-2]`（B） |

**动态 shape 示例**：
```python
# 3D batch matmul，batch 动态
a = pypto.from_torch(a_tensor, dynamic_axis=[0])  # [B, M, K], B 动态
b = pypto.from_torch(b_tensor, dynamic_axis=[0])  # [B, K, N], B 动态

# 4D batch matmul，所有轴动态
a = pypto.from_torch(a_tensor, dynamic_axis=[0, 1, 2, 3])  # [B1, B2, M, K]
b = pypto.from_torch(b_tensor, dynamic_axis=[0, 1, 2, 3])  # [B1, B2, K, N]
```

### 3.6 JIT 装饰器配置

```python
@pypto.frontend.jit(
    runtime_options={"run_mode": pypto.RunMode.NPU}
)
def matmul_kernel(
    a: pypto.Tensor([], pypto.DT_FP32),
    b: pypto.Tensor([], pypto.DT_FP32),
    out: pypto.Tensor([], pypto.DT_FP32)
):
    ...
```

**配置说明**：
- `run_mode`：NPU 模式（生产环境）或 SIM 模式（调试）
- 无需特殊的 `pass_options` 或 `runtime_options` 配置

---

## 4. Tiling 策略

### 4.1 算子类型判断

- **类型**: Cube
- **判断依据**: 算子核心操作为矩阵乘法（matmul），根据 quick_ref.md §1.1，含 matmul 操作的算子属于 Cube 类型，必须调用 `pypto.set_cube_tile_shapes()`。3D/4D 场景还需额外调用 `pypto.set_vec_tile_shapes()` 处理 batch 维度。

### 4.2 TileShape 初值设置

```python
# 2D 场景
pypto.set_cube_tile_shapes([128, 128], [128, 128], [128, 128])

# 3D/4D 场景
pypto.set_cube_tile_shapes([128, 128], [128, 128], [128, 128])
pypto.set_vec_tile_shapes(128, 128)  # 处理 batch 维度
```

### 4.3 设置依据

**Cube Tiling（set_cube_tile_shapes）**：
- **初值选择**：`[128, 128], [128, 128], [128, 128]`
- **理由**：
  1. 满足 32 字节对齐要求：128 * 2 bytes (FP16) = 256 bytes，是 32 的倍数
  2. 平衡 L0/L1 容量：128x128 tile 在大多数硬件上可容纳
  3. 参考实现验证：`examples/01_beginner/tiling/tiling_config.py` 中展示了 [128, 128] 配置在中等规模矩阵上性能良好
- **对齐约束验证**：
  - FP16: kL0=128, kL1=128, nL0=128, nL1=128 均满足 16 元素对齐（16*2=32B）
  - FP32: kL0=128, kL1=128, nL0=128, nL1=128 均满足 8 元素对齐（8*4=32B）

**Vector Tiling（set_vec_tile_shapes）**：
- **初值选择**：`[128, 128]`
- **理由**：
  1. 3D/4D 场景必须设置，否则使用默认值 [128, 128]
  2. 显式设置可提高代码可读性和可维护性
  3. 128 是常见的 batch 维度大小，覆盖多数场景

### 4.4 注意事项

- **必须调用**：`pypto.matmul` 前必须调用 `pypto.set_cube_tile_shapes()`，否则编译失败
- **3D/4D 必须额外设置**：矩阵维度为 3D 或 4D 时，必须额外调用 `pypto.set_vec_tile_shapes()`
- **对齐约束**：所有 tiling 参数需满足 32 字节对齐（FP32 场景 16 元素对齐）
- **多核切 K 限制**：3D/4D 场景不支持 `enable_split_k=True`

### 4.5 判断依据与适用条件

- **判断依据**：初值 `[128, 128]` 基于：
  1. docs/api/config/pypto-set_cube_tile_shapes.md 中的对齐约束
  2. examples/01_beginner/tiling/tiling_config.py 中的性能验证
  3. 平衡 L0/L1 容量和计算效率
- **适用条件**：
  - 中等规模矩阵（M, N, K 在 64-512 范围）
  - FP16/FP32 dtype
  - 2D/3D/4D 矩阵乘法
- **不适用场景**：
  - 极小矩阵（M, N, K < 16）：可能需要更小的 tile shape
  - 极大矩阵（M, N, K > 4096）：可能需要更大的 tile shape 或启用 `enable_split_k`
  - 特殊 dtype（INT8, BF16）：对齐要求不同，需调整

---

## 5. Loop 结构设计

### 5.1 Loop 判断结论

- **结论**: 需要 Loop
- **原因**: 命中 quick_ref.md §2.1 条件 1：存在动态轴（运行时才知道大小）。spec.md §2 和 §8 明确声明支持 batch, M, N, K 动态变化。
- **Loop 类型**: `pypto.loop`
- **适用条件**: 3D/4D batch matmul 场景，batch 维度为动态轴
- **限制**: 仅用于动态 batch 维度；静态轴使用 Python for

### 5.2 静态轴 vs 动态轴处理

| 轴 | 类型 | 处理方式 |
|----|------|----------|
| batch 维度 (3D/4D) | 动态 | `pypto.loop(batch_size, name="LOOP_BATCH")` |
| M, N, K | 静态或动态 | 如为动态，使用 `pypto.loop`；如为静态，使用 Python range |

**示例代码**：
```python
# 动态 batch 维度
for b_idx in pypto.loop(batch_size, name="LOOP_BATCH", idx_name="b_idx"):
    a_batch = pypto.view(a, [m, k], [b_idx * m, 0])
    b_batch = pypto.view(b, [k, n], [b_idx * k, 0])
    c_batch = pypto.matmul(a_batch, b_batch, out_dtype)
    out[b_idx * m:(b_idx + 1) * m, :] = c_batch
```

### 5.3 Loop 合并策略

对于多维 batch（如 4D [B1, B2, M, K]），使用嵌套 loop：
```python
for b1_idx in pypto.loop(b1_size, name="LOOP_B1"):
    for b2_idx in pypto.loop(b2_size, name="LOOP_B2"):
        # 计算单个 batch
        ...
```

### 5.4 数据依赖处理

- **无跨迭代依赖**：每个 batch 的计算相互独立，无需维护中间状态
- **广播处理**：PyPTO 的 `pypto.matmul` 自动处理 batch 维度广播，无需手动实现

### 5.5 尾块处理策略

- **动态轴尾块**：使用 `pypto.view()` 的 `valid_shape` 参数指定实际计算范围
- **示例**：
  ```python
  actual_batch_size = batch_size  # 动态获取
  for b_idx in pypto.loop(actual_batch_size, name="LOOP_BATCH"):
      # 自动处理尾块
      ...
  ```

### 5.6 loop_unroll 配置

不使用 `pypto.loop_unroll`。原因：
- 动态轴范围 [1, 65535] 虽然跨度大，但 matmul 计算密集，不需要为不同 batch 大小生成多版本代码
- 使用 `pypto.loop` 即可满足需求

---

## 6. 验证方案

### 6.1 Golden 函数设计

```python
def matmul_golden(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """matmul 参考实现

    使用 PyTorch 内置 matmul，支持 2D/3D/4D 和 batch 广播。
    """
    return torch.matmul(a, b)
```

**置信度**：⭐⭐⭐⭐⭐（使用 PyTorch 内置 API）

### 6.2 测试用例设计

#### 基于 spec.md 所有典型配置

| 配置名称 | 类型 | 优先级 | 参数 | 输入 Shape | 输出 Shape | 说明 |
|----------|------|--------|------|------------|------------|------|
| 性能_P0 | 性能 | P0 | - | A:[1024,4096,4096], B:[1024,4096,4096] | C:[1024,4096,4096] | 大规模 batch matmul，性能关键场景 |
| 功能_P0 | 功能 | P0 | - | A:[1,128,64,512], B:[1,128,512,256] | C:[1,128,64,256] | 典型 Attention QK^T 场景 |
| 广播_P1 | 功能 | P1 | - | A:[8,1,1024,4096], B:[1,4,4096,1024] | C:[8,4,1024,1024] | batch 广播场景 |
| 2D_P0 | 功能 | P0 | - | A:[4096,4096], B:[4096,4096] | C:[4096,4096] | 标准 2D 矩阵乘法 |
| 动态_M | 功能 | P0 | - | A:[B,M,K], B:[B,K,N] (动态) | C:[B,M,N] | 动态 shape 验证 |

#### 边界情况测试

| 场景 | 参数 | 说明 |
|------|------|------|
| 零值输入 | A:[32,64] 全零 | 验证 0 乘以任何数等于 0 |
| 极小矩阵 | A:[16,32], B:[32,16] | 验证小矩阵计算正确性 |
| 极大矩阵 | A:[8192,8192], B:[8192,8192] | 验证大矩阵计算稳定性 |
| 包含负数 | A:[64,128]-0.5, B:[128,64]-0.5 | 验证负数输入处理 |
| 非对齐 shape | A:[17,33], B:[33,65] | 验证非 2 的幂次 shape |

### 6.3 精度验证标准

| Dtype | atol | rtol | 说明 |
|-------|------|------|------|
| float32 | 0.001 | 0.001 | 默认，高精度 |
| float16 | 0.01 | 0.01 | 混合精度场景 |

**验证代码示例**：
```python
import torch
from numpy.testing import assert_allclose

# 测试 2D_P0
a = torch.randn(4096, 4096, dtype=torch.float32)
b = torch.randn(4096, 4096, dtype=torch.float32)
c_golden = matmul_golden(a, b)
c_impl = matmul_impl(a, b)
assert_allclose(c_impl.cpu().numpy(), c_golden.cpu().numpy(), rtol=1e-3, atol=1e-3)
```

---

## 7. 性能指标与开箱配置

### 7.1 性能目标

基于 spec.md 典型配置（性能类）的预期性能：

| 配置名称 | 类型 | 优先级 | 参数 | 输入 Shape | 输出 Shape | 性能目标 |
|----------|------|--------|------|------------|------------|----------|
| 性能_P0 | 性能 | P0 | - | A:[1024,4096,4096], B:[1024,4096,4096] | C:[1024,4096,4096] | 首跑精度成功性能的 2 倍 |

**说明**：
- "首跑精度成功性能" 指首次通过精度验证时的 kernel 耗时
- "2 倍" 目标指优化后性能应达到首跑的 2 倍（即耗时减半）

### 7.2 开箱性能配置

```python
# 推荐 Tiling 配置（平衡性能与通用性）
# 2D 场景
pypto.set_cube_tile_shapes([128, 128], [128, 128], [128, 128])

# 3D/4D 场景
pypto.set_cube_tile_shapes([128, 128], [128, 128], [128, 128])
pypto.set_vec_tile_shapes(128, 128)
```

**性能调优建议**：
- 小矩阵（M, N, K < 64）：尝试 `[32, 32], [64, 64], [64, 64]`
- 大矩阵（M, N, K > 2048）：尝试 `[256, 256], [256, 256], [256, 256]`
- 参考 `examples/01_beginner/tiling/tiling_config.py` 中的性能对比实验

### 7.3 pass_options 配置

无特殊 pass_options 配置需求。

### 7.4 runtime_options 配置

```python
@pypto.frontend.jit(
    runtime_options={"run_mode": pypto.RunMode.NPU}
)
def matmul_kernel(...):
    ...
```

**说明**：
- 生产环境使用 `RunMode.NPU`
- 调试环境可使用 `RunMode.SIM`

---

## 8. 风险点与注意事项

### 8.1 已知约束

- **Tiling 配置必须**：调用 `pypto.matmul` 前必须调用 `pypto.set_cube_tile_shapes()`，否则编译失败
- **3D/4D 需 Vector Tiling**：矩阵维度为 3D 或 4D 时，必须额外调用 `pypto.set_vec_tile_shapes()`
- **Contiguous 要求**：输入 tensor 必须连续（`tensor.is_contiguous() == True`），否则 `from_torch` 失败
- **K 维度匹配**：A 的 K 维度（倒数第一维）必须等于 B 的 K 维度（倒数第二维），否则运行时错误
- **对齐约束**：Tiling 参数需满足 32 字节对齐（FP32 场景 16 元素对齐）

### 8.2 常见错误规避

| 风险 / 错误 | 触发场景 | 影响 / 原因 | 规避方法 |
|-------------|----------|-------------|----------|
| 未设置 cube_tile_shapes | 调用 matmul 前未调用 set_cube_tile_shapes | 编译失败，报错 "cube tile shapes not set" | 在 matmul 前添加 `pypto.set_cube_tile_shapes([128, 128], [128, 128], [128, 128])` |
| 3D/4D 未设置 vec_tile_shapes | 3D/4D 场景未调用 set_vec_tile_shapes | 使用默认 [128, 128]，可能性能不佳 | 显式调用 `pypto.set_vec_tile_shapes(128, 128)` |
| 输入不连续 | torch tensor 经过 transpose 等操作 | `from_torch` 失败，报错 "tensor must be contiguous" | 使用 `tensor.contiguous()` 确保连续 |
| K 维度不匹配 | A.shape[-1] != B.shape[-2] | 运行时错误，shape mismatch | 在 kernel 前检查 shape 一致性 |
| Tiling 对齐错误 | tile shape 不满足 32B 对齐 | 编译失败或性能劣化 | 确保 tile shape 为 16 的倍数（FP16）或 8 的倍数（FP32） |
| 多核切 K 误用 | 3D/4D 场景设置 enable_split_k=True | 编译失败，3D/4D 不支持多核切 K | 3D/4D 场景保持 `enable_split_k=False`（默认） |

### 8.3 特殊场景处理

**动态 shape 处理**：
- 使用 `pypto.from_torch(tensor, dynamic_axis=[...])` 标记动态维度
- 在 kernel 中使用 `pypto.loop()` 遍历动态 batch 维度
- 使用 `pypto.view()` 的 `valid_shape` 参数处理尾块

**batch 广播处理**：
- PyPTO 的 `pypto.matmul` 自动处理 batch 维度广播
- 无需手动实现广播逻辑

**混合 dtype 处理**：
- 当前设计仅支持相同 dtype 的输入输出
- 如需混合精度（如 FP16 输入、FP32 输出），设置 `out_dtype=pypto.DT_FP32`

### 8.4 实现建议

| 建议项 | 说明 |
|--------|------|
| 优先使用 PyPTO 内置 matmul | `pypto.matmul` 已优化，无需手动实现矩阵乘法逻辑 |
| Tiling 配置从 [128, 128] 开始 | 平衡性能与通用性，后续根据实际 shape 调优 |
| 动态 shape 必须标记 | 使用 `dynamic_axis` 参数，否则编译失败或性能不佳 |
| 验证覆盖所有典型配置 | 按 P0 → P1 顺序验证，确保核心场景通过 |
| 参考 examples/ 中的实现 | `examples/01_beginner/compute/matmul_ops.py` 提供了完整示例 |

---

## 9. 交付件清单

### 9.1 目录结构

```
custom/matmul/
├── spec.md                          # 需求规范（已有）
├── api_report.md                    # API 探索报告（已有）
├── design.md                        # 设计文档（本文件）
├── matmul_golden.py                 # Golden 参考实现（已有）
├── matmul_impl.py                   # 算子实现代码（待实现）
├── test_matmul.py                   # 测试代码（待实现）
└── output/                          # 运行输出（自动生成）
```

### 9.2 文件清单

| 文件 | 类型 | 说明 | 生成方式 |
|------|------|------|----------|
| spec.md | 需求 | 算子需求规范 | pypto-intent-understanding |
| api_report.md | 探索 | API 探索报告 | pypto-api-explorer |
| design.md | 设计 | 算子设计文档 | pypto-op-design（本 skill） |
| matmul_golden.py | 代码 | Golden 参考实现 | pypto-golden-generator |
| matmul_impl.py | 代码 | 算子核心实现 | 后续实现 |
| test_matmul.py | 代码 | 测试用例 | 后续实现 |

### 9.3 命名规范

| 项目 | 规范 | 示例 |
|------|------|------|
| 算子名称 | 小写字母 | `matmul` |
| 目录名 | 与算子名称一致 | `custom/matmul/` |
| Golden 文件 | `{op}_golden.py` | `matmul_golden.py` |
| 实现文件 | `{op}_impl.py` | `matmul_impl.py` |
| 测试文件 | `test_{op}.py` | `test_matmul.py` |

### 9.4 生成顺序

```
spec.md → api_report.md → design.md → matmul_golden.py → matmul_impl.py → test_matmul.py
```

**当前进度**：已完成 spec.md, api_report.md, design.md, matmul_golden.py

**下一步**：实现 matmul_impl.py（Stage 5）
