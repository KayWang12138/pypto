# GroupedMatmulFinalizeRoutingV3 算子设计文档

> **算子名称**: GroupedMatmulFinalizeRoutingV3
> **算子分类**: MoE 融合算子（Cube + Vector 混合）
> **生成时间**: 2026-04-14
> **基于**: SPEC.md, API_REPORT.md, gmm_mxfp8.py

---

## 1. 概述

### 1.1 功能描述

GroupedMatmulFinalizeRoutingV3 是 MoE（Mixture of Experts）场景下的融合算子，将原本需要多个独立算子的操作合并为一个高效算子：

1. **分组矩阵乘法（GMM）**: 对每个专家组执行 MXFP8/MXFP4 量化矩阵乘法
2. **路由分配（Routing）**: 按 rowIndex 将专家输出 scatter add 到对应位置
3. **共享专家融合**: 将共享专家输出与 MoE 专家结果加权融合

适用场景：
- MoE 模型推理（GLM、LLaMA 等）
- 大规模专家模型的路由计算
- MXFP8/MXFP4 量化推理（Ascend 950PR/950DT）

### 1.2 数学公式

**整体计算流程**：

$$
y[rowIndex[i], :] = \sum_{i \in \mathcal{E}[j]} y_i [j - start_i] + sharedInputWeight \times sharedInput[j, :]
$$

**分解公式**：

1. 分组矩阵乘法：
$$
y_i = (x_i \times weight_i) \times scale_i \times perTokenScale_i
$$

2. 路由分配：
$$
out[rowIndex[j], :] \mathrel{+}= intermediate[j, :]
$$

3. 共享专家融合：
$$
out[row,:] \mathrel{+}= sharedInputWeight \times sharedInput[j,:]
$$

### 1.3 算法描述

```
Algorithm: GroupedMatmulFinalizeRoutingV3
─────────────────────────────────────────────────────────────
输入: x1(M,K), x2(E,K,N), scale(E,Ceil(K/64),N,2), 
      pertokenScale(M,Ceil(K/64),2), groupList(E), 
      rowIndex(M), logit(M), sharedInput(bsdp,N)
输出: out(batch,N)

Step 1: 初始化
    out = zeros(batch, N)
    intermediate = zeros(M, N)
    begin = 0, end = 0

Step 2: 分组矩阵乘法（遍历专家组）
    for i in [0, E):
        begin = end
        end = end + groupList[i]
        x_i = x1[begin:end, :]           # shape: (groupList[i], K)
        weight_i = x2[i]                  # shape: (K, N) or (N, K)
        scale_i = scale[i]                # shape: (Ceil(K/64), N, 2)
        pertoken_scale_i = pertokenScale[begin:end]  # shape: (groupList[i], Ceil(K/64), 2)
        
        # MXFP8 scaled_matmul
        intermediate[begin:end, :] = scaled_mm(x_i, weight_i, scale_i, pertoken_scale_i)

Step 3: 路由分配（Scatter Add）
    out = index_add_(out, dim=0, index=rowIndex, source=intermediate, alpha=1.0)

Step 4: 共享专家融合（可选）
    if sharedInput is not None:
        for j in [0, bsdp):
            target_row = sharedInputOffset + j
            out[target_row, :] += sharedInputWeight * sharedInput[j, :]

输出: out
```

### 1.4 数据流图

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│    x1       │────▶│   GMM       │────▶│intermediate │
│  (M, K)     │     │  (Loop E)   │     │  (M, N)     │
│  MXFP8      │     │ scaled_mm   │     │   FP32      │
└─────────────┘     └─────────────┘     └─────────────┘
                           │                    │
                           │                    │
┌─────────────┐           │                    │
│    x2       │───────────┤                    │
│  (E,K,N)    │           │                    │
│  MXFP8      │           │                    ▼
└─────────────┘           │              ┌─────────────┐
                          │              │  Routing    │
┌─────────────┐           │              │ index_add_  │
│   scale     │───────────┤              │  (Scatter)  │
│  (E,CeilK/64│           │              └─────────────┘
│   ,N,2)     │           │                    │
│  E8M0       │           │                    │
└─────────────┘           │                    ▼
                          │              ┌─────────────┐
┌─────────────┐           │              │    out      │
│pertokenScale│───────────┘              │ (batch, N)  │
│(M,CeilK/64, │                          │    FP32     │
│    2)       │                          └─────────────┘
│  E8M0       │                                 │
└─────────────┘                                 │
                                                │
┌─────────────┐                                 │
│  rowIndex   │─────────────────────────────────┤
│    (M)      │                                 │
│   INT64     │                                 ▼
└─────────────┘                          ┌─────────────┐
                                         │ Shared Fusion│
┌─────────────┐                          │   (可选)     │
│sharedInput  │──────────────────────────│ add + mul   │
│ (bsdp, N)   │                          └─────────────┘
│  BF16       │                                 │
└─────────────┘                                 ▼
                                         ┌─────────────┐
                                         │ Final out   │
                                         │ (batch, N)  │
                                         │    FP32     │
                                         └─────────────┘
```

---

## 2. API 映射设计

### 2.1 数学公式分解

将整体计算分解为原子操作步骤：

| 步骤 | 数学表达 | 说明 |
|------|----------|------|
| 1 | $begin_i = \sum_{j=0}^{i-1} groupList[j]$ | 计算专家起始位置 |
| 2 | $x_i = x1[begin_i:end_i, :]$ | 提取专家输入 |
| 3 | $weight_i = x2[i]$ | 提取专家权重 |
| 4 | $scale_i = scale[i]$ | 提取专家缩放因子 |
| 5 | $pertoken\_scale_i = pertokenScale[begin_i:end_i]$ | 提取 token 级缩放因子 |
| 6 | $y_i = scaled\_mm(x_i, weight_i, scale_i, pertoken\_scale_i)$ | MXFP8 矩阵乘法 |
| 7 | $intermediate[begin_i:end_i, :] = y_i$ | 存储中间结果 |
| 8 | $out = index\_add\_(out, 0, rowIndex, intermediate)$ | Scatter Add |
| 9 | $out = out + sharedInputWeight \times sharedInput$ | 共享专家融合 |

### 2.2 PyPTO API 映射表

| 步骤 | 数学表达 | PyPTO API | 参数 | 文档路径 |
|------|----------|-----------|------|----------|
| 1-5 | Tensor切片 | `pypto.Tensor.__getitem__` | `tensor[begin:end, :]` | docs/api/tensor/ |
| 6 | scaled_mm | `pypto.scaled_mm` | `mat_a, mat_b, out_dtype, scale_a, scale_b` | docs/api/math/pypto-scaled_mm.md |
| 6-pre | Tile配置 | `pypto.set_cube_tile_shapes` | `m_tile, k_tile, n_tile` | docs/api/config/pypto-set_cube_tile_shapes.md |
| 6-pre | Vector配置 | `pypto.set_vec_tile_shapes` | `*args` | docs/api/config/pypto-set_vec_tile_shapes.md |
| 7 | Tensor赋值 | `pypto.Tensor.__setitem__` | `tensor[begin:end, :] = value` | docs/api/tensor/ |
| 8 | Scatter Add | `pypto.index_add_` | `input, dim, index, source, alpha` | docs/api/tensor/pypto-index_add_.md |
| 9 | 加权融合 | `pypto.mul` + `pypto.add` | `input, other, alpha` | docs/api/math/pypto-mul.md |

### 2.3 计算步骤序列

```python
@pypto.frontend.jit
def grouped_matmul_finalize_routing_v3_kernel(
    x1, x2, scale, pertoken_scale, group_list,
    row_index, out, shared_input=None,
    shared_input_weight=1.0, shared_input_offset=0,
    tile_config
):
    # Step 1-7: 分组矩阵乘法（遍历专家组）
    num_experts = x2.shape[0]
    begin = 0
    end = 0
    
    for i in pypto.loop(0, num_experts):  # 需要动态循环（groupList 运行时确定）
        begin = end
        end = end + group_list[i]
        
        # Tensor切片：提取专家数据
        x = x1[begin:end, :]
        weight = x2[i]
        scaled_x = pertoken_scale[begin:end, :, :]
        scaled_weight = scale[i]
        
        # 设置 Vector Tile Shapes（处理 scale）
        pypto.set_vec_tile_shapes(
            tile_config.vector_tile_shape[0],
            tile_config.vector_tile_shape[1],
            tile_config.vector_tile_shape[2],
            tile_config.vector_tile_shape[3]
        )
        
        # 设置 Cube Tile Shapes（scaled_mm）
        pypto.set_cube_tile_shapes(
            tile_config.m_tile_shape,
            tile_config.k_tile_shape,
            tile_config.n_tile_shape
        )
        
        # MXFP8 scaled_matmul
        intermediate[begin:end, :] = pypto.scaled_mm(
            x, weight, pypto.DT_FP32, scaled_x, scaled_weight
        )
    
    # Step 8: 路由分配（Scatter Add）
    pypto.set_vec_tile_shapes(1, tile_config.n_tile_shape[0])
    out = pypto.index_add_(out, 0, row_index, intermediate, alpha=1.0)
    
    # Step 9: 共享专家融合（可选）
    if shared_input is not None:
        for j in pypto.loop(0, shared_input.shape[0]):
            target_row = shared_input_offset + j
            weighted = pypto.mul(shared_input[j, :], shared_input_weight)
            out[target_row, :] = pypto.add(out[target_row, :], weighted)
```

### 2.4 设计依据

- **来源**: API_REPORT.md + gmm_mxfp8.py 参考实现 + docs/
- **API选择理由**:
  - `scaled_mm`: 唯一支持 MXFP8/MXFP4 量化矩阵乘法的 API，支持 DT_FP8E8M0 缩放因子
  - `index_add_`: 专为 scatter add 场景设计，性能优于循环+切片方案
  - `set_cube_tile_shapes`: scaled_mm 的前置必要配置（硬约束）
  - `pypto.loop`: groupList 在运行时确定，需动态循环遍历专家组

---

## 3. 数据规格设计

### 3.1 OperatorInput dataclass

```python
from dataclasses import dataclass
from typing import Optional, List
import pypto

@dataclass
class GroupedMatmulFinalizeRoutingV3Input:
    # 必选参数（MXFP8场景）
    x1: pypto.Tensor              # 输入左矩阵, shape: (M, K), dtype: FLOAT8_E4M3FN/E5M2
    x2: pypto.Tensor              # 权重矩阵, shape: (E, K, N) or (E, N, K), dtype: FLOAT8_E4M3FN/E5M2
    scale: pypto.Tensor           # 权重缩放因子, shape: (E, Ceil(K/64), N, 2), dtype: FLOAT8_E8M0
    pertoken_scale: pypto.Tensor  # Token级缩放因子, shape: (M, Ceil(K/64), 2), dtype: FLOAT8_E8M0
    group_list: List[int]         # 专家分组列表, shape: (E)
    row_index: pypto.Tensor       # 路由索引, shape: (M), dtype: INT64
    logit: pypto.Tensor           # MoE logit, shape: (M), dtype: FLOAT32
    
    # 可选参数
    bias: Optional[pypto.Tensor] = None           # 偏置, shape: (E, N), dtype: BFLOAT16
    shared_input: Optional[pypto.Tensor] = None   # 共享专家输出, shape: (bsdp, N), dtype: BFLOAT16
    
    # 配置参数
    batch: int                    # 输出 batch 维度
    shared_input_weight: float = 1.0    # 共享专家融合系数
    shared_input_offset: int = 0        # 共享专家偏移
    transpose_x1: bool = False          # x1 是否转置（仅支持 False）
    transpose_x2: bool = False          # x2 是否转置
    group_list_type: int = 1            # 分组模式：0=cumsum, 1=count
```

### 3.2 OperatorOutput dataclass

```python
@dataclass
class GroupedMatmulFinalizeRoutingV3Output:
    out: pypto.Tensor  # 最终输出, shape: (batch, N), dtype: FLOAT32
```

### 3.3 中间 Tensor 定义

| 名称 | Shape | Dtype | 说明 | 存储位置 |
|------|-------|-------|------|----------|
| intermediate | (M, N) | FLOAT32 | GMM 中间结果（专家输出拼接） | GM（全局内存） |
| weighted_shared | (bsdp, N) | FLOAT32 | 加权后的共享专家输出 | UB（临时） |

**Workspace 估算**：
- intermediate: M × N × 4 bytes（FP32）
- 典型配置（M=16, N=7168）：16 × 7168 × 4 = 460 KB

### 3.4 数据格式选择

| Tensor | 格式 | 说明 |
|--------|------|------|
| x1 | ND | MXFP8 标准格式，不支持 NZ |
| x2 | ND | MXFP8 标准格式，非转置为 (E,K,N)，转置为 (E,N,K) |
| scale | ND | FLOAT8_E8M0 格式，需与 x2 转置属性一致 |
| pertoken_scale | ND | FLOAT8_E8M0 格式 |
| intermediate | ND | FP32 输出格式 |
| out | ND | FP32 输出格式 |

**格式约束**：
- MXFP8/MXFP4 场景仅支持 ND 格式，不支持 NZ
- 所有输入 tensor 必须连续（is_contiguous() == True）

### 3.5 动态轴定义

| 轴名称 | 含义 | 取值范围 | 动态原因 |
|--------|------|----------|----------|
| M (x1.shape[0]) | Token 数量 | [1, 16×1024×8] | 由 groupList 累加决定 |
| groupList[i] | 每个专家处理的 token 数 | [0, M] | 运行时动态分配 |
| batch (out.shape[0]) | 输出 batch 维度 | [0, INT32_MAX] | 由 rowIndex 和 sharedInputOffset 决定 |

**动态轴处理策略**：
- 使用 `pypto.loop` 遍历专家组（groupList 运行时确定）
- 使用 `pypto.DYNAMIC` 标记 M 维度，支持编译期不确定的场景

### 3.6 JIT 装饰器配置

```python
@pypto.frontend.jit(
    runtime_options={
        "run_mode": pypto.RunMode.NPU,  # NPU 模式（MXFP8 需真实 NPU）
    }
)
def grouped_matmul_finalize_routing_v3_kernel(
    inputs: GroupedMatmulFinalizeRoutingV3Input
) -> GroupedMatmulFinalizeRoutingV3Output:
    ...
```

---

## 4. Tiling 策略

### 4.1 算子类型判断

- **类型**: Cube + Vector 混合算子
- **判断依据**:
  - 核心计算 scaled_mm 是 Cube 操作（矩阵乘法）
  - index_add_ 和 add/mul 是 Vector 操作
  - 需同时设置 cube_tile_shapes 和 vec_tile_shapes
- **占比估算**: Cube（scaled_mm）占比 70%+，Vector（routing + fusion）占比 30%-

### 4.2 TileShape 初值设置

参考 gmm_mxfp8.py 的成熟配置：

```python
# Cube Tile Shapes（用于 scaled_mm）
m_tile_shape = [9, 9]      # [mL0, mL1] - M 维度切分
k_tile_shape = [256, 256]  # [kL0, kL1] - K 维度切分（32字节对齐）
n_tile_shape = [256, 256]  # [nL0, nL1] - N 维度切分（32字节对齐）

pypto.set_cube_tile_shapes(m_tile_shape, k_tile_shape, n_tile_shape)

# Vector Tile Shapes（用于 scale 处理）
vector_tile_shape = [1, 8, 256, 32]  # 处理 scale tensor

pypto.set_vec_tile_shapes(
    vector_tile_shape[0],
    vector_tile_shape[1],
    vector_tile_shape[2],
    vector_tile_shape[3]
)
```

### 4.3 设置依据

**Cube TileShape 选择理由**：

| 参数 | 值 | 理由 |
|------|-----|------|
| kL0 = 256 | K 维度切分基础块 | 32字节对齐（FP8: 256×1=256B > 32B），匹配 MXFP8 块大小（64元素对齐） |
| nL0 = 256 | N 维度切分基础块 | 32字节对齐，提高 L0B 利用率，减少切分次数 |
| mL0 = 9 | M 维度切分基础块 | 参考典型配置 groupList 平均值，适配动态 M 维度 |
| kL1 = kL0 | L1 切分等于 L0 | 减少层级切分复杂度，简化编译 |
| nL1 = nL0 | L1 切分等于 L0 | 同上 |

**Vector TileShape 选择理由**：

- 用于处理 scale tensor（shape: (M, Ceil(K/64), 2)）
- [1, 8, 256, 32] 适配 scale 的多维结构
- 确保 UB 内存不溢出

**约束验证**：

```python
# 对齐验证
kL0 % 16 == 0   # ✓ 256 % 16 = 0
nL0 % 16 == 0   # ✓ 256 % 16 = 0
kL1 % kL0 == 0  # ✓ 256 % 256 = 0
nL1 % nL0 == 0  # ✓ 256 % 256 = 0

# Buffer 空间估算（FP8 输入）
L0A_usage = ceil(9, 16) * ceil(256, 16) * 1 = 9 * 256 * 1 = 2304 B
L0B_usage = ceil(256, 16) * ceil(256, 16) * 1 = 256 * 256 * 1 = 64 KB
L0C_usage = ceil(9, 16) * ceil(256, 16) * 4 = 9 * 256 * 4 = 9 KB (FP32)
# 符合 L0 buffer 容量约束（A2/A3 系列约 128KB）
```

### 4.4 注意事项

- **mL0 动态适配**: 当 groupList[i] < mL0 时，需处理尾块（运行时自动调整）
- **K 维度对齐**: MXFP8 要求 K % 64 == 0，否则需要 padding
- **UB 内存**: index_add_ 需确保 input + source + index 总大小不超过 UB
- **scale 形状匹配**: scale 的转置属性必须与 x2 保持一致

### 4.5 判断依据与适用条件

- **判断依据**: 参考 gmm_mxfp8.py 的成熟配置（已验证可行）
- **适用条件**:
  - MXFP8 场景（Ascend 950PR/950DT）
  - K 维度 >= 256，N 维度 >= 256
  - 专家数量 E <= 1024
- **不适用场景**:
  - MXFP4 场景（K/N 有偶数约束，需调整）
  - 极小 M 维度（groupList 平均值 < 4，需减小 mL0）
  - K/N 维度小于 256（需减小 kL0/nL0）

---

## 5. Loop 结构设计

### 场景 B：需要 Loop

#### 5.1 Loop 判断结论

- **结论**: 需要 Loop
- **原因**:
  1. groupList 在运行时确定（动态轴）
  2. 多步骤分组计算（遍历 E 个专家组）
  3. 每个专家组处理的 token 数不同（动态 M 维度）
- **Loop 类型**: `pypto.loop`
- **适用条件**: 专家数量 E > 1，groupList 运行时已知
- **限制**: E <= 1024（硬件约束）

#### 5.2 静态轴 vs 动态轴处理

| 轴 | 类型 | 处理方式 | 说明 |
|----|------|----------|------|
| 专家索引 i | 静态（E 固定） | Python for 或 pypto.loop | E 编译期已知时可使用 Python for |
| groupList[i] | 动态 | pypto.loop 迭代变量 | 运行时确定每个专家的 token 数 |
| M 维度 | 动态 | 不直接 loop，通过 groupList 累加 | M = sum(groupList) |

**推荐写法**：

```python
# 方案 A: pypto.loop（推荐，支持动态边界）
for i in pypto.loop(0, num_experts):
    begin = end
    end = end + group_list[i]
    ...

# 方案 B: Python for（仅当 E 编译期已知且固定）
for i in range(num_experts):
    begin = end
    end = end + group_list[i]
    ...
```

#### 5.3 Loop 合并策略

- **专家组 Loop**: 遍历 E 个专家执行 GMM
- **共享专家 Loop**: 单独处理（可选）
- **不建议合并**: 两个 Loop 处理不同数据流，合并会增加复杂度

#### 5.4 数据依赖处理

- **GMM Loop**: 每个专家组独立计算，写入 intermediate 不同位置，无依赖
- **Scatter Add**: 等待所有 GMM 完成，读取完整 intermediate
- **融合**: 等待 Scatter Add 完成，读取完整 out

**数据流依赖图**：

```
GMM Loop (并行) → intermediate[begin:end] → Scatter Add → out → Shared Fusion
```

#### 5.5 尾块处理策略

当 groupList[i] < mL0 时（尾块场景）：
- **自动处理**: scaled_mm 内部自动调整 TileShape
- **无需手动**: PyPTO 编译器会自动处理尾块切分
- **注意**: 确保 intermediate 正确写入对应位置

#### 5.6 loop_unroll 配置

**不推荐使用**: 动态轴范围跨度大但由 groupList 决定，不适合编译期多版本生成

---

## 6. 验证方案

### 6.1 Golden 函数设计

已生成 `grouped_matmul_finalize_routing_v3_golden.py`，包含：

```python
def grouped_matmul_finalize_routing_v3_golden(
    x1, x2, scale, pertoken_scale, group_list, row_index, logit,
    batch, n, bias=None, shared_input=None,
    shared_input_weight=1.0, shared_input_offset=0,
    transpose_x1=False, transpose_x2=False, group_list_type=1
) -> torch.Tensor:
    """
    纯 PyTorch 参考实现，包含：
    1. MXFP8 缩放矩阵乘法（32倍广播 + padding）
    2. Scatter Add（index_add_）
    3. 共享专家融合
    """
```

**关键实现细节**：
- K 维度对齐：Ceil(k/32) % 2 != 0 时裁剪 scale
- Scale 广播：repeat_interleave 32 倍扩展
- Padding 处理：匹配广播后的 K 维度

### 6.2 测试用例设计

#### 基于 SPEC.md 典型配置

| 配置名称 | 类型 | 优先级 | 参数 | 输入 Shape | 输出 Shape | 说明 |
|----------|------|--------|------|------------|------------|------|
| MXFP8_基础 | 功能 | P0 | e=2, group=[7,9], batch=8 | x1(16,512), x2(2,512,7168) | (8,7168) | 基础验证 |
| MXFP8_共享专家 | 功能 | P1 | e=2, group=[7,9], shared=True | x1(16,512), x2(2,512,7168), shared(4,7168) | (8,7168) | 含共享专家 |
| MXFP8_转置 | 功能 | P1 | e=2, transpose_x2=True | x1(16,512), x2(2,7168,512) | (8,7168) | 权重转置 |
| MXFP8_bias | 功能 | P2 | e=2, bias=True | x1(16,512), x2(2,512,7168), bias(2,7168) | (8,7168) | 含 bias |
| MXFP8_大专家 | 性能 | P0 | e=8, group=[16,16,...], batch=32 | x1(128,512), x2(8,512,7168) | (32,7168) | 性能测试 |
| MXFP8_大K | 性能 | P1 | e=4, k=2048 | x1(16,2048), x2(4,2048,4096) | (8,4096) | 大 K 维度 |
| MXFP8_空tensor | 功能 | P2 | m=0, group=[0,0] | x1(0,512), x2(2,512,7168) | (8,7168) | 空 tensor |

#### 边界情况测试

| 场景 | 参数 | 说明 |
|------|------|------|
| rowIndex 重复映射 | row_index 多个 token 映射到同一行 | 验证 scatter add 正确累加 |
| groupList 不均匀 | [1, 15] 极端分配 | 验证尾块处理 |
| sharedInputOffset 越界 | offset + bsdp > batch | 验证边界处理 |
| MXFP4 场景 | k=1024(偶数), n=4096(偶数), float4_e2m1 | MXFP4 量化验证 |

### 6.3 精度验证标准

| Dtype | atol | rtol |
|-------|------|------|
| FLOAT32（输出） | 1e-3 | 1e-3 |
| BFLOAT16（shared_input） | 1e-2 | 1e-2 |

**验证方法**：
1. 生成 MXFP8 格式输入数据
2. 计算 Golden 结果（纯 PyTorch FP32）
3. 计算 PyPTO 结果（NPU）
4. 使用 `numpy.testing.assert_allclose` 对比

---

## 7. 性能指标与开箱配置

### 7.1 性能目标

基于 SPEC.md 典型配置（性能类）的预期性能：

| 配置名称 | 类型 | 优先级 | 参数 | 输入 Shape | 输出 Shape | 预期 kernel 耗时 |
|----------|------|--------|------|------------|------------|------------------|
| MXFP8_大专家 | 性能 | P0 | e=8, m=128 | (128,512), (8,512,7168) | (32,7168) | < 5ms（参考 gmm_mxfp8） |
| MXFP8_大K | 性能 | P1 | e=4, k=2048 | (16,2048), (4,2048,4096) | (8,4096) | < 2ms |

**性能优化方向**：
- 减少 Loop 内部的 TileShape 切分次数
- 优化 index_add_ 的 UB 使用
- 考虑 shared expert fusion 原地操作

### 7.2 开箱性能配置

```python
# TileConfig dataclass
@dataclass
class TileConfig:
    m_tile_shape: list = [9, 9]
    k_tile_shape: list = [256, 256]
    n_tile_shape: list = [256, 256]
    vector_tile_shape: list = [1, 8, 256, 32]
    
    # 可选调优参数
    enable_split_k: bool = False  # 大 K 维度可启用
    submit_before_loop: bool = True  # 提前提交减少同步
```

### 7.3 pass_options 配置

```python
pass_options = {
    "submit_before_loop": True,  # 在 loop 开始前提交计算
}
```

### 7.4 runtime_options 配置

```python
runtime_options = {
    "run_mode": pypto.RunMode.NPU,  # MXFP8 需真实 NPU
    # 可选：动态轴范围提示
    "dynamic_axis_range": {
        "M": [1, 16*1024*8],
        "E": [1, 1024],
    }
}
```

---

## 8. 风险点与注意事项

### 8.1 已知约束

1. **MXFP8/MXFP4 数据类型约束**:
   - x1/x2 必须使用 FLOAT8_E4M3FN 或 FLOAT8_E5M2（MXFP8）
   - scale/pertokenScale 必须使用 FLOAT8_E8M0
   - 仅 Ascend 950PR/950DT 支持 MXFP8/MXFP4 量化

2. **K/N 维度对齐约束**:
   - MXFP8: K 维度需满足 64 元素对齐（K % 64 == 0）
   - MXFP4: K 必须为偶数且 K ≠ 2，N（非转置）必须为偶数

3. **scale 转置属性匹配**:
   - scale 的 shape 必须与 x2 的转置属性一致
   - 非转置: scale shape (E, Ceil(K/64), N, 2)
   - 转置: scale shape (E, N, Ceil(K/64), 2)

4. **Tensor 连续性**:
   - 所有输入 tensor 必须连续（is_contiguous() == True）
   - 不支持非连续 tensor

5. **index_add_ 约束**:
   - dim 轴不可切，需全载
   - index 必须为 INT32 或 INT64
   - UB 内存需容纳 input + source + index

### 8.2 常见错误规避

| 风险 / 错误 | 触发场景 | 影响 / 原因 | 规避方法 |
|-------------|----------|-------------|----------|
| K 维度不对齐 | K % 64 != 0 | scaled_mm 编译失败或精度异常 | 生成数据时确保 K % 64 == 0 |
| scale 形状不匹配 | transpose_x2 与 scale shape 不一致 | 计算结果错误 | 严格按 SPEC 生成 scale shape |
| rowIndex 类型错误 | rowIndex 非 INT64 | index_add_ 报错 | 确保 row_index dtype = INT64 |
| UB 内存溢出 | index_add_ TileShape 过大 | 编译失败或运行时错误 | 合理设置 vec_tile_shapes |
| 空 tensor 未处理 | M=0 或 groupList[i]=0 | 累加逻辑异常 | Golden 和实现都需处理空 tensor |
| Scale 广播错误 | 32 倍广播未正确执行 | MXFP8 量化精度异常 | 参考 golden 中的 repeat_interleave |
| Padding 未对齐 | Ceil(k/32) % 2 != 0 | 计算维度不匹配 | 参考 golden 中的裁剪逻辑 |

### 8.3 特殊场景处理

1. **空 Tensor 场景**:
   - M=0: intermediate 为空，直接初始化 out 并进行 shared fusion
   - groupList[i]=0: 跳过该专家组的 GMM 计算

2. **共享专家缺失**:
   - sharedInput=None: 直接返回 Scatter Add 结果
   - 无需执行 Step 9

3. **MXFP4 场景**:
   - K 维度必须为偶数（K % 2 == 0）
   - N 维度（非转置）必须为偶数
   - 数据类型使用 FLOAT4_E2M1

4. **groupListType 模式**:
   - cumsum 模式(0): groupList = [3, 7, 10] 表示前缀和
   - count 模式(1): groupList = [3, 4, 3] 表示每个专家的计数
   - 需在实现中正确转换

### 8.4 实现建议

| 建议项 | 说明 |
|--------|------|
| 参考 gmm_mxfp8.py | 循环结构、TileShape 设置、scaled_mm 调用方式均可复用 |
| 优先使用 index_add_ | 性能优于循环+切片方案，专为 scatter add 设计 |
| intermediate 在 GM 分配 | 避免 UB 溢出，M×N 可能较大 |
| bias 处理 | scaled_mm 后直接 add bias_i（专家级 bias） |
| shared fusion 原地操作 | 使用 add API 原地更新 out，减少数据搬运 |

---

## 9. 交付件清单

### 9.1 目录结构

```
custom/GroupedMatmulFinalizeRoutingV3/
├── SPEC.md                                    # 需求规范（已有）
├── API_REPORT.md                              # API 探索报告（已有）
├── DESIGN.md                                  # 设计文档（本文件）
├── grouped_matmul_finalize_routing_v3_golden.py  # Golden 参考实现（已有）
├── grouped_matmul_finalize_routing_v3_impl.py    # 算子实现代码（待生成）
├── test_grouped_matmul_finalize_routing_v3.py    # 测试代码（待生成）
└── output/                                    # 运行输出（自动生成）
```

### 9.2 文件清单

| 文件 | 类型 | 说明 | 生成方式 |
|------|------|------|----------|
| SPEC.md | 需求 | 算子需求规范 | pypto-intent-understand |
| API_REPORT.md | 分析 | API 映射探索报告 | pypto-api-explore |
| DESIGN.md | 设计 | 算子设计文档 | pypto-op-design（本 skill） |
| grouped_matmul_finalize_routing_v3_golden.py | 代码 | Golden 参考实现 | pypto-golden-generate |
| grouped_matmul_finalize_routing_v3_impl.py | 代码 | 算子核心实现 | pypto-op-develop |
| test_grouped_matmul_finalize_routing_v3.py | 代码 | 测试用例 | pypto-op-develop |

### 9.3 命名规范

| 项目 | 规范 | 示例 |
|------|------|------|
| 算子名称 | 大驼峰（类名风格） | GroupedMatmulFinalizeRoutingV3 |
| 目录名 | 与算子名称一致 | custom/GroupedMatmulFinalizeRoutingV3/ |
| Golden 文件 | `{op}_golden.py` | grouped_matmul_finalize_routing_v3_golden.py |
| 实现文件 | `{op}_impl.py` | grouped_matmul_finalize_routing_v3_impl.py |
| 测试文件 | `test_{op}.py` | test_grouped_matmul_finalize_routing_v3.py |

### 9.4 生成顺序

```
SPEC.md → API_REPORT.md → DESIGN.md → golden.py → impl.py → test.py
  │         │              │           │          │         │
  │         │              │           │          │         └─ 验证精度
  │         │              │           │          └─────────── 实现算子
  │         │              │           └─────────────────────── 生成 Golden
  │         │              └─────────────────────────────────── 设计文档（本文件）
  │         └────────────────────────────────────────────────── API 探索
  └─────────────────────────────────────────────────────────── 需求规格
```

---

## 10. 关键设计点确认

### 10.1 Loop 结构确认

- **结论**: 需要 `pypto.loop` 遍历专家组
- **原因**: groupList 运行时确定，每个专家处理的 token 数动态变化
- **实现方式**: `for i in pypto.loop(0, num_experts)`
- **数据依赖**: 各专家独立计算，写入 intermediate 不同位置

### 10.2 Tiling 策略确认

- **Cube TileShapes**: [mL0=9, kL0=256, nL0=256]
- **Vector TileShapes**: [1, 8, 256, 32]
- **依据**: 参考 gmm_mxfp8.py 成熟配置
- **约束**: K/N 维度 32 字节对齐，符合 L0 buffer 容量

### 10.3 API 映射确认

- **核心 API**: `scaled_mm`（MXFP8）+ `index_add_`（Scatter）
- **辅助 API**: `set_cube_tile_shapes`, `set_vec_tile_shapes`, `Tensor.__getitem__`
- **无 unsupported 项**: 所有操作均有对应 PyPTO API

---

**文档版本**: v1.0
**生成日期**: 2026-04-14
**参考实现**: /mnt/workspace/gitCode/cann/pypto/models/experimental/matmul/gmm_mxfp8.py
**产品支持**: Ascend 950PR/950DT（MXFP8/MXFP4），Atlas A2/A3（伪量化）