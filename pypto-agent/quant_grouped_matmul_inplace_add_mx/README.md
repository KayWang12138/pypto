# Grouped Matrix Multiplication with MXFP8 Quantization

本算子实现了基于 MXFP8 量化的分组矩阵乘法就地加法操作，支持多分组场景下的高效梯度累积计算。

---

## 产品支持情况

| 产品 | 是否支持 |
|:-----|:--------:|
| Ascend 950PR/Ascend 950DT | √ |

---

## 功能说明

`quant_grouped_matmul_inplace_add_mx` 算子将 GroupedMatMul 和 InplaceAdd 融合，使用 MXFP8 量化方式，支持按 K 轴分组进行矩阵乘法计算并就地累加到输出张量。

该算子的核心特性：

1. **MXFP8 量化**：数据支持 FP8E4M3FN 和 FP8E5M2 格式，缩放因子使用 FP8E8M0 格式
2. **灵活的数据格式**：支持转置（a_trans=True）和非转置（a_trans=False）两种输入矩阵格式
3. **K 轴分组**：支持将 K 轴按不同分组切分，每个分组对应不同的权重矩阵块，支持累加方式和累计值方式两种分组模式
4. **就地加法**：输出张量同时作为输入，执行就地加法操作 `y = y + result`，减少内存拷贝
5. **融合计算**：通过 `scaled_mm` 算子实现缩放和矩阵乘法的融合计算，计算公式为 `(mat_a * scale_a) @ (mat_b * scale_b)`
6. **新前端写法**：使用 PyPTO 新前端 API，支持类型注解和自动类型推导

### 支持的数据类型

| 数据类型 | PyPTO DataType | Torch dtype | 格式说明 | 适用场景 |
|----------|----------------|-------------|----------|----------|
| 输入数据 | DT_FP8E4M3 | torch.float8_e4m3fn | 4位指数+3位尾数，精度更高 | 训练场景，精度优先 |
| 输入数据 | DT_FP8E5M2 | torch.float8_e5m2 | 5位指数+2位尾数，动态范围更大 | 推理场景，动态范围优先 |
| 缩放因子 | DT_FP8E8M0 | torch.float8_e8m0fnu | 8位纯指数格式，仅包含指数部分 | MX 量化专用 |
| 输出数据 | DT_FP32 | torch.float32 | 标准 FP32 格式 | 累加输出 |

### 支持的矩阵格式

| 参数配置 | 输入矩阵形状 | Scale 形状 | 内轴（需32字节对齐） |
|----------|-------------|-----------|---------------------|
| a_trans=True | a=[K, M] | scaled_a=[(K//64)+g, M, 2] | M 维度 |
| a_trans=False | a=[M, K] | scaled_a=[M, (K//64)+g, 2] | K 维度（自动满足） |
| b_trans=False | b=[K, N] | scaled_b=[(K//64)+g, N, 2] | N 维度 |
| b_trans=True | b=[N, K] | scaled_b=[N, (K//64)+g, 2] | K 维度（自动满足） |

### MXFP8 量化说明

MX 量化（Microscaling Quantization）是一种基于块缩放的量化格式：

- **量化块大小**：每 64 个元素（K 轴）共享一个缩放因子
- **缩放因子格式**：FP8E8M0（8位纯指数），仅包含指数部分，隐含 mantissa=1.0
- **数据格式**：FP8E4M3FN（4位指数+3位尾数）或 FP8E5M2（5位指数+2位尾数）
- **Scale 存储格式**：连续存储，所有分组的 scale 存储在同一 tensor 中，分组间有间隔

---

## 计算公式

对于每个分组 i，计算公式为：

$$
y_i = y_i + ((a_i \times scale_{a_i}) @ (b_i \times scale_{b_i}))
$$

其中：
- `a_i` 和 `b_i` 为第 i 个分组的输入和权重矩阵块
- `scale_{a_i}` 和 `scale_{b_i}` 为对应的 MXFP8 量化缩放因子
- `@` 表示矩阵乘法
- 输出 `y` 形状为 `[num_groups, M, N]`，每个分组对应一个输出矩阵

分组切分由 `group_list` 和 `group_type` 决定：
- `group_type=0`：`group_list` 各元素为累计 K 值，最后一个元素等于 K
- `group_type=1`：`group_list` 各元素为单独的 group size，累加得到完整 K 轴

---

## 函数原型

```python
@pypto.frontend.jit
def scaled_matmul_kernel(
    a: pypto.Tensor(),
    b: pypto.Tensor(),
    scaled_a: pypto.Tensor(),
    scaled_b: pypto.Tensor(),
    y: pypto.Tensor(),
    tile_config: ShapeConfig
) -> None
```

---

## 参数说明

### 配置类 ShapeConfig

| 参数名 | 类型 | 描述 | 默认值 |
|--------|------|------|--------|
| ori_shape | list | 原始形状 [M, K, N] | - |
| group_list | list | K 轴分组列表 | - |
| m_tile_shape | list | M 维度的 tile shape [mL0, mL1] | - |
| k_tile_shape | list | K 维度的 tile shape [kL0, kL1] | - |
| n_tile_shape | list | N 维度的 tile shape [nL0, nL1] | - |
| vector_tile_shape | list | 向量操作的 tile shape | - |
| group_type | int | group_list 解释方式（0: 累计值方式, 1: 累加方式） | 0 |
| in_dtype | pypto.DataType | 输入数据类型（DT_FP8E4M3 或 DT_FP8E5M2） | DT_FP8E4M3 |
| a_trans | bool | 输入矩阵是否转置 | True |
| b_trans | bool | 权重矩阵是否转置 | False |
| a_format_nz | bool | 输入是否使用 NZ 格式 | False |
| b_format_nz | bool | 权重是否使用 NZ 格式 | False |
| c_format_nz | bool | 输出是否使用 NZ 格式 | False |
| description | str | 测试用例描述 | "" |

### in_dtype 数据类型说明

| pypto.DataType | torch dtype | 描述 | 适用场景 |
|----------------|-------------|------|----------|
| DT_FP8E4M3 | torch.float8_e4m3fn | E4M3FN 格式，4位指数3位尾数 | 训练场景，精度优先 |
| DT_FP8E5M2 | torch.float8_e5m2 | E5M2 格式，5位指数2位尾数 | 推理场景，动态范围优先 |

### group_list 和 group_type 说明

| group_type | 含义 | 示例 | 说明 |
|------------|------|------|------|
| 0 | 累计值方式 | `[256, 512]` | 各元素为累计 K 值，最后一个元素等于 K=512 |
| 1 | 累加方式 | `[256, 256]` | 各元素为单独的 group size，累加得到 K=512 |

### a_trans 和 b_trans 矩阵格式说明

| 参数 | 值 | 矩阵形状 | Scale 形状 | Scale_trans |
|------|-----|---------|-----------|-------------|
| a_trans | True | a=[K, M] | scaled_a=[(K//64)+g, M, 2] | scale_a_trans=True |
| a_trans | False | a=[M, K] | scaled_a=[M, (K//64)+g, 2] | scale_a_trans=False |
| b_trans | True | b=[N, K] | scaled_b=[N, (K//64)+g, 2] | scale_b_trans=True |
| b_trans | False | b=[K, N] | scaled_b=[(K//64)+g, N, 2] | scale_b_trans=False |

**注意：**
- `a_trans=True` 时，M 维度是内轴，需要 32 字节对齐
- `a_trans=False` 时，K 维度是内轴，但 MX 量化已要求 K>=64，自动满足对齐
- `b_trans=False` 时，N 维度是内轴，需要 32 字节对齐
- `b_trans=True` 时，K 维度是内轴，自动满足对齐

### Kernel 输入输出参数

| 参数名 | 输入/输出 | 描述 | 数据类型 | 维度(shape) |
|--------|-----------|------|----------|-------------|
| a | 输入 | 输入矩阵 | FP8E4M3 或 FP8E5M2 | a_trans=True: [K, M]; a_trans=False: [M, K] |
| b | 输入 | 权重矩阵 | FP8E4M3 或 FP8E5M2 | b_trans=True: [N, K]; b_trans=False: [K, N] |
| scaled_a | 输入 | a 的量化缩放因子 | FP8E8M0 | a_trans=True: [(K//64)+g, M, 2]; a_trans=False: [M, (K//64)+g, 2] |
| scaled_b | 输入 | b 的量化缩放因子 | FP8E8M0 | b_trans=True: [N, (K//64)+g, 2]; b_trans=False: [(K//64)+g, N, 2] |
| y | 输入输出 | 输入输出矩阵 | FP32 | [num_groups, M, N] |

其中 `g` 为分组数量（`len(group_list)`）。a 和 b 的数据类型由 `in_dtype` 参数决定。

---

## MXFP8 Scale 存储格式

Scale 张量采用连续存储格式，所有分组的 scale 存储在同一个 tensor 中：

```
scaled_a/scaled_b 形状: ((K//64)+g, M/N, 2)

第 i 个 group 的 scale 偏移量 = begin_i // 64 + i
第 i 个 group 的 scale 长度 = (end_i - begin_i) // 64
```

**示例：** K=512, g=2, group_list=[256, 256]

| 分组 | K 轴范围 | scale_offset | scale_length | scale 切片范围 |
|------|----------|--------------|--------------|----------------|
| group 0 | [0, 256] | 0 | 4 | [0:4, :, :] |
| group 1 | [256, 512] | 5 | 4 | [5:9, :, :] |

**注意：** 每个分组之间有一个间隔位置（如 offset=5 而非 4），这是 MX 量化存储格式的特殊要求。

---

## 约束说明

### 1. 基础约束

| 约束项 | 要求 | 说明 |
|--------|------|------|
| K 轴对齐 | K % 64 == 0 | MX 量化要求，每 64 个元素对应一个 scale |
| group_list | sum(group_list) == K | 分组切分必须覆盖完整 K 轴 |
| 每个 group | group_size % 64 == 0 | 每个分组大小必须 64 对齐 |

### 2. 内轴 32 字节对齐约束（重要）

根据 `scaled_mm` API 要求，内轴必须满足 32 字节对齐：

| 场景 | 内轴 | 要求 |
|------|------|------|
| a_trans=True | M（mat_a=[K,M] 的内轴） | M >= 32（FP8: 32 元素 = 32 字节） |
| b_trans=False | N（mat_b=[K,N] 的内轴） | N >= 32（FP8: 32 元素 = 32 字节） |

### 3. Tile Shape 约束（关键）

**⚠️ Buffer 空间约束（FP8 数据类型）**

Tile Shape 必须满足 L0A/L0B/L0C Buffer 空间约束，否则会导致精度问题！

**硬件 Buffer 大小：**
```
L0A_size = 64KB
L0B_size = 64KB
L0C_size = 128KB
```

**Buffer 需求计算（FP8，使用 32 元素对齐）：**
```
L0A需求: CeilAlign(mL0, 32) × CeilAlign(kL0, 32) × 1 字节
L0B需求: CeilAlign(nL0, 32) × CeilAlign(kL0, 32) × 1 字节
L0C需求: CeilAlign(mL0, 32) × CeilAlign(nL0, 32) × 4 字节
```

**推荐 Tile Shape 配置：**

| 参数 | 推荐值 | 约束 |
|------|--------|------|
| m_tile_shape | [32, 32] 或 [64, 64] | mL0 >= 32，满足内轴对齐 |
| k_tile_shape | [64, 64] 或 [64, 256] | **kL0 ≤ 256**（避免超出 L0B） |
| n_tile_shape | [128, 256] 或 [256, 256] | nL0 >= 32，**nL0 × kL0 ≤ 64KB** |
| vector_tile_shape | [1, 8, 256, 32] | 标准配置 |

**危险配置示例（会导致精度问题）：**

```python
# ❌ 错误配置：kL0=512, nL0=512 会导致 L0B 超出 64KB
k_tile_shape=[512, 512]   # L0B需求: 512×512×1=256KB > 64KB
n_tile_shape=[512, 512]   # 触发 Spill 机制，精度失败

# ✓ 正确配置：满足 Buffer 约束
k_tile_shape=[64, 512]    # L0B需求: 256×64×1=16KB < 64KB
n_tile_shape=[256, 512]   # L0B需求: 256×64×1=16KB < 64KB
```

### 4. 分组数量约束

| 约束项 | 要求 |
|--------|------|
| 分组数量 | 建议 ≤ 4，过多分组会增加调度开销 |

---

## 调用示例

### 基本用法（a_trans=True，转置格式）

```python
import pypto
import torch

# 定义配置（a_trans=True，转置格式）
tile_config = ShapeConfig(
    ori_shape=[32, 512, 7168],  # [M, K, N]
    group_list=[256, 256],       # 两个分组，每个 256
    m_tile_shape=[32, 32],       # M 维度 tile
    k_tile_shape=[256, 256],     # K 维度 tile
    n_tile_shape=[256, 256],     # N 维度 tile
    vector_tile_shape=[1, 8, 256, 32],
    group_type=1,                # 累加方式
    in_dtype=pypto.DT_FP8E4M3,   # 使用 FP8E4M3 数据类型
    a_trans=True,                # 转置格式
    b_trans=False,
    description="Basic test with 2 groups"
)

# 创建输入张量（根据 a_trans 选择形状）
K, M, N = 512, 32, 7168
num_groups = 2

# FP8E4M3: torch.float8_e4m3fn
# FP8E5M2: torch.float8_e5m2
torch_dtype = torch.float8_e4m3fn

# a_trans=True: a=[K, M], scaled_a=[(K//64)+g, M, 2]
a = torch.randn((K, M), dtype=torch_dtype, device='npu:0')
b = torch.randn((K, N), dtype=torch_dtype, device='npu:0')
scaled_a = torch.randn((K//64 + num_groups, M, 2), dtype=torch.float8_e8m0fnu, device='npu:0')
scaled_b = torch.randn((K//64 + num_groups, N, 2), dtype=torch.float8_e8m0fnu, device='npu:0')
y = torch.randn((num_groups, M, N), dtype=torch.float32, device='npu:0')

# 调用 kernel
scaled_matmul_kernel(a, b, scaled_a, scaled_b, y, tile_config)
```

### a_trans=False 用法（非转置格式）

```python
# 定义配置（a_trans=False，非转置格式）
tile_config = ShapeConfig(
    ori_shape=[32, 512, 1024],   # [M, K, N]
    group_list=[256, 256],
    m_tile_shape=[32, 32],
    k_tile_shape=[256, 256],
    n_tile_shape=[256, 256],
    vector_tile_shape=[1, 8, 256, 32],
    group_type=1,
    in_dtype=pypto.DT_FP8E4M3,
    a_trans=False,               # 非转置格式
    b_trans=False,
    description="a_trans=False test"
)

K, M, N = 512, 32, 1024
num_groups = 2

# a_trans=False: a=[M, K], scaled_a=[M, (K//64)+g, 2]
a = torch.randn((M, K), dtype=torch.float8_e4m3fn, device='npu:0')
b = torch.randn((K, N), dtype=torch.float8_e4m3fn, device='npu:0')
scaled_a = torch.randn((M, K//64 + num_groups, 2), dtype=torch.float8_e8m0fnu, device='npu:0')
scaled_b = torch.randn((K//64 + num_groups, N, 2), dtype=torch.float8_e8m0fnu, device='npu:0')
y = torch.randn((num_groups, M, N), dtype=torch.float32, device='npu:0')

# 调用 kernel
scaled_matmul_kernel(a, b, scaled_a, scaled_b, y, tile_config)
```

### 运行测试

```bash
# 运行所有测试用例
python quant_grouped_matmul_inplace_add_mx.py
```

---

## 测试用例说明

代码包含5个测试用例，覆盖不同场景。每个用例都经过性能优化配置。

### 用例 1：基础用例（FP8E4M3，N较大）

| 参数 | 值 | 优化说明 |
|------|-----|---------|
| M, K, N | 32, 512, 7168 | N=7168 很大 |
| group_list | [256, 256] | g=2 |
| m_tile_shape | [32, 32] | mL1=M=32，消除A矩阵重复载入 |
| k_tile_shape | [64, 256] | kL0减小至64，满足Buffer |
| n_tile_shape | [128, 1024] | nL1=1024增大，减少切分 |
| in_dtype | DT_FP8E4M3 | - |
| group_type | 1 | 累加方式 |

**性能优化特点：**
- mL1=M=32，消除A矩阵MTE2重复载入
- L0B需求: 128×64×1 = 8KB ✓（远小于64KB）
- nL1增大至1024，减少N轴切分次数
- kL1>kL0，使能大包搬运和double buffer

### 用例 2：更大 M 维度（FP8E4M3）

| 参数 | 值 | 优化说明 |
|------|-----|---------|
| M, K, N | 64, 1024, 4096 | M和K都较大 |
| group_list | [512, 512] | g=2 |
| m_tile_shape | [32, 64] | mL1接近M |
| k_tile_shape | [64, 1024] | kL1=K_group，使能大包搬运 |
| n_tile_shape | [128, 1024] | nL1增大 |
| in_dtype | DT_FP8E4M3 | - |
| group_type | 1 | 累加方式 |

**性能优化特点：**
- mL1=64接近M，减少A矩阵切分
- kL1=1024=K_group，A矩阵可驻留反复使用
- L0B需求: 128×64×1 = 8KB ✓

### 用例 3：3 个分组（FP8E4M3）

| 参数 | 值 | 优化说明 |
|------|-----|---------|
| M, K, N | 32, 768, 2048 | 3个分组 |
| group_list | [256, 256, 256] | g=3 |
| m_tile_shape | [32, 32] | mL1=M |
| k_tile_shape | [64, 256] | kL0减小 |
| n_tile_shape | [128, 512] | nL1增大 |
| in_dtype | DT_FP8E4M3 | - |
| group_type | 1 | 累加方式 |

**性能优化特点：**
- mL1=M=32，消除A矩阵重复载入
- L0B需求: 128×64×1 = 8KB ✓
- scale 存储：((768/64)+3, 32, 2) = (15, 32, 2)

### 用例 4：累计值模式（group_type=0）

| 参数 | 值 | 优化说明 |
|------|-----|---------|
| M, K, N | 32, 512, 1024 | 规模较小 |
| group_list | [256, 512] | 累计值方式 |
| m_tile_shape | [32, 32] | mL1=M |
| k_tile_shape | [64, 256] | kL0减小 |
| n_tile_shape | [64, 1024] | nL1=N，消除切分 |
| in_dtype | DT_FP8E4M3 | - |
| group_type | 0 | 累计值方式 |

**性能优化特点：**
- mL1=M=32，消除A矩阵重复载入
- nL1=N=1024，消除B矩阵切分
- L0B需求: 64×64×1 = 4KB ✓
- `[256, 512]` 表示第一个分组 K=[0,256]，第二个分组 K=[256,512]

### 用例 5：FP8E5M2 数据类型

| 参数 | 值 | 优化说明 |
|------|-----|---------|
| M, K, N | 32, 512, 1024 | FP8E5M2格式 |
| group_list | [128, 384] | 不均匀分组 |
| m_tile_shape | [32, 32] | mL1=M |
| k_tile_shape | [64, 256] | kL0减小 |
| n_tile_shape | [64, 1024] | nL1=N |
| in_dtype | DT_FP8E5M2 | 5位指数+2位尾数 |
| group_type | 1 | 累加方式 |

**性能优化特点：**
- 与Case4相同优化策略
- FP8E5M2格式具有更大的动态范围，适合推理场景

---

## 性能优化配置总结

| 优化策略 | 实现方式 | 效果 |
|----------|---------|------|
| 消除A矩阵重复载入 | 设置 mL1 = M | 减少MTE2载入量 |
| 消除B矩阵切分 | 设置 nL1 ≥ N | 减少切分开销 |
| 满足Buffer约束 | kL0 ≤ 64，nL0 × kL0 ≤ 64KB | 避免Spill精度问题 |
| 使能大包搬运 | kL1 > kL0 | 提高MTE2带宽利用率 |
| 使能Double buffer | L0A/L0B空间 ≤ 32KB | 流水并行 |

---

## a_trans=False 使用说明

代码已支持 `a_trans=False` 非转置格式的逻辑，但当前测试用例均使用 `a_trans=True`。如需测试非转置格式，可参考以下配置：

```python
# a_trans=False 配置示例（已优化）
tile_config = ShapeConfig(
    ori_shape=[32, 512, 1024],
    group_list=[256, 256],
    m_tile_shape=[32, 32],    # mL1=M
    k_tile_shape=[64, 256],   # kL0减小
    n_tile_shape=[64, 1024],  # nL1=N
    vector_tile_shape=[1, 8, 64, 1024],
    group_type=1,
    in_dtype=pypto.DT_FP8E4M3,
    a_trans=False,  # 非转置格式
    b_trans=False,
    description="a_trans=False test"
)

# 输入数据形状变化：
# - a: [M, K] = [32, 512]（非转置）
# - scaled_a: [M, (K//64)+g, 2] = [32, 10, 2]
```

---

## 性能优化建议

### 1. Tile Shape 配置

**关键原则：保证 Buffer 空间约束**

| Buffer | 约束公式（FP8） | 最大安全配置 |
|--------|----------------|--------------|
| L0A | mL0 × kL0 × 1 ≤ 64KB | mL0 × kL0 ≤ 65536 |
| L0B | **nL0 × kL0 × 1 ≤ 64KB** | **nL0 × kL0 ≤ 65536** ⚠️ |
| L0C | mL0 × nL0 × 4 ≤ 128KB | mL0 × nL0 ≤ 32768 |

**推荐配置策略：**

1. **kL0 保持较小值**（64-256），避免与 nL0 相乘超出 L0B
2. **mL0 >= 32**，满足内轴对齐
3. **nL0 >= 32**，满足内轴对齐，同时 nL0 × kL0 ≤ 65536

### 2. 分组策略

- 分组数量适中（2-4 个），避免过多分组增加调度开销
- 每个分组大小建议相同，便于切分优化

### 3. Double Buffer

当 mL1 × kL1 × sizeof(dtype) ≤ 32KB 时，可开启 MTE1 double buffer 实现流水并行：

```python
m_tile_shape=[32, 32]   # mL1=32, FP8下32KB，可开启 double buffer
k_tile_shape=[64, 64]   # kL1=64
```

---

## 常见问题

### Q1: 为什么 K 必须被 64 整除？

A: MX 量化以 64 个元素为一个量化块，每个块对应一个 scale 值（FP8E8M0 格式）。scale 张量的形状基于 `K//64` 计算。

### Q2: scale 张量的形状如何计算？

A: 
```
scaled_a: ((K//64) + num_groups, M, 2)
scaled_b: ((K//64) + num_groups, N, 2)
```
其中 `num_groups` 是额外增加的空间，用于分组间的间隔存储。

### Q3: 为什么 kL0 和 nL0 不能太大？

A: L0B Buffer 只有 64KB，需求为 `nL0 × kL0 × 1` 字节。当 kL0=512, nL0=512 时，需求 256KB，超出 4 倍，会触发 Spill 机制导致精度失败。

### Q4: 精度失败但没报错，原因是什么？

A: Tile Shape 超出 Buffer 约束时，编译器不会报错，但运行时会触发 Spill（数据换出换入），导致计算使用错误数据，精度失败。

### Q5: group_type=0 和 group_type=1 有什么区别？

A:
- **group_type=0**：`[256, 512]` 表示累计值，第一个分组 K=[0,256]，第二个分组 K=[256,512]
- **group_type=1**：`[256, 256]` 表示两个分组，大小分别为 256 和 256，K=512

### Q6: FP8E4M3 和 FP8E5M2 应该选择哪个？

A: 根据应用场景选择：

| 数据类型 | 特点 | 适用场景 |
|----------|------|----------|
| **DT_FP8E4M3** | 4位指数+3位尾数，精度更高，动态范围较小 | 训练场景，精度优先 |
| **DT_FP8E5M2** | 5位指数+2位尾数，动态范围更大，精度较低 | 推理场景，动态范围优先 |

- 训练场景推荐：`DT_FP8E4M3`（精度损失更小）
- 推理场景推荐：`DT_FP8E5M2`（支持更大数值范围）

---

## 参考资源

- [PyPTO scaled_mm API 文档](../../docs/api/operation/pypto-scaled_mm.md)
- [PyPTO set_cube_tile_shapes 文档](../../docs/api/config/pypto-set_cube_tile_shapes.md)
- [Matmul 性能编程指南](../../docs/tutorials/debug/matmul_performance_guide.md)