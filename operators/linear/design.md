# linear 算子设计文档

> **算子名称**: linear
> **算子分类**: matmul
> **生成时间**: 2026-03-28
> **基于**: spec.md, api_report.md

---

## 1. 概述

### 1.1 功能描述

线性层（全连接层）算子，对输入张量进行线性变换。支持任意维度的输入张量，对最后一维进行线性变换，保持其他维度不变。

### 1.2 数学公式

$$output = input \cdot weight^T + bias$$

### 1.3 算法描述

简单矩阵乘法算子，公式已足够描述计算逻辑。核心操作为 matmul 后可选的 bias add。

### 1.4 数据流图

```
     输入 input         权重 weight       偏置 bias (可选)
 ┌──────────────┐   ┌──────────────┐   ┌──────────────┐
 │[..., in_feat]│   │[out, in_feat]│   │   [out]      │
 │   float32    │   │   float32    │   │  float32     │
 └──────┬───────┘   └──────┬───────┘   └──────┬───────┘
        │                  │                   │
        │                  ▼                   │
        │           ┌────────────┐             │
        │           │  weight^T  │             │
        │           └──────┬─────┘             │
        │                  │                   │
        ▼                  ▼                   │
     ┌─────────────────────────┐               │
     │     input @ weight^T    │               │
     └───────────┬─────────────┘               │
                 │                             │
                 ▼                             ▼
          ┌─────────────────────────────────────┐
          │              + (bias)               │
          └──────────────┬──────────────────────┘
                         │
                         ▼
                  ┌──────────────┐
                  │ 输出 output   │
                  │[..., out_feat]│
                  │   float32    │
                  └──────────────┘

公式: output = input @ weight.T + bias
动态轴: batch, seq_len (input的前N-1维)
```

---

## 2. API 映射设计

### 2.1 数学公式分解

将公式拆解为基本操作步骤：

| 步骤 | 数学表达 | 说明 |
|------|----------|------|
| 1 | input @ weight^T | 矩阵乘法，weight 需要转置 |
| 2 | result + bias | 逐元素加法，bias 广播到 result shape（可选） |

### 2.2 PyPTO API 映射表

**方案一：2D 输入 + bias 融合（推荐）**

| 步骤 | 数学表达 | PyPTO API | 参数 | 文档路径 |
|------|----------|-----------|------|----------|
| 1 | input @ weight^T + bias | `pypto.matmul` | `a=input, b=weight, dtype=DT_FP32, b_trans=True, extend_params={'bias_tensor': bias}` | docs/api/operation/pypto-matmul.md |

**方案二：3D/4D 输入或无 bias（通用）**

| 步骤 | 数学表达 | PyPTO API | 参数 | 文档路径 |
|------|----------|-----------|------|----------|
| 1 | input @ weight^T | `pypto.matmul` | `a=input, b=weight, dtype=DT_FP32, b_trans=True` | docs/api/operation/pypto-matmul.md |
| 2 | result + bias | `pypto.add` | `x=matmul_result, y=bias` | docs/api/operation/pypto-add.md |

### 2.3 计算步骤序列

**2D 输入 + bias 场景（融合方案）**：
```python
# 使用 b_trans=True 实现 weight 转置，extend_params 融合 bias
extend_params = {'bias_tensor': bias_reshaped}  # bias shape: [1, out_features]
out = pypto.matmul(input, weight, pypto.DT_FP32, b_trans=True, extend_params=extend_params)
```

**3D/4D 输入场景（分开方案）**：
```python
# 步骤 1: matmul with weight transpose
matmul_result = pypto.matmul(input, weight, pypto.DT_FP32, b_trans=True)

# 步骤 2: bias add (如果存在 bias)
if bias is not None:
    out = pypto.add(matmul_result, bias)
else:
    out = matmul_result
```

### 2.4 设计依据

- **来源**: api_report.md, docs/api/operation/pypto-matmul.md
- **说明**:
  - 使用 `b_trans=True` 参数直接实现 weight 转置，避免额外的 transpose 操作开销
  - 2D 场景下使用 `extend_params={'bias_tensor': bias}` 融合 bias，减少一次 kernel 启动
  - 3D/4D 场景下 bias_tensor 融合不支持，需分开调用 matmul + add
  - 参考实现 operators/matmul/matmul_impl.py 的多维度分发模式

---

## 3. 数据规格设计

### 3.1 OperatorInput dataclass

```python
@dataclass
class LinearInput:
    input: Tensor   # 输入张量, shape: [..., in_features], dtype: float32
    weight: Tensor  # 权重矩阵, shape: [out_features, in_features], dtype: float32
    bias: Optional[Tensor]  # 偏置向量, shape: [out_features], dtype: float32 (可选)
```

### 3.2 OperatorOutput dataclass

```python
@dataclass
class LinearOutput:
    output: Tensor  # 输出张量, shape: [..., out_features], dtype: float32
```

### 3.3 中间 Tensor 定义

| 名称 | Shape | Dtype | 说明 |
|------|-------|-------|------|
| matmul_result | [..., out_features] | float32 | matmul 中间结果（仅分开方案使用） |

### 3.4 数据格式选择

| Tensor | 格式 | 说明 |
|--------|------|------|
| input | ND | ND 格式适配动态 shape，无需特殊对齐 |
| weight | ND | ND 格式，b_trans=True 时硬件自动处理 |
| output | ND | 与输入格式一致 |

### 3.5 动态轴定义

| 轴名称 | 含义 | 取值范围 |
|--------|------|----------|
| batch | 批次维度，表示样本数量 | [1, INT32_MAX] |
| seq_len | 序列长度维度，表示 token 数量 | [1, INT32_MAX] |

### 3.6 JIT 装饰器配置

```python
@pypto.frontend.jit(
    runtime_options={"run_mode": pypto.RunMode.NPU}
)
def linear_kernel(...):
    ...
```

---

## 4. Tiling 策略

### 4.1 算子类型判断

- **类型**: Cube + Vector 混合
- **判断依据**: 公式包含 matmul 操作（Cube 类型）和可选的 bias add 操作（Vector 类型）。matmul 是核心计算，需使用 set_cube_tile_shapes；3D/4D 输入时 matmul 本身需要 set_vec_tile_shapes，add 操作也需要 Vector Tiling。

### 4.2 TileShape 初值设置

**Cube Tiling（matmul 必需）**：
```python
# 根据矩阵大小动态选择 tiling
m, k, n = input.shape[-2], input.shape[-1], weight.shape[0]

if m <= 64 and k <= 64 and n <= 64:
    # 小矩阵场景
    pypto.set_cube_tile_shapes(m=[32, 32], k=[64, 64], n=[64, 64])
elif m <= 2048 and k <= 2048 and n <= 2048:
    # 中等矩阵场景
    pypto.set_cube_tile_shapes(m=[128, 128], k=[128, 128], n=[128, 128])
else:
    # 大矩阵场景
    pypto.set_cube_tile_shapes(m=[256, 256], k=[256, 256], n=[256, 256])
```

**Vector Tiling（3D/4D 输入必需）**：
```python
# 3D 输入
pypto.set_vec_tile_shapes(batch_tile, seq_tile, feature_tile)

# 4D 输入
pypto.set_vec_tile_shapes(b1_tile, b2_tile, seq_tile, feature_tile)
```

### 4.3 设置依据

1. **Cube Tiling 选择逻辑**：
   - 小矩阵 (M,N,K <= 64): 使用较小的 tile 提高并行度，减少尾块浪费
   - 中等矩阵 (<= 2048): 平衡 L0/L1 利用率和计算效率
   - 大矩阵 (> 2048): 使用大 tile 最大化内存带宽利用率

2. **FP32 对齐要求**：
   - kL0, kL1, nL0, nL1 需 16 元素对齐（FP32 类型，32 字节 = 8 个 FP32 元素，但文档要求 16 元素对齐）
   - mL0, mL1 需满足 0 < mL0 <= mL1, mL1 % mL0 == 0

3. **bias_tensor 融合约束**（仅 2D 场景）：
   - nL0 * 4 <= BTBuffer_size (1KB)，即 nL0 <= 256

### 4.4 注意事项

- 尾轴 tile 需满足 16 元素对齐（FP32）
- bias_tensor 融合仅支持 2D matmul，3D+ 需分开实现
- 3D/4D 场景需同时设置 cube tiling 和 vector tiling

### 4.5 判断依据与适用条件

- **判断依据**: 根据输入 shape 的 M、K、N 维度大小动态选择 tiling 配置，平衡 L0/L1 buffer 利用率和计算效率
- **适用条件**:
  - 2D 输入: 仅需 cube tiling，可使用 bias_tensor 融合
  - 3D/4D 输入: 需要 cube tiling + vector tiling，bias 需分开 add
- **不适用场景**:
  - 非 FP32 dtype（如 FP16/BF16）需调整对齐要求
  - 极端大矩阵（单维超过 65535）需特殊处理

---

## 5. Loop 结构设计

### 场景 A：不需要 Loop

> 适用于所有轴编译期已知、单次 Tile 可处理的算子。

- **结论**: 不需要 pypto.loop
- **原因**: linear 算子的动态轴（batch, seq_len）由 matmul API 内部处理，PyPTO 的 matmul 实现会自动处理动态维度的数据切分和循环迭代，无需用户手动编写 loop
- **适用条件**:
  - 输入维度 2-4 维
  - 动态轴范围 [1, INT32_MAX]
  - 使用 from_torch 转换的 tensor
- **限制**:
  - 超过 4 维输入不支持
  - 需要为动态轴在 Tensor 定义中使用 pypto.DYNAMIC
- **处理方式**: 编译器自动处理数据切分，matmul API 内部处理动态维度遍历

---

## 6. 验证方案

### 6.1 Golden 函数设计

```python
def linear_golden(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """linear 参考实现"""
    return torch.nn.functional.linear(input, weight, bias)
```

### 6.2 测试用例设计

#### 基于 spec.md 所有典型配置

| 配置名称 | 类型 | 优先级 | 参数 | 输入 Shape | 输出 Shape | 说明 |
|----------|------|--------|------|------------|------------|------|
| 性能_P0_2D | 性能 | P0 | in_features=4096, out_features=4096, has_bias=True | input: [batch, 4096], weight: [4096, 4096], bias: [4096] | output: [batch, 4096] | 核心 2D 性能场景，使用 bias 融合 |
| 性能_P0_3D | 性能 | P0 | in_features=1024, out_features=4096, has_bias=True | input: [batch, seq_len, 1024], weight: [4096, 1024], bias: [4096] | output: [batch, seq_len, 4096] | 核心 3D 性能场景（含动态轴），分开实现 |
| 功能_P0_no_bias | 功能 | P0 | in_features=512, out_features=512, has_bias=False | input: [batch, 512], weight: [512, 512] | output: [batch, 512] | 无偏置功能验证 |
| 功能_P1_4D | 功能 | P1 | in_features=256, out_features=512, has_bias=True | input: [batch, heads, seq_len, 256], weight: [512, 256], bias: [512] | output: [batch, heads, seq_len, 512] | 4D 输入功能验证 |

#### 边界情况测试

| 场景 | 参数 | 说明 |
|------|------|------|
| 最小 batch | batch=1, in=128, out=64 | 测试动态轴下界 |
| 最小 seq_len | batch=4, seq=1, in=256, out=128 | 测试 seq_len 下界 |
| 零值输入 | input=zeros | 验证输出等于 bias |
| 大值输入 | input * 1000 | 验证无 NaN/Inf |

### 6.3 精度验证标准

| Dtype | atol | rtol |
|-------|------|------|
| float32 | 0.001 | 0.001 |

---

## 7. 性能指标与开箱配置

### 7.1 性能目标

基于 spec.md 典型配置（性能类）的预期性能：

| 配置名称 | 类型 | 优先级 | 参数 | 输入 Shape | 输出 Shape | 预期 kernel 耗时 |
|----------|------|--------|------|------------|------------|------------------|
| 性能_P0_2D | 性能 | P0 | in=4096, out=4096, bias=True | [batch, 4096] | [batch, 4096] | 待实测 |
| 性能_P0_3D | 性能 | P0 | in=1024, out=4096, bias=True | [batch, seq, 1024] | [batch, seq, 4096] | 待实测 |

性能目标: 首跑精度成功性能的 2 倍

### 7.2 开箱性能配置

```python
# 2D 场景默认配置
def get_tiling_2d(m, k, n):
    if m <= 64 and k <= 64 and n <= 64:
        return ([32, 32], [64, 64], [64, 64])
    elif m <= 2048 and k <= 2048 and n <= 2048:
        return ([128, 128], [128, 128], [128, 128])
    else:
        return ([256, 256], [256, 256], [256, 256])

# 3D/4D 场景需额外设置 vector tiling
# 3D: pypto.set_vec_tile_shapes(128, 128, 128)
# 4D: pypto.set_vec_tile_shapes(64, 64, 64, 64)
```

### 7.3 pass_options 配置

无需特殊 pass_options 配置。

### 7.4 runtime_options 配置

```python
runtime_options = {"run_mode": pypto.RunMode.NPU}
```

---

## 8. 风险点与注意事项

### 8.1 已知约束

- **bias_tensor 融合限制**: extend_params 的 bias_tensor 仅支持 2D matmul，3D+ 需分开用 add
- **contiguous 输入**: 必须确保 torch.Tensor.is_contiguous() == True
- **FP32 对齐要求**: set_cube_tile_shapes 的 kL0, kL1, nL0, nL1 需 16 元素对齐
- **维度限制**: matmul 支持 2-4 维输入

### 8.2 常见错误规避

| 风险 / 错误 | 触发场景 | 影响 / 原因 | 规避方法 |
|-------------|----------|-------------|----------|
| 非连续输入 | torch tensor 经过 transpose/slice 等操作 | from_torch 失败或数据错误 | 调用 input.contiguous() 确保连续 |
| bias_tensor 3D+ 融合 | 3D/4D 输入尝试使用 extend_params | 编译失败或运行时错误 | 3D/4D 使用分开方案（matmul + add） |
| 对齐不满足 | tiling 参数未 16 元素对齐 | 编译失败 | 确保 kL0, kL1, nL0, nL1 为 16 的倍数 |
| 动态轴未声明 | Tensor 定义中使用固定值 | 动态 shape 无法正常工作 | 使用 pypto.DYNAMIC 声明动态轴 |

### 8.3 特殊场景处理

- **无 bias 场景**: 直接调用 matmul，无需 add 操作
- **大矩阵场景**: 使用大 tile (256, 256, 256)，可能需要多次迭代
- **混合精度**: 当前仅支持 FP32，FP16/BF16 需额外调整对齐要求

### 8.4 实现建议

| 建议项 | 说明 |
|--------|------|
| 多维度分发 | 参考 matmul_impl.py，根据输入维度选择不同 kernel |
| b_trans=True | 使用 matmul 的 b_trans 参数，避免额外 transpose 开销 |
| 动态 tiling | 根据 M/K/N 大小动态选择 tiling 配置 |

---

## 9. 交付件清单

### 9.1 目录结构

```
operators/linear/
├── spec.md                          # 需求规范（已有）
├── api_report.md                    # API 探索报告（已有）
├── design.md                        # 设计文档（本文件）
├── linear_golden.py                 # Golden 参考实现（已有）
├── linear_impl.py                   # 算子实现代码
├── test_linear.py                   # 测试代码
└── output/                          # 运行输出（自动生成）
```

### 9.2 文件清单

| 文件 | 类型 | 说明 | 生成方式 |
|------|------|------|----------|
| spec.md | 需求 | 算子需求规范 | pypto-intent-understanding |
| api_report.md | API | API 探索报告 | pypto-api-explorer |
| design.md | 设计 | 算子设计文档 | pypto-op-design（本 skill） |
| linear_golden.py | 代码 | Golden 参考实现 | pypto-golden-generator |
| linear_impl.py | 代码 | 算子核心实现 | pypto-op-develop |
| test_linear.py | 代码 | 测试用例 | pypto-op-develop |

### 9.3 命名规范

| 项目 | 规范 | 示例 |
|------|------|------|
| 算子名称 | 小写字母 | `linear` |
| 目录名 | 与算子名称一致 | `operators/linear/` |
| Golden 文件 | `{op}_golden.py` | `linear_golden.py` |
| 实现文件 | `{op}_impl.py` | `linear_impl.py` |
| 测试文件 | `test_{op}.py` | `test_linear.py` |
| wrapper 函数 | `{op}_wrapper` | `linear_wrapper` |

### 9.4 生成顺序

```
spec.md → api_report.md → linear_golden.py → design.md → linear_impl.py → test_linear.py
```
