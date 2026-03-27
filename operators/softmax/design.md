# Softmax 算子设计方案

> **生成时间**: 2026-03-27
> **算子复杂度**: medium

---

## 1. 概述

### 1.1 基础信息

- **算子名称**: softmax
- **算子分类**: normalization
- **功能**: 对输入张量沿指定轴进行 softmax 归一化，输出概率分布

### 1.2 数学公式

$$\text{softmax}(x_i) = \frac{\exp(x_i - \max(x))}{\sum_j \exp(x_j - \max(x))}$$

### 1.3 算法描述

```
Algorithm: Safe Softmax
────────────────────────────────────
输入: x ∈ R^{d_1 × d_2 × ... × d_n}, 归一化轴 dim
输出: y ∈ R^{d_1 × d_2 × ... × d_n}

1. max_val = max(x, axis=dim, keepdims=True)  // 沿指定轴求最大值
2. x_shifted = x - max_val                     // 数值稳定：减去最大值
3. exp_x = exp(x_shifted)                      // 计算指数
4. sum_exp = sum(exp_x, axis=dim, keepdims=True)  // 沿指定轴求和
5. y = exp_x / sum_exp                         // 归一化
6. return y
```

### 1.4 数据流图

```
    输入 x
┌──────────────┐
│  [b, s, n, d] │
│   float32     │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│  max(x,dim)  │ ──┐
└──────┬───────┘   │
       │           │
       ▼           │
┌──────────────┐   │
│  x - max     │   │
└──────┬───────┘   │
       │           │
       ▼           │
┌──────────────┐   │
│   exp(x)     │   │
└──────┬───────┘   │
       │           │
       ▼           │
┌──────────────┐   │
│  sum(exp)    │ ◄─┘
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ exp / sum    │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│  输出 y       │
│  [b, s, n, d] │
│   float32     │
└──────────────┘

动态轴: b, s
```

---

## 2. API 映射设计

### 2.1 数学公式分解

将 softmax 公式拆解为基本操作步骤：

| 步骤 | 数学表达 | 说明 |
|------|----------|------|
| 1 | softmax(x, dim) | 沿指定轴归一化 |

### 2.2 PyPTO API 映射表

**方案 A（推荐）：直接使用 softmax API**

| 步骤 | 数学表达 | PyPTO API | 参数 | 文档路径 |
|------|----------|-----------|------|----------|
| 1 | softmax(x, dim) | `pypto.softmax(x, dim)` | input, dim | docs/api/operation/pypto-softmax.md |

**方案 B（备选）：手动实现**

| 步骤 | 数学表达 | PyPTO API | 参数 | 文档路径 |
|------|----------|-----------|------|----------|
| 1 | max(x, dim, keepdim) | `pypto.amax(x, dim, keepdim=True)` | input, dim, keepdim | docs/api/operation/pypto-amax.md |
| 2 | x - max_val | `pypto.sub(x, max_val)` | x, max_val | docs/api/operation/pypto-sub.md |
| 3 | exp(x_shifted) | `pypto.exp(x_shifted)` | x | docs/api/operation/pypto-exp.md |
| 4 | sum(exp_x, dim, keepdim) | `pypto.sum(exp_x, dim, keepdim=True)` | input, dim, keepdim | docs/api/operation/pypto-sum.md |
| 5 | exp_x / sum_exp | `pypto.div(exp_x, sum_exp)` | x, y | docs/api/operation/pypto-div.md |

### 2.3 计算步骤序列

```python
# 方案 A（推荐）：直接 API
output = pypto.softmax(input_tensor, dim)

# 方案 B（备选）：手动实现
max_val = pypto.amax(input_tensor, dim=dim, keepdim=True)
x_shifted = input_tensor - max_val
exp_x = pypto.exp(x_shifted)
sum_exp = pypto.sum(exp_x, dim=dim, keepdim=True)
output = exp_x / sum_exp
```

### 2.4 设计依据

- **来源**: api_report.md §3.3
- **说明**: 
  - PyPTO 已提供优化的 `pypto.softmax` API，推荐直接使用
  - 方案 B 提供更灵活的实现，可支持 FP16/BF16
  - 方案 A 代码简洁，官方优化，稳定性高

---

## 3. 数据规格设计

### 3.1 OperatorInput dataclass

```python
@dataclass
class SoftmaxInput:
    x: Tensor  # 输入张量, shape: [batch, seq, ...dims], dtype: float32
    dim: int   # 归一化轴（默认 -1）
```

### 3.2 OperatorOutput dataclass

```python
@dataclass
class SoftmaxOutput:
    y: Tensor  # 归一化后的概率分布, shape: [batch, seq, ...dims], dtype: float32
```

### 3.3 中间 Tensor 定义

**方案 A（直接 API）**：无中间 Tensor

**方案 B（手动实现）**：

| 名称 | Shape | Dtype | 说明 |
|------|-------|-------|------|
| max_val | [batch, seq, ...dims[:-1], 1] | float32 | 沿 dim 轴的最大值 |
| x_shifted | [batch, seq, ...dims] | float32 | 减去最大值后的输入 |
| exp_x | [batch, seq, ...dims] | float32 | 指数值 |
| sum_exp | [batch, seq, ...dims[:-1], 1] | float32 | 沿 dim 轴的指数和 |

### 3.4 数据格式选择

| Tensor | 格式 | 说明 |
|--------|------|------|
| x | ND | 默认格式，无需特殊处理 |
| y | ND | 与输入格式一致 |

### 3.5 动态轴定义

| 轴名称 | 含义 | 取值范围 |
|--------|------|----------|
| batch | 批次大小 | [1, 65536] |
| seq | 序列长度 | [1, 65536] |

### 3.6 JIT 装饰器配置

```python
@pypto.frontend.jit(
    runtime_options={"run_mode": pypto.RunMode.NPU}  # 优先使用 NPU 模式
)
def softmax_kernel(
    input_tensor: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_FP32),
    output_tensor: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_FP32),
    dim: int = -1
):
    ...
```

---

## 4. Tiling 策略

### 4.1 算子类型判断

- **类型**: Vector
- **判断依据**: 仅包含逐元素操作（exp, sub, div）和归约操作（amax, sum），不涉及矩阵乘法

### 4.2 TileShape 初值设置

```python
# 基于 spec.md 性能_P0 配置: [1, 4096, 4096]
# 推荐配置: [tile_b, tile_s, tile_h, tile_d]
pypto.set_vec_tile_shapes(1, 4, 1, 64)
```

### 4.3 设置依据

- **参考**: examples/02_intermediate/operators/softmax/softmax.py
- **理由**: 
  - 尾轴 64 满足 32B 对齐（FP32: 64*4=256B）
  - 次尾轴 4 ≤ 255，满足约束
  - 批次轴设为 1，配合 Loop 处理动态批次

### 4.4 注意事项

- 尾轴必须 32B 对齐（FP32: 尾轴大小必须是 8 的倍数）
- TileShape 维度数 ≤ 输入维度数
- 动态轴通过 Loop 处理，TileShape 固定

### 4.5 判断依据与适用条件

- **判断依据**: 基于 spec.md 典型配置和官方示例
- **适用条件**: 适用于 2-4 维输入，FP32 dtype
- **不适用场景**: 输入维度 > 4 或 dtype 为 FP16/BF16（需调整 TileShape）

---

## 5. Loop 结构设计

### 5.1 是否需要 Loop

**结论**: 需要 Loop

**判断依据**（按 quick_ref.md §2.1 判据表）:
- ✅ 命中条件 1: 存在动态轴（batch, seq），运行时才知道大小
- **Loop 类型**: `pypto.loop`
- **原因**: 编译期无法展开动态维度，必须用运行时循环遍历

### 5.2 Loop 结构设计

```python
# 输入: [batch, seq, head, dim]
bs, seqlen, head, dim = input_tensor.shape
tile_b = 1  # 每次处理 1 个 batch
b_loop = bs // tile_b

for idx in pypto.loop(0, b_loop, 1, name="LOOP_L0_bIdx", idx_name="idx"):
    b_offset = idx * tile_b
    b_offset_end = (idx + 1) * tile_b
    
    # 切片获取当前批次
    input_view = input_tensor[b_offset:b_offset_end, :seqlen, :head, :dim]
    
    # 执行 softmax
    softmax_out = pypto.softmax(input_view, dim=dim)
    
    # 写回输出
    output_tensor[b_offset:b_offset_end, ...] = softmax_out
```

### 5.3 静态/动态轴处理

| 轴 | 类型 | 处理方式 |
|-----|------|----------|
| batch | 动态 | 使用 `pypto.loop` 遍历 |
| seq | 动态 | 切片处理，由编译器优化 |
| head | 静态 | 直接处理 |
| dim | 静态 | 直接处理 |

### 5.4 尾块处理

- 当前设计假设 batch 能被 tile_b 整除
- 如需处理尾块，可添加判断逻辑：
  ```python
  if bs % tile_b != 0:
      # 处理剩余批次
      remaining = bs % tile_b
      input_view = input_tensor[b_loop*tile_b:, ...]
      softmax_out = pypto.softmax(input_view, dim=dim)
      output_tensor[b_loop*tile_b:, ...] = softmax_out
  ```

---

## 6. 验证方案

### 6.1 Golden 函数设计

- **文件**: `softmax_golden.py`
- **函数**: `softmax_golden(x: torch.Tensor, dim: int = -1) -> torch.Tensor`
- **实现**: 使用 `torch.nn.functional.softmax`

### 6.2 测试用例

基于 spec.md §11 典型配置：

| 配置名称 | 类型 | 优先级 | 参数 | 输入 Shape | 输出 Shape | 说明 |
|----------|------|--------|------|------------|------------|------|
| 性能_P0 | 性能 | P0 | dim=-1 | [1, 4096, 4096] | [1, 4096, 4096] | Attention 场景大 token |
| 功能_P0 | 功能 | P0 | dim=-1 | [2, 1024, 512] | [2, 1024, 512] | 常规 batch 推理 |
| 功能_P1 | 功能 | P1 | dim=1 | [4, 256, 128] | [4, 256, 128] | 中间轴归一化 |
| 功能_P1 | 功能 | P1 | dim=-1 | [8, 512, 64] | [8, 512, 64] | 小批量多 head |

#### 边界情况测试（可选）

| 场景 | 参数 | 说明 |
|------|------|------|
| 最小 batch | batch=1 | 验证边界情况 |
| 最大 seq | seq=65536 | 验证大序列（可选） |
| dim=0 | dim=0 | 验证不同轴 |

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
| 性能_P0 | 性能 | P0 | dim=-1 | [1, 4096, 4096] | [1, 4096, 4096] | 待实测（首跑性能的 2 倍） |

### 7.2 开箱性能配置

```python
# 推荐配置
pypto.set_vec_tile_shapes(1, 4, 1, 64)
```

### 7.3 pass_options 配置

无特殊 pass_options 配置

### 7.4 runtime_options 配置

```python
runtime_options={"run_mode": pypto.RunMode.NPU}  # 优先 NPU，无 NPU 时使用 sim
```

---

## 8. 风险点与注意事项

### 8.1 已知约束

- **dtype 限制**: 方案 A（直接 API）仅支持 FP32，方案 B 可支持 FP16/BF16
- **contiguous**: 输入 tensor 必须连续（`is_contiguous() == True`）
- **dim 范围**: 需在 `[-input.dim, input.dim-1]` 范围内
- **shape 限制**: 方案 B 的 amax/sum 仅支持 2-4 维输入

### 8.2 常见错误规避

| 风险 / 错误 | 触发场景 | 影响 / 原因 | 规避方法 |
|-------------|----------|-------------|----------|
| dtype 不支持 | 输入为 FP16/BF16 | 方案 A 编译失败 | 使用方案 B 或转换 dtype |
| 维度不匹配 | 输入维度 > 4 | 方案 B 的 amax/sum 失败 | 使用方案 A 或 reshape |
| 尾轴未对齐 | TileShape 尾轴非 8 倍数 | 编译失败 | 调整 TileShape 尾轴为 8 倍数 |
| 动态轴未标记 | JIT 装饰器中未声明 | 编译失败 | 使用 `pypto.DYNAMIC` 标记 |

### 8.3 特殊场景处理

- **大 batch**: 使用 Loop 分批处理，避免内存溢出
- **dim 参数**: 支持负索引，需在实现中处理
- **数值稳定性**: 使用 safe softmax（减去最大值），避免 exp 溢出

### 8.4 实现建议

| 建议项 | 说明 |
|--------|------|
| 优先使用方案 A | 代码简洁，官方优化，稳定性高 |
| 动态轴标记 | 在 kernel 函数签名中使用 `pypto.DYNAMIC` |
| Loop 边界处理 | 检查是否能整除，处理尾块 |
| dim 参数验证 | 检查 dim 在有效范围内 |

---

## 9. 交付件清单

### 9.1 目录结构

```
operators/softmax/
├── spec.md                          # 需求规范（已有）
├── api_report.md                    # API 探索报告（已有）
├── design.md                        # 设计文档（本文件）
├── softmax_golden.py                # Golden 参考实现（已有）
├── softmax_impl.py                  # 算子实现代码（待生成）
├── test_softmax.py                  # 测试代码（待生成）
└── output/                          # 运行输出（自动生成）
```

### 9.2 文件清单

| 文件 | 类型 | 说明 | 生成方式 |
|------|------|------|----------|
| spec.md | 需求 | 算子需求规范 | pypto-intent-understanding |
| api_report.md | 分析 | API 探索报告 | pypto-api-explorer |
| design.md | 设计 | 算子设计文档 | pypto-op-design（本 skill） |
| softmax_golden.py | 代码 | Golden 参考实现 | pypto-golden-generator |
| softmax_impl.py | 代码 | 算子核心实现 | 后续实现 |
| test_softmax.py | 代码 | 测试用例 | 后续实现 |

### 9.3 命名规范

| 项目 | 规范 | 示例 |
|------|------|------|
| 算子名称 | 小写字母 + 下划线 | `softmax` |
| 目录名 | 与算子名称一致 | `operators/softmax/` |
| Golden 文件 | `{op}_golden.py` | `softmax_golden.py` |
| 实现文件 | `{op}_impl.py` | `softmax_impl.py` |
| 测试文件 | `test_{op}.py` | `test_softmax.py` |

### 9.4 生成顺序

```
spec.md → api_report.md → design.md → softmax_golden.py → softmax_impl.py → test_softmax.py
```

---

## 质量自检

- [x] API 映射是否具体：通过（提供两种方案，明确 API 名称）
- [x] Tiling / Loop 是否说明理由：通过（说明判断依据和适用条件）
- [x] 验证方案是否覆盖典型配置：通过（覆盖 4 个典型配置）
- [x] 风险点是否具体：通过（列出 4 个已知约束和 4 个常见错误）
- [x] 是否存在空话或占位符：通过（无空话或占位符）
