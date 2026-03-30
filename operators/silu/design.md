# silu 算子设计文档

> **算子名称**: silu
> **算子分类**: element-wise activation
> **生成时间**: 2026-03-28
> **基于**: spec.md, api_report.md

---

## 1. 概述

### 1.1 功能描述

SiLU (Sigmoid Linear Unit) 激活函数，也称为 Swish。将输入逐元素乘以 sigmoid 值，在 LLM 中广泛使用（如 LLaMA、Mistral）。该算子为简单的 element-wise 操作，计算公式为 `y = x * sigmoid(x)`。

### 1.2 数学公式

$$y = x \cdot \sigma(x) = \frac{x}{1 + e^{-x}}$$

其中 $\sigma(x) = \frac{1}{1 + e^{-x}}$ 为 sigmoid 函数。

### 1.3 数据流图

```
    输入 x                    输出 y
+--------------+         +--------------+
|  [b, s, n, d] | ------->|  [b, s, n, d] |
|   float32     |  silu   |   float32     |
+--------------+         +--------------+

公式: y = x * sigmoid(x) = x / (1 + exp(-x))
动态轴: b, s (batch, seq)
```

---

## 2. API 映射设计

### 2.1 数学公式分解

将公式拆解为基本操作步骤：

| 步骤 | 数学表达 | 说明 |
|------|----------|------|
| 1 | $\sigma(x) = \frac{1}{1 + e^{-x}}$ | 计算 sigmoid |
| 2 | $y = x \cdot \sigma(x)$ | 逐元素乘法 |

**dtype 策略说明**:
- **FP32**: 可直接使用 `pypto.sigmoid(x)` API（步骤简化为 2 步）
- **FP16/BF16**: 由于 `pypto.sigmoid` 仅支持 FP32，需手动展开 sigmoid（步骤展开为 5 步）

### 2.2 PyPTO API 映射表

#### 方案 A：FP32 路径（直接 sigmoid）

| 步骤 | 数学表达 | PyPTO API | 参数 | 文档路径 |
|------|----------|-----------|------|----------|
| 1 | $\sigma(x)$ | `pypto.sigmoid(x)` | x: Tensor | docs/api/operation/pypto-sigmoid.md |
| 2 | $x \cdot \sigma(x)$ | `pypto.mul(x, sigmoid_x)` | x: Tensor, sigmoid_x: Tensor | docs/api/operation/pypto-mul.md |

#### 方案 B：FP16/BF16 路径（手动展开）

| 步骤 | 数学表达 | PyPTO API | 参数 | 文档路径 |
|------|----------|-----------|------|----------|
| 1 | $-x$ | `pypto.mul(x, -1.0)` | x: Tensor, -1.0: float | docs/api/operation/pypto-mul.md |
| 2 | $e^{-x}$ | `pypto.exp(neg_x)` | neg_x: Tensor | docs/api/operation/pypto-exp.md |
| 3 | $1 + e^{-x}$ | `pypto.add(exp_neg_x, 1.0)` | exp_neg_x: Tensor, 1.0: float | docs/api/operation/pypto-add.md |
| 4 | $\frac{1}{1 + e^{-x}}$ | `pypto.reciprocal(one_plus_exp)` | one_plus_exp: Tensor | docs/api/operation/pypto-reciprocal.md |
| 5 | $x \cdot \sigma(x)$ | `pypto.mul(x, sigmoid_x)` | x: Tensor, sigmoid_x: Tensor | docs/api/operation/pypto-mul.md |

### 2.3 计算步骤序列

```python
# 方案 A：FP32 路径
sigmoid_x = pypto.sigmoid(x)
y = pypto.mul(x, sigmoid_x)

# 方案 B：FP16/BF16 路径
neg_x = pypto.mul(x, -1.0)
exp_neg_x = pypto.exp(neg_x)
one_plus_exp = pypto.add(exp_neg_x, 1.0)
sigmoid_x = pypto.reciprocal(one_plus_exp)
y = pypto.mul(x, sigmoid_x)
```

### 2.4 设计依据

- **来源**: api_report.md §3 API 映射 + §5 实现方案
- **说明**:
  - 选择条件分支方案：根据 dtype 选择最优路径
  - FP32 路径优先使用 `pypto.sigmoid`（性能更优，API 更简洁）
  - FP16/BF16 路径必须手动展开（`pypto.sigmoid` 不支持这两种 dtype）
  - 参考实现：`examples/02_intermediate/operators/activation/activation.py` (行 110-123)

---

## 3. 数据规格设计

### 3.1 OperatorInput dataclass

```python
@dataclass
class SiluInput:
    x: Tensor  # 输入张量，任意 shape，dtype: float16 / float32 / bfloat16
```

### 3.2 OperatorOutput dataclass

```python
@dataclass
class SiluOutput:
    y: Tensor  # 输出张量，shape 和 dtype 与输入相同
```

### 3.3 中间 Tensor 定义

| 名称 | Shape | Dtype | 说明 |
|------|-------|-------|------|
| sigmoid_x | 与 x 相同 | 与 x 相同 | sigmoid(x) 的计算结果 |
| neg_x | 与 x 相同 | 与 x 相同 | -x（仅 FP16/BF16 路径） |
| exp_neg_x | 与 x 相同 | 与 x 相同 | exp(-x)（仅 FP16/BF16 路径） |
| one_plus_exp | 与 x 相同 | 与 x 相同 | 1 + exp(-x)（仅 FP16/BF16 路径） |

### 3.4 数据格式选择

| Tensor | 格式 | 说明 |
|--------|------|------|
| x, y | ND (NCHW) | PyPTO 默认格式，element-wise 操作无需特殊格式 |
| 中间 Tensor | ND | 与输入保持一致 |

**理由**: silu 为纯 element-wise 操作，无需 Cube 计算，使用 ND 格式即可。

### 3.5 动态轴定义

| 轴名称 | 含义 | 取值范围 |
|--------|------|----------|
| b | batch size | [1, INT32_MAX] |
| s | sequence length | [1, INT32_MAX] |
| n | num_heads / hidden dim split | [1, INT32_MAX] |
| d | head_dim | [1, INT32_MAX] |

**动态轴**: b, s (batch, seq) - 在编译期未知，运行时确定

### 3.6 JIT 装饰器配置

```python
@pypto.frontend.jit(
    runtime_options={"run_mode": pypto.RunMode.NPU}
)
def silu(x: pypto.Tensor(), out: pypto.Tensor()):
    """SiLU activation: x * sigmoid(x)"""
    ...
```

**说明**:
- `run_mode`: NPU 模式（需 CANN 环境），可选 SIM 模式用于调试
- 无需特殊 `pass_options` 配置

---

## 4. Tiling 策略

### 4.1 算子类型判断

- **类型**: Vector
- **判断依据**: silu 为纯 element-wise 操作，仅包含逐元素的 mul、exp、add、reciprocal，无 matmul 操作

### 4.2 TileShape 初值设置

```python
def configure_tiling(x):
    """根据输入 shape 动态设置 TileShape"""
    if len(x.shape) >= 2:
        # 多维输入：每个维度设置 tile size 为 32
        tile_list = [32 for _ in range(len(x.shape))]
        pypto.set_vec_tile_shapes(*tile_list)
    else:
        # 1D 输入：使用默认 tile
        pypto.set_vec_tile_shapes(32, 128)
```

**推荐 TileShape 配置**:

| 输入维度 | TileShape 示例 | 说明 |
|---------|---------------|------|
| 3D [1, 4096, 4096] | [1, 32, 128] | 性能_P0 配置 |
| 3D [2, 1024, 512] | [1, 32, 128] | 功能_P0 配置 |
| 4D [b, s, n, d] | [1, 32, 32, 32] | 通用 4D 配置 |

### 4.3 设置依据

1. **尾轴对齐**: fp16/bf16 尾轴设为 16 的倍数（32B 对齐），fp32 设为 8 的倍数
2. **TileShape 维度**: 与输入 shape 维度一致，最多 4 维
3. **参考来源**: `examples/02_intermediate/operators/activation/activation.py` (行 80-86)

### 4.4 注意事项

- 尾轴必须满足 32B 对齐要求，否则编译失败或性能劣化
- 非尾轴无对齐要求，可根据 L0 容量灵活设置
- 大 shape 场景（如 [1, 4096, 4096]）需合理设置 tile size 以平衡并行度和内存占用

### 4.5 判断依据与适用条件

- **判断依据**: silu 为 Vector 算子，使用 `set_vec_tile_shapes` 即可
- **适用条件**: 适用于所有 shape（1D-4D）和 dtype（FP16/BF16/FP32）
- **不适用场景**: 无（element-wise 操作无特殊限制）

---

## 5. Loop 结构设计

### 场景 A：不需要 Loop

> 适用于所有轴编译期已知、单次 Tile 可处理的算子（如逐元素运算）。

- **结论**: 不需要 pypto.loop
- **原因**: silu 为纯 element-wise 操作，编译器可自动处理数据切分和并行，无需手动循环
- **适用条件**: 适用于所有 shape 和 dtype 的 silu 计算
- **限制**: 无
- **处理方式**: 编译器自动处理数据切分，无需手动循环

**判断依据** (依据 quick_ref.md §2.1):
- ✅ 命中条件 4：所有轴编译期已知 & 单次 Tile 可处理
- 虽然存在动态轴 b, s，但 element-wise 操作的并行性由编译器自动优化，无需显式 loop

**参考实现验证**: `examples/02_intermediate/operators/activation/activation.py` 中的 `silu_activation_kernel` 未使用任何 loop 结构。

---

## 6. 验证方案

### 6.1 Golden 函数设计

```python
def silu_golden(x: torch.Tensor) -> torch.Tensor:
    """silu 参考实现"""
    return torch.nn.functional.silu(x)
```

**说明**: 使用 PyTorch 内置 API `torch.nn.functional.silu`，置信度 ⭐⭐⭐⭐⭐

### 6.2 测试用例设计

#### 基于 spec.md 所有典型配置

| 配置名称 | 类型 | 优先级 | 参数 | 输入 Shape | 输出 Shape | 说明 |
|----------|------|--------|------|------------|------------|------|
| 性能_P0 | 性能 | P0 | dtype=float16 | [1, 4096, 4096] | [1, 4096, 4096] | LLaMA MLP 典型规模 |
| 功能_P0 | 功能 | P0 | dtype=float32 | [2, 1024, 512] | [2, 1024, 512] | 功能验证基础配置 |
| 功能_P1 | 功能 | P1 | dtype=bfloat16 | [4, 2048, 1024] | [4, 2048, 1024] | BF16 精度验证 |
| 边界_P0 | 边界 | P0 | dtype=float32 | [1, 1, 1] | [1, 1, 1] | 最小 shape 验证 |

**验证顺序**: 性能_P0 → 功能_P0 → 功能_P1 → 边界_P0

#### 边界情况测试

| 场景 | 参数 | 说明 |
|------|------|------|
| 零值 | x = 0 | 验证 silu(0) = 0 * 0.5 = 0 |
| 大正值 | x = 100 | 验证 silu(100) ≈ 100 * 1 = 100 |
| 大负值 | x = -100 | 验证 silu(-100) ≈ -100 * 0 ≈ 0 |
| NaN 传播 | x = NaN | 验证 NaN 按 IEEE 754 传播 |
| Inf 传播 | x = ±Inf | 验证 Inf 按 IEEE 754 传播 |

### 6.3 精度验证标准

| Dtype | atol | rtol |
|-------|------|------|
| float32 | 0.001 | 0.001 |
| float16 | 0.001 | 0.001 |
| bfloat16 | 0.001 | 0.001 |

**说明**: 精度标准来自 spec.md §7

---

## 7. 性能指标与开箱配置

### 7.1 性能目标

基于 spec.md 典型配置（性能类）的预期性能：

| 配置名称 | 类型 | 优先级 | 参数 | 输入 Shape | 输出 Shape | 预期 kernel 耗时 |
|----------|------|--------|------|------------|------------|------------------|
| 性能_P0 | 性能 | P0 | dtype=float16 | [1, 4096, 4096] | [1, 4096, 4096] | < 1ms (参考值) |

**性能目标**: 首跑精度成功性能的 2 倍（来自 spec.md §10）

### 7.2 开箱性能配置

```python
# 针对 [1, 4096, 4096] shape 的推荐配置
pypto.set_vec_tile_shapes(1, 32, 128)  # 尾轴 128 满足 32B 对齐
```

**说明**:
- 尾轴 128: fp16 下 128 * 2 bytes = 256B，满足 32B 对齐
- 中间轴 32: 平衡并行度和内存占用
- 首轴 1: batch 维度，保持不变

### 7.3 pass_options 配置

```python
# 无需特殊 pass_options 配置
pass_options = {}
```

**说明**: silu 为简单 element-wise 操作，无需特殊 pass 优化。

### 7.4 runtime_options 配置

```python
runtime_options = {
    "run_mode": pypto.RunMode.NPU  # 或 pypto.RunMode.SIM 用于调试
}
```

**说明**: 默认使用 NPU 模式，SIM 模式用于功能验证和调试。

---

## 8. 风险点与注意事项

### 8.1 已知约束

- `pypto.sigmoid` 仅支持 FP32，FP16/BF16 必须使用手动展开方案
- 所有输入 tensor 必须连续（`is_contiguous() == True`）
- 输入 tensor 不支持空 tensor（shape size ≥ 1）

### 8.2 常见错误规避

| 风险 / 错误 | 触发场景 | 影响 / 原因 | 规避方法 |
|-------------|----------|-------------|----------|
| dtype 不匹配 | FP16/BF16 使用 pypto.sigmoid | 编译失败或运行时错误 | 使用 dtype 判断，FP16/BF16 采用手动展开 |
| TileShape 不对齐 | 尾轴不满足 32B 对齐 | 编译失败或性能劣化 | fp16/bf16 尾轴设为 16 的倍数，fp32 设为 8 的倍数 |
| 输入不连续 | tensor.is_contiguous() == False | from_torch 失败 | 调用前确保 tensor 连续或调用 tensor.contiguous() |
| 特殊值处理 | x = NaN/Inf | 按 IEEE 754 标准传播 | 无需特殊处理，API 自动处理 |

### 8.3 特殊场景处理

1. **大值输入 (x > 20)**: exp(-x) 接近 0，sigmoid 接近 1，silu 接近 x。PyPTO exp API 已处理数值稳定性。
2. **小值输入 (x < -20)**: exp(-x) 接近 +inf，sigmoid 接近 0，silu 接近 0。PyPTO reciprocal API 已处理边界。
3. **零值 (x = 0)**: silu(0) = 0 * 0.5 = 0，无需特殊处理。

### 8.4 实现建议

| 建议项 | 说明 |
|--------|------|
| 采用条件分支 | 根据 dtype 选择 FP32 路径或 FP16/BF16 路径 |
| 复用 configure_tiling | 参考 `activation.py` 中的 tiling 配置函数 |
| 使用 torch.nn.functional.silu | Golden 实现使用 PyTorch 内置 API，确保正确性 |
| 测试覆盖所有 dtype | 验证 FP32/FP16/BF16 三种 dtype 的精度和性能 |

---

## 9. 交付件清单

### 9.1 目录结构

```
operators/silu/
├── spec.md                          # 需求规范（已有）
├── api_report.md                    # API 探索报告（已有）
├── design.md                        # 设计文档（本文件）
├── silu_golden.py                   # Golden 参考实现（已有）
├── silu_impl.py                     # 算子实现代码（待生成）
├── test_silu.py                     # 测试代码（待生成）
└── .orchestrator_state.json         # 状态文件（已有）
```

### 9.2 文件清单

| 文件 | 类型 | 说明 | 生成方式 |
|------|------|------|----------|
| spec.md | 需求 | 算子需求规范 | pypto-intent-understanding |
| api_report.md | 探索 | API 探索报告 | pypto-api-explorer |
| design.md | 设计 | 算子设计文档 | pypto-op-design（本 skill） |
| silu_golden.py | 代码 | Golden 参考实现 | pypto-golden-generator |
| silu_impl.py | 代码 | 算子核心实现 | pypto-op-develop（后续） |
| test_silu.py | 代码 | 测试用例 | pypto-op-develop（后续） |

### 9.3 命名规范

| 项目 | 规范 | 示例 |
|------|------|------|
| 算子名称 | 小写字母 | `silu` |
| 目录名 | 与算子名称一致 | `operators/silu/` |
| Golden 文件 | `{op}_golden.py` | `silu_golden.py` |
| 实现文件 | `{op}_impl.py` | `silu_impl.py` |
| 测试文件 | `test_{op}.py` | `test_silu.py` |

### 9.4 生成顺序

```
spec.md → api_report.md → design.md → silu_golden.py → silu_impl.py → test_silu.py
```

---

*生成时间: 2026-03-28*
*生成工具: pypto-op-design*
