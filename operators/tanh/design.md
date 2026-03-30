# TanH 算子设计文档

> **算子名称**: tanh
> **算子分类**: element-wise activation
> **生成时间**: 2026-03-29
> **基于**: spec.md, api_report.md

---

## 1. 概述

### 1.1 功能描述

Tanh（Hyperbolic Tangent）双曲正切激活函数。将输入逐元素映射到 [-1, 1] 区间，在 RNN/LSTM 中广泛用于门控机制。

### 1.2 数学公式

$$\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$$

### 1.3 数据流图

```
    输入 x                    输出 y
+--------------+         +--------------+
|  [m, n] 或   | ------->|  [m, n] 或   |
|  [b, m, n] 或|  tanh   |  [b, m, n] 或|
|  [b, s, m, n]|         |  [b, s, m, n]|
|  float16/32  |         |  float16/32  |
|  /bfloat16   |         |  /bfloat16   |
+--------------+         +--------------+

公式: tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))
输出范围: [-1, 1]
动态轴: b, s (batch, seq)
```

---

## 2. API 映射设计

### 2.1 数学公式分解

将公式拆解为基本操作步骤：

| 步骤 | 数学表达 | 说明 |
|------|----------|------|
| 1 | $e^x$ | 计算 exp(x) |
| 2 | $-x$ | 取负 |
| 3 | $e^{-x}$ | 计算 exp(-x) |
| 4 | $e^x - e^{-x}$ | 分子 |
| 5 | $e^x + e^{-x}$ | 分母 |
| 6 | $\frac{e^x - e^{-x}}{e^x + e^{-x}}$ | 最终结果 |

**dtype 策略说明**:
- PyPTO 没有直接的 `pypto.tanh` API
- 需要使用 `exp`, `mul`, `sub`, `add`, `div` 组合实现
- 所有 API 支持 FP16/BF16/FP32

### 2.2 PyPTO API 映射表

| 步骤 | 数学表达 | PyPTO API | 参数 | 文档路径 |
|------|----------|-----------|------|----------|
| 1 | $e^x$ | `pypto.exp(x)` | x: Tensor | docs/api/operation/pypto-exp.md |
| 2 | $-x$ | `pypto.mul(x, -1.0)` | x: Tensor, -1.0: float | docs/api/operation/pypto-mul.md |
| 3 | $e^{-x}$ | `pypto.exp(neg_x)` | neg_x: Tensor | docs/api/operation/pypto-exp.md |
| 4 | $e^x - e^{-x}$ | `pypto.sub(exp_x, exp_neg_x)` | exp_x: Tensor, exp_neg_x: Tensor | docs/api/operation/pypto-sub.md |
| 5 | $e^x + e^{-x}$ | `pypto.add(exp_x, exp_neg_x)` | exp_x: Tensor, exp_neg_x: Tensor | docs/api/operation/pypto-add.md |
| 6 | $\frac{numerator}{denominator}$ | `pypto.div(numerator, denominator)` | numerator: Tensor, denominator: Tensor | docs/api/operation/pypto-div.md |

### 2.3 计算步骤序列

```python
# Tanh 实现步骤
# Step 1: 计算 e^x
exp_x = pypto.exp(x)

# Step 2: 计算 -x
neg_x = pypto.mul(x, -1.0)

# Step 3: 计算 e^(-x)
exp_neg_x = pypto.exp(neg_x)

# Step 4: 计算分子 e^x - e^(-x)
numerator = pypto.sub(exp_x, exp_neg_x)

# Step 5: 计算分母 e^x + e^(-x)
denominator = pypto.add(exp_x, exp_neg_x)

# Step 6: 计算最终结果 (e^x - e^(-x)) / (e^x + e^(-x))
y = pypto.div(numerator, denominator)
```

### 2.4 设计依据

- **来源**: api_report.md §5 实现方案
- **说明**:
  - PyPTO 无直接 tanh API，需组合实现
  - 使用 exp/mul/sub/add/div 组合，完整支持 FP16/BF16/FP32
  - 参考实现：`examples/02_intermediate/operators/activation/activation.py` 中的激活函数模式

---

## 3. 数据规格设计

### 3.1 OperatorInput dataclass

```python
@dataclass
class TanhInput:
    x: Tensor  # 输入张量，2-4维，dtype: float16 / float32 / bfloat16
```

### 3.2 OperatorOutput dataclass

```python
@dataclass
class TanhOutput:
    y: Tensor  # 输出张量，shape 和 dtype 与输入相同，值域 [-1, 1]
```

### 3.3 中间 Tensor 定义

| 名称 | Shape | Dtype | 说明 |
|------|-------|-------|------|
| exp_x | 与 x 相同 | 与 x 相同 | exp(x) 的计算结果 |
| neg_x | 与 x 相同 | 与 x 相同 | -x |
| exp_neg_x | 与 x 相同 | 与 x 相同 | exp(-x) 的计算结果 |
| numerator | 与 x 相同 | 与 x 相同 | 分子 e^x - e^(-x) |
| denominator | 与 x 相同 | 与 x 相同 | 分母 e^x + e^(-x) |

### 3.4 数据格式选择

| Tensor | 格式 | 说明 |
|--------|------|------|
| x, y | ND (NCHW) | PyPTO 默认格式，element-wise 操作无需特殊格式 |
| 中间 Tensor | ND | 与输入保持一致 |

**理由**: tanh 为纯 element-wise 操作，无需 Cube 计算，使用 ND 格式即可。

### 3.5 动态轴定义

| 轴名称 | 含义 | 取值范围 |
|--------|------|----------|
| m | 特征维度 | [1, INT32_MAX] |
| n | 特征维度 | [1, INT32_MAX] |
| b | batch size | [1, INT32_MAX] |
| s | sequence length | [1, INT32_MAX] |

**动态轴**: b, s (batch, seq) - 在编译期未知，运行时确定

**约束**: Shape Size 不大于 INT32_MAX

### 3.6 JIT 装饰器配置

```python
@pypto.frontend.jit(
    runtime_options={"run_mode": pypto.RunMode.NPU}
)
def tanh_wrapper(x: pypto.Tensor(), out: pypto.Tensor()):
    """Tanh activation: (e^x - e^(-x)) / (e^x + e^(-x))"""
    ...
```

**说明**:
- `run_mode`: NPU 模式（需 CANN 环境），可选 SIM 模式用于调试
- 无需特殊 `pass_options` 配置

---

## 4. Tiling 策略

### 4.1 算子类型判断

- **类型**: Vector
- **判断依据**: tanh 为纯 element-wise 操作，仅包含逐元素的 exp、mul、sub、add、div，无 matmul 操作

### 4.2 TileShape 初值设置

```python
def configure_tiling(x):
    """根据输入 shape 动态设置 TileShape"""
    if len(x.shape) == 2:
        pypto.set_vec_tile_shapes(32, 128)
    elif len(x.shape) == 3:
        pypto.set_vec_tile_shapes(1, 32, 128)
    elif len(x.shape) == 4:
        pypto.set_vec_tile_shapes(1, 1, 32, 128)
```

**推荐 TileShape 配置**:

| 输入维度 | TileShape 示例 | 说明 |
|---------|---------------|------|
| 2D [1024, 1024] | [32, 128] | 性能_P0 配置 |
| 2D [32, 64] | [32, 128] | 功能_P0 配置 |
| 3D [2, 128, 256] | [1, 32, 128] | 功能_P1 配置 |
| 4D [1, 1, 64, 64] | [1, 1, 32, 128] | 功能_P2 配置 |

### 4.3 设置依据

1. **尾轴对齐**: fp16/bf16 尾轴设为 16 的倍数（32B 对齐），fp32 设为 8 的倍数
2. **TileShape 维度**: 与输入 shape 维度一致，最多 4 维
3. **参考来源**: relu/silu 算子的 tiling 配置

### 4.4 注意事项

- 尾轴必须满足 32B 对齐要求，否则编译失败或性能劣化
- 非尾轴无对齐要求，可根据 L0 容量灵活设置
- TileShape 维度数必须与输入 shape 维度数一致

### 4.5 判断依据与适用条件

- **判断依据**: tanh 为 Vector 算子，使用 `set_vec_tile_shapes` 即可
- **适用条件**: 适用于所有 shape（2D-4D）和 dtype（FP16/BF16/FP32）
- **不适用场景**: 输入维度超过 4 维时不支持（spec 约束）

---

## 5. Loop 结构设计

### 场景 A：不需要 Loop

> 适用于所有轴编译期已知、单次 Tile 可处理的算子（如逐元素运算）。

- **结论**: 不需要 pypto.loop
- **原因**: tanh 为纯 element-wise 操作，编译器可自动处理数据切分和并行，无需手动循环
- **适用条件**: 适用于所有 2-4D shape 和 dtype 的 tanh 计算
- **限制**: 无
- **处理方式**: 编译器自动处理数据切分，无需手动循环

**判断依据** (依据 quick_ref.md):
- 命中条件 4：所有轴编译期已知 & 单次 Tile 可处理
- element-wise 操作的并行性由编译器自动优化，无需显式 loop

---

## 6. 验证方案

### 6.1 Golden 函数设计

```python
def tanh_golden(x: torch.Tensor) -> torch.Tensor:
    """Tanh 参考实现"""
    return torch.tanh(x)
```

**说明**: 使用 PyTorch 内置 API `torch.tanh`，置信度最高

### 6.2 测试用例设计

#### 基于 spec.md 所有典型配置

| 配置名称 | 类型 | 优先级 | 参数 | 输入 Shape | 输出 Shape | 说明 |
|----------|------|--------|------|------------|------------|------|
| 性能_P0 | 性能 | P0 | - | [1024, 1024] | [1024, 1024] | 核心性能场景 |
| 功能_P0 | 功能 | P0 | - | [32, 64] | [32, 64] | 核心功能验证 |
| 功能_P1 | 功能 | P1 | - | [2, 128, 256] | [2, 128, 256] | 3维输入验证 |
| 功能_P2 | 功能 | P2 | - | [1, 1, 64, 64] | [1, 1, 64, 64] | 4维输入验证 |

**验证顺序**: 性能_P0 -> 功能_P0 -> 功能_P1 -> 功能_P2

#### 边界情况测试

| 场景 | 参数 | 说明 |
|------|------|------|
| 零值 | x = 0 | 验证 tanh(0) = 0 |
| 大正值 | x -> +inf | 验证 tanh -> 1 |
| 大负值 | x -> -inf | 验证 tanh -> -1 |
| 奇函数性质 | tanh(-x) = -tanh(x) | 验证奇函数性质 |
| 各 dtype | FP16/FP32/BF16 | 覆盖所有支持的数据类型 |

### 6.3 精度验证标准

| Dtype | atol | rtol |
|-------|------|------|
| float32 | 3e-3 | 3e-3 |
| float16 | 3e-3 | 3e-3 |
| bfloat16 | 3e-3 | 3e-3 |

**说明**: 精度标准来自 spec.md Section 7

---

## 7. 性能指标与开箱配置

### 7.1 性能目标

基于 spec.md 典型配置（性能类）的预期性能：

| 配置名称 | 类型 | 优先级 | 参数 | 输入 Shape | 输出 Shape | 预期目标 |
|----------|------|--------|------|------------|------------|----------|
| 性能_P0 | 性能 | P0 | - | [1024, 1024] | [1024, 1024] | 首跑精度成功性能的 2 倍 |

**性能目标**: 首跑精度成功性能的 2 倍（来自 spec.md Section 10）

### 7.2 开箱性能配置

```python
# 针对 [1024, 1024] shape 的推荐配置
pypto.set_vec_tile_shapes(32, 128)  # 尾轴 128 满足 32B 对齐
```

**说明**:
- 尾轴 128: fp16 下 128 * 2 bytes = 256B，满足 32B 对齐
- 首轴 32: 平衡并行度和内存占用

### 7.3 pass_options 配置

```python
# 无需特殊 pass_options 配置
pass_options = {}
```

**说明**: tanh 为简单 element-wise 操作，无需特殊 pass 优化。

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

- PyPTO 无直接 tanh API，需组合实现
- 所有输入 tensor 必须连续（`is_contiguous() == True`）
- 输入 tensor 不支持空 tensor（shape size >= 1）
- Shape 仅支持 2-4 维
- Shape Size 不超过 INT32_MAX

### 8.2 常见错误规避

| 风险 / 错误 | 触发场景 | 影响 / 原因 | 规避方法 |
|-------------|----------|-------------|----------|
| TileShape 不对齐 | 尾轴不满足 32B 对齐 | 编译失败或性能劣化 | fp16/bf16 尾轴设为 16 的倍数，fp32 设为 8 的倍数 |
| 输入不连续 | tensor.is_contiguous() == False | from_torch 失败 | 调用前确保 tensor 连续或调用 tensor.contiguous() |
| 特殊值处理 | x = NaN/Inf | 按 IEEE 754 标准传播 | 无需特殊处理，API 自动处理 |
| exp 溢出 | x 值过大 | 数值不稳定 | PyPTO exp API 已处理边界情况 |

### 8.3 特殊场景处理

1. **大值输入 (x > 20)**: exp(x) 可能溢出，但 tanh(x) -> 1。PyPTO div API 会处理。
2. **小值输入 (x < -20)**: exp(-x) 可能溢出，但 tanh(x) -> -1。PyPTO div API 会处理。
3. **零值 (x = 0)**: tanh(0) = 0，无需特殊处理。

### 8.4 实现建议

| 建议项 | 说明 |
|--------|------|
| 复用 configure_tiling | 参考 relu/silu 算子中的 tiling 配置函数 |
| 使用 torch.tanh | Golden 实现使用 PyTorch 内置 API，确保正确性 |
| 测试覆盖所有 dtype | 验证 FP32/FP16/BF16 三种 dtype 的精度和性能 |
| 参考 silu 实现模式 | silu 同样是组合实现，可参考其代码结构 |

---

## 9. 交付件清单

### 9.1 目录结构

```
operators/tanh/
├── spec.md                          # 需求规范（已有）
├── api_report.md                    # API 探索报告（已有）
├── design.md                        # 设计文档（本文件）
├── tanh_golden.py                   # Golden 参考实现（已有）
├── tanh_impl.py                     # 算子实现代码（待生成）
├── test_tanh.py                     # 测试代码（待生成）
├── README.md                        # 实现说明（待生成）
└── .orchestrator_state.json         # 状态文件（已有）
```

### 9.2 文件清单

| 文件 | 类型 | 说明 | 生成方式 |
|------|------|------|----------|
| spec.md | 需求 | 算子需求规范 | pypto-intent-understanding |
| api_report.md | 探索 | API 探索报告 | pypto-api-explorer |
| design.md | 设计 | 算子设计文档 | pypto-op-design（本 skill） |
| tanh_golden.py | 代码 | Golden 参考实现 | pypto-golden-generator |
| tanh_impl.py | 代码 | 算子核心实现 | pypto-op-develop（后续） |
| test_tanh.py | 代码 | 测试用例 | pypto-op-develop（后续） |
| README.md | 文档 | 实现说明 | pypto-op-develop（后续） |

### 9.3 命名规范

| 项目 | 规范 | 示例 |
|------|------|------|
| 算子名称 | 小写字母 | `tanh` |
| 目录名 | 与算子名称一致 | `operators/tanh/` |
| Golden 文件 | `{op}_golden.py` | `tanh_golden.py` |
| 实现文件 | `{op}_impl.py` | `tanh_impl.py` |
| 测试文件 | `test_{op}.py` | `test_tanh.py` |

### 9.4 生成顺序

```
spec.md -> api_report.md -> design.md -> tanh_golden.py -> tanh_impl.py -> test_tanh.py -> README.md
```

---

*生成时间: 2026-03-29*
*生成工具: pypto-op-orchestrator*
