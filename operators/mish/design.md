# mish 算子设计文档

> **算子名称**: mish
> **算子分类**: element-wise (activation)
> **生成时间**: 2026-03-30T09:35:00Z
> **基于**: spec.md, api_report.md

---

## 1. 概述

### 1.1 功能描述

Mish 是一种平滑的非单调激活函数，公式为 `mish(x) = x * tanh(softplus(x))`。相比 ReLU，Mish 提供了更平滑的梯度流，在深度网络中表现更好，广泛应用于 YOLOv4、ResNet 变体、EfficientNet 等现代神经网络。

### 1.2 数学公式

$$mish(x) = x \cdot \tanh(\text{softplus}(x)) = x \cdot \tanh(\ln(1 + e^x))$$

### 1.3 算法描述

不适用（简单逐元素计算，公式已完整描述计算逻辑）

### 1.4 数据流图

```
    输入 x                         输出 y
┌──────────────────┐         ┌──────────────────┐
│  [1D-4D tensor]  │         │  [1D-4D tensor]  │
│    float32       │ ──────▶ │    float32       │
└──────────────────┘  mish   └──────────────────┘

计算分解:
  step1: exp_x = exp(x)              # 指数运算
  step2: one_plus_exp = 1 + exp_x    # 加 1
  step3: softplus = ln(one_plus_exp) # 自然对数，即 softplus
  step4: sp_2x = 2 * softplus        # softplus 乘以 2
  step5: sigmoid_2x = sigmoid(sp_2x) # sigmoid 函数
  step6: tanh_out = 2 * sigmoid_2x - 1  # tanh 近似: tanh(x) = 2*sigmoid(2x)-1
  step7: y = x * tanh_out            # 逐元素乘法得到输出

动态轴: 支持所有维度动态 (使用隐式shape推断)
特殊处理: 4D输入需reshape为2D避免tiling编译问题
```

---

## 2. API 映射设计

### 2.1 数学公式分解

将 mish 公式拆解为基本操作步骤：

| 步骤 | 数学表达 | 说明 |
|------|----------|------|
| 1 | $e^x$ | 计算 x 的指数 |
| 2 | $1 + e^x$ | 加 1 |
| 3 | $\ln(1 + e^x)$ | 自然对数，即 softplus(x) |
| 4 | $2 \cdot \text{softplus}(x)$ | softplus 乘以 2 |
| 5 | $\sigma(2 \cdot \text{softplus}(x))$ | sigmoid 函数 |
| 6 | $2 \cdot \sigma(\cdot) - 1$ | tanh 近似: $\tanh(x) = 2\sigma(2x) - 1$ |
| 7 | $x \cdot \tanh(\cdot)$ | 逐元素乘法得到输出 |

### 2.2 PyPTO API 映射表

| 步骤 | 数学表达 | PyPTO API | 参数 | 文档路径 |
|------|----------|-----------|------|----------|
| 1 | $e^x$ | `pypto.exp(x)` | input: Tensor | `docs/api/operation/pypto-exp.md` |
| 2 | $1 + e^x$ | `pypto.add(exp_x, 1.0)` | input, other: 1.0 | `docs/api/operation/pypto-add.md` |
| 3 | $\ln(1 + e^x)$ | `pypto.log(one_plus_exp)` | input: Tensor | `docs/api/operation/pypto-log.md` |
| 4 | $2 \cdot \text{softplus}$ | `pypto.mul(softplus, 2.0)` | input, other: 2.0 | `docs/api/operation/pypto-mul.md` |
| 5 | $\sigma(\cdot)$ | `pypto.sigmoid(sp_2x)` | input: Tensor | `docs/api/operation/pypto-sigmoid.md` |
| 6 | $2 \cdot \sigma - 1$ | `pypto.add(pypto.mul(sigmoid_2x, 2.0), -1.0)` | - | `docs/api/operation/pypto-mul.md`, `pypto-add.md` |
| 7 | $x \cdot \tanh(\cdot)$ | `pypto.mul(x, tanh_out)` | input, other: Tensor | `docs/api/operation/pypto-mul.md` |

### 2.3 计算步骤序列

```python
# 伪代码展示计算流程
def mish_core(x: pypto.Tensor) -> pypto.Tensor:
    # Step 1-3: softplus(x) = ln(1 + exp(x))
    exp_x = pypto.exp(x)
    one_plus_exp = pypto.add(exp_x, 1.0)
    softplus = pypto.log(one_plus_exp)

    # Step 4-6: tanh(softplus) = 2 * sigmoid(2 * softplus) - 1
    # 使用数学恒等式: tanh(x) = 2 * sigmoid(2x) - 1
    sp_2x = pypto.mul(softplus, 2.0)
    sigmoid_2x = pypto.sigmoid(sp_2x)
    tanh_out = pypto.add(pypto.mul(sigmoid_2x, 2.0), -1.0)

    # Step 7: mish(x) = x * tanh(softplus(x))
    result = pypto.mul(x, tanh_out)

    return result
```

### 2.4 设计依据

- **来源**: api_report.md + docs/api/operation/ 官方文档
- **说明**:
  - PyPTO 没有直接的 tanh API，使用数学恒等式 $\tanh(x) = 2\sigma(2x) - 1$ 实现
  - 该方法与 GELU 算子中的 tanh 近似实现一致，已在 `operators/gelu/gelu_impl.py` 中验证可行
  - 所有操作均为 element-wise，无 matmul，因此为 Vector 类型算子

---

## 3. 数据规格设计

### 3.1 OperatorInput dataclass

```python
@dataclass
class MishInput:
    x: Tensor  # 输入张量, shape: [1D-4D], dtype: float32
```

### 3.2 OperatorOutput dataclass

```python
@dataclass
class MishOutput:
    y: Tensor  # 输出张量, shape: [与输入相同], dtype: float32
```

### 3.3 中间 Tensor 定义

| 名称 | Shape | Dtype | 说明 |
|------|-------|-------|------|
| exp_x | [与输入相同] | float32 | exp(x) 的结果 |
| one_plus_exp | [与输入相同] | float32 | 1 + exp(x) 的结果 |
| softplus | [与输入相同] | float32 | ln(1 + exp(x)) 的结果 |
| sp_2x | [与输入相同] | float32 | 2 * softplus 的结果 |
| sigmoid_2x | [与输入相同] | float32 | sigmoid(2 * softplus) 的结果 |
| tanh_out | [与输入相同] | float32 | 2 * sigmoid - 1 的结果 (即 tanh 近似) |

### 3.4 数据格式选择

| Tensor | 格式 | 说明 |
|--------|------|------|
| x (输入) | ND | 默认格式，from_torch 自动推导 |
| y (输出) | ND | 与输入格式一致 |

### 3.5 动态轴定义

| 轴名称 | 含义 | 取值范围 |
|--------|------|----------|
| dim0 | 第一维度 (batch/row) | [1, INT32_MAX] |
| dim1 | 第二维度 (col/seq) | [1, INT32_MAX] |
| dim2 | 第三维度 (可选) | [1, INT32_MAX] |
| dim3 | 第四维度 (可选) | [1, INT32_MAX] |

**动态 shape 声明方式**:
```python
x: pypto.Tensor([], pypto.DT_FP32)  # 隐式 shape 推断
```

### 3.6 JIT 装饰器配置

```python
@pypto.frontend.jit(
    runtime_options={"run_mode": pypto.RunMode.NPU}
)
def mish_kernel(x: pypto.Tensor([], pypto.DT_FP32), out: pypto.Tensor([], pypto.DT_FP32)):
    ...
```

---

## 4. Tiling 策略

### 4.1 算子类型判断

- **类型**: Vector
- **判断依据**: mish 算子仅包含逐元素操作 (exp, log, add, mul, sigmoid)，无 matmul 操作

### 4.2 TileShape 初值设置

```python
# 用户指定配置
pypto.set_vec_tile_shapes(64, 128)
```

### 4.3 设置依据

1. **用户明确要求**: spec.md 中指定使用 `pypto.set_vec_tile_shapes(64, 128)` 用于 2D kernels
2. **4D 输入处理**: 4D 输入需要 reshape 为 2D 避免 tiling 编译问题
3. **尾轴对齐**: 128 是 8 的倍数，满足 fp32 尾轴 32B 对齐要求（8 个 fp32 = 32B）

### 4.4 注意事项

- 1D 输入需要 reshape 为 2D（因为 exp/mul/add 限制 2-4 维）
- 4D 输入需要 flatten 为 2D 避免 tiling 编译问题
- TileShape 维度数不能超过输入维度数

### 4.5 判断依据与适用条件

- **判断依据**: 算子为纯 Vector 类型，无 Cube 操作
- **适用条件**: 适用于所有 1D-4D float32 输入
- **不适用场景**: 超过 4D 的输入需要额外处理

---

## 5. Loop 结构设计

### 场景 A：不需要 Loop

> 适用于所有轴编译期已知、单次 Tile 可处理的算子（如逐元素运算）。

- **结论**：不需要 pypto.loop
- **原因**：mish 是逐元素运算，编译器自动处理数据切分，无需手动循环
- **适用条件**：所有 1D-4D float32 输入
- **限制**：需要确保输入 tensor 连续（contiguous）
- **处理方式**：编译器自动处理数据切分，无需手动循环

**Loop 判断依据**（按 quick_ref.md §2.1 判据表）:
- 不命中条件 1（动态轴）：虽然支持动态轴，但逐元素运算不需要显式循环遍历
- 不命中条件 2（多步骤分块）：单次 Tile 可处理
- 不命中条件 3（动态轴范围大）：编译器自动处理
- **命中条件 4**：所有轴编译期已知 & 单次 Tile 可处理 → 不需要 Loop

---

## 6. 验证方案

### 6.1 Golden 函数设计

```python
def mish_golden(x: torch.Tensor) -> torch.Tensor:
    """Mish 参考实现

    公式: mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))
    """
    softplus_x = torch.nn.functional.softplus(x)
    return x * torch.tanh(softplus_x)
```

### 6.2 测试用例设计

#### 基于 spec.md 所有典型配置

| 配置名称 | 类型 | 优先级 | 参数 | 输入 Shape | 输出 Shape | 说明 |
|----------|------|--------|------|------------|------------|------|
| 功能_1D | 功能 | P0 | - | [1024] | [1024] | 1D基础功能验证 |
| 功能_2D | 功能 | P0 | - | [128, 1024] | [128, 1024] | 2D基础功能验证 |
| 功能_3D | 功能 | P0 | - | [2, 128, 1024] | [2, 128, 1024] | 3D功能验证 |
| 功能_4D | 功能 | P0 | - | [2, 4, 128, 1024] | [2, 4, 128, 1024] | 4D功能验证(reshape为2D处理) |
| 性能_2D | 性能 | P0 | - | [4096, 4096] | [4096, 4096] | 核心性能场景 |

#### 边界情况测试

| 场景 | 参数 | 说明 |
|------|------|------|
| 零值输入 | x = zeros(128, 128) | 验证 mish(0) = 0 |
| 大值输入 | x = randn(128, 128) * 100 | 验证数值稳定性 |
| 小值输入 | x = randn(128, 128) * 0.001 | 验证小值精度 |

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
| 性能_2D | 性能 | P0 | - | [4096, 4096] | [4096, 4096] | 首跑成功后优化 2x |

### 7.2 开箱性能配置

```python
# 用户指定的 tiling 配置
pypto.set_vec_tile_shapes(64, 128)
```

### 7.3 pass_options 配置

不适用（本算子不需要特殊 pass 配置）

### 7.4 runtime_options 配置

```python
@pypto.frontend.jit(
    runtime_options={"run_mode": pypto.RunMode.NPU}
)
def mish_kernel(x: pypto.Tensor([], pypto.DT_FP32), out: pypto.Tensor([], pypto.DT_FP32)):
    ...
```

---

## 8. 风险点与注意事项

### 8.1 已知约束

- PyPTO 没有 tanh API，必须使用 sigmoid 近似: $\tanh(x) = 2\sigma(2x) - 1$
- exp/mul/add 限制 2-4 维，1D 输入需要 reshape
- 4D 输入需要 reshape 为 2D 避免 tiling 编译问题
- sigmoid 公开文档仅支持 DT_FP32，FP16 需验证

### 8.2 常见错误规避

| 风险 / 错误 | 触发场景 | 影响 / 原因 | 规避方法 |
|-------------|----------|-------------|----------|
| shape 维度错误 | 1D 输入直接传入 | API 限制 2-4 维，触发编译错误 | 1D 输入 reshape 为 2D |
| 4D tiling 编译失败 | 4D 输入直接计算 | tiling 编译问题 | 4D flatten 为 2D 计算 |
| 输入不连续 | 非连续 tensor 传入 | from_torch 要求连续 | 调用 `.contiguous()` |
| 标量乘法类型错误 | 使用 `x * 2.0` 语法 | 可能触发类型错误 | 使用 `pypto.mul(x, 2.0)` |

### 8.3 特殊场景处理

**1D 输入处理**:
```python
if x.ndim == 1:
    x_2d = x.unsqueeze(0)  # [N] -> [1, N]
    # ... 计算 ...
    y = y_2d.squeeze(0)  # [1, N] -> [N]
```

**4D 输入处理**:
```python
if x.ndim == 4:
    original_shape = x.shape
    x_2d = x.reshape(-1, x.shape[-1])  # [B, H, W, C] -> [B*H*W, C]
    # ... 计算 ...
    y = y_2d.reshape(original_shape)
```

### 8.4 实现建议

| 建议项 | 说明 |
|--------|------|
| 复用 gelu_impl.py 的 tanh 近似模式 | gelu 中已验证 tanh(x) = 2*sigmoid(2x)-1 的正确性 |
| 使用隐式 shape 推断 | 支持动态轴，使用 `pypto.Tensor([], pypto.DT_FP32)` |
| wrapper 函数处理输入 | 检查连续性、处理 1D/4D reshape |

---

## 9. 交付件清单

### 9.1 目录结构

```
operators/mish/
├── spec.md                          # 需求规范（已有）
├── api_report.md                    # API 探索报告（已有）
├── design.md                        # 设计文档（本文件）
├── mish_golden.py                   # Golden 参考实现（已有）
├── mish_impl.py                     # 算子实现代码
├── test_mish.py                     # 测试代码
└── output/                          # 运行输出（自动生成）
```

### 9.2 文件清单

| 文件 | 类型 | 说明 | 生成方式 |
|------|------|------|----------|
| spec.md | 需求 | 算子需求规范 | pypto-intent-understanding |
| api_report.md | 设计 | API 探索报告 | pypto-api-explorer |
| design.md | 设计 | 算子设计文档 | pypto-op-design（本文件） |
| mish_golden.py | 代码 | Golden 参考实现 | pypto-golden-generator |
| mish_impl.py | 代码 | 算子核心实现 | 后续实现 |
| test_mish.py | 代码 | 测试用例 | 后续实现 |

### 9.3 命名规范

| 项目 | 规范 | 示例 |
|------|------|------|
| 算子名称 | 小写字母 | `mish` |
| 目录名 | 与算子名称一致 | `operators/mish/` |
| Golden 文件 | `{op}_golden.py` | `mish_golden.py` |
| 实现文件 | `{op}_impl.py` | `mish_impl.py` |
| 测试文件 | `test_{op}.py` | `test_mish.py` |

### 9.4 生成顺序

```
spec.md → api_report.md → design.md → mish_golden.py → mish_impl.py → test_mish.py
```
