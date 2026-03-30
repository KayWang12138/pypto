# exp 算子设计文档

> **算子名称**: exp
> **算子分类**: element-wise
> **生成时间**: 2026-03-30T05:11:00Z
> **基于**: spec.md

---

## 1. 概述

### 1.1 功能描述

逐元素计算指数函数，对输入张量中的每个元素计算 e 的该元素次方。

### 1.2 数学公式

$y = \exp(x) = e^x$

### 1.3 算法描述

```
Algorithm: Exp (Element-wise with 4D Reshape)
────────────────────────────────────
输入: x ∈ R^{shape}, dtype=FP32/FP16
输出: y ∈ R^{shape}, dtype=与输入相同

1. 检测输入维度:
   - 如果是 4D [b, s, n, d]: 记录原始 shape，reshape 为 2D [-1, d]
   - 如果是 2D [m, n]: 直接处理
2. 设置 2D tiling: pypto.set_vec_tile_shapes(64, 128)
3. 计算: y = pypto.exp(x)
4. 如果执行了 reshape: 恢复原始 shape
5. 输出写回: output[:] = y
```

### 1.4 数据流图

```
    输入 x                          输出 y
+------------------+           +------------------+
|  [b, s, n, d]    |           |  [b, s, n, d]    |
|  或 [m, n]       | --------> |  或 [m, n]       |
|  float32/float16 |   exp     |  float32/float16 |
+------------------+           +------------------+

公式: y = exp(x) = e^x (逐元素计算)
动态轴: b, s (4D) 或 m (2D)
```

---

## 2. API 映射设计

### 2.1 数学公式分解

将公式拆解为基本操作步骤：

| 步骤 | 数学表达 | 说明 |
|------|----------|------|
| 1 | y = exp(x) | 逐元素计算 e 的 x 次方 |

### 2.2 PyPTO API 映射表

| 步骤 | 数学表达 | PyPTO API | 参数 | 文档路径 |
|------|----------|-----------|------|----------|
| 1 | y = exp(x) | `pypto.exp(x)` | input: Tensor | `docs/api/operation/pypto-exp.md` |
| 2 | reshape 4D→2D | `pypto.reshape(x, [-1, d])` | input, shape | `docs/api/operation/pypto-reshape.md` |
| 3 | 输出写回 | `output[:] = y` | slice assignment | PyPTO 语法 |

### 2.3 计算步骤序列

```python
# 伪代码展示计算流程
# 1. 设置 tiling
pypto.set_vec_tile_shapes(64, 128)

# 2. 直接计算 exp
result = pypto.exp(x)

# 3. 输出写回
output[:] = result
```

### 2.4 设计依据

- 来源：api_report.md + spec.md + docs/api/operation/pypto-exp.md
- 说明：pypto.exp 是 PyPTO 提供的直接 API，支持 FP32/FP16/BF16，2-4 维输入。使用 2D tiling 避免编译问题。

---

## 3. 数据规格设计

### 3.1 OperatorInput dataclass

```python
@dataclass
class ExpInput:
    x: Tensor  # 输入张量, shape: [b, s, n, d] 或 [m, n], dtype: FP32/FP16
```

### 3.2 OperatorOutput dataclass

```python
@dataclass
class ExpOutput:
    y: Tensor  # 输出张量, shape: 与输入相同, dtype: 与输入相同
```

### 3.3 中间 Tensor 定义

| 名称 | Shape | Dtype | 说明 |
|------|-------|-------|------|
| result | 与输入相同 | 与输入相同 | exp 计算结果 |

### 3.4 数据格式选择

| Tensor | 格式 | 说明 |
|--------|------|------|
| x | ND | 默认格式，from_torch 自动推导 |
| y | ND | 与输入格式一致 |

### 3.5 动态轴定义

| 轴名称 | 含义 | 取值范围 |
|--------|------|----------|
| b | batch size | [1, INT32_MAX] |
| s | sequence length | [1, INT32_MAX] |
| m | rows (2D) | [1, INT32_MAX] |

### 3.6 JIT 装饰器配置

```python
@pypto.frontend.jit
def exp_kernel(
    x: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC], pypto.DT_FP32),
    output: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC], pypto.DT_FP32)
):
    ...
```

---

## 4. Tiling 策略

### 4.1 算子类型判断

- **类型**: Vector
- **判断依据**: 纯逐元素计算，不涉及矩阵乘法，使用 `pypto.set_vec_tile_shapes()` 配置 tiling

### 4.2 TileShape 初值设置

```python
pypto.set_vec_tile_shapes(64, 128)
```

### 4.3 设置依据

1. **尾轴 128**:
   - FP32: 128 * 4 = 512 字节，满足 32B 对齐要求
   - FP16: 128 * 2 = 256 字节，满足 32B 对齐要求
   - 较大的尾轴可提高向量化效率

2. **首轴 64**:
   - 平衡 L0 容量与并行度
   - 避免过大的 tile 导致 L0 溢出

3. **使用 2D tiling**:
   - spec 要求 4D 输入 reshape 到 2D 后调用 2D kernel
   - 避免 4D tiling 可能的编译问题

### 4.4 注意事项

- 尾轴 128 满足 FP16/FP32 的 32B 对齐要求
- 4D 输入需先 reshape 到 2D，再使用此 tiling 配置

### 4.5 判断依据与适用条件

- 判断依据：exp 是 Vector 类型逐元素操作，使用 set_vec_tile_shapes
- 适用条件：2D 输入或 reshape 后的 4D 输入，FP32/FP16 dtype
- 不适用场景：BF16 dtype 时需验证对齐要求

---

## 5. Loop 结构设计

### 场景 A：不需要 Loop

> 适用于所有轴编译期已知、单次 Tile 可处理的算子（如逐元素运算）。

- **结论**：不需要 pypto.loop
- **原因**：exp 是简单逐元素操作，编译器自动处理数据切分，单次 Tile 可覆盖全部数据
- **适用条件**：2D 或 reshape 后的 2D 输入，无跨 tile 依赖
- **限制**：超大 shape 可能需要调整 TileShape
- **处理方式**：编译器自动处理数据切分，无需手动循环

---

## 6. 验证方案

### 6.1 Golden 函数设计

```python
def exp_golden(x: torch.Tensor) -> torch.Tensor:
    """exp 参考实现"""
    return torch.exp(x)
```

### 6.2 测试用例设计

#### 基于 spec.md 所有典型配置

| 配置名称 | 类型 | 优先级 | 参数 | 输入 Shape | 输出 Shape | 说明 |
|----------|------|--------|------|------------|------------|------|
| 功能_2D_FP32 | 功能 | P0 | - | [1024, 1024] | [1024, 1024] | 2D 基础功能验证 |
| 功能_4D_FP32 | 功能 | P0 | - | [2, 4096, 16, 128] | [2, 4096, 16, 128] | 4D 动态轴验证 |
| 功能_2D_FP16 | 功能 | P0 | - | [1024, 1024] | [1024, 1024] | FP16 功能验证 |
| 性能_FP32 | 性能 | P1 | - | [4, 8192, 32, 128] | [4, 8192, 32, 128] | FP32 性能场景 |
| 性能_FP16 | 性能 | P1 | - | [4, 8192, 32, 128] | [4, 8192, 32, 128] | FP16 性能场景 |

#### 边界情况测试（可选）

| 场景 | 参数 | 说明 |
|------|------|------|
| 零值输入 | x = 0 | 验证 exp(0) = 1 |
| 大正值 | x = 100 | 验证 overflow 处理 |
| 大负值 | x = -100 | 验证 underflow 处理 |
| NaN 输入 | x = nan | 验证 NaN 传播 |

### 6.3 精度验证标准

| Dtype | atol | rtol |
|-------|------|------|
| float32 | 1e-5 | 1e-3 |
| float16 | 1e-3 | 1e-2 |

---

## 7. 性能指标与开箱配置

### 7.1 性能目标

基于 spec.md 典型配置（性能类）的预期性能：

| 配置名称 | 类型 | 优先级 | 参数 | 输入 Shape | 输出 Shape | 预期 kernel 耗时 |
|----------|------|--------|------|------------|------------|------------------|
| 性能_FP32 | 性能 | P1 | - | [4, 8192, 32, 128] | [4, 8192, 32, 128] | 首跑后确定 |
| 性能_FP16 | 性能 | P1 | - | [4, 8192, 32, 128] | [4, 8192, 32, 128] | 首跑后确定 |

### 7.2 开箱性能配置

```python
# Vector tiling 配置
pypto.set_vec_tile_shapes(64, 128)
```

### 7.3 pass_options 配置

默认无需特殊 pass_options 配置。

### 7.4 runtime_options 配置

```python
@pypto.frontend.jit
def exp_kernel(x, output):
    ...
```

默认使用 NPU 模式，无需显式 runtime_options。

---

## 8. 风险点与注意事项

### 8.1 已知约束

- pypto.exp 仅支持 2-4 维输入
- 输入必须 contiguous（is_contiguous() == True）
- 不支持空 Tensor
- Shape Size 不超过 INT32_MAX

### 8.2 常见错误规避

| 风险 / 错误 | 触发场景 | 影响 / 原因 | 规避方法 |
|-------------|----------|-------------|----------|
| 非连续输入 | torch.transpose 后直接传入 | from_torch 报错 | wrapper 中检查并调用 contiguous() |
| 4D tiling 编译失败 | 直接对 4D 使用 tiling | 编译错误 | 先 reshape 到 2D 再计算 |
| 数值溢出 | 输入值过大 | exp 结果为 inf | 输入已做数值稳定化处理 |

### 8.3 特殊场景处理

- **4D 输入**: 使用 reshape(-1, last_dim) 策略，避免 4D tiling 编译问题
- **动态轴**: 在 from_torch 时声明 dynamic_axis 参数

### 8.4 实现建议

| 建议项 | 说明 |
|--------|------|
| 使用 pypto.exp 直接 API | 无需手动实现，直接调用 |
| 4D reshape 策略 | 4D 输入 reshape(-1, last_dim) 后调用 2D kernel |
| wrapper 函数检查 | 在 wrapper 中检查 contiguous 并处理 |

---

## 9. 交付件清单

### 9.1 目录结构

```
operators/exp/
├── spec.md                          # 需求规范（已有）
├── api_report.md                    # API 探索报告（已有）
├── design.md                        # 设计文档（本文件）
├── exp_golden.py                    # Golden 参考实现（已有）
├── exp_impl.py                      # 算子实现代码
├── test_exp.py                      # 测试代码
└── README.md                        # 实现说明
```

### 9.2 文件清单

| 文件 | 类型 | 说明 | 生成方式 |
|------|------|------|----------|
| spec.md | 需求 | 算子需求规范 | pypto-intent-understanding |
| api_report.md | 设计 | API 探索报告 | pypto-api-explorer |
| design.md | 设计 | 算子设计文档 | pypto-op-design（本 skill） |
| exp_golden.py | 代码 | Golden 参考实现 | pypto-golden-generator |
| exp_impl.py | 代码 | 算子核心实现 | 后续实现 |
| test_exp.py | 代码 | 测试用例 | 后续实现 |

### 9.3 命名规范

| 项目 | 规范 | 示例 |
|------|------|------|
| 算子名称 | 小写字母 + 下划线 | `exp` |
| 目录名 | 与算子名称一致 | `operators/exp/` |
| Golden 文件 | `{op}_golden.py` | `exp_golden.py` |
| 实现文件 | `{op}_impl.py` | `exp_impl.py` |
| 测试文件 | `test_{op}.py` | `test_exp.py` |

### 9.4 生成顺序

```
spec.md → api_report.md → design.md → exp_golden.py → exp_impl.py → test_exp.py
```
