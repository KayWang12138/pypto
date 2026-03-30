# Sigmoid 算子设计文档

> **算子名称**: sigmoid
> **算子分类**: element-wise
> **生成时间**: 2026-03-29
> **基于**: spec.md, api_report.md

---

## 1. 概述

### 1.1 功能描述

Sigmoid 是深度学习中常用的激活函数。对输入 tensor 的每个元素执行 sigmoid 运算，将输入映射到 (0, 1) 区间，常用于二分类输出层和门控机制。

### 1.2 数学公式

$$\sigma(x) = \frac{1}{1 + e^{-x}}$$

### 1.3 算法描述

简单 element-wise 算子，公式已完整描述计算逻辑，无需复杂算法伪代码。

### 1.4 数据流图

```
    输入 x                    输出 y
┌──────────────────┐         ┌──────────────────┐
│  [m, n] 或       │         │  [m, n] 或       │
│  [b, m, n] 或    │ ──────▶ │  [b, m, n] 或    │
│  [b, s, m, n]    │ sigmoid │  [b, s, m, n]    │
│  float16/float32 │         │  float16/float32 │
│  /bfloat16       │         │  /bfloat16       │
└──────────────────┘         └──────────────────┘

公式: sigmoid(x) = 1 / (1 + exp(-x))
      输出范围: (0, 1)
约束: Shape仅支持2-4维，Shape Size <= INT32_MAX
      不支持空Tensor
```

---

## 2. API 映射设计

### 2.1 数学公式分解

将公式拆解为基本操作步骤：

| 步骤 | 数学表达 | 说明 |
|------|----------|------|
| 1 | sigmoid(x) = 1 / (1 + exp(-x)) | 直接 sigmoid 计算 |

### 2.2 PyPTO API 映射表

| 步骤 | 数学表达 | PyPTO API | 参数 | 文档路径 |
|------|----------|-----------|------|----------|
| 1 | sigmoid(x) | `pypto.sigmoid(x)` | x: Tensor | `docs/api/operation/pypto-sigmoid.md` |

### 2.3 计算步骤序列

```python
# 伪代码展示计算流程
out[:] = pypto.sigmoid(x)
```

### 2.4 设计依据

- **来源**: api_report.md §3.1 核心 API 列表
- **说明**: PyPTO 提供直接的 `pypto.sigmoid()` API。文档声明仅支持 DT_FP32，若实际测试发现不支持 FP16/BF16，需使用组合方案：`exp/mul/add/div`

**备选组合方案**（如直接 API 不支持 FP16/BF16）：

```python
# sigmoid(x) = 1 / (1 + exp(-x))
neg_x = pypto.mul(x, -1.0)              # -x
exp_neg_x = pypto.exp(neg_x)            # e^(-x)
denominator = pypto.add(exp_neg_x, 1.0) # 1 + e^(-x)
out[:] = pypto.div(1.0, denominator)    # 1 / (1 + e^(-x))
```

---

## 3. 数据规格设计

### 3.1 OperatorInput dataclass

```python
@dataclass
class SigmoidInput:
    x: Tensor  # 源操作数, shape: [m, n] 或 [b, m, n] 或 [b, s, m, n], dtype: DT_FP16/DT_FP32/DT_BF16
```

### 3.2 OperatorOutput dataclass

```python
@dataclass
class SigmoidOutput:
    y: Tensor  # 输出Tensor, shape: 与x相同, dtype: 与x相同, 值域: (0, 1)
```

### 3.3 中间 Tensor 定义

**方案 A（直接 API）**: 无中间 Tensor。Sigmoid 是单步 element-wise 操作，输入直接映射到输出。

**方案 B（组合实现）**:

| Tensor | Shape | Dtype | 说明 |
|--------|-------|-------|------|
| neg_x | 与 x 相同 | 与 x 相同 | -x |
| exp_neg_x | 与 x 相同 | 与 x 相同 | e^(-x) |
| denominator | 与 x 相同 | 与 x 相同 | 1 + e^(-x) |

### 3.4 数据格式选择

| Tensor | 格式 | 说明 |
|--------|------|------|
| x | ND | 默认格式，由 from_torch 自动推导 |
| y | ND | 与输入格式一致 |

### 3.5 动态轴定义

| 轴名称 | 含义 | 取值范围 |
|--------|------|----------|
| m | 特征维度 | [1, INT32_MAX] |
| n | 特征维度 | [1, INT32_MAX] |
| b | batch维度 | [1, INT32_MAX] |
| s | sequence维度 | [1, INT32_MAX] |

**约束**: Shape Size 不大于 INT32_MAX

### 3.6 JIT 装饰器配置

```python
@pypto.frontend.jit(
    runtime_options={"run_mode": pypto.RunMode.NPU}
)
def sigmoid_kernel(x: pypto.Tensor(), out: pypto.Tensor()):
    ...
```

---

## 4. Tiling 策略

### 4.1 算子类型判断

- **类型**: Vector
- **判断依据**: Sigmoid 是纯逐元素操作（element-wise），不涉及矩阵乘法，仅需要对每个输入元素执行 sigmoid 计算，因此使用 Vector 类型的 tiling 配置。

### 4.2 TileShape 初值设置

```python
# 2D 输入
pypto.set_vec_tile_shapes(32, 128)

# 3D 输入
pypto.set_vec_tile_shapes(1, 32, 128)

# 4D 输入
pypto.set_vec_tile_shapes(1, 1, 32, 128)

# 动态配置（根据输入维度自动适配）
def configure_tiling(x):
    if len(x.shape) == 2:
        pypto.set_vec_tile_shapes(32, 128)
    elif len(x.shape) == 3:
        pypto.set_vec_tile_shapes(1, 32, 128)
    elif len(x.shape) == 4:
        pypto.set_vec_tile_shapes(1, 1, 32, 128)
```

### 4.3 设置依据

1. **TileShape 维度匹配**: TileShape 维度数应与输入 shape 维度数一致（pypto.sigmoid 文档要求）
2. **尾轴 32B 对齐**: fp16/bf16 尾轴需为 16 的倍数，fp32 需为 8 的倍数。选择 128 满足所有 dtype 对齐要求
3. **非尾轴选择**: 32 或 1，便于处理各种 shape 大小
4. **来源**: api_report.md §5 Tiling 需求 + `docs/api/config/pypto-set_vec_tile_shapes.md`

### 4.4 注意事项

- TileShape 维度数必须与输入 shape 维度数一致
- 每个维度必须 > 0
- 最多 4 维

### 4.5 判断依据与适用条件

- **判断依据**: Sigmoid 是简单 element-wise 操作，框架可自动处理数据切分
- **适用条件**: 2-4 维输入，所有 dtype（FP16/FP32/BF16）
- **不适用场景**: 输入维度超过 4 维时不支持（spec 约束）

---

## 5. Loop 结构设计

### 场景 A：不需要 Loop

> 适用于所有轴编译期已知、单次 Tile 可处理的算子（如逐元素运算）。

- **结论**: 不需要 pypto.loop
- **原因**: Sigmoid 是简单 element-wise 操作，框架自动处理数据切分，无需手动循环
- **适用条件**: 所有 2-4 维输入，满足 spec 约束的任意 shape
- **限制**: 无
- **处理方式**: 编译器自动处理数据切分，无需手动循环

**依据**: quick_ref.md §2.1 条件 4 - "所有轴编译期已知 & 单次 Tile 可处理 -> 不需要 Loop"

---

## 6. 验证方案

### 6.1 Golden 函数设计

```python
def sigmoid_golden(x: torch.Tensor) -> torch.Tensor:
    """Sigmoid 参考实现"""
    return torch.sigmoid(x)
```

Golden 实现已生成于 `sigmoid_golden.py`。

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
| 零值 | x = 0 | 验证 sigmoid(0) = 0.5 |
| 大正值 | x -> +inf | 验证 sigmoid -> 1 |
| 大负值 | x -> -inf | 验证 sigmoid -> 0 |
| 对称性 | sigmoid(-x) = 1 - sigmoid(x) | 验证对称性质 |
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

**说明**: sigmoid 为简单 element-wise 操作，无需特殊 pass 优化。

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

- `pypto.sigmoid` 文档声明仅支持 DT_FP32，需测试验证 FP16/BF16 支持
- 若直接 API 不支持 FP16/BF16，需使用组合方案（exp/mul/add/div）
- 所有输入 tensor 必须连续（`is_contiguous() == True`）
- 输入 tensor 不支持空 tensor（shape size >= 1）
- Shape 仅支持 2-4 维
- Shape Size 不超过 INT32_MAX

### 8.2 常见错误规避

| 风险 / 错误 | 触发场景 | 影响 / 原因 | 规避方法 |
|-------------|----------|-------------|----------|
| TileShape 不对齐 | 尾轴不满足 32B 对齐 | 编译失败或性能劣化 | fp16/bf16 尾轴设为 16 的倍数，fp32 设为 8 的倍数 |
| 输入不连续 | tensor.is_contiguous() == False | from_torch 失败 | 调用前确保 tensor 连续或调用 tensor.contiguous() |
| dtype 不支持 | pypto.sigmoid 不支持 FP16/BF16 | 运行时错误 | 使用组合方案或降级到 FP32 |
| exp 溢出 | x 值过小 (< -100) | exp(-x) 可能溢出 | sigmoid 的 exp 参数为负，大负值时 exp -> 0，安全 |

### 8.3 特殊场景处理

1. **大正值输入 (x > 20)**: sigmoid(x) -> 1（趋近于 1）
2. **大负值输入 (x < -20)**: sigmoid(x) -> 0（趋近于 0）
3. **零值 (x = 0)**: sigmoid(0) = 0.5，无需特殊处理

### 8.4 实现建议

| 建议项 | 说明 |
|--------|------|
| 优先使用直接 API | `pypto.sigmoid(x)` 更简洁高效 |
| 备选组合方案 | 若 dtype 不支持，使用 exp/mul/add/div 组合 |
| 复用 configure_tiling | 参考 relu/tanh 算子中的 tiling 配置函数 |
| 使用 torch.sigmoid | Golden 实现使用 PyTorch 内置 API，确保正确性 |
| 测试覆盖所有 dtype | 验证 FP32/FP16/BF16 三种 dtype 的精度和性能 |

---

## 9. 交付件清单

### 9.1 目录结构

```
operators/sigmoid/
├── spec.md                          # 需求规范（已有）
├── api_report.md                    # API 探索报告（已有）
├── design.md                        # 设计文档（本文件）
├── sigmoid_golden.py                # Golden 参考实现（已有）
├── sigmoid_impl.py                  # 算子实现代码（待生成）
├── test_sigmoid.py                  # 测试代码（待生成）
├── README.md                        # 实现说明（待生成）
└── .orchestrator_state.json         # 状态文件（已有）
```

### 9.2 文件清单

| 文件 | 类型 | 说明 | 生成方式 |
|------|------|------|----------|
| spec.md | 需求 | 算子需求规范 | pypto-intent-understanding |
| api_report.md | 探索 | API 探索报告 | pypto-api-explorer |
| design.md | 设计 | 算子设计文档 | pypto-op-design（本 skill） |
| sigmoid_golden.py | 代码 | Golden 参考实现 | pypto-golden-generator |
| sigmoid_impl.py | 代码 | 算子核心实现 | pypto-op-develop（后续） |
| test_sigmoid.py | 代码 | 测试用例 | pypto-op-develop（后续） |
| README.md | 文档 | 实现说明 | pypto-op-develop（后续） |

### 9.3 命名规范

| 项目 | 规范 | 示例 |
|------|------|------|
| 算子名称 | 小写字母 | `sigmoid` |
| 目录名 | 与算子名称一致 | `operators/sigmoid/` |
| Golden 文件 | `{op}_golden.py` | `sigmoid_golden.py` |
| 实现文件 | `{op}_impl.py` | `sigmoid_impl.py` |
| 测试文件 | `test_{op}.py` | `test_sigmoid.py` |

### 9.4 生成顺序

```
spec.md -> api_report.md -> design.md -> sigmoid_golden.py -> sigmoid_impl.py -> test_sigmoid.py -> README.md
```

---

*生成时间: 2026-03-29*
*生成工具: pypto-op-orchestrator*
