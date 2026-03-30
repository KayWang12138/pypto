# clip 算子设计文档

> **算子名称**: clip
> **算子分类**: element-wise
> **生成时间**: 2026-03-29T00:00:00Z
> **基于**: spec.md, api_report.md

---

## 1. 概述

### 1.1 功能描述

clip 算子将输入张量的每个元素限制在 [min_val, max_val] 范围内：
- 如果 x < min_val，输出 min_val
- 如果 x > max_val，输出 max_val
- 否则输出 x

### 1.2 数学公式

$$y = \min(\max(x, \text{min\_val}), \text{max\_val})$$

### 1.3 算法描述

简单逐元素操作，无需复杂算法描述。

### 1.4 数据流图

```
    输入 x              输入 min_val       输入 max_val
┌──────────────┐   ┌─────────┐       ┌─────────┐
│ [*, dynamic] │   │ scalar  │       │ scalar  │
│   float32    │   │ float32 │       │ float32 │
└──────┬───────┘   └────┬────┘       └────┬────┘
       │                │                 │
       │                │                 │
       ▼                ▼                 ▼
    ┌─────────────────────────────────────────────┐
    │            y = clip(x, min, max)            │
    │    if x < min: min                          │
    │    elif x > max: max                        │
    │    else: x                                  │
    └─────────────────────┬───────────────────────┘
                          │
                          ▼
                   ┌──────────────┐
                   │ 输出 y        │
                   │ [*, dynamic] │
                   │   float32    │
                   └──────────────┘

公式: y = min(max(x, min_val), max_val)
动态轴: 所有维度
```

---

## 2. API 映射设计

### 2.1 数学公式分解

| 步骤 | 数学表达 | 说明 |
|------|----------|------|
| 1 | y = clip(x, min_val, max_val) | 单步逐元素裁剪操作 |

### 2.2 PyPTO API 映射表

| 步骤 | 数学表达 | PyPTO API | 参数 | 文档路径 |
|------|----------|-----------|------|----------|
| 1 | y = clip(x, min_val, max_val) | `pypto.clip(input, min, max)` | input: Tensor, min: float/scalar, max: float/scalar | `docs/api/operation/pypto-clip.md` |

### 2.3 计算步骤序列

```python
# 伪代码展示计算流程
# clip 是单步操作，直接调用 PyPTO API
y = pypto.clip(x, min_val, max_val)
```

### 2.4 设计依据

- 来源：api_report.md + docs/api/operation/pypto-clip.md
- 说明：PyPTO 提供直接的 `pypto.clip` API，语义与需求完全匹配，无需组合 `maximum` + `minimum`。该 API 支持标量 min/max，支持动态 shape，支持 FP32/FP16/BF16 dtype。

---

## 3. 数据规格设计

### 3.1 ClipInput dataclass

```python
@dataclass
class ClipInput:
    x: Tensor  # 输入张量，shape: [*, dynamic], dtype: float32/float16/bfloat16
    min_val: float  # 最小值边界，标量
    max_val: float  # 最大值边界，标量
```

### 3.2 ClipOutput dataclass

```python
@dataclass
class ClipOutput:
    y: Tensor  # 输出张量，shape 与 x 相同，dtype 与 x 相同
```

### 3.3 中间 Tensor 定义

无需中间 Tensor，clip 是单步操作。

### 3.4 数据格式选择

| Tensor | 格式 | 说明 |
|--------|------|------|
| x | ND | 默认格式，from_torch 自动推导 |
| y | ND | 与输入格式一致 |

### 3.5 动态轴定义

| 轴名称 | 含义 | 取值范围 |
|--------|------|----------|
| dim_0 | 第0维 | [1, INT32_MAX] |
| dim_1 | 第1维 | [1, INT32_MAX] |
| dim_2 | 第2维（如存在） | [1, INT32_MAX] |
| dim_3 | 第3维（如存在） | [1, INT32_MAX] |

### 3.6 JIT 装饰器配置

```python
@pypto.frontend.jit(
    runtime_options={"run_mode": pypto.RunMode.NPU}
)
def clip_wrapper(
    x: pypto.Tensor([], pypto.DT_FP32),
    min_val: float,
    max_val: float,
    y: pypto.Tensor([], pypto.DT_FP32)
):
    ...
```

---

## 4. Tiling 策略

### 4.1 算子类型判断

| 判断依据 | 结论 |
|----------|------|
| 是否含 matmul | 否 |
| 算子类型 | **Vector** |
| 需调用 API | `pypto.set_vec_tile_shapes()` |

**判断理由**：clip 是逐元素操作，不涉及矩阵乘法，属于 Vector 类型算子。

### 4.2 TileShape 配置

| 维度 | TileShape | 说明 |
|------|-----------|------|
| 2D | `[64, 128]` | 尾轴 128 满足 FP32 的 8 倍数对齐（128 * 4 = 512B > 32B） |
| 3D | `[8, 64, 128]` | 典型配置适配 |
| 4D | `[1, 8, 64, 128]` | 4D 场景 |

### 4.3 设置依据

- 来源：`docs/api/config/pypto-set_vec_tile_shapes.md` + `examples/01_beginner/compute/elementwise_ops.py`
- 尾轴对齐：FP32 尾轴需为 8 的倍数（32B / 4B = 8），128 满足要求
- 参考实现使用 `pypto.set_vec_tile_shapes(2, 8)` 作为基础配置

### 4.4 推荐配置

```python
# 2D 输入
pypto.set_vec_tile_shapes(64, 128)

# 3D 输入
pypto.set_vec_tile_shapes(8, 64, 128)

# 4D 输入
pypto.set_vec_tile_shapes(1, 8, 64, 128)
```

---

## 5. Loop 结构设计

### 5.1 是否需要 Loop

| 检查条件 | 结果 | 说明 |
|----------|------|------|
| 存在动态轴 | 是 | spec.md 声明所有维度动态 |
| 多步骤分块计算 | 否 | 单步操作 |
| 动态轴范围跨度大 | 否 | 单 TileShape 可处理 |
| 编译期已知 & 单次可处理 | 部分 | 动态轴需要运行时处理 |

**结论**：**不需要显式 Loop**

**理由**：
1. clip 是逐元素操作，编译器会自动处理 tile 遍历
2. PyPTO 的 Vector 算子在设置 TileShape 后，编译器自动生成遍历代码
3. 参考 `examples/01_beginner/compute/elementwise_ops.py` 中的 clip 实现，无需手动 Loop

### 5.2 静态/动态轴处理

| 轴类型 | 处理方式 |
|--------|----------|
| 静态轴 | 编译器自动展开 |
| 动态轴 | 通过 `dynamic_axis` 参数标记，编译器生成动态遍历代码 |

### 5.3 尾块处理

无需特殊处理。PyPTO 编译器自动处理非对齐的尾块。

---

## 6. 验证方案

### 6.1 Golden 函数设计

使用 `clip_golden.py` 中的 `clip_golden()` 函数：

```python
def clip_golden(
    x: torch.Tensor,
    min_val: Optional[Union[float, int]] = None,
    max_val: Optional[Union[float, int]] = None
) -> torch.Tensor:
    return torch.clamp(x, min=min_val, max=max_val)
```

### 6.2 测试用例（基于典型配置）

| 配置名称 | 类型 | 优先级 | 参数 | 输入 Shape | 验证内容 |
|----------|------|--------|------|------------|----------|
| 性能_P0 | 性能 | P0 | min=-1.0, max=1.0 | [1024, 1024] | 功能正确性 + 性能基准 |
| 功能_P0 | 功能 | P0 | min=-1.0, max=1.0 | [32, 64, 128] | 功能正确性 |
| 动态_P0 | 功能 | P0 | min=0.0, max=1.0 | [batch, seq, hidden] | 动态 shape 支持 |
| 边界_1 | 功能 | P1 | min=-1.0, max=1.0 | [-2.0, -1.0, 0.0, 1.0, 2.0] | 边界值处理 |
| dtype_FP16 | 功能 | P1 | min=-1.0, max=1.0 | [64, 64] float16 | FP16 支持 |

### 6.3 精度标准

| Dtype | atol | rtol |
|-------|------|------|
| float32 | 0.001 | 0.001 |
| float16 | 0.01 | 0.01 |
| bfloat16 | 0.01 | 0.01 |

### 6.4 验证流程

1. 运行 golden 验证确保参考实现正确
2. 运行 PyPTO 实现并与 golden 对比
3. 检查输出 shape 一致性
4. 检查数值精度满足 atol/rtol 要求
5. 输出 `[PRECISION_PASS]` 或 `[PRECISION_FAIL]` 标记

---

## 7. 性能指标与开箱配置

### 7.1 性能目标

| 配置名称 | 类型 | 优先级 | 参数 | 输入 Shape | 输出 Shape | 预期时间 |
|----------|------|--------|------|------------|------------|----------|
| 性能_P0 | 性能 | P0 | min=-1.0, max=1.0 | [1024, 1024] | [1024, 1024] | 基准测试后确定 |

### 7.2 开箱性能配置

```python
# 2D 性能配置
pypto.set_vec_tile_shapes(64, 128)

# 针对 [1024, 1024] 的大矩阵
# TileShape [64, 128] 可有效利用 L0/L1 缓存
```

### 7.3 pass_options 配置

```python
# 无特殊 pass_options 需求
pass_options = {}
```

### 7.4 runtime_options 配置

```python
runtime_options = {
    "run_mode": pypto.RunMode.NPU  # NPU 模式运行
}
```

---

## 8. 风险点与注意事项

### 8.1 已知约束

1. **维度限制**：PyPTO clip API 仅支持 2-4 维 Tensor，1 维输入需先 reshape
2. **dtype 限制**：仅支持 DT_FP32, DT_FP16, DT_BF16, DT_INT32, DT_INT16
3. **contiguous 要求**：输入 Tensor 必须连续（from_torch 要求）
4. **元素个数限制**：不超过 UINT32_MAX

### 8.2 常见错误规避

| 风险 / 错误 | 触发场景 | 影响 / 原因 | 规避方法 |
|-------------|----------|-------------|----------|
| 维度超限 | 输入为 1D 或 >4D | API 调用失败 | 输入检查，必要时 reshape |
| dtype 不支持 | 输入为 INT8/UINT8 | 编译失败 | 限制支持 dtype，提前转换 |
| 非连续 Tensor | 输入经过 transpose | from_torch 失败 | 调用前 ensure contiguous |
| min > max | 参数配置错误 | 输出全为 max | 参数校验或文档说明 |
| TileShape 不对齐 | 尾轴不满足 32B 对齐 | 性能劣化或编译失败 | 尾轴设为 8/16/32 的倍数 |

### 8.3 特殊场景处理

- **None min/max**：支持单侧裁剪（仅 min 或仅 max），但不可同时为 None
- **NaN/Inf 输入**：按正常计算处理，结果保持不变
- **min == max**：输出为常数 Tensor，值为 min/max

---

## 9. 交付件清单

### 9.1 目录结构

```text
operators/clip/
├── spec.md           # 算子规格文档（已生成）
├── api_report.md     # API 探索报告（已生成）
├── design.md         # 设计方案文档（本文档）
├── clip_golden.py    # Golden 参考实现（已生成）
├── clip_impl.py      # PyPTO 实现（待生成）
├── test_clip.py      # 测试文件（待生成）
└── README.md         # 使用说明（待生成）
```

### 9.2 文件命名规范

| 文件 | 命名规则 | 说明 |
|------|----------|------|
| 规格文档 | `spec.md` | 算子需求规格 |
| API 报告 | `api_report.md` | API 映射与约束分析 |
| 设计文档 | `design.md` | 实现设计方案 |
| Golden | `{op}_golden.py` | PyTorch 参考实现 |
| 实现 | `{op}_impl.py` | PyPTO kernel 实现 |
| 测试 | `test_{op}.py` | 精度验证测试 |
| 说明 | `README.md` | 使用说明 |

### 9.3 生成顺序

1. spec.md (已完成)
2. api_report.md (已完成)
3. clip_golden.py (已完成)
4. design.md (本文档)
5. clip_impl.py (下一步)
6. test_clip.py (下一步)
7. README.md (最后)

---

## 10. 总结

clip 是一个简单的逐元素裁剪算子，PyPTO 提供直接的 `pypto.clip` API 支持。设计要点：

1. **API 选择**：直接使用 `pypto.clip`，无需组合 `maximum` + `minimum`
2. **Tiling 策略**：Vector 类型，使用 `set_vec_tile_shapes`，尾轴 128 满足对齐要求
3. **Loop 策略**：不需要显式 Loop，编译器自动处理 tile 遍历
4. **动态轴**：通过 `dynamic_axis` 参数标记，编译器生成动态遍历代码
5. **风险点**：维度限制（2-4D）、dtype 限制、contiguous 要求
