# avg_pool2d 算子设计文档

> **算子名称**: avg_pool2d
> **算子分类**: reduction
> **生成时间**: 2026-03-29
> **基于**: spec.md, api_report.md

---

## 1. 概述

### 1.1 功能描述

对输入张量应用 2D 平均池化操作，支持 SAME 和 VALID 两种填充模式。该算子对输入特征图应用滑动窗口进行平均计算,常用于卷积神经网络中的下采样操作。

### 1.2 数学公式

$$\text{output}[n, c, oh, ow] = \frac{1}{k_h \times k_w} \sum_{i=0}^{k_h-1} \sum_{j=0}^{k_w-1} \text{input}[n, c, oh \times s_h + i, ow \times s_w + j]$$

### 1.3 算法描述

<!-- 简单算子,公式足以描述计算逻辑,无需详细算法描述 -->

### 1.4 数据流图

```
                    输入 x
            ┌──────────────────┐
            │ [b, c, in_h, in_w] │
            │     float32        │
            └─────────┬──────────┘
                      │
                      ▼
            ┌─────────────────────┐
            │    2D Avg Pooling   │
            │ kernel: (k_h, k_w)  │
            │ stride: (s_h, s_w)  │
            │ padding: SAME/VALID │
            └─────────┬───────────┘
                      │
                      ▼
                    输出 y
            ┌────────────────────┐
            │ [b, c, out_h, out_w] │
            │      float32         │
            └──────────────────────┘

动态轴: b (batch_size), c (channels)

SAME padding 输出尺寸:
  out_h = ceil(in_h / s_h)
  out_w = ceil(in_w / s_w)

VALID padding 输出尺寸:
  out_h = ceil((in_h - k_h + 1) / s_h)
  out_w = ceil((in_w - k_w + 1) / s_w)
```

---

## 2. API 映射设计

### 2.1 数学公式分解
将公式拆解为基本操作步骤:

| 步骤 | 数学表达 | 说明 |
|------|----------|------|
| 1 | `window = input[n, c, oh*s_h:oh*s_h+k_h, ow*s_w:ow*s_w+k_w]` | 提取滑动窗口 |
| 2 | `sum_h = sum(window, axis=h)` | 对高度维度求和 |
| 3 | `sum_w = sum(sum_h, axis=w)` | 对宽度维度求和 |
| 4 | `avg = sum_w / (k_h * k_w)` | 除以窗口面积得到平均值 |
| 5 | `output[n, c, oh, ow] = avg` | 将结果写入输出位置 |

### 2.2 PyPTO API 映射表

| 步骤 | 数学表达 | PyPTO API | 参数 | 文档路径 |
|------|----------|-----------|------|----------|
| 1 | 窗口切片 | `Tensor.__getitem__` | `[b:b+unroll, h_start:h_end, w_start:w_end]` | `docs/api/tensor/pypto-Tensor-getitem.md` |
| 2 | 高度方向求和 | `pypto.sum` | `input, dim=1, keepdim=True` | `docs/api/operation/pypto-sum.md` |
| 3 | 宽度方向求和 | `pypto.sum` | `input, dim=2, keepdim=True` | `docs/api/operation/pypto-sum.md` |
| 4 | 除法运算 | `pypto.div` | `sum_val, k_h * k_w` | `docs/api/operation/pypto-div.md` |
| 5 | 结果写入 | `pypto.assemble` | `avg_val, [bc_idx, oh, ow], output` | `docs/api/operation/pypto-assemble.md` |
| 6 | 动态轴定义 | `pypto.frontend.dynamic` | `"batch_size"`, `"channels"` | `docs/api/config/pypto-jit.md` |
| 7 | reshape | `pypto.reshape` | `input, [b*c, h, w], inplace=True` | `docs/api/operation/pypto-reshape.md` |
| 8 | 零填充 | `pypto.zeros` | `[unroll_length, 1, 1], dtype=DT_FP32` | `docs/api/operation/pypto-zeros.md` |

### 2.3 计算步骤序列
```python
# 伪代码展示计算流程
# 1. Reshape 输入为 [b*c, h, w]
input_reshaped = pypto.reshape(input_tensor, [batch_size * channels, in_h, in_w], inplace=True)

# 2. Loop unroll 遍历 batch*channel 维度
for bc_idx, unroll_length in pypto.loop_unroll(0, bc_total, 1, unroll_list=[8, 4, 2, 1]):
    input_cur = input_reshaped[bc_idx:bc_idx+unroll_length, :, :]

    # 3. 遍历输出高度
    for oh in range(out_h):
        h_start = oh * s_h - t_pad
        h_end = h_start + k_h
        h_start_clamped = max(h_start, 0)
        h_end_clamped = min(h_end, in_h)

        # 4. 提取高度方向窗口并求和
        if cur_k_h > 0:
            input_single_row = input_cur[:, h_start_clamped:h_end_clamped, :]
            input_single_row_1 = pypto.sum(input_single_row, 1, keepdim=True)
        else:
            input_single_row_1 = None

            # 5. 遍历输出宽度
            for ow in range(out_w):
                w_start = ow * s_w - l_pad
                w_end = w_start + k_w
                w_start_clamped = max(w_start, 0)
                w_end_clamped = min(w_end, in_w)

                # 6. 提取宽度方向窗口、求和、计算平均值
                if cur_k_h > 0 and cur_k_w > 0:
                    window = input_single_row_1[:, :, w_start_clamped:w_end_clamped]
                    sum_val = pypto.sum(window, dim=2, keepdim=True)
                    avg_val = sum_val / (k_h * k_w)
                    pypto.assemble(avg_val, [bc_idx, oh, ow], output_tmp)
                else:
                    zero_val = pypto.zeros([unroll_length, 1, 1], dtype=pypto.DT_FP32)
                    pypto.assemble(zero_val, [bc_idx, oh, ow], output_tmp)

```

### 2.4 设计依据
- 来源: api_report.md §6 参考实现分析
- 说明: 参考 `models/experimental/vector/AvgPool2d/avg_pool2d.py` 的成熟实现模式,使用分步 sum 和 div 宋建平均池化计算,避免一次性加载整个窗口造成内存压力

同时使用 loop_unroll 优化 batch*channel 维度的并行度

---

## 3. 数据规格设计
### 3.1 OperatorInput dataclass
```python
@dataclass
class AvgPool2dInput:
    x: Tensor  # 输入特征图, shape: [batch_size, channels, in_h, in_w], dtype: float32
    kernel_size: Tuple[int, int]  # 池化窗口大小 (k_h, k_w)
    stride: Tuple[int, int]  # 步长 (s_h, s_w)
    padding_mode: str  # 填充模式: 'SAME' 或 'VALID'
```

### 3.2 OperatorOutput dataclass
```python
@dataclass
class AvgPool2dOutput:
    y: Tensor  # 输出特征图, shape: [batch_size, channels, out_h, out_w], dtype: float32
```

### 3.3 中间 Tensor 定义
| 名称 | Shape | Dtype | 说明 |
|------|-------|-------|------|
| input_reshaped | [batch_size * channels, in_h, in_w] | float32 | Res reshape 后的输入 |
| output_tmp | [batch_size * channels, out_h, out_w] | float32 | 中间输出张量 |
| input_single_row | [unroll_length, cur_k_h, in_w] | float32 | 高度方向窗口 |
| input_single_row_1 | [unroll_length, 1, in_w] | float32 | 高度求和后的中间结果 |
| window | [unroll_length, 1, cur_k_w] | float32 | 宽度方向窗口 |
| sum_val | [unroll_length, 1, 1] | float32 | 窗口求和结果 |
| avg_val | [unroll_length, 1, 1] | float32 | 平均值结果 |

### 3.4 数据格式选择
| Tensor | 格式 | 说明 |
|--------|------|------|
| input_tensor | ND | 输入张量使用 ND 格式 |
| output_result | ND | 输出张量使用 ND 格式 |
| input_reshaped | ND | reshape 后保持 ND 格式 |
| output_tmp | ND | 中间张量使用 ND 格式 |

### 3.5 动态轴定义
| 轴名称 | 含义 | 取值范围 |
|--------|------|----------|
| batch_size | 批次大小 | [1, INT32_MAX] |
| channels | 通道数 | [1, INT32_MAX] |

### 3.6 JIT 装饰器配置
```python
@pypto.frontend.jit(
    pass_options={"vec_nbuffer_setting": {-1: 2, 0: 8}},
    runtime_options={"run_mode": pypto.RunMode.NPU, "stitch_function_num_initial": 128, "stitch_function_outcast_memory": 1024, "stitch_function_inner_memory": 1024},
    debug_options=dict(runtime_debug_mode=1, compile_debug_mode=1)
)
def avg_pool2d_kernel(inputs: AvgPool2dInput) -> AvgPool2dOutput:
    ...
```

---

## 4. Tiling 策略
### 4.1 算子类型判断
- **类型**: Vector
- **判断依据**: 公式仅含索引切片、sum 归约和 div 逐元素操作,无矩阵乘法,属于纯 Vector 类型

### 4.2 TileShape 初值设置
```python
# 主循环前
pypto.set_vec_tile_shapes(16, 16, 4, 128)

# 窗口内高度求和
pypto.set_vec_tile_shapes(16, 16, 128)

# 结果组装
pypto.set_vec_tile_shapes(unroll_length, 4, 128)
```
### 4.3 设置依据
- 主循环前 `(16, 16, 4, 128)`: 处理 4D 输入 [b, c, h, w]
 tile 16 元素覆盖 b 维度, 16 覆盖 c 维度, 4 覆盖 h 维度, 128 覆盖 w 维度
- 窗口内高度求和 `(16, 16, 128)`: 处理 3D 窗口 [bc, h, w], tile 16 覆盖 bc 维度, 16 覆盖 h 维度, 128 覆盖 w 维度
- 结果组装 `(unroll_length, 4, 128)`: 处理 3D 输出 [bc, oh, ow], unroll_length 覆盖 bc 维度, 4 覆盖 oh 维度, 128 覆盖 ow 维度

### 4.4 注意事项
- 尾轴需 32B 对齐: w 维度的 tile 大小 (128 * 4 bytes = 512 bytes) 满足 32B 对齐要求
- TileShape 次尾轴需 <= 255: h 维度的 tile 大小 (16/4) 满足 <= 255 约束
- TileShape 总大小不超过 64KB: 黺议配置远小于 64KB 以避免内存溢出
### 4.5 判断依据与适用条件
- 判断依据: 算子为 Vector 类型,需要使用 `set_vec_tile_shapes` 配置 tile 大小
- 适用条件: 适用于所有 shape/dtype/动态轴场景,TileShape 可根据实际 shape 动态调整
- 不适用场景: 无不适用场景,但需要注意当输入 shape 鷨大时可能需要调整 tile 大小以优化性能
---
## 5. Loop 结构设计
### 场景 B: 需要 Loop
#### 5.1 Loop 判断结论
- **结论**: 需要 Loop
- **原因**:
  1. batch_size 和 channels 为动态轴,取值范围大 (1-INT32_MAX), 需要循环处理
  2. out_h 和 out_w 需要遍历每个输出位置
  3. 滑动窗口操作需要对每个输出位置计算平均值
- **Loop 类型**: `pypto.loop_unroll` + Python for
- **适用条件**: 动态轴 batch_size 和 channels, 以及需要遍历 out_h 和 out_w
- **限制**: loop_unroll 的 unroll_list 选择需要平衡编译时间和性能
#### 5.2 静态轴 vs 动态轴处理
| 轴 | 类型 | 处理方式 |
|----|------|----------|
| batch_size | 动态 | `pypto.loop_unroll` with unroll_list=[8, 4, 2, 1] |
| channels | 动态 | 与 batch_size 合并为 bc_total, 使用 `pypto.loop_unroll` |
| out_h | 静态 | Python `for` loop |
| out_w | 静态 | Python `for` loop |
#### 5.3 Loop 合并策略
将 batch_size 和 channels 合并为 bc_total = batch_size * channels, 使用单个 `pypto.loop_unroll` 遍历,这样可以:
1. 减少循环嵌套层数
2. 提高并行度
3. 简化代码结构
#### 5.4 数据依赖处理
- oh 循环依赖于 input_cur (bc_idx 循环的输出)
- ow 循环依赖于 input_single_row_1 (oh 循环的输出)
- assemble 依赖于 avg_val (ow 循环的输出)
#### 5.5 尾块处理策略
- 使用 `loop_unroll` 的 unroll_list 参数自动处理尾块
- unroll_list=[8, 4, 2, 1] 确保最后总能找到合适的 unroll_factor 夻理剩余元素
#### 5.6 loop_unroll 配置
```python
for bc_idx, unroll_length in pypto.loop_unroll(
    0, bc_total, 1,
    name="LOOP_BC",
    idx_name="bc_idx",
    unroll_list=[8, 4, 2, 1]
):
    # bc_idx: 当前循环索引
    # unroll_length: 当前展开因子
    ...
```
---
## 6. 验证方案
### 6.1 Golden 函数设计
```python
def avg_pool2d_golden(
    x: torch.Tensor,
    kernel_size: Tuple[int, int],
    stride: Optional[Tuple[int, int]] = None,
    padding_mode: str = 'SAME',
) -> torch.Tensor:
    """avg_pool2d 参考实现"""
    # 处理步长
    if stride is None:
        stride = kernel_size
    k_h, k_w = kernel_size
    s_h, s_w = stride
    batch_size, channels, in_h, in_w = x.shape

    # 计算输出尺寸和 padding
    if padding_mode.upper() == 'VALID':
        out_h = (in_h - k_h + s_h) // s_h
        out_w = (in_w - k_w + s_w) // s_w
        padding = 0
    elif padding_mode.upper() == 'SAME':
        out_h = math.ceil(in_h / s_h)
        out_w = math.ceil(in_w / s_w)
        pad_h = max(0, (out_h - 1) * s_h + k_h - in_h)
        pad_w = max(0, (out_w - 1) * s_w + k_w - in_w)
        t_pad = pad_h // 2
        l_pad = pad_w // 2
        padding = (t_pad, l_pad)

    ceil_mode = (padding_mode.upper() == 'SAME')
    return torch.nn.functional.avg_pool2d(
        x,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        ceil_mode=ceil_mode
    )
```
### 6.2 测试用例设计
#### 基于 spec.md 所有典型配置
| 配置名称 | 类型 | 优先级 | 参数 | 输入 Shape | 输出 Shape | 说明 |
|----------|------|--------|------|------------|------------|------|
| SAME_P0 | 性能 | P0 | kernel=(2,2), stride=(2,2), padding=SAME | [2, 3, 6, 6] | [2, 3, 3, 3] | SAME 模式核心场景 |
| VALID_P0 | 功能 | P0 | kernel=(3,3), stride=(2,2), padding=VALID | [4, 8, 12, 12] | [4, 8, 5, 5] | VALID 模式功能验证 |
| SAME_large | 性能 | P1 | kernel=(3,3), stride=(2,2), padding=SAME | [8, 64, 56, 56] | [8, 64, 28, 28] | 大尺寸特征图性能测试 |
| VALID_stride1 | 功能 | P2 | kernel=(2,2), stride=(1,1), padding=VALID | [2, 16, 8, 8] | [2, 16, 7, 7] | stride=1 边界场景 |

#### 边界情况测试
| 场景 | 参数 | 说明 |
|------|------|------|
| 零值输入 | x = zeros([2, 3, 6, 6]) | 验证零值输出 |
| 大值输入 | x = randn([2, 3, 6, 6]) * 100 | 验证数值稳定性 |
| padding 边界 | SAME padding with odd size | 验证边界 padding 计算 |

### 6.3 精度验证标准
| Dtype | atol | rtol |
|-------|------|------|
| float32 | 0.001 | 0.001 |
---
## 7. 性能指标与开箱配置
### 7.1 性能目标
基于 spec.md 典型配置（性能类）的预期性能:
| 配置名称 | 类型 | 优先级 | 参数 | 输入 Shape | 输出 Shape | 预期 kernel 耗时 |
|----------|------|--------|------|------------|------------|------------------|
| SAME_P0 | 性能 | P0 | kernel=(2,2), stride=(2,2), padding=SAME | [2, 3, 6, 6] | [2, 3, 3, 3] | < 1ms |
| SAME_large | 性能 | P1 | kernel=(3,3), stride=(2,2), padding=SAME | [8, 64, 56, 56] | [8, 64, 28, 28] | < 10ms |

### 7.2 开箱性能配置
```python
# TileShape 配置
pypto.set_vec_tile_shapes(16, 16, 4, 128)  # 主循环前
pypto.set_vec_tile_shapes(16, 16, 128)     # 窗口处理
pypto.set_vec_tile_shapes(unroll_length, 4, 128)  # 结果组装
```

### 7.3 pass_options 配置
```python
pass_options={
    "vec_nbuffer_setting": {-1: 2, 0: 8}  # vector buffer 配置
}
```

### 7.4 runtime_options 配置
```python
runtime_options={
    "run_mode": pypto.RunMode.NPU,
    "stitch_function_num_initial": 128,
    "stitch_function_outcast_memory": 1024,
    "stitch_function_inner_memory": 1024
}
```
---
## 8. 风险点与注意事项
### 8.1 已知约束
- sum API 要求: Shape 仅支持 2-4 维, Shape Size 不大于 INT32_MAX
- div API 要求: other 不支持 nan/inf
- TileShape 要求: 每维 > 0, 尾轴 32B 对齐, 次尾轴 <= 255

### 8.2 常见错误规避
| 风险 / 错误 | 触发场景 | 影响 / 原因 | 规避方法 |
|-------------|----------|-------------|----------|
| sum 后维度减少 | keepdim=False | TileShape 与实际维度不匹配 | 使用 keepdim=True 保持维度,或在 sum 后重置 TileShape |
| padding 边界越界 | SAME padding | 访问超出输入范围的索引 | 使用 max/min clamp 边界 |
| 空窗口处理 | cur_k_h <= 0 或 cur_k_w <= 0 | None 传给 sum 导致错误 | 使用 pypto.zeros 填充零值 |
| TileShape 不对齐 | 尾轴非 32B 对齐 | 编译错误 | 确保 w 维度 tile 大小为 4/8/16/32/64/128 等 |

### 8.3 特殊场景处理
- **SAME padding 边界**: 使用 `max(h_start, 0)` 和 `min(h_end, in_h)` clamp 边界,避免越界访问
- **空窗口**: 当 `cur_k_h <= 0` 或 `cur_k_w <= 0` 时,使用 `pypto.zeros` 填充零值,避免 None 传给 sum
- **动态轴**: 使用 `pypto.frontend.dynamic()` 定义 batch_size 和 channels 为动态维度

### 8.4 实现建议
| 建议项 | 说明 |
|--------|------|
| 分步 sum | 先对高度维度求和,再对宽度维度求和,避免一次性加载整个窗口 |
| loop_unroll | 使用 unroll_list=[8, 4, 2, 1] 平衡编译时间和性能 |
| TileShape 配置 | 根据实际 unroll_length 动态调整 TileShape |

---
## 9. 交付件清单
### 9.1 目录结构
```
custom/avg_pool2d/
├── spec.md                    # 需求规范（已有）
├── api_report.md              # API 探索报告（已有）
├── design.md                  # 设计文档（本文件）
├── avg_pool2d_golden.py      # Golden 参考实现（已有）
├── avg_pool2d_impl.py        # 算子实现代码
├── test_avg_pool2d.py        # 测试代码
└── output/                    # 运行输出（自动生成）
```
### 9.2 文件清单
| 文件 | 类型 | 说明 | 生成方式 |
|------|------|------|----------|
| spec.md | 需求 | 算子需求规范 | pypto-intent-understanding |
| api_report.md | 设计 | API 探索报告 | pypto-api-explorer |
| design.md | 设计 | 算子设计文档 | pypto-op-design（本 skill） |
| avg_pool2d_golden.py | 代码 | Golden 参考实现 | pypto-golden-generator |
| avg_pool2d_impl.py | 代码 | 算子核心实现 | 后续实现 |
| test_avg_pool2d.py | 代码 | 测试用例 | 后续实现 |

### 9.3 命名规范
| 项目 | 规范 | 示例 |
|------|------|------|
| 算子名称 | 小写字母 + 下划线 | `avg_pool2d` |
| 目录名 | 与算子名称一致 | `custom/avg_pool2d/` |
| Golden 文件 | `{op}_golden.py` | `avg_pool2d_golden.py` |
| 实现文件 | `{op}_impl.py` | `avg_pool2d_impl.py` |
| 测试文件 | `test_{op}.py` | `test_avg_pool2d.py` |

### 9.4 生成顺序
```
spec.md → api_report.md → design.md → avg_pool2d_golden.py → avg_pool2d_impl.py → test_avg_pool2d.py
```
