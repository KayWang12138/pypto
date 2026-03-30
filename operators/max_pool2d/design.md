# max_pool2d 算子设计文档

> **算子名称**: max_pool2d
> **算子分类**: pooling
> **生成时间**: 2026-03-28
> **基于**: spec.md, api_report.md

---

## 1. 概述

### 1.1 功能描述

max_pool2d 是对输入 tensor 在空间维度（H 和 W）上进行最大池化操作。对于输入 tensor 的每个滑动窗口，输出窗口内的最大值。常用于 CNN 网络中下采样特征图。

### 1.2 数学公式

$$\text{output}[n, c, h, w] = \max_{i \in [0, kH), j \in [0, kW)} \text{input}[n, c, s_h \cdot h + i \cdot d_h - p_h, s_w \cdot w + j \cdot d_w - p_w]$$

其中：
- $kH, kW$ 为 kernel_size 的高度和宽度
- $s_h, s_w$ 为 stride 的高度和宽度
- $p_h, p_w$ 为 padding 的高度和宽度偏移
- $d_h, d_w$ 为 dilation 的高度和宽度

### 1.3 算法描述

```
Algorithm: max_pool2d (Forward)
────────────────────────────────────
输入: X [N, C, H_in, W_in], kernel_size (kH, kW), stride (sH, sW), padding (pH, pW), dilation (dH, dW), ceil_mode
输出: Y [N, C, H_out, W_out]

1. 计算输出大小:
   if ceil_mode:
     H_out = ceil((H_in + 2*pH - dH*(kH-1) - 1) / sH + 1)
     W_out = ceil((W_in + 2*pW - dW*(kW-1) - 1) / sW + 1)
   else:
     H_out = floor((H_in + 2*pH - dH*(kH-1) - 1) / sH + 1)
     W_out = floor((W_in + 2*pW - dW*(kW-1) - 1) / sW + 1)

2. 对 batch*channel 维度进行分块处理 (loop_unroll)

3. 对每个输出位置 (oh, ow):
   3.1 计算输入窗口范围:
       h_start = oh * sH - top_pad
       w_start = ow * sW - left_pad
       h_end = h_start + (kH-1)*dH + 1
       w_end = w_start + (kW-1)*dW + 1

   3.2 边界 clamp (处理 padding):
       h_start_clamped = max(h_start, 0)
       w_start_clamped = max(w_start, 0)
       h_end_clamped = min(h_end, H_in)
       w_end_clamped = min(w_end, W_in)

   3.3 提取有效窗口并计算 max:
       使用 pypto.amax 对窗口内数据进行归约

   3.4 使用 pypto.assemble 将结果写入输出位置

4. return Y
```

### 1.4 数据流图

```
                    输入 input
           ┌─────────────────────┐
           │  [N, C, H_in, W_in]  │
           │   float16/float32    │
           └──────────┬───────────┘
                      │
                      ▼
              ┌───────────────┐
              │ reshape to    │
              │ [N*C, H, W]   │
              └───────┬───────┘
                      │
          ┌───────────┴───────────┐
          │ loop_unroll on N*C    │
          └───────────┬───────────┘
                      │
                      ▼
              ┌───────────────┐
              │ for each oh   │
              │   for each ow │
              └───────┬───────┘
                      │
                      ▼
              ┌───────────────┐
              │ extract window│
              │ with dilation │
              └───────┬───────┘
                      │
                      ▼
              ┌───────────────┐
              │ pypto.amax    │
              │ max reduce    │
              └───────┬───────┘
                      │
                      ▼
              ┌───────────────┐
              │ pypto.assemble│
              │ write output  │
              └───────┬───────┘
                      │
                      ▼
               ┌──────────────┐
               │ 输出 output   │
               │[N,C,H_out,W_out]│
               └──────────────┘
```

---

## 2. API 映射设计

### 2.1 数学公式分解

将公式拆解为基本操作步骤：

| 步骤 | 数学表达 | 说明 |
|------|----------|------|
| 1 | reshape | 将 4D 输入 reshape 为 3D [N*C, H, W] |
| 2 | sliding window | 根据参数提取滑动窗口 |
| 3 | amax(dim=H) | 对窗口 H 维度求最大值 |
| 4 | amax(dim=W) | 对窗口 W 维度求最大值 |
| 5 | assemble | 将结果写入输出位置 |

### 2.2 PyPTO API 映射表

| 步骤 | 数学表达 | PyPTO API | 参数 | 文档路径 |
|------|----------|-----------|------|----------|
| 1 | reshape | `pypto.reshape` | input, [bc, H, W], inplace=True | `docs/api/operation/pypto-reshape.md` |
| 2 | sliding window | Python 切片 | input[:, h_start:h_end, :] | - |
| 3 | max reduce (H) | `pypto.amax` | window, dim=1, keepdim=True | `docs/api/operation/pypto-amax.md` |
| 4 | max reduce (W) | `pypto.amax` | window, dim=2, keepdim=True | `docs/api/operation/pypto-amax.md` |
| 5 | assemble | `pypto.assemble` | max_val, [bc_idx, oh, ow], output | `docs/api/operation/pypto-assemble.md` |
| 6 | loop control | `pypto.loop_unroll` | 0, bc_total, 1, unroll_list=[8,4,2,1] | `docs/api/controlflow/pypto-loop_unroll.md` |
| 7 | create tensor | `pypto.tensor` | (shape, dtype) | `docs/api/operation/pypto-zeros.md` |
| 8 | output move | `pypto.Tensor.move` | reshape(output_tmp, ...) | - |

### 2.3 计算步骤序列

```python
# 伪代码展示计算流程
# 1. Reshape 输入
input_reshaped = pypto.reshape(input_tensor, [batch_size * channels, in_h, in_w], inplace=True)
output_tmp = pypto.tensor((bc_total, out_h, out_w), dtype)

# 2. Loop over batch*channel
for bc_idx, unroll_length in pypto.loop_unroll(0, bc_total, 1, unroll_list=[8, 4, 2, 1]):
    input_cur = input_reshaped[bc_idx:bc_idx+unroll_length, :, :]

    # 3. Loop over output height
    for oh in range(out_h):
        h_start = oh * stride_h - top_pad
        # ... 边界处理 ...

        # 4. Loop over output width
        for ow in range(out_w):
            w_start = ow * stride_w - left_pad
            # ... 边界处理 ...

            # 5. 提取窗口
            window = input_cur[:, h_start_clamped:h_end_clamped, w_start_clamped:w_end_clamped]

            # 6. 计算 max (两步归约)
            window_h_max = pypto.amax(window, dim=1, keepdim=True)  # H 维度
            max_val = pypto.amax(window_h_max, dim=2, keepdim=True)  # W 维度

            # 7. 写入输出
            pypto.assemble(max_val, [bc_idx, oh, ow], output_tmp)

# 8. Reshape 输出
output_result.move(pypto.reshape(output_tmp, [batch_size, channels, out_h, out_w], inplace=True))
```

### 2.4 设计依据

- 来源：api_report.md + models/experimental/vector/AvgPool2d/avg_pool2d.py
- 说明：
  - 参考 AvgPool2d 的 loop_unroll 模式和边界处理方式
  - 将 sum 替换为 amax 实现最大池化
  - 添加 dilation 和 ceil_mode 支持

---

## 3. 数据规格设计

### 3.1 OperatorInput dataclass

```python
@dataclass
class MaxPool2dInput:
    input_tensor: Tensor  # 输入特征图, shape: [N, C, H_in, W_in] 或 [C, H_in, W_in], dtype: float16/float32
    kernel_size: Tuple[int, int]  # 池化窗口大小 (kH, kW)
    stride: Tuple[int, int]  # 步长 (sH, sW)
    padding: Tuple[int, int, int, int]  # 填充 (top, bottom, left, right)
    dilation: Tuple[int, int]  # 空洞率 (dH, dW), 默认 (1, 1)
    ceil_mode: bool  # 是否使用 ceil 模式, 默认 False
```

### 3.2 OperatorOutput dataclass

```python
@dataclass
class MaxPool2dOutput:
    output_tensor: Tensor  # 池化输出, shape: [N, C, H_out, W_out] 或 [C, H_out, W_out], dtype: 与输入相同
```

### 3.3 中间 Tensor 定义

| 名称 | Shape | Dtype | 说明 |
|------|-------|-------|------|
| input_reshaped | [N*C, H_in, W_in] | 与输入相同 | Reshape 后的输入 |
| output_tmp | [N*C, H_out, W_out] | 与输入相同 | 临时输出 Tensor |
| window | [unroll, kH, kW] | 与输入相同 | 当前窗口数据 |
| window_h_max | [unroll, 1, kW] | 与输入相同 | H 维度 max 归约结果 |
| max_val | [unroll, 1, 1] | 与输入相同 | 最终 max 值 |

### 3.4 数据格式选择

| Tensor | 格式 | 说明 |
|--------|------|------|
| input_tensor | ND | 默认格式，保持原始布局 |
| output_tensor | ND | 与输入格式一致 |

### 3.5 动态轴定义

| 轴名称 | 含义 | 取值范围 |
|--------|------|----------|
| batch_size (N) | batch 维度 | [1, INT32_MAX] |
| channels (C) | 通道数 | [1, 2048] |
| H_in | 输入高度 | [kernel_size, INT32_MAX] |
| W_in | 输入宽度 | [kernel_size, INT32_MAX] |

### 3.6 JIT 装饰器配置

```python
@pypto.frontend.jit(
    pass_options={"vec_nbuffer_setting": {-1: 2, 0: 8}},
    runtime_options={
        "run_mode": pypto.RunMode.NPU,
        "stitch_function_num_initial": 128,
        "stitch_function_outcast_memory": 1024,
        "stitch_function_inner_memory": 1024
    },
    debug_options={"runtime_debug_mode": 1, "compile_debug_mode": 1}
)
def max_pool2d_kernel(
    input_tensor: pypto.Tensor((batch_size, channels, in_h, in_w), pypto.DT_FP32),
    output_result: pypto.Tensor((batch_size, channels, out_h, out_w), pypto.DT_FP32),
):
    ...
```

---

## 4. Tiling 策略

### 4.1 算子类型判断

- **类型**: Vector
- **判断依据**: 核心计算为滑动窗口内的 amax 归约操作，不涉及 matmul，属于 Vector 类型算子

### 4.2 TileShape 初值设置

```python
# 主循环前 - 4D tile shape
pypto.set_vec_tile_shapes(16, 16, 4, 128)

# 单行处理时 - 3D tile shape
pypto.set_vec_tile_shapes(16, 16, 128)

# 输出组装时 - 根据当前 unroll_length 调整
pypto.set_vec_tile_shapes(unroll_length, 4, 128)
```

### 4.3 设置依据

1. **主循环 TileShape (16, 16, 4, 128)**：
   - 16: batch*channel 维度的切分大小
   - 16: H 维度的切分大小
   - 4: W 维度的切分大小（用于窗口处理）
   - 128: 尾轴大小，满足 32B 对齐（fp32: 128*4=512B）

2. **单行处理 TileShape (16, 16, 128)**：
   - 用于 H 维度的 sum/amax 归约
   - 尾轴 128 满足对齐要求

3. **输出组装 TileShape (unroll_length, 4, 128)**：
   - unroll_length: 当前处理的 batch*channel 数量（8/4/2/1）
   - 4: H 维度
   - 128: W 维度

### 4.4 注意事项

- 尾轴必须 32B 对齐：fp32 尾轴需为 8 的倍数，fp16 需为 16 的倍数
- TileShape 次尾轴要小于等于 255
- amax 后维度变化，需重设 TileShape

### 4.5 判断依据与适用条件

- 判断依据：参考 AvgPool2d 实现的成功经验，采用相同的 tiling 策略
- 适用条件：适用于 4D 输入 [N, C, H, W]，H/W 在 32-512 范围内效果最佳
- 不适用场景：超大 H/W (>1024) 可能需要调整 tile shape

---

## 5. Loop 结构设计

### 场景 B：需要 Loop

#### 5.1 Loop 判断结论

- **结论**: 需要 Loop
- **原因**: 存在动态轴（batch_size, H, W 运行时才知道大小），需要用运行时循环遍历动态维度
- **Loop 类型**: `pypto.loop_unroll` + Python for
- **适用条件**: 输入包含动态轴，batch*channel 范围跨度大
- **限制**: 大量 loop_unroll 配置会增加编译时间

#### 5.2 静态轴 vs 动态轴处理

| 轴 | 类型 | 处理方式 |
|----|------|----------|
| batch_size | 动态 | `pypto.frontend.dynamic("batch_size")` |
| channels | 动态 | `pypto.frontend.dynamic("channels")` |
| out_h | 静态 | Python for |
| out_w | 静态 | Python for |
| batch*channel | 动态 | `pypto.loop_unroll` |

#### 5.3 Loop 合并策略

将 batch 和 channel 合并为一个 loop（batch*channel），减少嵌套层级：
- 使用 `bc_total = batch_size * channels`
- 单层 `loop_unroll` 遍历所有 batch*channel

#### 5.4 数据依赖处理

- 每个 batch*channel 的计算相互独立，无数据依赖
- 输出位置通过 `assemble` 的 offset 参数精确定位

#### 5.5 尾块处理策略

使用 `loop_unroll` 的 `unroll_list=[8, 4, 2, 1]` 自动处理尾块：
- 优先使用 unroll_length=8 处理
- 剩余部分依次用 4, 2, 1 处理
- 确保所有数据都被处理

#### 5.6 loop_unroll 配置

```python
for bc_idx, unroll_length in pypto.loop_unroll(
    0, bc_total, 1,
    name="LOOP_BC",
    idx_name="bc_idx",
    unroll_list=[8, 4, 2, 1]
):
    input_cur = input_reshaped[bc_idx:bc_idx+unroll_length, :, :]
    # ... 处理逻辑 ...
```

---

## 6. 验证方案

### 6.1 Golden 函数设计

```python
def max_pool2d_golden(
    input: torch.Tensor,
    kernel_size: Union[int, Tuple[int, int]],
    stride: Optional[Union[int, Tuple[int, int]]] = None,
    padding: Union[int, Tuple[int, int]] = 0,
    dilation: Union[int, Tuple[int, int]] = 1,
    ceil_mode: bool = False,
) -> torch.Tensor:
    """max_pool2d 参考实现"""
    return F.max_pool2d(
        input=input,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        ceil_mode=ceil_mode,
    )
```

### 6.2 测试用例设计

#### 基于 spec.md 所有典型配置

| 配置名称 | 类型 | 优先级 | 参数 | 输入 Shape | 输出 Shape | 说明 |
|----------|------|--------|------|------------|------------|------|
| resnet_pool | 性能 | P0 | kernel=3, stride=2, padding=1 | [32, 64, 112, 112] | [32, 64, 56, 56] | ResNet 典型池化层 |
| vgg_pool | 性能 | P0 | kernel=2, stride=2, padding=0 | [32, 128, 56, 56] | [32, 128, 28, 28] | VGG 典型池化层 |
| dynamic_batch | 功能 | P0 | kernel=3, stride=2, padding=1 | [?, 64, 224, 224] | [?, 64, 112, 112] | 动态 batch 测试 |
| dynamic_hw | 功能 | P0 | kernel=2, stride=2, padding=0 | [16, 32, ?, ?] | [16, 32, H/2, W/2] | 动态 H/W 测试 |
| float16_test | 功能 | P1 | kernel=3, stride=2, padding=1 | [16, 64, 56, 56] (fp16) | [16, 64, 28, 28] (fp16) | float16 精度验证 |
| dilation_test | 功能 | P1 | kernel=5, stride=1, padding=2, dilation=2 | [8, 32, 32, 32] | [8, 32, 28, 28] | 空洞池化测试 |
| ceil_mode_test | 功能 | P1 | kernel=2, stride=2, padding=0, ceil_mode=True | [8, 32, 7, 7] | [8, 32, 4, 4] | ceil_mode 测试 |

#### 边界情况测试（可选）

| 场景 | 参数 | 说明 |
|------|------|------|
| 全零输入 | input=zeros | 验证输出为全零 |
| 单元素 kernel | kernel=1, stride=1 | 输出应等于输入 |
| 3D 输入 | [C, H, W] | 验证无 batch 维度场景 |

### 6.3 精度验证标准

| Dtype | atol | rtol |
|-------|------|------|
| float32 | 0.001 | 0.001 |
| float16 | 0.005 | 0.005 |

---

## 7. 性能指标与开箱配置

### 7.1 性能目标

基于 spec.md 典型配置（性能类）的预期性能：

| 配置名称 | 类型 | 优先级 | 参数 | 输入 Shape | 输出 Shape | 预期 kernel 耗时 |
|----------|------|--------|------|------------|------------|------------------|
| resnet_pool | 性能 | P0 | kernel=3, stride=2, padding=1 | [32, 64, 112, 112] | [32, 64, 56, 56] | < 1ms |
| vgg_pool | 性能 | P0 | kernel=2, stride=2, padding=0 | [32, 128, 56, 56] | [32, 128, 28, 28] | < 0.5ms |

### 7.2 开箱性能配置

```python
# Tiling 配置
pypto.set_vec_tile_shapes(16, 16, 4, 128)  # 主循环
pypto.set_vec_tile_shapes(16, 16, 128)      # 单行处理
pypto.set_vec_tile_shapes(unroll_length, 4, 128)  # 输出组装
```

### 7.3 pass_options 配置

```python
pass_options={"vec_nbuffer_setting": {-1: 2, 0: 8}}
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

- `pypto.amax` 仅支持单轴归约，需要对 H 和 W 维度分别归约
- `pypto.pad` 不支持左/上填充，需在 kernel 内通过边界检查处理
- TileShape 次尾轴要小于等于 255
- 尾轴需要 32B 对齐

### 8.2 常见错误规避

| 风险 / 错误 | 触发场景 | 影响 / 原因 | 规避方法 |
|-------------|----------|-------------|----------|
| TileShape 不对齐 | 尾轴非 8/16 倍数 | 编译失败或性能劣化 | 确保尾轴为 8 (fp32) 或 16 (fp16) 的倍数 |
| amax 后维度变化 | keepdim=False | 后续操作 shape 不匹配 | amax 后重设 TileShape |
| 边界越界 | padding 导致窗口超出边界 | 访问非法内存 | 使用 clamp 限制窗口范围 |
| dilation 计算错误 | dilation > 1 | 窗口索引错误 | 窗口大小 = (kernel-1)*dilation + 1 |

### 8.3 特殊场景处理

1. **padding 边界处理**：
   - 通过 clamp 将窗口起始/结束位置限制在 [0, H/W] 范围内
   - 窗口无效时（cur_k_h <= 0 或 cur_k_w <= 0）使用极小值填充

2. **dilation 支持**：
   - 窗口索引计算时乘以 dilation 因子
   - 有效窗口大小 = (kernel_size - 1) * dilation + 1

3. **ceil_mode 支持**：
   - 在参数计算阶段根据 ceil_mode 计算输出 shape
   - 使用 `math.ceil` 而非默认的 `math.floor`

### 8.4 实现建议

| 建议项 | 说明 |
|--------|------|
| 参考 AvgPool2d | 复用其 loop_unroll 模式和边界处理逻辑 |
| 分步归约 | 使用两次 amax 分别处理 H 和 W 维度 |
| 动态轴声明 | 使用 `pypto.frontend.dynamic()` 声明动态维度 |
| 边界 clamp | 使用 Python max/min 进行边界限制 |

---

## 9. 交付件清单

### 9.1 目录结构

```
operators/max_pool2d/
├── spec.md                          # 需求规范（已有）
├── api_report.md                    # API 探索报告（已有）
├── design.md                        # 设计文档（本文件）
├── max_pool2d_golden.py             # Golden 参考实现（已有）
├── max_pool2d_impl.py               # 算子实现代码
├── test_max_pool2d.py               # 测试代码
└── .orchestrator_state.json         # 状态文件
```

### 9.2 文件清单

| 文件 | 类型 | 说明 | 生成方式 |
|------|------|------|----------|
| spec.md | 需求 | 算子需求规范 | pypto-intent-understanding |
| api_report.md | 设计 | API 探索报告 | pypto-api-explorer |
| design.md | 设计 | 算子设计文档 | pypto-op-design（本 skill） |
| max_pool2d_golden.py | 代码 | Golden 参考实现 | pypto-golden-generator |
| max_pool2d_impl.py | 代码 | 算子核心实现 | 后续实现 |
| test_max_pool2d.py | 代码 | 测试用例 | 后续实现 |

### 9.3 命名规范

| 项目 | 规范 | 示例 |
|------|------|------|
| 算子名称 | 小写字母 + 下划线 | `max_pool2d` |
| 目录名 | 与算子名称一致 | `operators/max_pool2d/` |
| Golden 文件 | `{op}_golden.py` | `max_pool2d_golden.py` |
| 实现文件 | `{op}_impl.py` | `max_pool2d_impl.py` |
| 测试文件 | `test_{op}.py` | `test_max_pool2d.py` |

### 9.4 生成顺序

```
spec.md → api_report.md → design.md → max_pool2d_golden.py → max_pool2d_impl.py → test_max_pool2d.py
```
