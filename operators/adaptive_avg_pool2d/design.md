# adaptive_avg_pool2d 算子设计文档

> **算子名称**: adaptive_avg_pool2d
> **算子分类**: pooling
> **生成时间**: 2026-03-29
> **基于**: spec.md

---

## 1. 概述

### 1.1 功能描述

自适应平均池化，将任意尺寸的输入池化到指定的输出尺寸。池化窗口的大小和位置根据输入和输出尺寸动态计算，支持非对齐的池化窗口。

### 1.2 数学公式

$$output[b,c,h,w] = \frac{1}{(h_{end} - h_{start}) \times (w_{end} - w_{start})} \sum_{i=h_{start}}^{h_{end}-1} \sum_{j=w_{start}}^{w_{end}-1} input[b,c,i,j]$$

其中池化窗口位置动态计算：
- $h_{start} = \lfloor h \times iH / oH \rfloor$
- $h_{end} = \lceil (h+1) \times iH / oH \rceil$
- $w_{start} = \lfloor w \times iW / oW \rfloor$
- $w_{end} = \lceil (w+1) \times iW / oW \rceil$

### 1.3 算法描述

```
Algorithm: Adaptive Average Pooling 2D
────────────────────────────────────────
输入: input [N, C, H, W], output_size (oH, oW)
输出: output [N, C, oH, oW]

1. 解析 output_size:
   - 若为单个 int: oH = oW = output_size
   - 若为 tuple: (oH, oW)

2. 合并 batch 和 channel 维度: bc_total = N * C

3. for bc_idx in [0, bc_total) with loop_unroll:
     input_cur = input[bc_idx : bc_idx + unroll_length]
     for oh in [0, oH):
       for ow in [0, oW):
         3.1 计算输入窗口范围 (编译时常量):
             h_start = floor(oh * H / oH)
             h_end = ceil((oh + 1) * H / oH)
             w_start = floor(ow * W / oW)
             w_end = ceil((ow + 1) * W / oW)

         3.2 提取窗口数据:
             window = input_cur[:, h_start:h_end, w_start:w_end]

         3.3 计算平均值:
             sum_h = pypto.sum(window, dim=1, keepdim=True)
             sum_hw = pypto.sum(sum_h, dim=2, keepdim=True)
             win_size = (h_end - h_start) * (w_end - w_start)
             avg_val = sum_hw / win_size

         3.4 写入输出:
             pypto.assemble(avg_val, [bc_idx, oh, ow], output_tmp)

4. reshape output_tmp -> [N, C, oH, oW]
5. return output
```

### 1.4 数据流图

```
    输入 input                              输出 output
┌──────────────────┐                   ┌──────────────────┐
│  [N, C, H, W]    │                   │  [N, C, oH, oW]  │
│   float32        │  adaptive_avg_    │   float32        │
│                  │  pool2d           │                  │
└────────┬─────────┘                   └────────┬─────────┘
         │                                      ▲
         │    ┌────────────────────────────┐    │
         └───▶│ 参数: output_size (oH, oW) │────┘
              │                            │
              │ 动态计算每个输出位置对应的  │
              │ 输入窗口范围和窗口大小      │
              │                            │
              │ h_start = floor(h*H/oH)    │
              │ h_end = ceil((h+1)*H/oH)   │
              └────────────────────────────┘

动态轴: N, H, W
```

---

## 2. API 映射设计

### 2.1 数学公式分解

将公式拆解为基本操作步骤：

| 步骤 | 数学表达 | 说明 |
|------|----------|------|
| 1 | `input_cur = input[bc_idx:bc_idx+unroll]` | 提取当前 batch/channel 数据 |
| 2 | `window = input_cur[:, h_start:h_end, w_start:w_end]` | 提取池化窗口 |
| 3 | `sum_h = sum(window, dim=1)` | 沿 H 维度求和 |
| 4 | `sum_hw = sum(sum_h, dim=2)` | 沿 W 维度求和 |
| 5 | `avg = sum_hw / win_size` | 计算平均值 |
| 6 | `output[bc, oh, ow] = avg` | 写入输出位置 |

### 2.2 PyPTO API 映射表

| 步骤 | 数学表达 | PyPTO API | 参数 | 文档路径 |
|------|----------|-----------|------|----------|
| 1 | 切片提取 | `Tensor.__getitem__` | `input[start:end]` | `docs/api/tensor/pypto-Tensor-getitem.md` |
| 2 | 切片窗口 | `Tensor.__getitem__` | `input[:, h1:h2, w1:w2]` | `docs/api/tensor/pypto-Tensor-getitem.md` |
| 3 | H 维度求和 | `pypto.sum` | `dim=1, keepdim=True` | `docs/api/operation/pypto-sum.md` |
| 4 | W 维度求和 | `pypto.sum` | `dim=2, keepdim=True` | `docs/api/operation/pypto-sum.md` |
| 5 | 除法 | `pypto.div` 或 `/` | `sum / win_size` | `docs/api/operation/pypto-div.md` |
| 6 | 组装输出 | `pypto.assemble` | `assemble(val, idx, out)` | `docs/api/operation/pypto-assemble.md` |
| 7 | 循环展开 | `pypto.loop_unroll` | `unroll_list=[8,4,2,1]` | `docs/api/controlflow/pypto-loop_unroll.md` |

### 2.3 计算步骤序列

```python
# 伪代码展示计算流程
bc_total = batch_size * channels
output_tmp = pypto.tensor((bc_total, out_h, out_w), pypto.DT_FP32)

for bc_idx, unroll_length in pypto.loop_unroll(0, bc_total, 1, name="LOOP_BC",
                                                idx_name="bc_idx", unroll_list=[8, 4, 2, 1]):
    input_cur = input_reshaped[bc_idx: bc_idx + unroll_length, :, :]

    for oh in range(out_h):
        h_start = (oh * in_h) // out_h
        h_end = ((oh + 1) * in_h + out_h - 1) // out_h  # ceil

        for ow in range(out_w):
            w_start = (ow * in_w) // out_w
            w_end = ((ow + 1) * in_w + out_w - 1) // out_w  # ceil

            window = input_cur[:, h_start:h_end, w_start:w_end]
            sum_h = pypto.sum(window, dim=1, keepdim=True)
            sum_hw = pypto.sum(sum_h, dim=2, keepdim=True)
            win_size = (h_end - h_start) * (w_end - w_start)
            avg_val = sum_hw / win_size

            pypto.assemble(avg_val, [bc_idx, oh, ow], output_tmp)

output = pypto.reshape(output_tmp, [batch_size, channels, out_h, out_w])
```

### 2.4 设计依据

- 来源：参考 `models/experimental/vector/AvgPool2d/avg_pool2d.py` 的实现模式
- 说明：adaptive_avg_pool2d 与 avg_pool2d 类似，但窗口大小和位置需要动态计算。参考 avg_pool2d 使用 loop_unroll 遍历 batch/channel，使用 Python for 遍历输出位置的模式。

---

## 3. 数据规格设计

### 3.1 OperatorInput dataclass

```python
@dataclass(frozen=True)
class AdaptiveAvgPool2dInput:
    input: Tensor      # 输入张量, shape: [N, C, H, W], dtype: float32
    output_size: Tuple[int, int]  # 输出尺寸 (oH, oW)，编译时常量
```

### 3.2 OperatorOutput dataclass

```python
@dataclass
class AdaptiveAvgPool2dOutput:
    output: Tensor     # 输出张量, shape: [N, C, oH, oW], dtype: float32
```

### 3.3 中间 Tensor 定义

| 名称 | Shape | Dtype | 说明 |
|------|-------|-------|------|
| input_reshaped | [N*C, H, W] | float32 | 合并 batch/channel 后的输入 |
| output_tmp | [N*C, oH, oW] | float32 | 中间输出张量 |
| input_cur | [unroll, H, W] | float32 | 当前循环切片 |
| window | [unroll, win_h, win_w] | float32 | 池化窗口 |
| sum_h | [unroll, 1, win_w] | float32 | H 维度求和结果 |
| sum_hw | [unroll, 1, 1] | float32 | 窗口总和 |
| avg_val | [unroll, 1, 1] | float32 | 窗口平均值 |

### 3.4 数据格式选择

| Tensor | 格式 | 说明 |
|--------|------|------|
| input/output | ND | 4D 张量，标准 NCHW 格式 |
| 中间 tensor | ND | 保持维度顺序 |

### 3.5 动态轴定义

| 轴名称 | 含义 | 取值范围 |
|--------|------|----------|
| N | batch size | [1, INT32_MAX] |
| H | 输入高度 | [1, INT32_MAX] |
| W | 输入宽度 | [1, INT32_MAX] |

### 3.6 JIT 装饰器配置

```python
@pypto.frontend.jit(
    pass_options={"vec_nbuffer_setting": {-1: 2, 0: 8}},
    runtime_options={"run_mode": pypto.RunMode.NPU}
)
def adaptive_avg_pool2d_kernel(
    input_tensor: pypto.Tensor((batch_size, channels, in_h, in_w), pypto.DT_FP32),
    output_result: pypto.Tensor((batch_size, channels, out_h, out_w), pypto.DT_FP32),
):
    ...
```

---

## 4. Tiling 策略

### 4.1 算子类型判断

- **类型**: Vector
- **判断依据**: 不含 matmul 操作，主要是逐元素和归约操作

### 4.2 TileShape 初值设置

```python
# 主循环 tiling (batch/channel 维度)
pypto.set_vec_tile_shapes(16, 16, 4, 128)

# 窗口操作 tiling
pypto.set_vec_tile_shapes(16, 16, 128)

# 组装输出 tiling
pypto.set_vec_tile_shapes(unroll_length, 4, 128)
```

### 4.3 设置依据

1. **尾轴 32B 对齐**: fp32 尾轴需为 8 的倍数，设置 128 满足要求
2. **次尾轴 <= 255**: 16 满足约束
3. **参考 avg_pool2d**: 使用相同的 tiling 配置模式

### 4.4 注意事项

- 在 sum 操作后，keepdim=False 会导致维度减少，需要重新设置 TileShape
- 不同操作间需要根据输出 shape 调整 TileShape

### 4.5 判断依据与适用条件

- 判断依据：参考 avg_pool2d 的 tiling 配置，该算子已在生产环境验证
- 适用条件：4D 输入 [N, C, H, W]，fp32 数据类型
- 不适用场景：5D 输入需要调整 tiling 配置

---

## 5. Loop 结构设计

### 场景 B：需要 Loop

#### 5.1 Loop 判断结论

- **结论**: 需要 Loop
- **原因**: 存在动态轴 N（batch size 运行时确定），需要遍历所有 batch/channel 组合
- **Loop 类型**: `pypto.loop_unroll` (遍历 batch/channel) + Python for (遍历输出位置)
- **适用条件**: 动态 batch size，编译期未知的 N
- **限制**: 输出尺寸 oH, oW 需要是编译时常量

#### 5.2 静态轴 vs 动态轴处理

| 轴 | 类型 | 处理方式 |
|----|------|----------|
| N (batch) | 动态 | `pypto.loop_unroll` |
| C (channel) | 动态 | 与 N 合并为 bc_total，使用 `loop_unroll` |
| oH (输出高度) | 静态 | Python for 循环 |
| oW (输出宽度) | 静态 | Python for 循环 |

#### 5.3 Loop 合并策略

将 N 和 C 维度合并为 bc_total = N * C，使用单个 `loop_unroll` 遍历，减少循环嵌套层级。

#### 5.4 数据依赖处理

每个输出位置 (oh, ow) 的计算相互独立，无数据依赖，可以并行处理。

#### 5.5 尾块处理策略

使用 `loop_unroll` 的 `unroll_list=[8, 4, 2, 1]` 自动处理尾块，最后一个迭代会自动调整 unroll_length。

#### 5.6 loop_unroll 配置

```python
for bc_idx, unroll_length in pypto.loop_unroll(
    0, bc_total, 1,
    name="LOOP_BC",
    idx_name="bc_idx",
    unroll_list=[8, 4, 2, 1]
):
    # unroll_length 自动处理尾块
    input_cur = input_reshaped[bc_idx: bc_idx + unroll_length, :, :]
    ...
```

---

## 6. 验证方案

### 6.1 Golden 函数设计

```python
def adaptive_avg_pool2d_golden(
    input: torch.Tensor,
    output_size: Union[int, Tuple[int, int]],
) -> torch.Tensor:
    """adaptive_avg_pool2d 参考实现 (PyTorch)"""
    return F.adaptive_avg_pool2d(input, output_size)
```

### 6.2 测试用例设计

#### 基于 spec.md 所有典型配置

| 配置名称 | 类型 | 优先级 | 参数 | 输入 Shape | 输出 Shape | 说明 |
|----------|------|--------|------|------------|------------|------|
| 性能_P0 | 性能 | P0 | output_size=(7,7) | [16, 256, 14, 14] | [16, 256, 7, 7] | 典型 CNN 特征图下采样 |
| 功能_P0 | 功能 | P0 | output_size=(1,1) | [8, 512, 7, 7] | [8, 512, 1, 1] | 全局平均池化 |
| 动态轴_P0 | 功能 | P0 | output_size=(14,14) | [N, 64, H, W] | [N, 64, 14, 14] | 动态轴验证 |
| 单值尺寸_P1 | 功能 | P1 | output_size=8 | [4, 128, 16, 16] | [4, 128, 8, 8] | 单值输出尺寸 |
| 非对齐_P1 | 功能 | P1 | output_size=(5,5) | [2, 64, 7, 7] | [2, 64, 5, 5] | 非对齐池化窗口 |

#### 边界情况测试（可选）

| 场景 | 参数 | 说明 |
|------|------|------|
| 最小输入 | input=[1,1,1,1], output_size=(1,1) | 单元素输入 |
| 大 batch | input=[128,64,7,7], output_size=(1,1) | 大 batch 验证 |
| 非方形输出 | input=[2,4,8,8], output_size=(3,5) | 非方形输出尺寸 |

### 6.3 精度验证标准

| Dtype | atol | rtol |
|-------|------|------|
| float32 | 0.001 | 0.001 |
| float16 | 0.001 | 0.001 |
| bfloat16 | 0.01 | 0.01 |

---

## 7. 性能指标与开箱配置

### 7.1 性能目标

基于 spec.md 典型配置（性能类）的预期性能：

| 配置名称 | 类型 | 优先级 | 参数 | 输入 Shape | 输出 Shape | 预期 kernel 耗时 |
|----------|------|--------|------|------------|------------|------------------|
| 性能_P0 | 性能 | P0 | output_size=(7,7) | [16, 256, 14, 14] | [16, 256, 7, 7] | 待实测 |
| 功能_P0 | 性能 | P0 | output_size=(1,1) | [8, 512, 7, 7] | [8, 512, 1, 1] | 待实测 |

### 7.2 开箱性能配置

```python
@pypto.frontend.jit(
    pass_options={"vec_nbuffer_setting": {-1: 2, 0: 8}},
    runtime_options={
        "run_mode": pypto.RunMode.NPU,
        "stitch_function_num_initial": 128,
        "stitch_function_outcast_memory": 1024,
        "stitch_function_inner_memory": 1024
    }
)
```

### 7.3 pass_options 配置

```python
pass_options={"vec_nbuffer_setting": {-1: 2, 0: 8}}
```

- `-1: 2`: 尾轴 buffer 数量
- `0: 8`: 首轴 buffer 数量

### 7.4 runtime_options 配置

```python
runtime_options={
    "run_mode": pypto.RunMode.NPU,  # NPU 运行模式
    "stitch_function_num_initial": 128,  # 初始函数数量
    "stitch_function_outcast_memory": 1024,  # 外部内存
    "stitch_function_inner_memory": 1024  # 内部内存
}
```

---

## 8. 风险点与注意事项

### 8.1 已知约束

- 输出尺寸 output_size 必须是编译时常量
- 输入 H, W 必须能被 output_size 整除或处理非对齐情况
- pypto.sum 的 keepdim=False 会导致维度减少，需要重新设置 TileShape

### 8.2 常见错误规避

| 风险 / 错误 | 触发场景 | 影响 / 原因 | 规避方法 |
|-------------|----------|-------------|----------|
| TileShape 不匹配 | sum 后未重设 | 编译失败 | 每次 sum 操作后调用 set_vec_tile_shapes |
| 尾轴不对齐 | fp32 尾轴非 8 倍数 | 编译失败或性能劣化 | 确保尾轴为 8 的倍数 |
| 窗口越界 | 动态窗口计算 | 运行时错误 | 使用 min/max 裁剪窗口边界 |
| 除零错误 | win_size = 0 | NaN 输出 | 确保 H, W >= oH, oW |

### 8.3 特殊场景处理

1. **非对齐窗口**: 当 H 不能被 oH 整除时，不同输出位置的窗口大小可能不同
2. **全局池化**: output_size=(1,1) 时，窗口覆盖整个输入
3. **边界处理**: 窗口可能超出输入边界，需要使用 min/max 裁剪

### 8.4 实现建议

| 建议项 | 说明 |
|--------|------|
| 参考 avg_pool2d | `models/experimental/vector/AvgPool2d/avg_pool2d.py` 提供了类似实现模式 |
| 动态窗口计算 | 窗口位置使用整数除法计算，避免浮点运算 |
| 分离 H/W 求和 | 先 sum H 维度，再 sum W 维度，减少单次计算量 |

---

## 9. 交付件清单

### 9.1 目录结构

```
operators/adaptive_avg_pool2d/
├── spec.md                          # 需求规范（已有）
├── design.md                        # 设计文档（本文件）
├── adaptive_avg_pool2d_golden.py    # Golden 参考实现（已有）
├── adaptive_avg_pool2d_impl.py      # 算子实现代码
├── test_adaptive_avg_pool2d.py      # 测试代码
└── README.md                        # 使用说明
```

### 9.2 文件清单

| 文件 | 类型 | 说明 | 生成方式 |
|------|------|------|----------|
| spec.md | 需求 | 算子需求规范 | pypto-intent-understanding |
| design.md | 设计 | 算子设计文档 | pypto-op-design（本 skill） |
| adaptive_avg_pool2d_golden.py | 代码 | Golden 参考实现 | pypto-golden-generator |
| adaptive_avg_pool2d_impl.py | 代码 | 算子核心实现 | pypto-op-develop |
| test_adaptive_avg_pool2d.py | 代码 | 测试用例 | pypto-op-develop |
| README.md | 文档 | 使用说明 | 手动编写 |

### 9.3 命名规范

| 项目 | 规范 | 示例 |
|------|------|------|
| 算子名称 | 小写字母 + 下划线 | `adaptive_avg_pool2d` |
| 目录名 | 与算子名称一致 | `operators/adaptive_avg_pool2d/` |
| Golden 文件 | `{op}_golden.py` | `adaptive_avg_pool2d_golden.py` |
| 实现文件 | `{op}_impl.py` | `adaptive_avg_pool2d_impl.py` |
| 测试文件 | `test_{op}.py` | `test_adaptive_avg_pool2d.py` |

### 9.4 生成顺序

```
spec.md → design.md → {op}_golden.py → {op}_impl.py → test_{op}.py
```
