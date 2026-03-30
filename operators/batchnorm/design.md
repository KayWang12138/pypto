# BatchNorm 算子设计文档

> **生成时间**: 2026-03-29T14:25:00Z
> **算子名称**: batchnorm

---

## 1. 概述

### 1.1 基本信息

- **算子名称**: batchnorm
- **算子分类**: normalization
- **数学公式**: $y = \frac{x - E[x]}{\sqrt{Var[x] + \epsilon}} \cdot \gamma + \beta$
- **功能描述**: Batch Normalization 对输入张量在通道维度上进行归一化处理。训练时使用当前 batch 的均值和方差，推理时使用 running mean 和 running variance.

### 1.2 数据流图

```
         输入 x                  gamma                beta
    ┌──────────────┐      ┌────────────┐      ┌────────────┐
    │[b,s,c,h,w]   │      │    [c]     │      │    [c]     │
    │   float32    │      │  float32   │      │  float32   │
    └──────┬───────┘      └─────┬──────┘      └─────┬──────┘
           │                    │                   │
           ▼                    │                   │
    ┌──────────────┐            │                   │
    │  Reshape 2D  │            │                   │
    │  [N, c]       │            │                   │
    └──────┬───────┘            │                   │
           │                    │                   │
           ▼                    │                   │
    ┌──────────────┐            │                   │
    │  sum(x,dim=0) │            │                   │
    │  [1, c]       │            │                   │
    └──────┬───────┘            │                   │
           │                    │                   │
           ▼                    │                   │
    ┌──────────────┐            │                   │
    │  mean/N      │            │                   │
    │  [1, c]       │            │                   │
    └──────┬───────┘            │                   │
           │                    │                   │
           ▼                    │                   │
    ┌──────────────┐            │                   │
    │  x - mean    │            │                   │
    │  [N, c]       │            │                   │
    └──────┬───────┘            │                   │
           │                    │                   │
           ▼                    │                   │
    ┌──────────────┐            │                   │
    │ (x-mean)^2   │            │                   │
    │  [N, c]       │            │                   │
    └──────┬───────┘            │                   │
           │                    │                   │
           ▼                    │                   │
    ┌──────────────┐            │                   │
    │ sum(^2,dim=0)│            │                   │
    │  [1, c]       │            │                   │
    └──────┬───────┘            │                   │
           │                    │                   │
           ▼                    │                   │
    ┌──────────────┐            │                   │
    │  var/N       │            │                   │
    │  [1, c]       │            │                   │
    └──────┬───────┘            │                   │
           │                    │                   │
           ▼                    │                   │
    ┌──────────────┐            │                   │
    │sqrt(var+eps) │            │                   │
    │  [1, c]       │            │                   │
    └──────┬───────┘            │                   │
           │                    │                   │
           ▼                    │                   │
    ┌──────────────┐            │                   │
    │(x-mean)/std  │            │                   │
    │  [N, c]       │            │                   │
    └──────┬───────┘            │                   │
           │                    │                   │
           ├────────────────────┘                   │
           ▼                                        │
    ┌──────────────┐                                │
    │  * gamma     │                                │
    │  [N, c]       │                                │
    └──────┬───────┘                                │
           │                                        │
           ├────────────────────────────────────────┘
           ▼
    ┌──────────────┐
    │  + beta      │
    │  [N, c]       │
    └──────┬───────┘
           │
           ▼
    ┌──────────────┐
    │Reshape 5D    │
    │[b,s,c,h,w]   │
    └──────┬───────┘
           │
           ▼
       输出 y
    ┌──────────────┐
    │  [b,s,c,h,w] │
    │   float32    │
    └──────────────┘

动态轴: b (batch), s (seq_len)
归一化轴: c (channels) - 沿此维度归约计算 mean/var
实现策略: Reshape 到 2D [N, c]，归一化后再 reshape 回 5D
```

---

## 2. API 映射设计

### 2.1 公式分解

| 步骤 | 数学表达 | 说明 |
|------|----------|------|
| 1 | x_2d = reshape(x, [N, c]) | 将 5D 输入 reshape 为 2D |
| 2 | sum_x = pypto.sum(x_2d, dim=0, keepdim=True) | 沿 dim=0 归约计算总和 |
| 3 | mean = pypto.mul(sum_x, 1.0/N) | 计算均值 |
| 4 | centered = pypto.sub(x_2d, mean) | 中心化 |
| 5 | squared = pypto.mul(centered, centered) | 计算平方 |
| 6 | sum_sq = pypto.sum(squared, dim=0, keepdim=True) | 沿 dim=0 归约计算平方和 |
| 7 | var = pypto.mul(sum_sq, 1.0/N) | 计算方差 |
| 8 | var_eps = pypto.add(var, eps) | 加 epsilon |
| 9 | std = pypto.sqrt(var_eps) | 计算标准差 |
| 10 | normalized = pypto.div(centered, std) | 归一化 |
| 11 | scaled = pypto.mul(normalized, gamma) | 缩放 |
| 12 | output_2d = pypto.add(scaled, beta) | 偏移 |
| 13 | output = pypto.reshape(output_2d, [batch, seq_len, channels, H, W]) | reshape 回 5D |

### 2.2 PyPTO API 映射表

| 步骤 | 数学表达 | PyPTO API | 参数 | 文档路径 |
|------|----------|-----------|------|----------|
| 1 | reshape 5D to 2D | `pypto.reshape` | input, [N, channels] | `docs/api/operation/pypto-reshape.md` |
| 2 | sum(x, dim=0) | `pypto.sum` | input, dim=0, keepdim=True | `docs/api/operation/pypto-sum.md` |
| 3 | mean = sum / N | `pypto.mul` | sum_result, 1.0/N | `docs/api/operation/pypto-mul.md` |
| 4 | x - mean | `pypto.sub` | x_2d, mean | `docs/api/operation/pypto-sub.md` |
| 5 | centered^2 | `pypto.mul` | centered, centered | `docs/api/operation/pypto-mul.md` |
| 6 | sum(squared, dim=0) | `pypto.sum` | squared, dim=0, keepdim=True | `docs/api/operation/pypto-sum.md` |
| 7 | var = sum_sq / N | `pypto.mul` | sum_sq, 1.0/N | `docs/api/operation/pypto-mul.md` |
| 8 | var + eps | `pypto.add` | var, eps | `docs/api/operation/pypto-add.md` |
| 9 | sqrt(var_eps) | `pypto.sqrt` | var_eps | `docs/api/operation/pypto-sqrt.md` |
| 10 | centered / std | `pypto.div` | centered, std | `docs/api/operation/pypto-div.md` |
| 11 | normalized * gamma | `pypto.mul` | normalized, gamma | `docs/api/operation/pypto-mul.md` |
| 12 | scaled + beta | `pypto.add` | scaled, beta | `docs/api/operation/pypto-add.md` |
| 13 | reshape 2D to 5D | `pypto.reshape` | output_2d, [batch, seq_len, channels, H, W] | `docs/api/operation/pypto-reshape.md` |

### 2.3 计算步骤序列

```python
def batchnorm_core(x, gamma, beta, eps, N, channels):
    # Step 1: Reshape 5D to 2D [N, channels]
    x_2d = pypto.reshape(x, [N, channels])

    # Step 2-3: Compute mean along dim=0
    sum_x = pypto.sum(x_2d, dim=0, keepdim=True)
    mean = pypto.mul(sum_x, 1.0 / N)

    # Step 4: Center
    centered = pypto.sub(x_2d, mean)

    # Step 5-7: Compute variance
    squared = pypto.mul(centered, centered)
    sum_sq = pypto.sum(squared, dim=0, keepdim=True)
    var = pypto.mul(sum_sq, 1.0 / N)

    # Step 8-10: Normalize
    var_eps = pypto.add(var, eps)
    std = pypto.sqrt(var_eps)
    normalized = pypto.div(centered, std)

    # Step 11-12: Affine transform
    scaled = pypto.mul(normalized, gamma)
    output_2d = pypto.add(scaled, beta)

    # Step 13: Reshape 2D to 5D
    return pypto.reshape(output_2d, [batch, seq_len, channels, H, W])
```

### 2.4 设计依据

- 来源: `examples/02_intermediate/basic_nn/layer_normalization/layer_norm.py` 的 `layernorm_core` 函数
- 说明: BatchNorm 与 LayerNorm 结构相似，区别在于归一化轴。采用 reshape 到 2D 方案规避 pypto.var 多轴归约约束。

---

## 3. 数据规格设计

### 3.1 输入数据规格

| 变量 | Shape | Dtype | 动态轴 | 说明 |
|------|-------|-------|--------|------|
| x | [batch, seq_len, channels, H, W] | float32/float16/bfloat16 | batch, seq_len | 输入张量 (5D) |
| gamma | [channels] | float32/float16/bfloat16 | 无 | 缩放参数 (1D) |
| beta | [channels] | float32/float16/bfloat16 | 无 | 偏移参数 (1D) |

### 3.2 输出数据规格

| 变量 | Shape | Dtype | 动态轴 | 说明 |
|------|-------|-------|--------|------|
| output | [batch, seq_len, channels, H, W] | float32/float16/bfloat16 | batch, seq_len | 归一化输出 (5D) |

### 3.3 中间 Tensor 定义

| 名称 | Shape | Dtype | 说明 |
|------|-------|-------|------|
| x_2d | [N, channels] | float32/float16/bfloat16 | Reshape 后的 2D 输入 |
| mean | [1, channels] | float32/float16/bfloat16 | 均值 (FP32 建议用于精度) |
| centered | [N, channels] | float32/float16/bfloat16 | 中心化结果 |
| squared | [N, channels] | float32/float16/bfloat16 | 平方结果 |
| var | [1, channels] | float32/float16/bfloat16 | 方差 |
| std | [1, channels] | float32/float16/bfloat16 | 标准差 |
| normalized | [N, channels] | float32/float16/bfloat16 | 归一化结果 |
| scaled | [N, channels] | float32/float16/bfloat16 | 缩放结果 |
| output_2d | [N, channels] | float32/float16/bfloat16 | 2D 输出 |

### 3.4 动态轴定义

| 轴名称 | 含义 | 取值范围 |
|--------|------|----------|
| batch | 批次大小 | [1, INT32_MAX] |
| seq_len | 序列长度 | [1, INT32_MAX] |

### 3.5 JIT 装饰器配置

```python
@pypto.frontend.jit(
    runtime_options={"run_mode": global_run_mode}
)
def batchnorm_kernel(x, gamma, beta, output, config):
    ...
```

---

## 4. Tiling 策略

### 4.1 算子类型判断

- **类型**: Vector
- **判断依据**: 算子仅包含逐元素运算（sub, mul, div, add）和归约运算（sum），无矩阵乘法操作

### 4.2 TileShape 初值设置

```python
pypto.set_vec_tile_shapes(64, channels)
```

### 4.3 设置依据

- **tile_rows = 64**: 每次处理 64 行数据，平衡 L0 容量和并行度
- **channels**: 尾轴完整保留 channels 维度，确保 sum 归约精度
- **来源**: 参考 `layer_norm.py` 的 `pypto.set_vec_tile_shapes(64, 128)`

### 4.4 注意事项

- channels 必须完整处理，不能被切分（用于 sum 归约精度）
- 尾轴 channels 需满足 32B 对齐要求
- 若 channels 不满足对齐，可能需要 padding

### 4.5 判断依据与适用条件

- 判断依据: channels 维度完整保留以保证归约精度，行维度可切分
- 适用条件: channels * dtype_size 是 32 的倍数
- 不适用场景: channels 过大超过 L0 容量时需要调整

---

## 5. Loop 结构设计

### 场景 B: 需要 Loop

> 由于存在动态轴 batch 和 seq_len，需要使用 pypto.loop 处理动态维度

- **结论**: 需要 `pypto.loop`
- **原因**: batch 和 seq_len 为动态轴，运行时才知道大小，需要循环处理动态维度的数据
- **适用条件**: 动态 shape 场景
- **限制**: Loop 会增加编译复杂度，需要合理设置 loop body

### Loop 实现方案

```python
# 计算动态维度的循环次数
N = batch * seq_len * H * W
tile_rows = 64
n_loop = (N + tile_rows - 1) // tile_rows

for i in pypto.loop(n_loop, name="LOOP_BATCHNORM", idx_name="loop_idx"):
    # 处理每个 tile
    start_idx = loop_idx * tile_rows
    x_tile = pypto.view(x_2d, [tile_rows, channels], [start_idx, 0])
    # ... 核心计算 ...
    pypto.assemble(output_tile, [start_idx, 0], output_2d)
```

---

## 6. 验证方案

### 6.1 Golden 函数设计

Golden 函数 `batchnorm_golden()` 已在 `batchnorm_golden.py` 中实现:
- 支持 3D 输入 [batch, seq_len, channels] 和 5D 输入 [batch, seq_len, channels, H, W]
- 支持 training 和 inference 模式
- 使用 PyTorch 内置 mean/var 函数

### 6.2 测试用例

| 配置名称 | 类型 | 优先级 | 输入 Shape | 验证重点 |
|----------|------|--------|------------|----------|
| 功能_P0 | 功能 | P0 | [2, 32, 64, 28, 28] | ResNet 典型配置，基础精度 |
| 性能_P0 | 性能 | P0 | [8, 64, 256, 56, 56] | 大 batch 性能，高精度 |
| 动态shape_1 | 功能 | P1 | [1, 16, 128, 14, 14] | 小 batch 边界 |
| 动态shape_2 | 功能 | P1 | [16, 128, 512, 7, 7] | 大 batch 边界 |
| 1D_BN | 功能 | P1 | [2, 512, 768] | 3D 输入支持 |

### 6.3 精度验证标准

| Dtype | atol | rtol |
|-------|------|------|
| float32 | 0.001 | 0.001 |
| float16 | 0.01 | 0.01 |
| bfloat16 | 0.01 | 0.01 |

---

## 7. 性能指标与开箱配置

### 7.1 性能目标

基于 spec.md 典型配置（性能类）的预期性能:

| 配置名称 | 类型 | 优先级 | 输入 Shape | 预期 kernel 耗时 |
|----------|------|--------|------------|------------------|
| 性能_P0 | 性能 | P0 | [8, 64, 256, 56, 56] | 首跑精度成功性能的2倍 |

### 7.2 开箱性能配置

```python
# Tiling 配置
pypto.set_vec_tile_shapes(64, channels)

# 对于大 channels 场景
pypto.set_vec_tile_shapes(32, channels)
```

### 7.3 pass_options 配置

无特殊 pass_options 需求。

### 7.4 runtime_options 配置

```python
runtime_options={"run_mode": global_run_mode}
```

---

## 8. 风险点与注意事项

### 8.1 已知约束

- **pypto.sum 尾轴 32B 对齐**: 尾轴 channels * dtype_size 必须是 32 的倍数
- **pypto.var 多轴归约约束**: dim 轴不可切，使用 reshape 方案规避
- **Shape 仅支持 2-4 维**: 5D 输入需要先 reshape 到 2D

### 8.2 常见错误规避

| 风险 / 错误 | 触发场景 | 影响 / 原因 | 规避方法 |
|-------------|----------|-------------|----------|
| 尾轴不对齐 | channels 不是 16 的倍数 (FP16) | 编译失败或性能劣化 | 确保 channels 满足对齐，或添加 padding |
| reshape 失败 | N 计算溢出 | 运行时错误 | 确保 N = batch * seq_len * H * W 不超过 INT32_MAX |
| 精度问题 | 使用 FP16/BF16 直接计算 | 数值溢出 | 中间计算使用 FP32，最后转回原类型 |

### 8.3 特殊场景处理

- **3D 输入**: 直接使用 [batch, seq_len, channels] 格式，无需处理 H, W 维度
- **小 batch**: 可能导致统计量不稳定，但算法仍正常工作
- **推理模式**: 使用 running_mean 和 running_var 替代 batch 统计量

### 8.4 实现建议

| 建议项 | 说明 |
|--------|------|
| 中间精度使用 FP32 | 在 mean/var 计算中使用 FP32 保证数值稳定性 |
| 简化 3D 场景 | 3D 输入可直接处理，无需复杂的 reshape |
| 优先实现训练模式 | 训练模式是核心功能，先实现并验证 |

---

## 9. 交付件清单

### 9.1 目录结构

```
operators/batchnorm/
├── spec.md                          # 需求规范（已有）
├── api_report.md                    # API 探索报告（已有）
├── design.md                         # 设计文档（本文件）
├── batchnorm_golden.py               # Golden 参考实现（已有）
├── batchnorm_impl.py                 # 算子实现代码（待实现）
├── test_batchnorm.py                # 测试代码（待实现）
└── output/                           # 运行输出（自动生成）
```

### 9.2 文件清单

| 文件 | 类型 | 说明 | 生成方式 |
|------|------|------|----------|
| spec.md | 需求 | 算子需求规范 | pypto-intent-understanding |
| api_report.md | 探索 | API 映射与约束 | pypto-api-explorer |
| design.md | 设计 | 算子设计文档 | pypto-op-design（本 skill） |
| batchnorm_golden.py | 代码 | Golden 参考实现 | pypto-golden-generator |
| batchnorm_impl.py | 代码 | 算子核心实现 | 后续实现 |
| test_batchnorm.py | 代码 | 测试用例 | 后续实现 |

### 9.3 命名规范

| 项目 | 规范 | 示例 |
|------|------|------|
| 算子名称 | 小写字母 | `batchnorm` |
| 目录名 | 与算子名称一致 | `operators/batchnorm/` |
| Golden 文件 | `{op}_golden.py` | `batchnorm_golden.py` |
| 实现文件 | `{op}_impl.py` | `batchnorm_impl.py` |
| 测试文件 | `test_{op}.py` | `test_batchnorm.py` |

### 9.4 生成顺序

```
spec.md -> api_report.md -> design.md -> batchnorm_golden.py -> batchnorm_impl.py -> test_batchnorm.py
```

---

*设计文档生成完成*
