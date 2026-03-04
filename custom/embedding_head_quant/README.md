# Embedding Head Quantization Operator

## 算子概述

### 功能
对 embedding 权重进行量化操作，支持训练时的 STE (Straight-Through Estimator) 梯度传递。该算子实现了权重的对称量化，将浮点权重量化到整数范围后再缩放回浮点表示。

### 数学公式
```
# BF16 输入转 FP32 进行计算
weight_fp32 = cast(weight_bf16, FP32)
scale_fp32 = cast(scale_bf16, FP32)

# Scale 保护（避免除零）
protected_scale = max(scale_fp32, eps)

# 量化过程（FP32 精度）
normalized = weight_fp32 / protected_scale
rounded = round(normalized)
clamped = clip(rounded, min_v, max_v)
output_fp32 = clamped * protected_scale

# 转回 BF16 输出
output_bf16 = cast(output_fp32, BF16)
clamped_bf16 = cast(clamped, BF16)
protected_scale_bf16 = cast(protected_scale, BF16)
```

### 参数说明
| 参数 | 类型 | 说明 | 默认值 |
|------|------|------|---------|
| **weight** | Tensor[BF16] | 输入权重张量 | - |
| **scale** | Tensor[BF16] | 量化缩放因子张量 | - |
| **eps** | float | 最小缩放因子阈值，用于保护 scale 不为 0 | 1e-4 |
| **min_v** | float | 量化下限（对应 int8 范围） | -128.0 |
| **max_v** | float | 量化上限（对应 int8 范围） | 127.0 |

### 输出说明
返回三元组 `(output, clamped, protected_scale)`：

| 输出 | 类型 | 说明 |
|------|------|------|
| **output** | Tensor[BF16] | 量化后的权重，与输入 weight 相同 shape |
| **clamped** | Tensor[BF16] | 截断后的量化值，用于反向传播 |
| **protected_scale** | Tensor[BF16] | 保护后的缩放因子，用于反向传播 |

## 编译运行指南

### 环境准备
```bash
# 设置 Ascend 环境变量
source /usr/local/Ascend/ascend-toolkit/latest/bin/setenvsetenv.bash

# 设置 NPU 设备 ID（必需）
export TILE_FWK_DEVICE_ID=0

# 可选：设置 PTO 库路径
export PTO_TILE_LIB_CODE_PATH=/mnt/workspace/pto_isa/pto-isa/
```

### 编译安装
```bash
# 在项目根目录执行
cd /workspace/sher/pypto

# 编译 whl 包并安装
python3 build_ci.py -f python3 --disable_auto_execute
```

### 运行测试

#### NPU 模式（需要真实硬件）
```bash
cd custom/embedding_head_quant

# 运行所有测试用例
python3 embedding_head_quant.py

# 运行特定级别的测试
python3 embedding_head_quant.py --test_level 0  # 基础功能测试
python3 embedding_head_quant.py --test_level 1  # 典型规模测试
python3 embedding_head_quant.py --test_level 2  # 边界条件测试
python3 embedding_head_quant.py --test_level 3  # 性能测试
```

#### 仿真模式（CPU，无需 NPU 硬件）
```bash
cd custom/embedding_head_quant

# 运行所有测试用例
python3 embedding_head_quant.py --run_mode sim

# 运行特定级别的测试
python3 embedding_head_quant.py --run_mode sim --test_level 0
```

## 测试结果

### 测试用例设计

| 测试级别 | 描述 | 输入 Shape | 测试目的 |
|-----------|------|-------------|----------|
| Level 0 | 基础功能测试 | (8, 8) | 验证核心逻辑正确性 |
| Level 1 | 典型规模测试 | (32, 32) | 验证实际使用场景（~1K 元素） |
| Level 2 | 边界条件测试 | (16, 16) | 验证小缩放因子、大值、零值等边界情况 |
| Level 3 | 性能测试 | (256, 256) | 验证大规模张量的 NPU 性能 |

### 边界条件测试详情

1. **极小缩放因子测试**：当 scale < eps 时，应自动使用 eps 替代，避免除零错误
2. **大值截断测试**：当量化后的值超过 [min_v, max_v] 范围时，应正确截断
3. **零值测试**：输入权重全为 0 时，输出应正确处理
4. **均匀缩放因子测试**：所有元素使用相同的 scale 值，验证广播机制

### 精度验证标准
- **相对误差容忍度**: 3e-3 (0.3%)
- **绝对误差容忍度**: 3e-3
- **验证方法**: 使用 NumPy 的 `assert_allclose` 与 PyTorch golden 函数对比

### 测试通过标准
所有测试用例必须满足以下条件：
- 与 PyTorch golden 函数的输出误差在容忍度范围内
- 无编译错误和运行时错误
- NPU 模式下无硬件报错

## 实现细节

### BF16 I/O + FP32 计算模式

为确保量化精度，算子采用 **BF16 输入/输出 + FP32 内部计算** 模式：

```python
# 1. BF16 → FP32 类型转换
weight_fp32 = pypto.cast(weight, pypto.DT_FP32)
scale_fp32 = pypto.cast(scale, pypto.DT_FP32)

# 2. FP32 精度量化计算
protected_scale = pypto.maximum(scale_fp32, eps)
normalized = pypto.div(weight_fp32, protected_scale)
rounded = pypto.round(normalized, decimals=0)
clamped = pypto.clip(rounded, min_v, max_v)
output = pypto.mul(clamped, protected_scale)

# 3. FP32 → BF16 类型转换
output_bf16 = pypto.cast(output, pypto.DT_BF16)
clamped_bf16 = pypto.cast(clamped, pypto.DT_BF16)
protected_scale_bf16 = pypto.cast(protected_scale, pypto.DT_BF16)
```

### STE (Straight-Through Estimator) 处理

PyPTO 是前向计算内核框架，不包含自动微分功能。输出 `clamped` 和 `protected_scale` 用于反向传播计算。

### PyPTO API 映射

| 操作 | PyTorch | PyPTO API | 说明 |
|------|----------|------------|------|
| 类型转换 | `.float()` / `.bfloat16()` | `pypto.cast(x, dtype)` | BF16 ↔ FP32 |
| Scale 保护 | `torch.where(scale > eps, scale, eps)` | `pypto.maximum(scale, eps)` | 使用 maximum 更简洁 |
| 除法 | `weight / scale` | `pypto.div(weight, scale)` | 逐元素除法 |
| 四舍五入 | `weight.round()` | `pypto.round(weight, decimals=0)` | 银行家舍入法 |
| 截断 | `torch.clamp(weight, min_v, max_v)` | `pypto.clip(weight, min_v, max_v)` | 限制在范围内 |
| 乘法 | `weight * scale` | `pypto.mul(weight, scale)` | 逐元素乘法 |

### TileShape 设置
```python
pypto.set_vec_tile_shapes(64, 64)
```
优化后的向量化计算 Tile 形状（性能调优结果）。

## 已知限制

1. **STE 梯度处理**：PyPTO 算子仅负责前向计算，不包含自动微分功能。STE 的梯度传递需要在更高层框架（如 PyTorch）中实现。

2. **数据类型**：输入/输出为 BF16，内部计算为 FP32。

3. **维度限制**：当前实现支持 2D 张量输入。

4. **量化范围**：默认使用 int8 范围 [-128, 127]。如需其他范围，需修改参数。

## 常见问题

### Q: 为什么使用 BF16 I/O + FP32 计算？
**A**: BF16 节省内存带宽和存储空间，FP32 保证量化计算精度。这是训练场景的常见实践。

### Q: 如何处理训练时的梯度传递？
**A**: 使用返回的 `clamped` 和 `protected_scale` 在 PyTorch 中实现自定义 autograd 函数。

### Q: 支持哪些数据类型？
**A**: 输入/输出为 BF16，不支持其他数据类型。

### Q: 如何调整量化范围？
**A**: 修改 `create_embedding_head_quant_kernel` 调用时的 `min_v` 和 `max_v` 参数：
```python
kernel = create_embedding_head_quant_kernel(shape, min_v=0.0, max_v=255.0)  # uint8 范围
```

### Q: NPU 模式下报错 "Invalid Device" 怎么办？
**A**: 检查并设置正确的 NPU 设备 ID：
```bash
npu-smi info
export TILE_FWK_DEVICE_ID=0
```

### Q: 如何在仿真模式下调试？
**A**: 使用 `--run_mode sim` 参数在 CPU 上运行：
```bash
python3 embedding_head_quant.py --run_mode sim --test_level 0
```

## 性能数据

### 优化后性能 (BF16 I/O)

| Shape | 元素数 | 延迟 (ms) | 吞吐量 (M/s) | 误差 |
|-------|--------|-----------|-------------|------|
| (8, 8) | 64 | 0.62 | 0.10 | 0.000000 |
| (32, 32) | 1,024 | 0.68 | 1.51 | 0.000000 |
| (64, 64) | 4,096 | 0.63 | 6.54 | 0.000000 |
| (128, 128) | 16,384 | 0.62 | 26.55 | 0.000000 |
| (256, 256) | 65,536 | 0.63 | 103.46 | 0.000000 |
| (512, 512) | 262,144 | 0.63 | 418.10 | 0.000000 |

- **最佳吞吐量**: 418.10 M elements/sec @ (512, 512)
- **扩展性**: 4,028x (从 64 到 262,144 元素)

## 参考实现

- **PyPTO API 文档**: `docs/api/operation/` 目录
- **类似算子示例**:
  - `models/deepseek_v32_exp/sparse_attention_antiquant_impl.py`
  - `models/glm_v4_5/glm_attention_pre_quant.py`

## 版本历史

- **v2.0** (2026-03-03): BF16 I/O + 多输出
  - 输入/输出改为 BF16，内部计算保持 FP32 精度
  - 返回三元组 `(output, clamped, protected_scale)` 用于反向传播
  - 性能调优: vec_tile=(64, 64)
  - 添加 stitch 参数优化

- **v1.0** (2026-03-03): 初始版本
  - 支持 FP32 数据类型
  - 支持 2D 张量
  - 实现基础量化功能
  - 提供完整测试用例
