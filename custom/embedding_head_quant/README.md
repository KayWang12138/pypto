# Embedding Head Quantization Operator

## 算子概述

### 功能
对 embedding 权重进行量化操作，支持训练时的 STE (Straight-Through Estimator) 梯度传递。该算子实现了权重的对称量化，将浮点权重量化到整数范围后再缩放回浮点表示。

### 数学公式
```
# Scale 保护（避免除零）
scale = max(scale, eps)

# 量化过程
weight = weight / scale           # 归一化
weight = round(weight)            # 四舍五入到整数
weight = clip(weight, min_v, max_v)  # 限制在量化范围内
weight = weight * scale           # 重新缩放
```

### 参数说明
| 参数 | 类型 | 说明 | 默认值 |
|------|------|------|---------|
| **weight** | Tensor | 输入权重张量，数据类型 FP32 | - |
| **scale** | Tensor | 量化缩放因子张量，数据类型 FP32 | - |
| **eps** | float | 最小缩放因子阈值，用于保护 scale 不为 0 | 1e-4 |
| **min_v** | float | 量化下限（对应 int8 范围） | -128.0 |
| **max_v** | float | 量化上限（对应 int8 范围） | 127.0 |

### 输出说明
输出张量与输入 weight 具有相同的 shape 和数据类型（FP32），包含量化后的权重值。

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

### STE (Straight-Through Estimator) 处理
PyPTO 是前向计算内核框架，不包含自动微分功能。原始 PyTorch 实现中的 STE 模式：
```python
weight = (weight.round() - weight).detach() + weight
```
该模式的作用是：
- **前向传播**：使用量化后的整数值
- **反向传播**：直接传递原始权重的梯度（跳过量化操作）

在 PyPTO 实现中，由于只关注前向计算，因此简化为直接使用四舍五入后的值：
```python
weight = pypto.round(weight, decimals=0)
```
如果需要训练时的梯度传递，STE 的梯度处理需要在更高层框架（如 PyTorch）中实现。

### PyPTO API 映射

| 操作 | PyTorch | PyPTO API | 说明 |
|------|----------|------------|------|
| Scale 保护 | `torch.where(scale > eps, scale, eps)` | `pypto.maximum(scale, eps)` | 使用 maximum 替代 where，更简洁 |
| 除法 | `weight / scale` | `pypto.div(weight, scale)` | 逐元素除法 |
| 四舍五入 | `weight.round()` | `pypto.round(weight, decimals=0)` | 银行家舍入法 |
| 截断 | `torch.clamp(weight, min_v, max_v)` | `pypto.clip(weight, min_v, max_v)` | 限制在范围内 |
| 乘法 | `weight * scale` | `pypto.mul(weight, scale)` | 逐元素乘法 |

### TileShape 设置
```python
pypto.set_vec_tile_shapes(32, 32)
```
设置向量化计算的 Tile 形状，用于优化 NPU 上的并行计算。

## 已知限制

1. **STE 梯度处理**：PyPTO 算子仅负责前向计算，不包含自动微分功能。STE 的梯度传递需要在更高层框架（如 PyTorch）中实现。

2. **数据类型支持**：当前仅支持 FP32 数据类型。FP16/BF16 支持待后续版本添加。

3. **维度限制**：当前实现支持 2D 张量输入。1D 和 3D/4D 支持待扩展。

4. **量化范围**：默认使用 int8 范围 [-128, 127]。如需其他范围（如 uint8 [0, 255]），需要修改参数。

## 常见问题

### Q: 如何处理训练时的梯度传递？
**A**: STE 的梯度传递需要在 PyTorch 层实现。PyPTO 算子仅负责前向计算。在 PyTorch 中实现自定义 autograd 函数即可。

### Q: 支持哪些数据类型？
**A**: 当前仅支持 FP32。FP16/BF16 支持待后续版本添加。

### Q: 如何调整量化范围？
**A**: 修改 `embedding_head_quant.py` 中的 `min_v` 和 `max_v` 参数。例如，使用 uint8 范围：
```python
min_v=0.0, max_v=255.0
```

### Q: 如何支持其他维度的张量？
**A**: 当前实现使用固定的 TileShape 设置 `pypto.set_vec_tile_shapes(32, 32)`。支持其他维度需要：
1. 修改 `create_embedding_head_quant_kernel` 函数中的 TileShape 设置
2. 确保输入 shape 与 TileShape 兼容

### Q: NPU 模式下报错 "Invalid Device" 怎么办？
**A**: 检查并设置正确的 NPU 设备 ID：
```bash
# 查看可用的 NPU 设备
npu-smi info

# 设置正确的设备 ID
export TILE_FWK_DEVICE_ID=0  # 或其他可用的设备号
```

### Q: 如何在仿真模式下调试？
**A**: 使用 `--run_mode sim` 参数在 CPU 上运行，无需 NPU 硬件：
```bash
python3 embedding_head_quant.py --run_mode sim --test_level 0
```

## 性能优化建议

1. **TileShape 调优**：根据实际输入 shape 调整 `set_vec_tile_shapes` 的参数以获得最佳 NPU 性能。

2. **批处理**：对于大规模 embedding，建议将多个 embedding 的量化操作合并为一个批处理操作。

3. **内存对齐**：确保输入张量的内存对齐（32 字节对齐）以获得最佳性能。

## 参考实现

- **原始 PyTorch 实现**：`/workspace/sher/pypto/qat.py` 中的 `Embedding_Head_Quant_new` 函数
- **PyPTO API 文档**：`/workspace/sher/pypto/docs/api/operation/` 目录
- **类似算子示例**：
  - `models/deepseek_v32_exp/sparse_attention_antiquant_impl.py` - DeepSeek V32 量化实现
  - `models/glm_v4_5/glm_attention_pre_quant.py` - GLM V4.5 量化实现

## 版本历史

- **v1.0** (2026-03-03): 初始版本
  - 支持 FP32 数据类型
  - 支持 2D 张量
  - 实现基础量化功能
  - 提供完整测试用例
