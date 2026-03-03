# QAT Asymmetric Quantization Operator

## 算子概述

实现了 Enhanced LSQ+ (Learned Step Size Quantization Plus) 非对称量化算子，用于量化感知训练(QAT)。支持分组量化 (Group-wise Quantization) 和可学习的 scale/offset 参数。

### 数学公式

```
n_levels = 2^(bit-1)
shift = 0.5
weight = weight - offset
alpha = scale * n_levels
weight = clamp(weight / alpha, -clip_val, clip_val) * n_levels - shift
weight = round(weight)  # STE forward
weight = (weight + shift) / n_levels
weight = weight * alpha + offset
```

### 参数说明

| 参数 | 类型 | 说明 | 默认值 |
|------|------|------|---------|
| **weight** | Tensor | 输入权重张量 (FP32) | - |
| **scale** | Tensor | 量化缩放因子 (per-group) | - |
| **offset** | Tensor | 量化偏移量 (per-group) | - |
| **group_size** | int | 每组元素数量 | - |
| **bit** | int | 量化位宽 (4, 8 等) | 4 |
| **eps** | float | scale 最小值保护 | 1e-4 |
| **clip_val** | float | 归一化裁剪值 | 0.99 |

## 编译运行指南

### 环境准备

```bash
# 设置 NPU 设备 ID
export TILE_FWK_DEVICE_ID=0

# 设置 PTO-ISA 源码路径
export PTO_TILE_LIB_CODE_PATH=/mnt/workspace/pto_isa/pto-isa

# 设置 CANN 环境
source /usr/local/Ascend/ascend-toolkit/setenv.sh
```

### 编译安装

```bash
# 在项目根目录执行
cd /mnt/workspace/pypto
python3 build_ci.py -f python3 --disable_auto_execute
```

### 运行测试

```bash
cd custom/qat_asymmetric

# 运行所有测试
python3 qat_asymmetric.py --run_mode npu

# 运行特定级别测试
python3 qat_asymmetric.py --run_mode npu --test_level 0  # 基础功能
python3 qat_asymmetric.py --run_mode npu --test_level 1  # 典型规模
python3 qat_asymmetric.py --run_mode npu --test_level 2  # 边界条件
python3 qat_asymmetric.py --run_mode npu --test_level 3  # 大规模性能
python3 qat_asymmetric.py --run_mode npu --test_level 4  # 多形状验证
```

## 测试结果

### 测试用例

| 级别 | 描述 | 输入 Shape | 结果 |
|------|------|-----------|------|
| Level 0 | 基础功能 | (16, 16) | PASSED |
| Level 1 | 典型规模 | (128, 64) | PASSED |
| Level 2 | 边界条件 | (8, 8) | PASSED |
| Level 3 | 大规模 | (1024, 1024) | PASSED |
| Level 4 | 多形状 | 多种 | PASSED |

### 精度验证

所有测试用例最大误差: **0.000000** (与 PyTorch golden 完全一致)

## 实现要点

### PyPTO 广播处理

PyPTO 二元操作不自动广播，需要显式使用 `expand_clone`:

```python
offset_2d = pypto.reshape(offset, [num_groups, 1])
offset_expanded = pypto.expand_clone(offset_2d, [num_groups, group_size])
```

### 分组量化

通过 reshape 实现 group-wise 量化:
- 输入: `(total_elements,)`
- 分组: `(num_groups, group_size)`
- 输出: `(total_elements,)`

## 使用示例

```python
import torch
import pypto

# 准备数据
weight = torch.randn(128, 64)  # 8192 elements
group_size = 128
num_groups = 64
scale = torch.rand(num_groups) * 0.1 + 0.01
offset = torch.randn(num_groups) * 0.1

# 创建内核
kernel = create_qat_asymmetric_kernel(
    weight_shape=(128, 64),
    num_groups=64,
    group_size=128,
    bit=4
)

# 执行量化
output = kernel(weight.view(-1), scale, offset).view(128, 64)
```

## 版本历史

- **v1.0** (2026-03-03): 初始版本
  - 支持 4-bit 和 8-bit 量化
  - 支持分组量化
  - 完整测试用例通过