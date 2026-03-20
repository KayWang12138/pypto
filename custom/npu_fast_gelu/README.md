# npu_fast_gelu 算子

## 概述

快速高斯误差线性单元激活函数（Fast Gaussian Error Linear Units）。

### 数学公式

在 Atlas A3 训练系列产品上：
```
fast_gelu(x) = x / (1 + e^(-1.702x))
            = x * sigmoid(1.702 * x)
```

## 实现方式

本算子提供三种实现：

| 实现 | 描述 |
|------|------|
| torch_npu | 使用 `torch_npu.npu_fast_gelu` |
| golden | 使用 NumPy 实现 |
| pypto | 使用 PyPTO sigmoid 组合实现 |

## 精度对比结果

### Float32
| 对比项 | 最大误差 | 平均误差 | 状态 |
|--------|----------|----------|------|
| torch_npu vs golden | ~2.4e-7 | ~4.2e-9 | ✓ 通过 |
| pypto vs golden | ~2.4e-7 | ~1.0e-8 | ✓ 通过 |
| pypto vs torch_npu | ~1.2e-7 | ~8.8e-9 | ✓ 通过 |

### BFloat16
| 对比项 | 最大误差 | 平均误差 | 状态 |
|--------|----------|----------|------|
| torch_npu vs golden | ~7.7e-3 | ~5.6e-4 | ✓ 通过 |
| pypto vs golden | ~1.3e-2 | ~7.3e-4 | ✓ 通过 |
| pypto vs torch_npu | ~1.6e-2 | ~4.8e-4 | ✓ 通过 |

## 使用方法

```bash
# 设置环境变量
export TILE_FWK_DEVICE_ID=0

# 运行测试
python3 custom/npu_fast_gelu/npu_fast_gelu.py all
```

## 支持的数据类型

- torch.float32
- torch.bfloat16

## 已知限制

- BFloat16 精度较低，误差容限需放宽至 rtol=0.02, atol=0.02