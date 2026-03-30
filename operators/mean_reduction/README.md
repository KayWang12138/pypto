# mean_reduction 算子

## 概述

沿指定轴计算张量均值的归约操作，类似于 PyTorch 的 `torch.mean()`。支持动态轴（batch、seq 等维度）和 keepdim 参数。

## 数学公式

$$y = \text{mean}(x, \text{dim}) = \frac{\sum_{i} x_i}{N}$$

其中 N 为 dim 轴上的元素数量。

## 目录结构

```
operators/mean_reduction/
├── spec.md                      # 需求规范
├── api_report.md                # API 探索报告
├── design.md                    # 设计文档
├── mean_reduction_golden.py     # Golden 参考实现
├── mean_reduction_impl.py       # 算子核心实现
├── test_mean_reduction.py       # 测试用例
└── README.md                    # 本文件
```

## 使用方式

### 运行测试

```bash
# 设置设备 ID
export TILE_FWK_DEVICE_ID=0

# 运行所有测试
python operators/mean_reduction/test_mean_reduction.py

# 运行特定测试
python operators/mean_reduction/test_mean_reduction.py mean_reduction::test_mean_reduction_level0

# 查看可用测试
python operators/mean_reduction/test_mean_reduction.py --list
```

### 调用接口

```python
from mean_reduction_impl import mean_reduction_wrapper

# 计算沿 dim 轴的均值
x = torch.randn(2, 512, 4096, dtype=torch.float32, device='npu:0')
result = mean_reduction_wrapper(x, dim=1, keepdim=False)
# result.shape = [2, 4096]
```

## 实现说明

### API 映射

PyPTO 无直接的 `mean` API，使用 substitute 方案实现：

1. `pypto.sum(input, dim, keepdim)` - 沿 dim 轴求和
2. `pypto.div(sum_result, N)` - 除以归约轴元素数量

### Tiling 策略

- **算子类型**: Vector
- **TileShape**: [8, 8, 8] (3D 输入)
- **尾轴对齐**: 8 元素 (float32 32B 对齐)
- **次尾轴限制**: 8 < 255

### 动态轴支持

- 支持动态轴: b (batch), s (seq_len)
- 通过 `x.shape[dim]` 获取归约轴元素数量（支持 SymbolicScalar）

## 测试覆盖

| 配置名称 | 类型 | 参数 | 输入 Shape | 说明 |
|----------|------|------|------------|------|
| Level 0 | 功能 | dim=-1, keepdim=False | [2, 4] | 基础功能验证 |
| Level 1 | 性能 | dim=1, keepdim=False | [2, 512, 4096] | 核心性能场景 |
| Level 2 | 功能 | dim=1, keepdim=True | [2, 512, 4096] | keepdim 验证 |
| Dynamic | 功能 | 动态 shape | [b, s, d] | 动态轴测试 |

## 精度标准

| Dtype | atol | rtol |
|-------|------|------|
| float32 | 0.001 | 0.001 |
| float16 | 0.01 | 0.01 |
| bfloat16 | 0.01 | 0.01 |

## 已知限制

1. 输入 shape 限制为 2-4 维
2. 仅支持单轴归约，不支持多轴同时归约
3. 尾轴需 32 bytes 对齐
4. 次尾轴 ≤ 255
5. 输入张量必须 contiguous

---
*生成时间: 2026-03-28*
