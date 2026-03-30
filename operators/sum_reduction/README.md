# sum_reduction

## 概述

沿指定轴计算张量元素求和的归约操作，类似于 PyTorch 的 `torch.sum()` 或 `tensor.sum()`。 支持动态轴（batch、seq 等维度)和 keepdim 参数。

## 数学公式

```
y = sum(x, dim) = Σᵢ xᵢ
```

## 目录结构

```
operators/sum_reduction/
├── spec.md                      # 需求规范
├── design.md                   # 设计文档
├── sum_reduction_golden.py     # Golden 参考实现
├── sum_reduction_impl.py       # 算子实现代码
├── test_sum_reduction.py       # 测试代码
└── README.md                   # 本文件
```

## 运行方式

### 环境要求

- 设置 NPU 设备 ID：
```bash
export TILE_FWK_DEVICE_ID=0
```

### 运行测试

```bash
# 运行所有测试
python3 operators/sum_reduction/test_sum_reduction.py

# 运行单个测试
python3 operators/sum_reduction/test_sum_reduction.py sum_reduction::test_sum_reduction_level0

# 列出所有测试用例
python3 operators/sum_reduction/test_sum_reduction.py --list
```

## 验证入口

| 测试用例 | 描述 |
|--------|------|
| `sum_reduction::test_sum_reduction_level0` | 小数据量基础功能验证 |
| `sum_reduction::test_sum_reduction_level1` | 典型场景验证 (4K 元素) |
| `sum_reduction::test_sum_reduction_level2` | 边界和动态 shape 测试 |
| `sum_reduction::test_sum_reduction_dynamic` | 动态 shape 测试 (多种 shape) |

## 已知限制

1. **输入必须是连续的**: 如果输入不连续, 会自动调用 `.contiguous()`
2. **只支持单轴归约**: 多轴归约需要多次调用或扩展设计
3. **维度限制**: 输入张量支持 2-4 维
4. **动态轴**: b (batch) 和 s (seq) 维度支持动态 shape

