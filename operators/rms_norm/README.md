# rms_norm 算子实现

## 算子概述

均方根归一化（Root Mean Square Normalization）。对输入张量在最后一个维度上计算均方根（RMS），然后用 RMS 进行归一化，最后乘以可学习的缩放参数。

与 Layer Norm 相比，RMS Norm 不进行均值中心化，计算更简单，在 Transformer 架构（如 LLaMA、GPT-NeoX）中广泛使用。

## 数学公式

```
y = x * gamma / sqrt(mean(x^2) + eps)
```

## 目录结构

```
operators/rms_norm/
├── spec.md                 # 需求规范
├── api_report.md           # API 探索报告
├── design.md               # 设计文档
├── rms_norm_golden.py      # Golden 参考实现
├── rms_norm_impl.py        # 算子实现代码
├── test_rms_norm.py        # 测试代码
└── README.md               # 本文件
```

## 运行方式

### 环境准备

```bash
# 设置 device id
export TILE_FWK_DEVICE_ID=0
```

### 执行测试

```bash
# 运行所有测试
python3 test_rms_norm.py

# 运行单个测试
python3 test_rms_norm.py rms_norm::test_rms_norm_level0

# 列出所有测试
python3 test_rms_norm.py --list
```

### 运行模式

```bash
# NPU 模式（默认）
python3 test_rms_norm.py --run_mode npu

# 模拟器模式
python3 test_rms_norm.py --run_mode sim
```

## 验证入口

| 测试级别 | 描述 | 输入 Shape |
|---------|------|------------|
| Level 0 | 小数据量基础功能验证 | [2, 128, 64] |
| Level 1 | LLaMA-7B 配置小规模 | [2, 128, 4096] |
| Level 2 | 动态轴测试 | [4, 64, 256] |
| Level 3 | LLaMA-13B 配置小规模 | [2, 64, 5120] |

## 实现说明

本实现使用 PyPTO 内置 `pypto.rms_norm` API，代码简洁、性能优化。

### API 调用

```python
result = pypto.rms_norm(x, gamma, eps)
```

### Tiling 配置

```python
pypto.set_vec_tile_shapes(64, 128)
```

## 已知限制

1. 输入 Tensor 必须是连续的（is_contiguous() == True）
2. x 和 weight 的 dtype 必须一致
3. eps 值必须为正数
4. 当 hidden_size > 8192 时，需要调整 tiling 参数

## 精度要求

| Dtype | atol | rtol |
|-------|------|------|
| FP32 | 0.001 | 0.001 |
| BF16 | 0.01 | 0.01 |
