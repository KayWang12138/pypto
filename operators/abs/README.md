# abs 算子

逐元素计算输入张量的绝对值。

## 数学公式

```
y = |x|
```

## 功能说明

- **算子分类**: element-wise
- **输入**: 任意 shape 的 tensor，支持 1-4 维
- **输出**: 与输入 shape 相同，每个元素为输入对应元素的绝对值
- **动态轴**: 支持所有维度为动态

## 目录结构

```
operators/abs/
├── spec.md           # 需求规格文档
├── api_report.md     # API 探索报告
├── design.md         # 设计方案
├── abs_golden.py     # PyTorch golden 参考实现
├── abs_impl.py       # PyPTO 实现代码
├── test_abs.py       # 测试脚本
└── README.md         # 本文档
```

## 运行方式

### 环境准备

```bash
# 设置 NPU 设备 ID
export TILE_FWK_DEVICE_ID=0
```

### 运行所有测试

```bash
cd operators/abs
python test_abs.py
```

### 运行单个测试

```bash
python test_abs.py abs::test_abs_level0
```

### 列出所有测试用例

```bash
python test_abs.py --list
```

## 测试用例

| 用例 ID | 描述 | 输入 Shape |
|---------|------|------------|
| abs::test_abs_level0 | 2D 基础功能验证 | [128, 1024] |
| abs::test_abs_level1 | 3D 典型场景验证 | [4, 64, 128] |
| abs::test_abs_level2 | 4D 性能场景验证 | [2, 4, 64, 128] |
| abs::test_abs_1d | 1D 输入验证 | [4096] |
| abs::test_abs_dtype_fp16 | FP16 dtype 支持 | [32, 64] |
| abs::test_abs_large | 大规模性能验证 | [4096, 4096] |

## 精度要求

- **atol**: 0.001
- **rtol**: 0.001

## 已知限制

1. **维度限制**: API 仅支持 2-4 维，其他维度会自动 reshape
2. **contiguous 要求**: 输入 tensor 必须连续，非连续输入会自动转换
3. **dtype**: 默认 FP32，支持 FP16 自动转换

## 参考文档

- PyPTO API: `docs/api/operation/pypto-abs.md`
- PyTorch 参考: `torch.abs()`
