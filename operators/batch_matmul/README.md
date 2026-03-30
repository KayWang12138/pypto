# batch_matmul 算子实现

## 概述

批量矩阵乘法算子，对 batch 维度的每个矩阵执行矩阵乘法。

### 数学公式

$$C[b,m,n] = \sum_{k} A[b,m,k] \times B[b,k,n]$$

### 功能特性

- 支持批量矩阵乘法
- 支持转置配置（transpose_x1, transpose_x2）
- 支持 batch 维度广播

## 目录结构

```
custom/batch_matmul/
├── spec.md                    # 需求规范
├── api_report.md              # API 探索报告
├── design.md                  # 设计文档
├── batch_matmul_golden.py     # Golden 参考实现
├── batch_matmul_impl.py       # 算子实现代码
├── test_batch_matmul.py       # 测试代码
└── README.md                  # 本文件
```

## 运行方式

### 环境准备

```bash
# 设置 NPU 设备 ID
export TILE_FWK_DEVICE_ID=0
```

### 执行测试

```bash
# 运行所有测试
python custom/batch_matmul/test_batch_matmul.py

# 运行特定测试
python custom/batch_matmul/test_batch_matmul.py batch_matmul::test_batch_matmul_level0

# 查看可用测试
python custom/batch_matmul/test_batch_matmul.py --list
```

### 使用示例

```python
import torch
from batch_matmul_impl import batch_matmul_wrapper

# 创建输入数据
x1 = torch.randn(4, 128, 256)  # [batch, m, k]
x2 = torch.randn(4, 256, 512)  # [batch, k, n]

# 执行批量矩阵乘法
result = batch_matmul_wrapper(x1, x2)
# result shape: [4, 128, 512]

# 使用转置功能
result_t = batch_matmul_wrapper(x1, x2.transpose(-1, -2), transpose_x2=True)
```

## 验证入口

| 测试级别 | 描述 | 命令 |
|---------|------|------|
| Level 0 | 小数据量基础功能验证 | `python test_batch_matmul.py batch_matmul::test_batch_matmul_level0` |
| Level 1 | 典型场景验证 | `python test_batch_matmul.py batch_matmul::test_batch_matmul_level1` |
| Level 2 | 转置功能验证 | `python test_batch_matmul.py batch_matmul::test_batch_matmul_transpose` |
| Level 3 | 广播场景验证 | `python test_batch_matmul.py batch_matmul::test_batch_matmul_broadcast` |
| Level 4 | 大规模性能验证 | `python test_batch_matmul.py batch_matmul::test_batch_matmul_large` |

## 精度要求

- float32: rtol=1e-5, atol=1e-5
- 与 torch.bmm 对比

## 已知限制

1. **输入连续性**: 输入 tensor 必须是连续的（contiguous）
2. **数据类型**: 当前仅支持 float32
3. **维度要求**: 输入必须是 3D tensor [batch, m, k] 和 [batch, k, n]
4. **K 轴匹配**: x1 的最后一维必须等于 x2 的倒数第二维

## API 参考

### batch_matmul_wrapper

```python
def batch_matmul_wrapper(
    x1: torch.Tensor,
    x2: torch.Tensor,
    transpose_x1: bool = False,
    transpose_x2: bool = False
) -> torch.Tensor
```

**参数**:
- `x1`: 左矩阵，shape [batch, m, k]
- `x2`: 右矩阵，shape [batch, k, n]
- `transpose_x1`: 是否对 x1 转置
- `transpose_x2`: 是否对 x2 转置

**返回**:
- 输出矩阵，shape [batch, m, n]

## 性能指标

| 配置 | 输入 Shape | 预期功能 |
|------|------------|----------|
| 小规模 | [2, 4, 8] | 功能验证 |
| 中规模 | [4, 1024, 1024] | 精度验证 |
| 大规模 | [2, 4096, 4096] | 性能验证 |
