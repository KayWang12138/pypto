# concat 算子实现

## 概述

concat 算子将多个张量沿指定维度拼接成一个张量。所有输入张量在非拼接维度上的 shape 必须相同，dtype 必须一致。

## 数学公式

沿 dim 维度拼接多个张量，保持其他维度不变：

```
output[i_0, ..., i_dim, ..., i_{n-1}] = tensor_k[i_0, ..., i'_dim, ..., i_{n-1}]
```

其中 k 是第 i_dim 所属的输入张量索引，i'_dim 是在该张量内的偏移。

输出 shape 计算：
```
output.shape[dim] = sum(tensor_k.shape[dim] for all k)
output.shape[other] = tensors[0].shape[other]
```

## 目录结构

```
operators/concat/
├── spec.md              # 需求规范
├── api_report.md        # API 探索报告
├── design.md            # 设计文档
├── concat_golden.py     # Golden 参考实现
├── concat_impl.py       # 算子实现代码
├── test_concat.py       # 测试代码
└── README.md            # 本文件
```

## 运行方式

### 环境准备

```bash
# 设置 NPU 设备 ID
export TILE_FWK_DEVICE_ID=0

# 确保已安装 pypto
python3 build_ci.py -f python3 --disable_auto_execute
```

### 运行测试

```bash
# 运行所有测试
cd operators/concat
python3 test_concat.py

# 运行单个测试用例
python3 test_concat.py concat::test_concat_level0

# 列出所有测试用例
python3 test_concat.py --list

# 使用 sim 模式
python3 test_concat.py --run_mode sim
```

## 验证入口

| 用例 ID | 描述 | 输入 Shape | dim |
|---------|------|------------|-----|
| test_concat_level0 | 2D tensor 基础验证 | [2,4], [2,4] | 1 |
| test_concat_level1 | 3D tensor 典型验证 | [2,64,256], [2,64,128] | -1 |
| test_concat_level2 | 3个张量拼接 | [4,64,128] * 3 | 0 |
| test_concat_level3 | 负数索引 | [4,8,16], [4,8,32] | -1 |
| test_concat_level4 | float16 dtype | [4,8,16], [4,8,32] | -1 |
| test_concat_level5 | 大规模张量 | [4096,1024], [4096,1024] | -1 |
| test_concat_level6 | 4D tensor | [2,4,8,16], [2,4,8,32] | -1 |

## 精度要求

| Dtype | atol | rtol |
|-------|------|------|
| float32 | 0.001 | 0.001 |
| float16 | 0.01 | 0.01 |
| bfloat16 | 0.01 | 0.01 |

## API 使用

```python
from concat_impl import concat_wrapper
from concat_golden import concat_golden
import torch

# 创建输入张量
a = torch.randn(2, 64, 256)
b = torch.randn(2, 64, 128)

# PyPTO 实现
result = concat_wrapper([a, b], dim=-1)

# Golden 参考
golden = concat_golden([a, b], dim=-1)

# 验证
assert result.shape == golden.shape
```

## 已知限制

1. **张量数量限制**: 2 <= len(tensors) <= 128
2. **维度限制**: 仅支持 2-4 维张量
3. **Shape Size 限制**: <= INT32_MAX (2147483647)
4. **非拼接维度约束**: 所有张量在非 dim 维度的 shape 必须相同
5. **dtype 约束**: 所有张量 dtype 必须相同
6. **空 Tensor**: 不支持空 Tensor（shape[dim] == 0）
7. **单张量输入**: 精度暂时不保证（建议至少 2 个张量）

## 实现说明

- 使用 `pypto.concat([tensors], dim=dim)` API 完成张量拼接
- 设置 `pypto.set_vec_tile_shapes(...)` 配置 Tiling
- 输出写回使用 `output[:] = ...` 方式
- 支持动态 batch 和 seq_len 轴
- 对于超过 2-3 个张量的情况，采用分批拼接策略
