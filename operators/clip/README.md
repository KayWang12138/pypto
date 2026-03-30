# clip 算子

## 概述

clip 算子将输入张量的每个元素限制在 [min_val, max_val] 范围内。

### 数学公式

$$y = \min(\max(x, \text{min\_val}), \text{max\_val})$$

### 功能说明

- 如果 x < min_val，输出 min_val
- 如果 x > max_val，输出 max_val
- 否则输出 x

## 目录结构

```text
operators/clip/
├── spec.md           # 算子规格文档
├── api_report.md     # API 探索报告
├── design.md         # 设计方案文档
├── clip_golden.py    # Golden 参考实现 (PyTorch)
├── clip_impl.py      # PyPTO 实现
├── test_clip.py      # 测试文件
└── README.md         # 本文档
```

## 运行方式

### 环境准备

1. 设置 NPU 设备 ID:
```bash
export TILE_FWK_DEVICE_ID=0
```

2. 确保已安装 PyPTO 和相关依赖。

### 运行测试

```bash
# 运行所有测试
python test_clip.py

# 运行指定测试
python test_clip.py clip::test_clip_level0

# 查看可用测试
python test_clip.py --list

# 指定运行模式
python test_clip.py --run_mode npu
python test_clip.py --run_mode sim
```

## 验证入口

| 测试名称 | 说明 |
|----------|------|
| test_clip_level0 | 2D 基础功能验证 |
| test_clip_level1 | 3D 功能验证 |
| test_clip_level2 | 4D 性能场景验证 |
| test_clip_boundary | 边界值验证 |
| test_clip_dtype_fp16 | FP16 dtype 支持 |
| test_clip_large | 大规模性能验证 |

## API 参考

### clip_wrapper

```python
def clip_wrapper(
    x: torch.Tensor,
    min_val: float = None,
    max_val: float = None
) -> torch.Tensor:
```

**参数**:
- `x`: 输入张量，支持 2-4 维
- `min_val`: 最小值边界（标量），默认 None 表示不限制下界
- `max_val`: 最大值边界（标量），默认 None 表示不限制上界

**返回**:
- 输出张量，shape 与输入相同

## 支持的数据类型

| Dtype | 说明 |
|-------|------|
| float32 | 默认支持 |
| float16 | 支持 |
| bfloat16 | 支持 |

## 已知限制

1. **维度限制**: 仅支持 2-4 维 Tensor，1 维输入需先 reshape
2. **contiguous 要求**: 输入 Tensor 必须连续
3. **元素个数限制**: 不超过 UINT32_MAX
4. **min/max 类型**: 类型必须一致（同为标量或同为 Tensor）

## 性能说明

- 算子类型: Vector（逐元素操作）
- Tiling 配置: `set_vec_tile_shapes(64, 128)` (2D), `(8, 64, 128)` (3D), `(1, 8, 64, 128)` (4D)
- 无需显式 Loop，编译器自动处理

## 参考

- PyTorch API: `torch.clamp(input, min=None, max=None)`
- PyPTO API: `pypto.clip(input, min, max)`
