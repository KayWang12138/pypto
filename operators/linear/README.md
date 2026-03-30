# linear 算子

## 概述

线性层（全连接层）算子，对输入张量进行线性变换。

**数学公式**:
$$output = input \cdot weight^T + bias$$

## 功能描述

- 支持任意维度的输入张量（2D、3D、4D）
- 对最后一维进行线性变换，保持其他维度不变
- bias 为可选参数

## 目录结构

```
operators/linear/
├── spec.md              # 需求规范
├── api_report.md        # API 探索报告
├── design.md            # 设计文档
├── linear_golden.py     # PyTorch 参考实现
├── linear_impl.py       # PyPTO kernel 实现
├── test_linear.py       # 测试代码
└── README.md            # 本文件
```

## 运行方式

### 环境准备

```bash
# 设置 NPU 设备 ID
export TILE_FWK_DEVICE_ID=0

# 确保 PyPTO 已编译安装
```

### 运行测试

```bash
cd operators/linear

# 运行所有测试
python test_linear.py

# 运行单个测试
python test_linear.py linear::test_linear_level0

# 列出所有测试用例
python test_linear.py --list

# 使用 sim 模式
python test_linear.py --run_mode sim
```

## 测试用例

| Level | 描述 | 输入 Shape | 输出 Shape |
|-------|------|------------|------------|
| Level 0 | 小数据量基础验证 (2D with bias) | [4, 64] | [4, 128] |
| Level 1 | 典型场景验证 (2D with bias) | [16, 512] | [16, 512] |
| Level 2 | 3D 场景验证 | [4, 64, 256] | [4, 64, 512] |
| Level 3 | 4D 场景验证 | [2, 4, 32, 128] | [2, 4, 32, 256] |
| Level 4 | 无 bias 场景验证 | [8, 256] | [8, 512] |
| Level 5 | 边界测试 - 最小 batch | [1, 128] | [1, 64] |

## 精度要求

- **容差**: rtol=1e-3, atol=1e-3
- **数据类型**: float32

## 实现策略

1. **2D 场景**: 使用 `matmul` 的 `extend_params={'bias_tensor': bias}` 融合 bias，性能更优
2. **3D/4D 场景**: 使用 `matmul(b_trans=True) + add` 分开实现（因为 bias_tensor 仅支持 2D）

## API 映射

| 操作 | PyPTO API | 说明 |
|------|-----------|------|
| 矩阵乘法（带转置） | `pypto.matmul(input, weight, dtype, b_trans=True)` | weight 自动转置 |
| bias 融合（仅 2D） | `extend_params={'bias_tensor': bias}` | 减少一次 kernel 启动 |
| bias 加法 | `pypto.add(matmul_result, bias)` | 3D/4D 场景 |

## 已知限制

1. **bias_tensor 融合限制**: `extend_params` 的 `bias_tensor` 仅支持 2D matmul，3D+ 需分开用 add
2. **输入必须 contiguous**: 必须确保 `torch.Tensor.is_contiguous() == True`
3. **FP32 对齐要求**: `set_cube_tile_shapes` 的 kL0, kL1, nL0, nL1 需 16 元素对齐
4. **维度限制**: matmul 支持 2-4 维输入

## 动态轴支持

- batch: 动态轴，范围 [1, INT32_MAX]
- seq_len: 动态轴，范围 [1, INT32_MAX]

## Tiling 策略

根据矩阵大小动态选择 tiling 配置：

| 场景 | cube_tile_shapes |
|------|------------------|
| 小矩阵 (M,N,K <= 64) | [32,32], [64,64], [64,64] |
| 中等矩阵 (<= 2048) | [128,128], [128,128], [128,128] |
| 大矩阵 (> 2048) | [256,256], [256,256], [256,256] |

3D/4D 场景还需设置 vector tiling。

## 验证入口

```python
from linear_impl import linear_wrapper
from linear_golden import linear_golden

# 准备数据
input_tensor = torch.randn(4, 64, dtype=torch.float32)
weight = torch.randn(128, 64, dtype=torch.float32)
bias = torch.randn(128, dtype=torch.float32)

# 执行
output = linear_wrapper(input_tensor, weight, bias)
expected = linear_golden(input_tensor, weight, bias)

# 验证
assert torch.allclose(output, expected, rtol=1e-3, atol=1e-3)
```
