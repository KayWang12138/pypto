# SiLU 算子实现

## 概述

SiLU (Sigmoid Linear Unit) 激活函数，也称为 Swish。该算子在 LLM 中广泛使用（如 LLaMA、Mistral）。

### 数学公式

$$y = x \cdot \sigma(x) = \frac{x}{1 + e^{-x}}$$

其中 $\sigma(x) = \frac{1}{1 + e^{-x}}$ 为 sigmoid 函数。

### 特性

- **算子类型**: element-wise 激活函数
- **支持数据类型**: float16, float32, bfloat16
- **输入**: 任意 shape 的 tensor
- **输出**: 与输入相同 shape 和 dtype

## 目录结构

```
operators/silu/
├── spec.md                # 需求规范
├── api_report.md          # API 探索报告
├── design.md              # 设计文档
├── silu_golden.py         # Golden 参考实现
├── silu_impl.py           # 算子实现代码
├── test_silu.py           # 测试代码
├── README.md              # 本文件
└── .orchestrator_state.json  # 状态文件
```

## 实现说明

### 核心逻辑

实现根据 dtype 选择不同的路径：

1. **FP32 路径**: 直接使用 `pypto.sigmoid` API
   ```python
   sigmoid_x = pypto.sigmoid(x)
   y = pypto.mul(x, sigmoid_x)
   ```

2. **FP16/BF16 路径**: 手动展开 sigmoid（因为 `pypto.sigmoid` 仅支持 FP32）
   ```python
   neg_x = pypto.mul(x, -1.0)
   exp_neg_x = pypto.exp(neg_x)
   one_plus_exp = pypto.add(exp_neg_x, 1.0)
   sigmoid_x = pypto.reciprocal(one_plus_exp)
   y = pypto.mul(x, sigmoid_x)
   ```

### Tiling 策略

- **算子类型**: Vector（纯 element-wise 操作）
- **TileShape 配置**: 根据 shape 维度动态设置
  - 多维输入：每个维度设置 tile size 为 32
  - 1D 输入：使用默认 tile (32, 128)

### 性能配置

针对性能_P0 配置 [1, 4096, 4096], float16：
- 推荐 TileShape: [1, 32, 128]
- 尾轴 128 满足 32B 对齐要求

## 运行方式

### 环境准备

1. 设置 NPU 设备 ID：
   ```bash
   export TILE_FWK_DEVICE_ID=0
   ```

2. 确保 CANN 环境已配置

### 执行测试

```bash
# 运行所有测试
python test_silu.py

# 运行特定测试
python test_silu.py silu::test_silu_function_p0

# 列出所有测试用例
python test_silu.py --list

# 使用 SIM 模式（调试用）
python test_silu.py --run_mode sim
```

## 测试用例

| 配置名称 | 类型 | Shape | Dtype | 说明 |
|---------|------|-------|-------|------|
| 边界_P0 | 边界 | [1, 1, 1] | float32 | 最小 shape 验证 |
| 功能_P0 | 功能 | [2, 1024, 512] | float32 | 功能验证基础配置 |
| 功能_P1 | 功能 | [4, 2048, 1024] | bfloat16 | BF16 精度验证 |
| 性能_P0 | 性能 | [1, 4096, 4096] | float16 | LLaMA MLP 典型规模 |

## 精度标准

- **atol**: 0.001
- **rtol**: 0.001

## 已知限制

1. **dtype 约束**: `pypto.sigmoid` 仅支持 FP32，FP16/BF16 需手动展开
2. **输入要求**: 输入 tensor 必须连续（`is_contiguous() == True`）
3. **shape 约束**: 不支持空 tensor，shape size ≤ INT32_MAX

## 参考资料

- **参考实现**: `examples/02_intermediate/operators/activation/activation.py`
- **论文**: "Searching for Activation Functions" (Ramachandran et al., 2017)
- **PyTorch API**: `torch.nn.functional.silu`
