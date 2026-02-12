# Sigmoid (Sigmoid 激活函数) 算子

## 算子概述

Sigmoid 算子实现了 Sigmoid 激活函数，也称为逻辑函数（Logistic Function），是神经网络中最常用的激活函数之一。它将任意实数映射到 (0, 1) 区间。

## 数学公式

```
sigmoid(x) = 1 / (1 + e^(-x))
```

其中：
- `e^(-x)` 是自然指数函数的负输入版本
- `1 + e^(-x)` 是分母
- `1 / (1 + e^(-x))` 将结果归一化到 (0, 1) 区间

## API 映射关系

| 数学操作 | PyPTO API | 文档位置 |
|---------|----------|---------|
| e^x | `pypto.exp(x)` | docs/api/operation/pypto-exp.md |
| -x | `pypto.neg(x)` | docs/api/operation/pypto-neg.md |
| a + b | `pypto.add(a, b)` | docs/api/operation/pypto-add.md |
| 1 / a | `pypto.reciprocal(a)` | docs/api/operation/pypto-reciprocal.md |

## 核心实现

```python
@pypto.frontend.jit(runtime_options={"run_mode": mode})
def sigmoid_activation_kernel(
    x: pypto.Tensor((m, n), pypto.DT_FP16),
) -> pypto.Tensor((m, n), pypto.DT_FP16):
    out = pypto.tensor((m, n), pypto.DT_FP16)
    configure_tiling(x)

    neg_x = pypto.neg(x)
    exp_neg_x = pypto.exp(neg_x)
    one_plus_exp = pypto.add(exp_neg_x, 1.0)
    out[:] = pypto.reciprocal(one_plus_exp)
    return out
```

### 实现步骤

1. **计算 -x**: 使用 `pypto.neg(x)` 计算输入的负数
2. **计算 e^(-x)**: 使用 `pypto.exp(neg_x)` 计算负输入的自然指数
3. **计算 1 + e^(-x)**: 使用 `pypto.add(exp_neg_x, 1.0)` 计算分母
4. **计算倒数**: 使用 `pypto.reciprocal(one_plus_exp)` 完成最终计算

## 编译运行指南

### 环境准备

```bash
# 设置 CANN 环境变量（如果未配置）
source /usr/local/Ascend/ascend-toolkit/latest/bin/setenv.bash

# 设置设备 ID 和 pto-isa 路径
export TILE_FWK_DEVICE_ID=0
export PTO_TILE_LIB_CODE_PATH=/mnt/workspace/pto_isa/pto-isa/
```

### 编译 whl 包

```bash
# 在项目根目录执行
python3 build_ci.py -f python3 --disable_auto_execute
```

### 运行测试

```bash
# 进入算子目录
cd custom/sigmoid/

# 运行测试（NPU 模式）
python3 sigmoid.py --run_mode npu

# 查看可用测试
python3 sigmoid.py --list
```

## 测试结果

### Level 0: 小规模测试 (8x8)
- 输入形状: (8, 8)
- 最大误差: 0.000488
- 平均误差: 0.000137
- 状态: ✅ 通过

### Level 1: 中等规模测试 (32x128)
- 输入形状: (32, 128)
- 最大误差: 0.000488
- 平均误差: 0.000103
- 状态: ✅ 通过

### Level 2: 大规模测试 (64x256)
- 输入形状: (64, 256)
- 最大误差: 0.000488
- 平均误差: 0.000102
- 状态: ✅ 通过

### 精度验证

所有测试用例均与 PyTorch 的 `torch.sigmoid()` 函数结果进行对比验证，最大误差小于 1e-3，满足精度要求。

**注意**: 由于使用 FP16 数据类型，精度误差略高于 FP32 实现，但在可接受范围内。

## 性能特性

- **数据类型**: FP16 (Float16)
- **支持形状**: 2-4 维张量
- **Tiling 策略**: 自适应张量形状
  - 2D 及以上: 每个维度使用 32 的 tile 大小
  - 1D: 使用 (32, 128) 的 tile 大小

## 已知限制

1. 当前实现仅支持 FP16 数据类型
2. 不支持动态形状（可通过修改代码支持）
3. 不支持复数类型输入

## 常见问题

### Q: 如何使用不同的运行模式？

A: 使用 `--run_mode` 参数：
```bash
# NPU 模式（需要 NPU 硬件）
python3 sigmoid.py --run_mode npu

# 仿真模式（无需 NPU 硬件）
python3 sigmoid.py --run_mode sim
```

### Q: 如何修改数据类型？

A: 修改 `sigmoid_activation_kernel` 函数中的 `pypto.DT_FP16` 为其他支持的数据类型，如 `pypto.DT_FP32` 或 `pypto.DT_BF16`。

### Q: 测试报错 "Invalid Device" 怎么办？

A: 检查 NPU 设备是否可用：
```bash
# 查看 NPU 设备信息
npu-smi info

# 设置正确的设备 ID
export TILE_FWK_DEVICE_ID=<正确的设备ID>
```

### Q: 为什么 FP16 的精度比 FP32 低？

A: FP16 是半精度浮点数，只有 16 位（1 位符号位 + 5 位指数位 + 10 位尾数位），而 FP32 是单精度浮点数，有 32 位（1 位符号位 + 8 位指数位 + 23 位尾数位）。因此 FP16 的精度会相对较低，但可以减少内存占用并提高计算速度。

## 数学背景

Sigmoid 函数是神经网络中最经典的激活函数之一，其性质包括：

- **值域**: (0, 1)
- **导数**: f'(x) = f(x) * (1 - f(x))
- **单调性**: 单调递增
- **对称性**: f(0) = 0.5

在神经网络中，Sigmoid 函数：
- 将输出映射到概率值范围 (0, 1)
- 适用于二分类问题的输出层
- 容易导致梯度消失问题（在两端导数接近 0）

## 参考资料

- PyPTO 官方文档: `docs/api/`
- 激活函数示例: `examples/02_intermediate/operators/activation/`
- Sigmoid 函数: [Wikipedia - Sigmoid function](https://en.wikipedia.org/wiki/Sigmoid_function)
