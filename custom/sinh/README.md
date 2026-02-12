# Sinh (双曲正弦函数) 算子

## 算子概述

Sinh 算子实现了双曲正弦函数（Hyperbolic Sine），这是一个常用的数学函数，在科学计算、工程和神经网络中广泛应用。

## 数学公式

```
sinh(x) = (e^x - e^(-x)) / 2
```

其中：
- `e^x` 是自然指数函数
- `e^(-x)` 是自然指数函数的负输入版本
- 最终结果除以 2 进行归一化

## API 映射关系

| 数学操作 | PyPTO API | 文档位置 |
|---------|----------|---------|
| e^x | `pypto.exp(x)` | docs/api/operation/pypto-exp.md |
| -x | `pypto.neg(x)` | docs/api/operation/pypto-neg.md |
| a - b | `pypto.sub(a, b)` | docs/api/operation/pypto-sub.md |
| a / b | `pypto.div(a, b)` | docs/api/operation/pypto-div.md |

## 核心实现

```python
@pypto.frontend.jit(runtime_options={"run_mode": mode})
def sinh_activation_kernel(
    x: pypto.Tensor((m, n), pypto.DT_FP32),
) -> pypto.Tensor((m, n), pypto.DT_FP32):
    out = pypto.tensor((m, n), pypto.DT_FP32)
    configure_tiling(x)

    exp_x = pypto.exp(x)
    neg_x = pypto.neg(x)
    exp_neg_x = pypto.exp(neg_x)
    diff = pypto.sub(exp_x, exp_neg_x)
    out[:] = pypto.div(diff, 2.0)
    return out
```

### 实现步骤

1. **计算 e^x**: 使用 `pypto.exp(x)` 计算输入的自然指数
2. **计算 -x**: 使用 `pypto.neg(x)` 计算输入的负数
3. **计算 e^(-x)**: 使用 `pypto.exp(neg_x)` 计算负输入的自然指数
4. **计算差值**: 使用 `pypto.sub(exp_x, exp_neg_x)` 计算 `e^x - e^(-x)`
5. **除以 2**: 使用 `pypto.div(diff, 2.0)` 完成最终计算

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
cd custom/sinh/

# 运行测试（NPU 模式）
python3 sinh.py --run_mode npu

# 查看可用测试
python3 sinh.py --list
```

## 测试结果

### Level 0: 小规模测试 (8x8)
- 输入形状: (8, 8)
- 最大误差: 0.000002
- 平均误差: 0.000000
- 状态: ✅ 通过

### Level 1: 中等规模测试 (32x128)
- 输入形状: (32, 128)
- 最大误差: 0.000002
- 平均误差: 0.000000
- 状态: ✅ 通过

### Level 2: 大规模测试 (64x256)
- 输入形状: (64, 256)
- 最大误差: 0.000004
- 平均误差: 0.000000
- 状态: ✅ 通过

### 精度验证

所有测试用例均与 PyTorch 的 `torch.sinh()` 函数结果进行对比验证，最大误差小于 1e-3，满足精度要求。

## 性能特性

- **数据类型**: FP32 (Float32)
- **支持形状**: 2-4 维张量
- **Tiling 策略**: 自适应张量形状
  - 2D 及以上: 每个维度使用 32 的 tile 大小
  - 1D: 使用 (32, 128) 的 tile 大小

## 已知限制

1. 当前实现仅支持 FP32 数据类型
2. 不支持动态形状（可通过修改代码支持）
3. 不支持复数类型输入

## 常见问题

### Q: 如何使用不同的运行模式？

A: 使用 `--run_mode` 参数：
```bash
# NPU 模式（需要 NPU 硬件）
python3 sinh.py --run_mode npu

# 仿真模式（无需 NPU 硬件）
python3 sinh.py --run_mode sim
```

### Q: 如何修改数据类型？

A: 修改 `sinh_activation_kernel` 函数中的 `pypto.DT_FP32` 为其他支持的数据类型，如 `pypto.DT_FP16` 或 `pypto.DT_BF16`。

### Q: 测试报错 "Invalid Device" 怎么办？

A: 检查 NPU 设备是否可用：
```bash
# 查看 NPU 设备信息
npu-smi info

# 设置正确的设备 ID
export TILE_FWK_DEVICE_ID=<正确的设备ID>
```

## 数学背景

双曲正弦函数是双曲函数的一种，其性质包括：

- 奇函数: sinh(-x) = -sinh(x)
- 泰勒展开: sinh(x) = x + x³/3! + x⁵/5! + ...
- 与普通正弦的关系: sinh(ix) = i·sin(x)

在神经网络中，sinh 函数有时被用作激活函数的变种，特别是在需要平滑、非线性变换的场景中。

## 参考资料

- PyPTO 官方文档: `docs/api/`
- 激活函数示例: `examples/02_intermediate/operators/activation/`
- 双曲函数: [Wikipedia - Hyperbolic functions](https://en.wikipedia.org/wiki/Hyperbolic_functions)
