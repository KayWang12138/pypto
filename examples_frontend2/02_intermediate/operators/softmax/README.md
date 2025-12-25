# Softmax 算子开发样例

本样例展示了如何使用 PyPTO 框架从底层逻辑开始实现一个高效的 Softmax 算子。

## 总览介绍

Softmax 是深度学习中极其常用的算子，尤其在 Attention 机制中。本样例详细介绍了以下实现步骤：
- **数值稳定的计算策略**: 通过减去最大值（Max Subtraction）防止指数运算溢出。
- **动态形状支持**: 处理不固定的 Batch Size。
- **显式分块 (Tiling) 配置**: 优化 NPU 硬件执行效率。
- **内核循环处理**: 在内核中通过循环遍历不同的数据分块。

## 核心算法实现

### 1. 核心计算逻辑 (`softmax_core`)

```python
def softmax_core(x: pypto.Tensor) -> pypto.Tensor:
    # 找到最后维度的最大值
    row_max = pypto.amax(x, dim=-1, keepdim=True)
    # 减去最大值，保证数值稳定性
    sub = x - row_max
    # 计算指数
    exp = pypto.exp(sub)
    # 计算指数之和
    esum = pypto.sum(exp, dim=-1, keepdim=True)
    # 归一化
    return exp / esum
```

### 2. JIT 内核封装 (`softmax`)

内核函数负责管理 Tiling 和循环：

```python
@pypto.frontend.jit()
def softmax(
    input_tensor: pypto.Tensor((b, n1, n2, dim), pypto.DT_FP32),
) -> pypto.Tensor((b, n1, n2, dim), pypto.DT_FP32):
    # 设置 Tiling 形状
    pypto.set_vec_tile_shapes(1, 4, 1, 64)
    # 使用 pypto.loop 处理数据分块
    for idx in pypto.loop(b_loop):
        # ... 视图划分与计算 ...
        pypto.assemble(softmax_out, [b_offset, 0, 0, 0], output_tensor)
```

## 代码文件说明

- **`softmax.py`**: 包含完整的 Softmax 实现代码及测试验证逻辑。

## 运行方法

### 环境准备

```bash
# 配置 CANN 环境变量
source /usr/local/Ascend/ascend-toolkit/latest/bin/setenv.bash

# 设置设备 ID
export TILE_FWK_DEVICE_ID=0
```

### 执行脚本

```bash
# 运行所有测试
python3 softmax.py

# 列出所有可用的测试
python3 softmax.py --list
```

## 新前端特性

本示例使用了 PyPTO 新前端 API，主要特点包括：

- **装饰器**: 使用 `@pypto.frontend.jit()` 替代 `@pypto.jit`
- **动态轴**: 使用 `pypto.frontend.dynamic("B")` 在模块级别定义动态维度
- **类型注解**: 函数签名中明确指定张量形状和数据类型
- **张量创建**: 函数内部使用 `pypto.tensor()` 创建输出张量
- **直接传入**: 测试时直接传入 torch 张量，无需使用 `pypto.from_torch()` 转换
- **循环处理**: 使用 `pypto.loop()` 和 `pypto.assemble()` 处理动态 batch size

## 关键特性

- **数值稳定性**: 通过 `amax` 和 `sub` 组合实现，这是在 NPU 上开发高性能激活算子的标准实践。
- **与 PyTorch 验证**: 实现结果通过 `assert_allclose` 与 `torch.softmax` 进行比对。
- **高性能 Tiling**: 展示了如何针对向量处理单元（Vector Core）配置最佳的计算分块。

## 注意事项

- 算子的性能高度依赖于 `set_vec_tile_shapes` 的设置，建议根据实际的隐层维度（Hidden Size）进行调优。
- 本样例展示的是在 dim=-1 上的 Softmax，如需在其他维度计算，需相应调整 `amax` 和 `sum` 的维度参数。

