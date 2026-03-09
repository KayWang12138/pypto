---
name: pypto-operator-accuracy-verify
description: PyPTO 算子精度验证技能。用于验证 PyPTO 算子的计算精度，提供多种容差配置、详细精度分析和调试方法。当用户需要验证算子精度、调试精度问题或选择合适的容差参数时使用此技能。
license: 完整条款见 LICENSE.txt
---

# PyPTO 算子精度验证技能 (PyPTO Operator Accuracy Verify)

此技能提供 PyPTO 算子精度验证的完整方法，包括容差配置、详细分析和调试技巧。

## 关于精度验证

PyPTO 算子精度验证通过对比 NPU 计算结果与 Golden/参考实现结果，验证算子的正确性。

### 精度验证的重要性

- 确保算子计算结果正确
- 验证不同数据类型的精度损失
- 发现和定位精度问题
- 为算子优化提供基准

## 核心原则

### 基础验证方法

使用 `numpy.testing.assert_allclose` 进行精度验证：

```python
from numpy.testing import assert_allclose

# 基本用法
assert_allclose(
    np.array(actual_result.cpu().flatten().tolist()),
    np.array(expected_result.cpu().flatten().tolist()),
    rtol=1e-3,  # 相对容差
    atol=1e-3   # 绝对容差
)
```

### 容差配置策略

根据算子特性选择合适的容差：

| 算子类型 | rtol | atol | 说明 |
|---------|------|------|------|
| **复杂算子** (Attention, FFN) | `0.0078125` | `0.0001` | 涉及 Softmax、除法等不稳定运算 |
| **简单算子** (Gate, MatMul) | `5e-3` | `5e-3` | 简单矩阵运算，容差稍宽 |
| **通用算子** | `1e-3` | `1e-3` | 默认容差 |
| **高精度要求** | `1e-5` | `1e-5` | 对精度要求极高的场景 |

### 容差选择原则

```python
# 策略1: 涉及指数/除法/Softmax 的算子
rtol, atol = 0.0078125, 0.0001

# 策略2: 简单矩阵运算
rtol, atol = 5e-3, 5e-3

# 策略3: 通用情况
rtol, atol = 1e-3, 1e-3

# 策略4: 极高精度要求
rtol, atol = 1e-5, 1e-5
```

## 详细精度分析

### 使用详细比较方法

使用 `utils/np_compare.py` 中的 `detailed_allclose_manual` 进行详细分析：

```python
from utils.np_compare import detailed_allclose_manual

# 详细比较，会打印异常元素信息
detailed_allclose_manual(
    cpu_result,      # CPU/Golden 结果
    npu_result,      # NPU 结果
    name="算子名称", # 算子名称
    rtol=1e-3,       # 相对容差
    atol=1e-3,       # 绝对容差
    max_prints=50,   # 最大打印异常数量
    force_print_first_n=5  # 强制打印前N个元素
)
```

### 输出信息说明

详细比较会输出以下信息：

- **前 N 个元素**：强制打印前几个元素的值，用于快速检查
- **异常元素**：超出容差的元素索引和值
- **NaN 统计**：NaN 值的数量和位置
- **异常比例**：异常元素占总元素的比例
- **比较结果**：是否通过 allclose 条件

### 输出示例

```
开始比较数组，形状: (32, 5120), 总元素数: 163840
容差条件: rtol=0.001, atol=0.001
================================================================================
强制打印前 5 个元素:
索引 (0, 0): cpu=1.234567e+00, npu=1.234568e+00, 差值=1.000000e-06
索引 (0, 1): cpu=2.345678e+00, npu=2.345679e+00, 差值=1.000000e-06
...
--------------------------------------------------------------------------------
================================================================================
算子名称 比较结果统计:
总元素数量: 163840
异常元素数量: 0
  - NaN 数量: 0
  - 超出容差数量: 0
异常比例: 0.0000%

np.allclose 等价结果: True
```

## 精度验证标准

### 通过条件

- ✅ 所有元素的误差满足：`|actual - expected| <= atol + rtol * |expected|`
- ✅ 无 NaN 值
- ✅ 形状一致

### 失败判定

- ❌ 误差超出容差
- ❌ 存在 NaN 值
- ❌ 形状不匹配

### 容差计算公式

```python
# 允许的误差范围
allowed_diff = atol + rtol * abs(expected_value)

# 实际误差
actual_diff = abs(actual_value - expected_value)

# 判断是否通过
is_pass = actual_diff <= allowed_diff
```

## 使用流程

### 步骤 1：准备测试数据

```python
import torch
import numpy as np

# 设置随机种子，保证可复现性
torch.manual_seed(0)
np.random.seed(0)

# 生成测试数据
input_data = torch.randn(shape, dtype=torch.bfloat16, device='npu:0')
```

### 步骤 2：执行 NPU 算子

```python
# 执行 NPU 算子
actual_result = torch.zeros(output_shape, dtype=torch.bfloat16, device='npu:0')
your_npu_operator(input_data, actual_result)
```

### 步骤 3：执行 Golden/参考实现

```python
# 执行 Golden 实现
expected_result = your_golden_implementation(input_data)
```

### 步骤 4：精度验证

```python
from numpy.testing import assert_allclose

# 基础验证
assert_allclose(
    np.array(actual_result.cpu().flatten().tolist()),
    np.array(expected_result.cpu().flatten().tolist()),
    rtol=0.0078125,  # 根据算子类型调整
    atol=0.0001
)

print("✓ 精度验证通过")
```

### 步骤 5：详细分析（可选）

如果精度验证失败，使用详细模式定位问题：

```python
from utils.np_compare import detailed_allclose_manual

detailed_allclose_manual(
    expected_result,
    actual_result,
    name="调试算子",
    rtol=1e-3,
    atol=1e-3,
    max_prints=100,  # 打印更多异常
    force_print_first_n=10  # 打印前10个元素
)
```

## 完整测试模板

### 基础测试模板

```python
def test_operator():
    # 1. 设置参数
    device_id = int(os.environ.get('TILE', 0))
    torch.npu.set_device(device_id)

    # 2. 准备测试数据
    torch.manual_seed(0)
    np.random.seed(0)
    input_data = torch.randn(shape, dtype=torch.bfloat16, device=f'npu:{device_id}')
    actual_result = torch.zeros(output_shape, dtype=torch.bfloat16, device=f'npu:{device_id}')

    # 3. 执行 NPU 算子
    your_npu_operator(input_data, actual_result)

    # 4. 执行 Golden/参考实现
    expected_result = your_golden_implementation(input_data)

    # 5. 精度验证
    assert_allclose(
        np.array(actual_result.cpu().flatten().tolist()),
        np.array(expected_result.cpu().flatten().tolist()),
        rtol=0.0078125,  # 根据算子类型调整
        atol=0.0001
    )

    print("✓ 精度验证通过")
```

### 多形状测试模板

```python
def test_operator_multiple_shapes():
    # 测试多种输入形状
    test_cases = [
        (1, 5120),    # 小规模
        (32, 5120),   # 中等规模
        (64, 5120),   # 大规模
    ]

    for batch_size, hidden_size in test_cases:
        print(f"Testing shape: ({batch_size}, {hidden_size})")

        # 准备数据
        input_data = torch.randn((batch_size, hidden_size), dtype=torch.bfloat16, device='npu:0')
        actual_result = torch.zeros((batch_size, hidden_size), dtype=torch.bfloat16, device='npu:0')

        # 执行算子
        your_npu_operator(input_data, actual_result)

        # 执行 Golden
        expected_result = your_golden_implementation(input_data)

        # 精度验证
        assert_allclose(
            np.array(actual_result.cpu().flatten().tolist()),
            np.array(expected_result.cpu().flatten().tolist()),
            rtol=0.0078125,
            atol=0.0001
        )

        print(f"✓ Shape ({batch_size}, {hidden_size}) 验证通过")
```

## 调试技巧

### 技巧 1：使用详细比较定位问题

当精度验证失败时，使用详细模式：

```python
from utils.np_compare import detailed_allclose_manual

detailed_allclose_manual(
    expected_result,
    actual_result,
    name="调试算子",
    rtol=1e-3,
    atol=1e-3,
    max_prints=100,  # 打印更多异常
    force_print_first_n=10  # 打印前10个元素
)
```

### 技巧 2：分段验证

对于复杂算子，可以分段验证中间结果：

```python
# 在算子实现中添加中间输出
def your_npu_operator(input_data, output_data, debug_intermediate=False):
    # 第一步计算
    intermediate1 = compute_step1(input_data)

    if debug_intermediate:
        # 输出中间结果用于验证
        print("Intermediate 1:", intermediate1)

    # 第二步计算
    intermediate2 = compute_step2(intermediate1)

    if debug_intermediate:
        print("Intermediate 2:", intermediate2)

    # 最终计算
    output_data[:] = compute_final(intermediate2)
```

### 技巧 3：数据类型检查

检查数据类型转换是否正确：

```python
# 检查输入数据类型
assert input_data.dtype == torch.bfloat16, f"Expected bfloat16, got {input_data.dtype}"

# 检查输出数据类型
assert output_data.dtype == torch.bfloat16, f"Expected bfloat16, got {output_data.dtype}"

# 检查中间计算数据类型
assert intermediate.dtype == torch.float32, f"Expected float32, got {intermediate.dtype}"
```

### 技巧 4：形状检查

检查 tensor 形状是否正确：

```python
# 检查输入形状
assert input_data.shape == expected_input_shape, f"Expected {expected_input_shape}, got {input_data.shape}"

# 检查输出形状
assert output_data.shape == expected_output_shape, f"Expected {expected_output_shape}, got {output_data.shape}"
```

## 容差调整建议

### 调整原则

**如果精度验证失败，按以下顺序排查：**

1. **检查实现逻辑**：确认算法实现是否正确
2. **检查数据类型**：确认数据类型转换是否正确
3. **检查 tile size**：确认 tile 配置是否合理
4. **检查边界条件**：确认边界值处理是否正确
5. **最后才考虑放宽容差**

### 容差调整策略

```python
# 策略1: 涉及指数/除法/Softmax 的算子
# 这些运算数值不稳定，需要较宽松的容差
rtol, atol = 0.0078125, 0.0001

# 策略2: 简单矩阵运算
# 矩阵乘法等运算相对稳定
rtol, atol = 5e-3, 5e-3

# 策略3: 通用情况
# 默认容差
rtol, atol = 1e-3, 1e-3

# 策略4: 极高精度要求
# 对精度要求极高的场景
rtol, atol = 1e-5, 1e-5
```

### BF16 vs FP32 容差

```python
# BF16 数据类型：精度较低，容差需要更宽松
if dtype == torch.bfloat16:
    rtol, atol = 0.0078125, 0.0001

# FP32 数据类型：精度较高，可以使用更严格的容差
elif dtype == torch.float32:
    rtol, atol = 1e-3, 1e-3
```

## 常见问题

### 问题 1：精度验证失败

**可能原因：**
- 算子实现逻辑错误
- 数据类型转换错误
- Tile size 配置不当
- 边界条件处理不当

**排查方法：**
1. 使用详细比较模式定位异常元素
2. 检查中间结果
3. 验证数据类型转换
4. 检查 tile 配置

### 问题 2：NaN 值

**可能原因：**
- 除以零
- 对负数求对数
- 指数运算溢出

**排查方法：**
1. 检查除数是否为零
2. 检查对数运算的输入是否为正
3. 检查指数运算是否溢出
4. 使用详细比较查看 NaN 位置

### 问题 3：形状不匹配

**可能原因：**
- 输入输出 shape 计算错误
- reshape/view 操作错误
- 动态 shape 计算错误

**排查方法：**
1. 检查输入输出 shape
2. 验证 reshape/view 操作
3. 检查动态 shape 计算

## 检查清单

使用精度验证时，确保：

- [ ] 设置正确的随机种子
- [ ] 准备合适的测试数据
- [ ] 执行 NPU 算子
- [ ] 执行 Golden/参考实现
- [ ] 选择合适的容差参数
- [ ] 进行精度验证
- [ ] 分析验证结果
- [ ] 处理异常情况

## 参考资料

- 精度验证示例: `models/glm_v4_5/`
- 详细比较工具: `models/glm_v4_5/utils/np_compare.py`
- numpy.testing 文档: https://numpy.org/doc/stable/reference/routines.testing.html
- PyPTO API: `docs/api/`
