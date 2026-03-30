# reshape 算子

## 概述

reshape 算子将输入张量变换为指定的形状，保持数据不变，仅改变维度视图。支持动态轴（batch、seq_len）和负维度自动推断。

### 数学公式

$$y = \text{reshape}(x, \text{shape})$$

### 功能特性

- **纯视图操作**: 不涉及数据搬运或计算，仅改变元数据
- **负维度支持**: 支持 `-1` 自动推断维度大小
- **动态轴支持**: 支持 batch、seq_len 等动态维度
- **多精度支持**: 支持 float32、bfloat16、float16

## 目录结构

```
operators/reshape/
├── spec.md                 # 需求规范
├── api_report.md           # API 探索报告
├── design.md               # 设计文档
├── reshape_golden.py       # Golden 参考实现
├── reshape_impl.py         # PyPTO kernel 实现
├── test_reshape.py         # 测试代码
├── README.md               # 本文件
└── .orchestrator_state.json # 状态文件
```

## 运行方式

### 环境准备

```bash
# 设置 NPU 设备 ID
export TILE_FWK_DEVICE_ID=0

# 配置 CANN 环境（如未配置）
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

### 运行测试

```bash
cd operators/reshape

# 运行所有测试
python test_reshape.py

# 列出所有测试用例
python test_reshape.py --list

# 运行特定测试
python test_reshape.py reshape::test_reshape_level0

# 使用模拟器模式（无 NPU 时）
python test_reshape.py --run_mode sim
```

### 可用测试用例

| 用例 ID | 描述 |
|---------|------|
| `reshape::test_reshape_level0` | 小数据量基础功能验证 |
| `reshape::test_reshape_level1` | 典型场景验证 |
| `reshape::test_reshape_negative_dim` | 负维度自动推断测试 |
| `reshape::test_reshape_flatten` | 展平为一维测试 |
| `reshape::test_reshape_bfloat16` | bfloat16 数据类型测试 |
| `reshape::test_reshape_float16` | float16 数据类型测试 |
| `reshape::test_reshape_dynamic` | 动态 shape 测试 |
| `reshape::test_reshape_boundary` | 边界情况测试 |

## 验证入口

### 使用方法

```python
from reshape_impl import reshape_wrapper, reshape_dynamic_wrapper

# 基础用法
x = torch.randn(2, 3, 4)
y = reshape_wrapper(x, [2, 12])  # [2, 3, 4] -> [2, 12]

# 负维度自动推断
y = reshape_wrapper(x, [2, -1])  # 自动推断为 [2, 12]

# 展平
y = reshape_wrapper(x, [-1])  # [2, 3, 4] -> [24]

# 动态 shape（带动态轴标注）
x = torch.randn(4, 16, 64)  # batch=4 是动态的
y = reshape_dynamic_wrapper(x, [4, 128, 8], dynamic_axes=[0])
```

### 精度标准

| Dtype | rtol | atol |
|-------|------|------|
| float32 | 1e-3 | 1e-3 |
| bfloat16 | 0.01 | 0.01 |
| float16 | 0.01 | 0.01 |

## 实现说明

### API 映射

reshape 算子直接使用 PyPTO 的 `pypto.reshape` API：

```python
# 在 kernel 中
result = pypto.reshape(input_tensor, target_shape)
output_tensor[:] = result
```

### Tiling 策略

reshape 是纯视图操作，不需要 Tiling 配置。

### Loop 结构

reshape 是单步元数据操作，不需要 Loop 结构。

### 约束条件

1. **输入必须 contiguous**: `from_torch` 要求输入 Tensor 必须是连续的
2. **元素总数必须匹配**: `prod(input.shape) == prod(target_shape)`
3. **-1 维度最多一个**: shape 中最多一个维度为 -1
4. **Shape Size 上限**: 不超过 INT32_MAX

## 已知限制

1. 非连续 Tensor 会在 wrapper 中自动调用 `.contiguous()` 进行转换
2. 动态 shape 场景需要使用 `reshape_dynamic_wrapper` 并指定 `dynamic_axes`
3. 当有效 shape 依赖其他 Tensor 标识时，需要显式传 `valid_shape`（高级用法）

## 性能说明

reshape 是 view 操作，性能开销极小（仅元数据操作），预期 kernel 耗时接近 0。
