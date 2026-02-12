# Sinh 算子

## 算子概述

Sinh（双曲正弦）激活算子，用于计算输入张量的双曲正弦值。

### 数学公式

```
y = sinh(x) = (e^x - e^{-x}) / 2
```

### 实现原理

本算子通过组合 PyPTO 的基础运算实现 sinh 函数：

1. **计算 e^x**：使用 `pypto.exp(x)` 计算 x 的指数
2. **计算 e^{-x}**：先使用 `pypto.neg(x)` 取负，再使用 `pypto.exp()` 计算指数
3. **计算差值**：使用 `pypto.sub(exp_x, exp_neg_x)` 计算 e^x - e^{-x}
4. **除以 2**：使用 `pypto.div(diff, 2.0)` 将结果除以 2

### API 映射关系

| 数学步骤 | PyPTO API | 说明 |
|---------|-----------|------|
| e^x | `pypto.exp(x)` | 指数运算 |
| -x | `pypto.neg(x)` | 取负运算 |
| e^{-x} | `pypto.exp(neg_x)` | 指数运算 |
| e^x - e^{-x} | `pypto.sub(exp_x, exp_neg_x)` | 减法运算 |
| (e^x - e^{-x}) / 2 | `pypto.div(diff, 2.0)` | 除法运算 |

## 输入输出规格

| 类型 | Shape | Dtype |
|------|-------|-------|
| 输入 x | [b, s, n, d] | float32 |
| 输出 y | [b, s, n, d] | float32 |

## 编译运行指南

### 环境准备

1. 设置环境变量：
```bash
export TILE_FWK_DEVICE_ID=0
export PTO_TILE_LIB_CODE_PATH=/mnt/workspace/pto_isa/pto-isa/
```

2. 检查 NPU 设备：
```bash
npu-smi info
```

### 编译安装

```bash
python3 build_ci.py -f python3 --disable_auto_execute
```

### 运行测试

```bash
# 运行所有测试
python3 custom/sinh/sinh.py --run_mode npu

# 运行特定测试
python3 custom/sinh/sinh.py basic::test_sinh_basic --run_mode npu
python3 custom/sinh/sinh.py 1k::test_sinh_1k --run_mode npu
python3 custom/sinh/sinh.py edge::test_sinh_edge_cases --run_mode npu
python3 custom/sinh/sinh.py 4d::test_sinh_4d --run_mode npu
```

## 测试结果说明

### 测试用例

| 测试用例 | 说明 | 输入规模 |
|---------|------|---------|
| basic::test_sinh_basic | 基础功能验证（Level 0） | 8 元素 |
| 1k::test_sinh_1k | 典型场景验证（Level 1） | 1K 元素 |
| edge::test_sinh_edge_cases | 边界情况验证（Level 2） | 6 元素（极值/零值） |
| 4d::test_sinh_4d | 4D 张量验证 | [2, 4, 8, 16] |

### 精度标准

- **atol**: 0.000025
- **rtol**: 0.005

### 测试结果

```
============================================================
PyPTO Sinh Activation Function Examples
============================================================

Running Example basic::test_sinh_basic: Basic Usage of sinh Function
============================================================
Test: Basic Usage of sinh Function (Level 0)
============================================================
Input shape: torch.Size([8])
Output shape: torch.Size([8])
Input: tensor([ 0.0000,  1.0000, -1.0000,  2.0000, -2.0000,  0.5000, -0.5000,  1.5000],
       device='npu:0')
Output: tensor([ 0.0000,  1.1752, -1.1752,  3.6269, -3.6269,  0.5211, -0.5211,  2.1293],
       device='npu:0')
Expected: tensor([ 0.0000,  1.1752, -1.1752,  3.6269, -3.6269,  0.5211, -0.5211,  2.1293],
       device='npu:0')
Max difference: 0.000000
✓ Basic usage of sinh function completed successfully

Running Example 1k::test_sinh_1k: Sinh with 1K Elements
============================================================
Test: Sinh with 1K Elements (Level 1)
============================================================
Input shape: torch.Size([1024])
Output shape: torch.Size([1024])
Max difference: 0.000000
✓ Sinh with 1K elements completed successfully

Running Example edge::test_sinh_edge_cases: Sinh with Edge Cases
============================================================
Test: Sinh with Edge Cases (Level 2)
============================================================
Input shape: torch.Size([6])
Output shape: torch.Size([6])
Input: tensor([ 0.0000e+00, -0.0000e+00,  1.0000e-06, -1.0000e-06,  1.0000e+01,
        -1.0000e+01], device='npu:0')
Output: tensor([ 0.0000e+00,  0.0000e+00,  9.8348e-07, -9.8348e-07,  1.1013e+04,
        -1.1013e+04], device='npu:0')
Expected: tensor([ 0.0000e+00,  0.0000e+00,  1.0000e-06, -1.0000e-06,  1.1013e+04,
        -1.1013e+04], device='npu:0')
Max difference: 0.000977
✓ Sinh with edge cases completed successfully

Running Example 4d::test_sinh_4d: Sinh with 4D Tensor
============================================================
Test: Sinh with 4D Tensor [b, s, n, d]
============================================================
Input shape: torch.Size([2, 4, 8, 16])
Output shape: torch.Size([2, 4, 8, 16])
Max difference: 0.000001
✓ Sinh with 4D tensor completed successfully

============================================================
All sinh tests passed!
============================================================
```

## 已知限制和注意事项

1. **数据类型**：当前仅支持 float32 数据类型
2. **输入维度**：支持 1-4 维张量
3. **精度**：在极值情况下（如 x=10 或 x=-10），可能会有微小的精度误差

## 常见问题

### Q: 如何切换到模拟模式运行？

A: 使用 `--run_mode sim` 参数：
```bash
python3 custom/sinh/sinh.py --run_mode sim
```

### Q: 编译时出现 "Invalid Device" 错误怎么办？

A: 检查 NPU 设备状态并设置正确的设备 ID：
```bash
npu-smi info
export TILE_FWK_DEVICE_ID=<设备ID>
```

### Q: 如何调试精度问题？

A: 可以使用渐进式调试方法，在每个中间步骤插入输出，检查 exp、neg、sub、div 各步骤的结果。

## 文件结构

```
custom/sinh/
├── sinh.py          # 测试用例文件
├── sinh_impl.py     # 算子实现文件
└── README.md        # 本文档
```
