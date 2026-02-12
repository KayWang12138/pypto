# sinh 算子

## 算子概述

`sinh` 算子实现了双曲正弦函数（hyperbolic sine），其数学公式为：

```
y = sinh(x) = (e^x - e^(-x)) / 2
```

## 功能说明

双曲正弦函数是双曲函数之一，与普通三角函数类似，但基于双曲线而非圆。它在各种数学和工程应用中广泛使用，特别是在信号处理、物理学和深度学习中的某些激活函数变体。

## 数学公式

| 数学符号 | PyPTO API | 说明 |
|---------|-----------|------|
| e^x | `pypto.exp(x)` | 指数函数 |
| e^(-x) | `pypto.exp(-x)` | 指数函数（负指数） |
| - | `pypto.sub(a, b)` | 减法 |
| / | `pypto.div(a, 2.0)` | 除以常数 2 |

## 输入输出规格

| 类型  | shape  | dtype  |
| ------------ | ------------ | ------------ |
| 输入 x| [b, s, n, d]  | float32  |
| 输出 y| [b, s, n, d]  | float32  |

## 编译运行指南

### 1. 环境准备

确保已设置必要的环境变量：

```bash
export TILE_FWK_DEVICE_ID=0
export PTO_TILE_LIB_CODE_PATH=/mnt/workspace/pto_isa/pto-isa/
```

### 2. 编译

```bash
python3 build_ci.py -f python3 --disable_auto_execute
```

### 3. 运行测试

```bash
python3 custom/sinh/sinh.py --run_mode npu
```

如果 NPU 不可用，可以使用模拟模式：

```bash
python3 custom/sinh/sinh.py --run_mode sim
```

## 测试结果说明

测试包含多个级别的用例：

- **Level 0**: 小数据量（8-16 元素），验证基础功能
- **Level 1**: 典型场景（1K 元素），验证常规应用
- **Level 2**: 边界情况，包括：
  - 零值测试
  - 大正数值测试
  - 大负数值测试

**精度验证标准**：
- 相对误差 < 1e-3
- NPU 模式下自动验证精度

## 已知限制和注意事项

1. **维度限制**：仅支持 2-4 维张量
2. **形状大小限制**：张量元素数量不超过 INT32_MAX (2147483647)
3. **数据类型**：当前实现仅支持 float32，扩展可支持 float16 和 bfloat16
4. **精度**：由于浮点运算精度，大数值可能存在一定误差

## 常见问题

### Q: 如何修改输入张量的维度？

A: 修改测试用例中的 `shape` 参数，例如：

```python
shape = (2, 4, 4, 2)  # [b, s, n, d]
```

### Q: 如何使用不同的数据类型？

A: 需要在 `sinh_impl.py` 和 `sinh.py` 中同时修改数据类型常量：

```python
# 从 pypto.DT_FP32 改为 pypto.DT_FP16 或 pypto.DT_BF16
# 从 torch.float32 改为 torch.float16 或 torch.bfloat16
```

### Q: 为什么大数值时精度会降低？

A: 由于浮点数的表示范围和精度限制，当 e^x 的值过大或过小时，可能会出现精度损失。这是浮点运算的正常现象。

## 参考资料

- PyPTO API 文档：`docs/api/operation/`
- 示例代码：`examples/01_beginner/compute/elementwise_ops.py`
