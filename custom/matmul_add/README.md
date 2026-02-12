# matmul_add 算子

## 算子概述

`matmul_add` 算子实现了矩阵乘法加偏置操作，其数学公式为：

```
y = a @ b^T + c
```

其中 `b^T` 表示矩阵 `b` 的转置。这个算子在深度学习中非常常见，特别是在全连接层、注意力机制等场景中。

## 功能说明

matmul_add 算子结合了矩阵乘法和元素加法操作，是神经网络中基本计算单元之一。其特点包括：

- 支持矩阵转置操作（`b^T`）
- 支持偏置加法（`+ c`）
- 使用 bfloat16 数据类型以优化内存和计算效率

## 数学公式

| 数学符号 | PyPTO API | 说明 |
|---------|-----------|------|
| b^T | `pypto.transpose(b, 0, 1)` | 转置操作（2维） |
| @ | `pypto.matmul(a, b_t, out_dtype)` | 矩阵乘法 |
| + | `pypto.add(matmul_result, c)` | 元素加法 |

## 输入输出规格

| 类型  | shape  | dtype  |
| ------------ | ------------ | ------------ |
| 输入 a| [m, k] | bfloat16  |
| 输入 b| [n, k] | bfloat16  |
| 输入 c| [m, n] | bfloat16  |
| 输出 y| [m, n] | bfloat16  |

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
python3 custom/matmul_add/matmul_add.py --run_mode npu
```

如果 NPU 不可用，可以使用模拟模式：

```bash
python3 custom/matmul_add/matmul_add.py --run_mode sim
```

## 测试结果说明

测试包含多个级别的用例：

- **Level 0**: 小矩阵（8x8），验证基础功能
- **Level 1**: 典型矩阵（128x128），验证常规应用
- **Level 2**: 非方阵，包括：
  - 非方阵测试（64x32）
  - 宽矩阵测试（32x128）
  - 高矩阵测试（128x32）

**精度验证标准**：
- 相对误差 < 1e-2（bfloat16 精度较低）
- NPU 模式下自动验证精度

## NPU 优化配置

本算子使用了 cube 单元优化，通过设置 `cube_tile_shapes` 来优化矩阵乘法性能：

```python
pypto.set_cube_tile_shapes([32, 32], [64, 64], [64, 64])
```

该配置在 M、N、K 轴上分别设置切分大小，以提高 NPU 立方体单元的利用率。

## 已知限制和注意事项

1. **维度限制**：仅支持 2 维矩阵（当前实现）
2. **数据类型**：使用 bfloat16，精度低于 float32
3. **矩阵维度约束**：
   - 矩阵 a 的列数（k）必须等于矩阵 b 的列数（k）
   - 偏置矩阵 c 的形状必须为 [m, n]
4. **转置约束**：2 维矩阵支持任意轴转置
5. **精度**：bfloat16 数据类型存在精度损失，对于对精度要求较高的场景建议使用 float32

## 常见问题

### Q: 如何修改矩阵的维度？

A: 修改测试用例中的 `m`, `k`, `n` 参数：

```python
m = 64  # 矩阵 a 的行数
k = 128  # 矩阵 a 的列数和矩阵 b 的列数
n = 32  # 矩阵 b 的行数和输出矩阵的列数
```

### Q: 如何使用 float32 数据类型？

A: 需要在 `matmul_add_impl.py` 和 `matmul_add.py` 中同时修改数据类型常量：

```python
# 从 pypto.DT_BF16 改为 pypto.DT_FP32
# 从 torch.bfloat16 改为 torch.float32
```

### Q: 为什么 bfloat16 精度较低？

A: bfloat16 (Brain Floating Point) 是一种 16 位浮点格式，牺牲了一定的精度以换取更宽的数值范围，适合深度学习场景。其指数位与 float32 相同，但尾数位较少。

### Q: cube_tile_shapes 如何影响性能？

A: `cube_tile_shapes` 决定了矩阵乘法在 NPU 立方体单元上的切分策略。合理的切分大小可以提高并行度和缓存利用率。不同的矩阵尺寸可能需要调整这些参数以达到最佳性能。

## 参考资料

- PyPTO API 文档：`docs/api/operation/`
- 矩阵乘法示例：`examples/01_beginner/compute/matmul_ops.py`
- Attention 示例（综合使用）：`examples/03_advanced/advanced_nn/attention/attention.py`
