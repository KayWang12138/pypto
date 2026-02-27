# PyPTO IR 类型系统和示例

本文档介绍类型系统并提供实际使用示例。

## 类型系统

### ScalarType

表示基本标量类型。

```python
from pypto import DataType, ir

int_type = ir.ScalarType(DataType.INT64)
float_type = ir.ScalarType(DataType.FP32)
```

**支持的 DataTypes：** INT8, INT16, INT32, INT64, UINT8, UINT16, UINT32, UINT64, FP16, FP32, FP64, BOOL

### TensorType

多维张量，可选内存引用。

```python
# 形状为 [10, 20] 的张量
shape = [ir.ConstInt(10, DataType.INT64, span), ir.ConstInt(20, DataType.INT64, span)]
tensor_type = ir.TensorType(shape, DataType.FP32)

# 带 MemRef 的张量
memref = ir.MemRef(ir.MemorySpace.DDR, ir.ConstInt(0x1000, DataType.INT64, span), 800)
tensor_with_memref = ir.TensorType(shape, DataType.FP32, memref)
```

### TileType

专用张量，带有可选的内存和视图信息，用于硬件优化操作。

```python
# 基本的 16x16 tile
shape = [ir.ConstInt(16, DataType.INT64, span)] * 2
tile_type = ir.TileType(shape, DataType.FP16)

# 3D tile（在 IR 层面支持）
shape_3d = [ir.ConstInt(4, DataType.INT64, span),
            ir.ConstInt(16, DataType.INT64, span),
            ir.ConstInt(16, DataType.INT64, span)]
tile_type_3d = ir.TileType(shape_3d, DataType.FP16)

# 带 MemRef 和 TileView 的 Tile
memref = ir.MemRef(ir.MemorySpace.L0A, ir.ConstInt(0, DataType.INT64, span), 512)

tile_view = ir.TileView()
tile_view.valid_shape = [ir.ConstInt(16, DataType.INT64, span)] * 2
tile_view.stride = [ir.ConstInt(1, DataType.INT64, span), ir.ConstInt(16, DataType.INT64, span)]
tile_view.start_offset = ir.ConstInt(0, DataType.INT64, span)

tile_with_view = ir.TileType(shape, DataType.FP16, memref, tile_view)
```

### TupleType

异构类型元组。

```python
# 标量元组: (int, float)
scalar_tuple = ir.TupleType([
    ir.ScalarType(DataType.INT64),
    ir.ScalarType(DataType.FP32)
])

# 嵌套元组
nested = ir.TupleType([
    ir.TupleType([ir.ScalarType(DataType.INT64)]),
    ir.ScalarType(DataType.FP32)
])
```

### PipeType

硬件执行流水线或同步屏障。

```python
pipe_s = ir.PipeType(ir.PipeType.S)    # 标量流水线
pipe_v = ir.PipeType(ir.PipeType.V)    # 向量流水线
pipe_m = ir.PipeType(ir.PipeType.M)    # 矩阵流水线
pipe_all = ir.PipeType(ir.PipeType.ALL) # 所有流水线
```

### UnknownType

未知或推断类型的占位符。

```python
unknown = ir.UnknownType()
```

### MemRefType

独立的内存引用类型节点,用于表示内存引用本身作为一个类型。

```python
# 创建 MemRefType
memref_type = ir.MemRefType()

# MemRefType 通常用于类型系统中表示内存引用
# 注意: MemRef 结构体(非类型)用作 TensorType/TileType 的可选字段
```

**区别说明**:
- **MemRefType**: 独立的类型节点,继承自 Type,用于类型系统
- **MemRef**: 结构体,用作 TensorType/TileType 的 `memref_` 字段,描述具体的内存分配

### MemorySpace 枚举

| 值 | 描述 |
|-------|-------------|
| `DDR` | 主内存（片外） |
| `UB` | 统一缓冲区（片上共享内存） |
| `L1` | L1 缓存 |
| `L0A` | L0A 缓冲区（矩阵 A） |
| `L0B` | L0B 缓冲区（矩阵 B） |
| `L0C` | L0C 缓冲区（矩阵 C/结果） |

## Python 使用示例

### 示例 1：构建表达式

```python
from pypto import DataType, ir

span = ir.Span.unknown()
dtype = DataType.INT64

# 变量和常量
x = ir.Var("x", ir.ScalarType(dtype), span)
y = ir.Var("y", ir.ScalarType(dtype), span)
one = ir.ConstInt(1, dtype, span)
two = ir.ConstInt(2, dtype, span)

# 构建: ((x + 1) * (y - 2)) / (x + y)
x_plus_1 = ir.Add(x, one, dtype, span)
y_minus_2 = ir.Sub(y, two, dtype, span)
numerator = ir.Mul(x_plus_1, y_minus_2, dtype, span)
denominator = ir.Add(x, y, dtype, span)
result = ir.FloatDiv(numerator, denominator, dtype, span)
```

### 示例 2：控制流（绝对值）

```python
# if (x >= 0) then { result = x } else { result = -x }
x = ir.Var("x", ir.ScalarType(dtype), span)
result = ir.Var("result", ir.ScalarType(dtype), span)
zero = ir.ConstInt(0, dtype, span)

condition = ir.Ge(x, zero, dtype, span)
then_assign = ir.AssignStmt(result, x, span)
else_assign = ir.AssignStmt(result, ir.Neg(x, dtype, span), span)

abs_stmt = ir.IfStmt(condition, then_assign, else_assign, [result], span)
```

### 示例 3：带累加的循环

```python
# for i, (sum,) in pl.range(0, n, 1, init_values=[0]):
#     sum = pl.yield_(sum + i)

n = ir.Var("n", ir.ScalarType(dtype), span)
i = ir.Var("i", ir.ScalarType(dtype), span)
zero = ir.ConstInt(0, dtype, span)
one = ir.ConstInt(1, dtype, span)

sum_iter = ir.IterArg("sum", ir.ScalarType(dtype), zero, span)
add_expr = ir.Add(sum_iter, i, dtype, span)
yield_stmt = ir.YieldStmt([add_expr], span)
sum_final = ir.Var("sum_final", ir.ScalarType(dtype), span)

loop = ir.ForStmt(i, zero, n, one, [sum_iter], yield_stmt, [sum_final], span)
```

### 示例 4：带算子调用的函数

```python
# def matmul(a, b) -> tensor:
#     result = tensor.matmul(a, b, out_dtype=FP32)

shape_m = ir.ConstInt(128, DataType.INT64, span)
shape_k = ir.ConstInt(64, DataType.INT64, span)
shape_n = ir.ConstInt(256, DataType.INT64, span)

a = ir.Var("a", ir.TensorType([shape_m, shape_k], DataType.FP16), span)
b = ir.Var("b", ir.TensorType([shape_k, shape_n], DataType.FP16), span)

matmul_call = ir.op.tensor.matmul(a, b, out_dtype=DataType.FP32)
result = ir.Var("result", ir.TensorType([shape_m, shape_n], DataType.FP32), span)
body = ir.AssignStmt(result, matmul_call, span)

return_types = [ir.TensorType([shape_m, shape_n], DataType.FP32)]
func = ir.Function("matmul", [a, b], return_types, body, span)
```

### 示例 5：包含多个函数的程序

```python
# 辅助函数: square(x) -> int { return x * x }
x = ir.Var("x", ir.ScalarType(dtype), span)
square_result = ir.Var("result", ir.ScalarType(dtype), span)
square_body = ir.AssignStmt(square_result, ir.Mul(x, x, dtype, span), span)
square_func = ir.Function("square", [x], [ir.ScalarType(dtype)], square_body, span)

# 主函数: sum_squares(a, b) -> int { return square(a) + square(b) }
a = ir.Var("a", ir.ScalarType(dtype), span)
b = ir.Var("b", ir.ScalarType(dtype), span)

program = ir.Program([square_func], "math", span)
square_gvar = program.get_global_var("square")

call_a = ir.Call(square_gvar, [a], span)
call_b = ir.Call(square_gvar, [b], span)
sum_expr = ir.Add(call_a, call_b, dtype, span)

main_result = ir.Var("result", ir.ScalarType(dtype), span)
main_body = ir.AssignStmt(main_result, sum_expr, span)
main_func = ir.Function("sum_squares", [a, b], [ir.ScalarType(dtype)], main_body, span)

program = ir.Program([square_func, main_func], "math", span)
```

### 示例 6：带 TileType 的内存布局

```python
# L0A 内存中的 32x32 tile，带自定义步长
shape = [ir.ConstInt(32, DataType.INT64, span)] * 2
memref = ir.MemRef(ir.MemorySpace.L0A, ir.ConstInt(0, DataType.INT64, span), 2048)

tile_view = ir.TileView()
tile_view.valid_shape = shape
tile_view.stride = [ir.ConstInt(1, DataType.INT64, span), ir.ConstInt(32, DataType.INT64, span)]
tile_view.start_offset = ir.ConstInt(0, DataType.INT64, span)

tile_type = ir.TileType(shape, DataType.FP16, memref, tile_view)
```

## 类型系统总结

| 类型 | 维度 | 内存信息 | 使用场景 |
|------|------------|-------------|----------|
| **ScalarType** | 0 | - | 单个值 |
| **TensorType** | N（任意） | 可选 MemRef | 通用张量 |
| **TileType** | N（任意）* | 可选 MemRef + TileView | 硬件优化的 tile |
| **TupleType** | - | - | 多返回值 |
| **PipeType** | - | - | 硬件同步 |
| **UnknownType** | - | - | 类型推断占位符 |

## 常见模式

**创建常量：**
```python
i32 = ir.ConstInt(42, DataType.INT32, span)
f32 = ir.ConstFloat(3.14, DataType.FP32, span)
```

**创建算子：**
```python
# 高级 API（推荐）
call = ir.op.tensor.matmul(a, b, out_dtype=DataType.FP32)

# 带 kwargs 的通用算子
call = ir.create_op_call("tensor.matmul", [a, b], {"out_dtype": DataType.FP32}, span)
```

**语句序列：**
```python
seq = ir.SeqStmts([stmt1, stmt2, stmt3], span)
```

## 类型检查和转换

```python
# 检查表达式类型
if isinstance(expr, ir.Var):
    print(expr.name_)

# 检查类型对象
if isinstance(type_obj, ir.TileType):
    # 访问 tile 特定的属性
    shape = type_obj.shape
```

## 相关文档

- [IR 概述](00-ir_overview_zh.md) - 核心概念和设计原则
- [IR 节点层次结构](01-ir_hierarchy_zh.md) - 完整的节点类型参考
- [结构化比较](03-structural_comparison_zh.md) - 相等性和哈希工具

## 总结

PyPTO 的类型系统提供：
- **标量类型** 用于基本值
- **张量/Tile 类型** 用于带内存布局的多维数据
- **元组类型** 用于异构集合
- **流水线类型** 用于硬件同步

IR 构建 API 支持：
- 使用共享指针创建不可变节点
- 带编译时检查的类型安全操作
- 通过 MemRef 和 TileView 进行硬件感知的内存管理
- 通过 GlobalVar 进行程序内函数调用
- 通过 IterArg 进行循环携带依赖
