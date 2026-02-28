# Python IR 语法规范

## 概述

PyPTO IR 的 Python 风格语法：
- **完整性**：包含重建 IR 所需的所有信息
- **可解析**：可以解析回 IR（参见 [IR 解析器](09-ir_parser_zh.md)）
- **Python 风格**：遵循 Python 风格，通过大多数代码检查工具
- **SSA 风格**：使用 SSA 形式，配合 `pl.yield_()` 和 `pl.range()`

## 模块结构

```python
# pypto.program: program_name
import pypto.language as pl
```

对于未命名的程序：`# pypto.program`

**注意：** 模块前缀可配置（默认 `pl`，旧版 `ir`，允许自定义）。

## 类型系统

### 标量类型

```python
x: pl.INT64
y: pl.FP32
z: pl.BOOL
```

可用类型：

| 类别 | 类型 |
|----------|-------|
| **整数** | `INT4`, `INT8`, `INT16`, `INT32`, `INT64` |
| **无符号整数** | `UINT4`, `UINT8`, `UINT16`, `UINT32`, `UINT64` |
| **浮点数** | `FP4`, `FP8`, `FP16`, `FP32` |
| **Brain Float** | `BF16` |
| **海思** | `HF4`, `HF8` |
| **布尔** | `BOOL` |

### 张量和块类型

```python
# 张量（下标表示法）
a: pl.Tensor[[4, 8], pl.FP32]      # 固定形状
b: pl.Tensor[[n, m], pl.INT64]     # 符号形状

# 块（统一缓冲区中的块）
t: pl.Tile[[16, 16], pl.FP16]
```

### 内存引用（MemRef）

```python
# 创建 MemRef
addr_expr = pl.ConstInt(0x1000, pl.INT64, span)
memref = pl.MemRef(pl.MemorySpace.DDR, addr_expr, 1024)

# 内存空间：DDR, UB, L1, L0A, L0B, L0C

# 带 memref 的张量
tensor: pl.Tensor[[64, 128], pl.FP32], memref=pl.MemRef(pl.MemorySpace.DDR, addr, 8192))
```

### 块视图（TileView）

```python
# 创建 TileView
valid_shape = [pl.ConstInt(16, pl.INT64, span)] * 2
stride = [pl.ConstInt(1, pl.INT64, span), pl.ConstInt(16, pl.INT64, span)]
start_offset = pl.ConstInt(0, pl.INT64, span)
tile_view = pl.TileView(valid_shape=valid_shape, stride=stride, start_offset=start_offset)

# 带 memref 和 tile_view 的块
tile: pl.Tile(
    (16, 16), pl.FP16,
    memref=pl.MemRef(pl.MemorySpace.L0A, addr, 512),
    tile_view=pl.TileView(valid_shape=..., stride=..., start_offset=...)
)
```

## 表达式

### 变量和常量

```python
x              # 变量引用
tensor_a       # 张量变量
42             # 整数字面量
3.14           # 浮点数字面量
```

### 二元运算

| Python 运算符 | PyPTO IR | 类别 |
|----------------|----------|----------|
| `+` | Add | 算术 |
| `-` | Sub | 算术 |
| `*` | Mul | 算术 |
| `//` | FloorDiv | 算术 |
| `%` | FloorMod | 算术 |
| `/` | FloatDiv | 算术 |
| `**` | Pow | 算术 |
| `==`, `!=`, `<`, `<=`, `>`, `>=` | Eq, Ne, Lt, Le, Gt, Ge | 比较 |
| `and`, `or` | And, Or | 逻辑 |
| `^` | Xor | 逻辑 |
| `&`, `|` | BitAnd, BitOr | 位运算 |
| `<<`, `>>` | BitShiftLeft, BitShiftRight | 位运算 |

### 一元运算和函数

```python
-x              # Neg
~x              # BitNot
not x           # Not
abs(x)          # Abs
min(a, b)       # Min
max(a, b)       # Max
```

### 函数/操作调用

```python
# 显式命名空间
pl.tensor.add(a, b)                  # 张量加法
pl.block.load(t, [0, 0], [64, 64])      # 块加载

# 统一分发（根据输入类型自动选择 tensor/block）
pl.add(a, b)                          # 张量或块 — 自动分发
pl.mul(tile, 2.0)                     # 块 + 标量 → block.muls
pl.exp(tile)                          # 块 → block.exp

# 提升的操作（单模块操作可通过 pl.* 访问）
pl.load(t, [0, 0], [64, 64])            # 从 block 提升
pl.create([64], dtype=pl.FP32)       # 从 tensor 提升
```

## 语句

### 赋值

```python
x: pl.INT64 = expr
y: pl.Tensor[[4], pl.FP32] = tensor_op(a)
```

### If 语句（SSA 风格）

```python
# 带两个分支的 if
if condition:
    y1 = pl.yield_(value1)
else:
    y1 = pl.yield_(value2)

# 多返回值（不使用内联类型注解）
if condition:
    y1, y2 = pl.yield_(value1, value2)
else:
    y1, y2 = pl.yield_(value3, value4)
```

**要点：**
- `pl.yield_()` 赋值给 SSA phi 节点
- yield 中定义的变量在 if 之后可访问
- 两个分支必须 yield 相同的变量
- 元组解包时不能使用类型注解

### For 循环（带 iter_args 的 SSA 风格）

```python
# 简单循环
for i in pl.range(start, stop, step):
    body_statements

# 带 iter_args 的循环（循环携带值）
sum_init: pl.INT64 = 0
for i, (sum,) in pl.range(0, n, 1, init_values=[sum_init]):
    sum = pl.yield_(sum + i)
sum_final = sum

# 并行 for 循环
for i in pl.parallel(start, stop, step):
    body_statements
```

**要点：** 循环携带值使用带 `init_values` 的 `pl.range()` 或 `pl.parallel()`，元组解包 `(sum,)` 声明 iter_args，`pl.yield_()` 更新下一次迭代的值，循环后 iter_args 包含最终值。`pl.parallel()` 产生 `ForKind.Parallel` 循环，而 `pl.range()` 产生 `ForKind.Sequential`（默认）。

### Yield 语句

```python
yield            # 无值
yield x          # 单个值
yield x, y       # 多个值
```

### 语句序列

```python
stmt1            # 自然的 Python 顺序
stmt2
stmt3
```

## 函数

```python
# 单返回类型
def function_name(param1: pl.INT64, param2: pl.FP32) -> pl.INT64:
    x: pl.INT64 = param1 + 1
    return x

# 多返回类型
def function_name(x: pl.INT64) -> tuple[pl.INT64, pl.INT64]:
    y: pl.INT64 = x + 1
    z: pl.INT64 = x * 2
    return y, z

# 无返回类型
def function_name(x: pl.INT64):
    y: pl.INT64 = x + 1

# 带函数类型
@pl.function(type=pl.FunctionType.Orchestration)
def orchestrator(n: pl.INT64) -> pl.INT64:
    return n + 1

@pl.function(type=pl.FunctionType.InCore)
def aicore_kernel(x: pl.INT64) -> pl.INT64:
    return x * 2
```

### 函数类型

| 类型 | 用途 | 描述 |
|------|-------|-------------|
| `pl.FunctionType.Opaque` | 默认 | 未指定函数类型 |
| `pl.FunctionType.Orchestration` | Host/AICPU | 控制流和依赖分析 |
| `pl.FunctionType.InCore` | AICore | 特定 AICore 上的子图 |

当未指定类型时，函数默认为 `Opaque`。

## 完整示例

### 张量操作（带 iter_args 的循环）

```python
# pypto.program: my_program
import pypto.language as pl

def loop_sum(n: pl.INT64) -> pl.INT64:
    sum_init: pl.INT64 = 0
    for i, (sum,) in pl.range(0, n, 1, init_values=[sum_init]):
        sum = pl.yield_(sum + i)
    return sum
```

### 块操作（基于块的计算）

```python
import pypto.language as pl

@pl.program
class BlockExample:
    @pl.function
    def tile_add(
        self,
        input_a: pl.Tensor[[64, 64], pl.FP32],
        input_b: pl.Tensor[[64, 64], pl.FP32],
        output: pl.Tensor[[64, 64], pl.FP32],
    ) -> pl.Tensor[[64, 64], pl.FP32]:
        tile_a: pl.Tile[[64, 64], pl.FP32] = pl.load(input_a, [0, 0], [64, 64])
        tile_b: pl.Tile[[64, 64], pl.FP32] = pl.load(input_b, [0, 0], [64, 64])
        tile_c: pl.Tile[[64, 64], pl.FP32] = pl.add(tile_a, tile_b)
        result: pl.Tensor[[64, 64], pl.FP32] = pl.store(tile_c, [0, 0], [64, 64], output)
        return result
```

## SSA 风格控制流

`pl.yield_()` 为 if/for 语句创建 SSA phi 节点：

```python
# If：在合并点创建 phi 节点
if condition:
    y1 = pl.yield_(x + 1)
else:
    y1 = pl.yield_(x + 2)
# y1 = phi(x + 1, x + 2)

# For：通过 iter_args 传递循环携带值
sum_init: pl.INT64 = 0
for i, (sum,) in pl.range(0, 10, 1, init_values=[sum_init]):
    sum = pl.yield_(sum + i)
sum_final: pl.INT64 = sum  # 捕获最终值
```

## 可配置的模块前缀

打印器支持可配置的模块前缀（`pl`、`pi`、`ir` 或自定义）：

```python
print(ir.python_print(stmt))          # "x: pl.INT64 = a + b" (默认)
print(ir.python_print(stmt, "ir"))    # "x: ir.INT64 = a + b"
```

## 参考

- [IR 概述](00-ir_overview_zh.md) - 核心 IR 结构
- [IR 解析器](09-ir_parser_zh.md) - 将 Python 语法解析回 IR
- [算子注册](05-operator_registration_zh.md) - 操作系统和类型推断
