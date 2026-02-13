# DSL 函数的 IR 解析器

## 概述

IR 解析器使用装饰器（`@pl.function`、`@pl.program`）将 Python DSL 代码转换为 PyPTO IR。它强制执行 SSA 属性，跟踪源代码位置，并支持嵌套控制流。

**核心组件**：装饰器 → AST 解析器 → IR 构建器 → 作用域管理器（SSA）→ ir.Function

参见 [IR 构建器](08-ir_builder_zh.md) 了解手动 IR 构造，参见 [Python IR 语法](07-python_syntax_zh.md) 了解完整语法。

## 使用方法

### 基本函数

```python
import pypto
import pypto.language as pl

@pl.function
def simple_add(
    x: pl.Tensor[[64, 128], pl.FP16],
    y: pl.Tensor[[64, 128], pl.FP16],
) -> pl.Tensor[[64, 128], pl.FP16]:
    result: pl.Tensor[[64, 128], pl.FP16] = pl.add(x, y)
    return result

# simple_add 现在是一个 ir.Function 对象
assert isinstance(simple_add, pypto.ir.Function)
```

### 类型注解

所有参数和局部变量都需要类型注解：

```python
x: pl.Tensor[[64, 128], pl.FP16]  # 推荐的下标语法
x: pl.Tensor((64, 128), pl.FP16)  # 旧版调用语法（也可接受）
```

两种语法是等价的；打印器始终输出下标表示法。

### 带迭代参数的 For 循环

使用 `pl.range()` 配合元组解包来处理循环携带值（iter_args）：

```python
for i, (sum_val,) in pl.range(10, init_values=[sum_init]):
    new_sum: pl.Tensor[[1], pl.INT32] = pl.add(sum_val, i)
    sum_out = new_sum
```

**语法**：`loop_var, (iter_arg1, ...)` - iter_args 的数量必须与 init_values 匹配。

## SSA 属性

解析器强制执行静态单赋值（Static Single Assignment）：

**单次赋值**：每个变量在每个作用域内只能赋值一次
```python
# ✓ 有效
y: pl.Tensor[[64], pl.FP32] = pl.add(x, 1.0)

# ✗ 无效 - SSA 违规
y: pl.Tensor[[64], pl.FP32] = pl.add(x, 1.0)
y = pl.mul(x, 2.0)  # 错误：y 已经定义
```

**作用域隔离**：内部作用域的变量必须通过返回值传递
```python
# ✗ 无效 - temp 未返回
for i, (sum_val,) in pl.range(10, init_values=[x]):
    temp: pl.Tensor[[64], pl.FP32] = pl.add(sum_val, i)
return temp  # 错误：temp 不在外部作用域

# ✓ 有效 - 通过迭代参数传递
for i, (sum_val,) in pl.range(10, init_values=[x]):
    result: pl.Tensor[[64], pl.FP32] = pl.add(sum_val, i)
return result  # 正确
```

**迭代参数**：通过 phi 节点在每次迭代中创建新的 SSA 值。

## Span 跟踪和操作

**Span 跟踪**：保留源代码位置以提供更好的错误消息
- 每个 IR 节点包含带有文件名、行/列范围的 `Span`
- 支持调试、错误报告和源代码到 IR 的映射

**支持的操作**：

| 类别 | 示例 |
|----------|----------|
| **张量操作** | `pl.{add, mul, sub, div, matmul, cast, view, ...}` |
| **二元表达式** | `a + b`、`a - b`、`a * b`、`a / b`、`i == 0`、`x < 10` |
| **字面量** | `42` → `ConstInt`、`3.14` → `ConstFloat` |

参见 [Python IR 语法](07-python_syntax_zh.md) 获取完整的操作列表。

## 完整示例

嵌套控制流示例：

```python
@pl.function
def flash_attn_simplified(
    q: pl.Tensor[[64, 128], pl.FP16],
    k: pl.Tensor[[1024, 128], pl.FP16],
) -> pl.Tensor[[64, 128], pl.FP32]:
    attn_init: pl.Tensor[[64, 128], pl.FP32] = pl.create([64, 128], dtype=pl.FP32)

    for i, (attn,) in pl.range(16, init_values=[attn_init]):
        k_block: pl.Tensor[[64, 128], pl.FP16] = pl.view(k, [64, 128], [i * 64, 0])
        scores: pl.Tensor[[64, 128], pl.FP16] = pl.matmul(q, k_block, b_trans=True)

        if i == 0:
            new_attn: pl.Tensor[[64, 128], pl.FP32] = pl.cast(scores, target_type=pl.FP32)
            result = new_attn
        else:
            updated: pl.Tensor[[64, 128], pl.FP32] = pl.add(attn, scores)
            result = updated

        final = result

    return final
```

## 使用 @pl.program 的多函数程序

定义包含多个可以相互调用的函数的程序：

```python
@pl.program
class MathOps:
    @pl.function
    def square(self, x: pl.Tensor[[1], pl.INT32]) -> pl.Tensor[[1], pl.INT32]:
        result: pl.Tensor[[1], pl.INT32] = pl.mul(x, x)
        return result

    @pl.function
    def sum_of_squares(self, a: pl.Tensor[[1], pl.INT32], b: pl.Tensor[[1], pl.INT32]) -> pl.Tensor[[1], pl.INT32]:
        a_squared: pl.Tensor[[1], pl.INT32] = self.square(a)  # 跨函数调用
        b_squared: pl.Tensor[[1], pl.INT32] = self.square(b)
        result: pl.Tensor[[1], pl.INT32] = pl.add(a_squared, b_squared)
        return result
```

**关键规则**：
- 使用 `@pl.program` 的基于类的语法
- 方法需要 `self` 参数（在 IR 中自动剥离）
- 跨函数调用使用 `self.method_name()` → 解析为 `GlobalVar` 引用
- 两遍解析：收集 `GlobalVar`，然后解析函数体（支持前向引用）
- 访问函数：`program.get_function("name")`
- 打印：`pypto.ir.python_print(program)` 生成有效的 `@pl.program` 类

**示例**：参见 `examples/ir_parser/program_example.py` 和 `examples/ir_builder/program_builder_example.py`

## 限制和测试

**当前限制**：
- if 条件中仅支持标量比较（不支持张量）
- `@pl.function` 内部不支持嵌套函数定义
- 有限的 Python 子集（函数内不支持类、装饰器）
- 所有变量都需要类型注解

**测试**：运行 `pytest tests/ut/language/parser/` 进行全面的解析器测试。

## 另请参阅

- [Python IR 语法](07-python_syntax_zh.md) - 完整语法规范
- [IR 构建器](08-ir_builder_zh.md) - 手动 IR 构造 API
- [IR 概述](00-ir_overview_zh.md) - 核心 IR 概念
