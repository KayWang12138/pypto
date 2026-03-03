# PyPTO IR 概述

## 概述

PyPTO 的中间表示（IR）是一种基于树的不可变数据结构，用于在编译过程中表示程序。IR 是程序转换、优化和代码生成的基础。

**核心设计原则：**

1. **不可变性**：所有 IR 节点一旦构造完成即不可变
2. **树结构**：形成有向无环图（DAG），节点可以被多个父节点共享
3. **共享指针**：所有节点通过 `std::shared_ptr<const T>` 管理
4. **引用相等性**：默认的 `==` 比较指针地址；使用 `structural_equal()` 进行结构比较

## 核心概念

### 源位置跟踪

每个 IR 节点都包含一个 `Span` 对象来跟踪其源位置：

```python
from pypto import ir

# 创建用于源位置跟踪的 span
span = ir.Span("example.py", 10, 5, 10, 20)
print(span.filename)      # "example.py"
print(span.begin_line)    # 10

# 当位置不可用时创建未知 span
unknown_span = ir.Span.unknown()
```

### 字段描述符和反射

IR 节点使用反射系统进行通用遍历。每个节点定义三种类型的字段：

| 字段类型 | 用途 | 示例用法 |
|---------|------|---------|
| **IgnoreField** | 遍历时忽略 | `Span`（源位置） |
| **DefField** | 引入新绑定的定义字段 | 循环变量、赋值目标 |
| **UsualField** | 正常遍历的常规字段 | 表达式操作数、语句体 |

```cpp
// 示例：AssignStmt 字段描述符
static constexpr auto GetFieldDescriptors() {
  return std::tuple_cat(
    Stmt::GetFieldDescriptors(),
    std::make_tuple(
      reflection::DefField(&AssignStmt::var_, "var"),      // 定义
      reflection::UsualField(&AssignStmt::value_, "value") // 普通字段
    )
  );
}
```

## 基于 Kind 机制的类型识别

PyPTO IR 使用高效的**基于 Kind 的类型识别机制**来避免 C++ RTTI（`dynamic_cast`）的开销。这提供了 O(1) 的类型检查和转换，零运行时开销。

### ObjectKind 枚举

所有 IR 节点类型都在统一的枚举中表示：

| 类别 | 种类 |
|------|------|
| **基类** | IRNode, Expr, Stmt, Type |
| **表达式** | Var, IterArg, Call, TupleGetItemExpr, ConstInt, ConstFloat, ConstBool |
| **二元操作** | Add, Sub, Mul, FloorDiv, FloorMod, FloatDiv, Min, Max, Pow, Eq, Ne, Lt, Le, Gt, Ge, And, Or, Xor, BitAnd, BitOr, BitXor, BitShiftLeft, BitShiftRight |
| **一元操作** | Abs, Neg, Not, BitNot, Cast |
| **语句** | AssignStmt, IfStmt, YieldStmt, ReturnStmt, ForStmt, SeqStmts, OpStmts, EvalStmt |
| **类型** | UnknownType, ScalarType, ShapedType, TensorType, TileType, TupleType, PipeType |
| **其他** | Function, Program, Op, GlobalVar |

### GetKind() 虚方法

每个 IR 节点都实现 `GetKind()` 方法：

```cpp
class IRNode {
 public:
  [[nodiscard]] virtual ObjectKind GetKind() const = 0;
};

class Var : public Expr {
 public:
  [[nodiscard]] ObjectKind GetKind() const override {
    return ObjectKind::Var;
  }
};
```

### 使用 IsA<T>() 进行类型检查

使用 `IsA<T>()` 检查节点是否为特定类型：

```cpp
#include "pypto/ir/kind_traits.h"

ExprPtr expr = ...;

// 检查 expr 是否为 Var
if (IsA<Var>(expr)) {
  // expr 是 Var
}

// 检查 expr 是否为 ConstInt
if (IsA<ConstInt>(expr)) {
  // expr 是 ConstInt
}

// 也适用于 TypePtr
TypePtr type = expr->GetType();
if (IsA<TileType>(type)) {
  // type 是 TileType
}
```

### 使用 As<T>() 进行类型转换

使用 `As<T>()` 安全地将节点转换为其具体类型：

```cpp
#include "pypto/ir/kind_traits.h"

ExprPtr expr = ...;

// 转换为 Var（如果不是 Var 则返回 nullptr）
if (auto var = As<Var>(expr)) {
  std::cout << "变量名: " << var->name_ << std::endl;
}

// 转换 ConstInt
if (auto const_int = As<ConstInt>(expr)) {
  std::cout << "整数值: " << const_int->value_ << std::endl;
}

// 类型转换
TypePtr type = expr->GetType();
if (auto tile_type = As<TileType>(type)) {
  // 访问 tile 特定属性
  auto shape = tile_type->GetShape();
}
```

**主要优势：**

- **O(1) 性能**：单次虚函数调用 vs. 多次 `dynamic_cast` 尝试
- **类型安全**：转换失败时返回 `nullptr`，无异常
- **简洁语法**：`IsA<T>()` 和 `As<T>()` 比 `dynamic_pointer_cast` 更易读
- **零开销**：编译器在许多情况下可以优化掉虚函数调用

## IRNode - 基类

```cpp
class IRNode {
  Span span_;                           // 源位置（IgnoreField）
  virtual ObjectKind GetKind() const;   // 返回节点的 kind 用于 O(1) 类型检查
  virtual std::string TypeName() const; // 返回节点类型名称（用于调试）
};
```

所有 IR 节点都继承自 `IRNode` 并且必须实现：
- `GetKind()`：返回节点的 `ObjectKind` 用于类型识别
- `TypeName()`：返回人类可读的类型名称（例如 "Var"、"AssignStmt"）

## 表达式基类

```cpp
class Expr : public IRNode {
  TypePtr type_;  // 表达式的结果类型
};
```

所有表达式都产生一个具有关联类型的值。

## 语句基类

```cpp
class Stmt : public IRNode {
  // 语句表示动作但不产生值
};
```

语句表示程序动作，如赋值、控制流和循环。

## 类型基类

```cpp
class Type : public IRNode {
  // 所有类型表示的基类
};
```

类型描述 IR 中数据的结构和属性。

## Python 使用模式

```python
from pypto import DataType, ir

# 创建基本 IR 节点
span = ir.Span.unknown()
dtype = DataType.INT64

# 变量
x = ir.Var("x", ir.ScalarType(dtype), span)
y = ir.Var("y", ir.ScalarType(dtype), span)

# 常量
one = ir.ConstInt(1, dtype, span)
pi = ir.ConstFloat(3.14, DataType.FP32, span)
flag = ir.ConstBool(True, span)

# 表达式
sum_expr = ir.Add(x, one, dtype, span)
product = ir.Mul(x, y, dtype, span)

# 语句
assign = ir.AssignStmt(x, sum_expr, span)
```

## 设计理念

**不可变性的好处：**
- 跨转换的线程安全共享
- 结构共享减少内存使用
- 更安全地推理程序语义

**Kind 机制的好处：**
- 无 RTTI 开销的快速类型检查
- 支持高效的访问者模式
- 支持通用转换和分析

**反射系统的好处：**
- 无代码重复的通用树遍历
- 结构相等性和哈希
- 美化打印和序列化

## 相关文档

- [IR 节点层次结构](01-ir_hierarchy_zh.md) - 完整的节点类型参考
- [IR 类型和示例](02-ir_types_examples_zh.md) - 类型系统和使用示例
- [结构比较](03-structural_comparison_zh.md) - 相等性和哈希工具

## 总结

PyPTO IR 提供：
- **不可变树结构**用于安全转换
- **高效的类型识别**通过 Kind 机制实现 O(1) 性能
- **基于反射的遍历**支持访问者、变换器和结构比较
- **Python 友好的 API**用于 IR 构造
- **源位置跟踪**用于错误报告
- **三层字段系统**（Ignore、Def、Usual）用于灵活遍历
