# PyPTO IR 节点层次结构

本文档提供了所有 IR 节点类型的完整参考，按类别组织。

## BNF 语法

```bnf
<program>    ::= [ identifier ":" ] { <function> }
<function>   ::= "def" identifier "(" [ <param_list> ] ")" [ "->" <type_list> ] ":" <stmt>
<param_list> ::= <var> { "," <var> }
<type_list>  ::= <type> { "," <type> }

<stmt>       ::= <assign_stmt> | <if_stmt> | <for_stmt> | <yield_stmt>
               | <eval_stmt> | <seq_stmts> | <op_stmts> | <return_stmt>

<assign_stmt> ::= <var> "=" <expr>
<if_stmt>    ::= "if" <expr> ":" <stmt_list> [ "else" ":" <stmt_list> ] [ "return" <var_list> ]
<for_stmt>   ::= "for" <var> [ "," "(" <iter_arg_list> ")" ] "in"
                 ( "range" | "pl.range" ) "(" <expr> "," <expr> "," <expr>
                 [ "," "init_values" "=" "[" <expr_list> "]" ] ")" ":" <stmt_list>
                 [ <return_assignments> ]

<yield_stmt> ::= "yield" [ <var_list> ]
<eval_stmt>  ::= <expr>
<seq_stmts>  ::= <stmt> { ";" <stmt> }
<op_stmts>   ::= <assign_stmt> { ";" <assign_stmt> }

<expr>       ::= <var> | <const_int> | <const_bool> | <const_float> | <call>
               | <binary_op> | <unary_op> | <tuple_get_item>

<call>       ::= <op> "(" [ <expr_list> ] ")"
<op>         ::= identifier | <global_var>

<type>       ::= <scalar_type> | <tensor_type> | <tile_type>
               | <tuple_type> | <pipe_type> | <unknown_type>

<scalar_type> ::= "ScalarType" "(" <data_type> ")"
<tensor_type> ::= "TensorType" "(" <data_type> "," <shape> [ "," <memref> ] ")"
<tile_type>   ::= "TileType" "(" <data_type> "," <shape> [ "," <memref> [ "," <tile_view> ] ] ")"
<tuple_type>  ::= "TupleType" "(" "[" <type_list> "]" ")"
<pipe_type>   ::= "PipeType" "(" <pipe_kind> ")"

<shape>       ::= "[" <expr_list> "]"
<data_type>   ::= "INT32" | "INT64" | "FP16" | "FP32" | "FP64" | "BOOL" | ...
<pipe_kind>   ::= "S" | "V" | "M" | "MTE1" | "MTE2" | "MTE3" | "ALL" | ...
```

## 表达式节点

| 节点类型 | 字段 | 描述 |
|-----------|--------|-------------|
| **Var** | `name_`, `type_` | 变量引用 |
| **IterArg** | `name_`, `type_`, `initValue_` | 循环迭代参数（扩展自 Var） |
| **ConstInt** | `value_`, `dtype_` | 整数常量 |
| **ConstBool** | `value_` | 布尔常量（始终为 BOOL 数据类型） |
| **ConstFloat** | `value_`, `dtype_` | 浮点常量 |
| **Call** | `op_`, `args_`, `kwargs_` | 函数/操作符调用 |
| **MakeTuple** | `fields_` | 从表达式列表创建元组 |
| **TupleGetItemExpr** | `tuple_`, `index_` | 元组元素访问 |

### 二元表达式节点

| 类别 | 节点 |
|----------|-------|
| **算术运算** | Add, Sub, Mul, FloorDiv, FloorMod, FloatDiv |
| **数学运算** | Min, Max, Pow |
| **比较运算** | Eq, Ne, Lt, Le, Gt, Ge |
| **逻辑运算** | And, Or, Xor |
| **位运算** | BitAnd, BitOr, BitXor, BitShiftLeft, BitShiftRight |

所有二元表达式都有：`lhs_`, `rhs_`, `dtype_`

### 一元表达式节点

| 节点 | 操作 |
|------|-----------|
| **Abs** | 绝对值 |
| **Neg** | 取负 |
| **Not** | 逻辑非 |
| **BitNot** | 按位取反 |
| **Cast** | 类型转换 |

所有一元表达式都有：`operand_`, `dtype_`

### Op 和 GlobalVar

| 节点类型 | 用途 | 使用场景 |
|-----------|---------|-------|
| **Op** | 通用操作/函数引用 | 外部操作符、内置函数 |
| **GlobalVar** | 程序内的函数引用 | 程序内函数调用 |

```python
# Op 用于外部函数
op = ir.Op("my_function")
call = ir.Call(op, [x, y], span)

# GlobalVar 用于程序函数
gvar = ir.GlobalVar("helper")  # 必须与函数名匹配
call = ir.Call(gvar, [x], span)
```

### IterArg - 循环携带值

`IterArg` 是用于 SSA 风格循环迭代的特殊变量：

**关键属性：**
- 扩展自 `Var`，增加了 `initValue_` 字段
- 作用域仅限于循环体内
- 通过 `yield` 语句更新
- 最终值在 `return_vars` 中捕获

```python
# for i, (sum,) in pl.range(0, n, 1, init_values=[0]):
#     sum = pl.yield_(sum + i)
# sum_final = sum

init_val = ir.ConstInt(0, DataType.INT64, span)
sum_iter = ir.IterArg("sum", ir.ScalarType(DataType.INT64), init_val, span)
sum_final = ir.Var("sum_final", ir.ScalarType(DataType.INT64), span)

for_stmt = ir.ForStmt(i, start, stop, step, [sum_iter], body, [sum_final], span)
```

## 语句节点

| 节点类型 | 字段 | 描述 |
|-----------|--------|-------------|
| **AssignStmt** | `var_` (DefField), `value_` (UsualField) | 变量赋值 |
| **IfStmt** | `condition_`, `then_stmts_`, `else_stmts_`, `return_vars_` | 条件分支 |
| **ForStmt** | `loop_var_` (DefField), `start_`, `stop_`, `step_`, `iter_args_` (DefField), `body_`, `return_vars_` (DefField) | 带可选迭代参数的 for 循环 |
| **YieldStmt** | `values_` | 在循环迭代中产出值 |
| **ReturnStmt** | `values_` | 从函数返回值 |
| **EvalStmt** | `expr_` | 求值表达式以产生副作用 |
| **SeqStmts** | `stmts_` | 通用语句序列 |
| **OpStmts** | `stmts_` | 赋值语句序列 |

### ForStmt 详细说明

**不带迭代参数：**
```python
# for i in range(0, 10, 1): x = x + i
for_stmt = ir.ForStmt(i, start, stop, step, [], body, [], span)
```

**带迭代参数：**
```python
# for i, (sum,) in pl.range(0, 10, 1, init_values=[0]):
#     sum = pl.yield_(sum + i)
# sum_final = sum
for_stmt = ir.ForStmt(i, start, stop, step, [sum_iter], body, [sum_final], span)
```

**要求：**
- yield 的值数量 = IterArgs 数量
- return_vars 数量 = IterArgs 数量
- IterArgs 仅在循环体内可访问
- 返回变量在循环后可访问

## 类型节点

| 节点类型 | 字段 | 描述 |
|-----------|--------|-------------|
| **UnknownType** | - | 未知或推断的类型 |
| **MemRefType** | `memory_space_`, `addr_`, `size_` | 内存引用类型（独立类型节点） |
| **ScalarType** | `dtype_` | 标量类型（INT64、FP32 等） |
| **TensorType** | `shape_`, `dtype_`, `memref_`（可选） | 多维张量 |
| **TileType** | `shape_`, `dtype_`, `memref_`（可选）, `tile_view_`（可选） | 统一缓冲区中的 Tile |
| **TupleType** | `types_` | 类型元组 |
| **PipeType** | `pipe_kind_` | 硬件流水线/屏障 |

### MemRef 和 MemRefType

**MemRefType** 是独立的类型节点,用于表示内存引用类型本身。

**MemRef** 是一个结构体,可以作为 TensorType/TileType 的可选字段,描述张量/tile 的内存分配。

#### MemRefType (独立类型节点)

| 字段 | 类型 | 描述 |
|-------|------|-------------|
| `memory_space_` | MemorySpace 枚举 | DDR、UB、L1、L0A、L0B、L0C |
| `addr_` | ExprPtr | 基地址 |
| `size_` | size_t | 字节大小 |

```python
# 创建 MemRefType 类型
memref_type = ir.MemRefType()
```

#### MemRef (作为 TensorType/TileType 的字段)

描述张量/tile 的内存分配：

| 字段 | 类型 | 描述 |
|-------|------|-------------|
| `memory_space_` | MemorySpace 枚举 | DDR、UB、L1、L0A、L0B、L0C |
| `addr_` | ExprPtr | 基地址 |
| `size_` | size_t | 字节大小 |

```python
memref = ir.MemRef(
    ir.MemorySpace.DDR,
    ir.ConstInt(0x1000, DataType.INT64, span),
    1024  # 字节
)
```

### TileView - Tile 布局

描述 tile 布局和访问模式：

| 字段 | 类型 | 描述 |
|-------|------|-------------|
| `valid_shape` | list[ExprPtr] | 有效维度 |
| `stride` | list[ExprPtr] | 每个维度的步长 |
| `start_offset` | ExprPtr | 起始偏移量 |

```python
tile_view = ir.TileView()
tile_view.valid_shape = [ir.ConstInt(16, DataType.INT64, span)] * 2
tile_view.stride = [ir.ConstInt(1, DataType.INT64, span), ir.ConstInt(16, DataType.INT64, span)]
tile_view.start_offset = ir.ConstInt(0, DataType.INT64, span)
```

## 函数节点

```python
# def add(x, y) -> int: return x + y
params = [
    ir.Var("x", ir.ScalarType(DataType.INT64), span),
    ir.Var("y", ir.ScalarType(DataType.INT64), span)
]
return_types = [ir.ScalarType(DataType.INT64)]
body = ir.AssignStmt(result, ir.Add(params[0], params[1], DataType.INT64, span), span)

func = ir.Function("add", params, return_types, body, span)

# 带函数类型
func_orch = ir.Function("orchestrator", params, return_types, body, span, ir.FunctionType.Orchestration)
```

| 字段 | 类型 | 描述 |
|-------|------|-------------|
| `name_` | string | 函数名 |
| `func_type_` | FunctionType | 函数类型（Opaque、Orchestration 或 InCore） |
| `params_` | list[VarPtr] | 参数（DefField） |
| `return_types_` | list[TypePtr] | 返回类型 |
| `body_` | StmtPtr | 函数体 |

### FunctionType 枚举

| 值 | 描述 |
|-------|-------------|
| `Opaque` | 未指定的函数类型（默认） |
| `Orchestration` | 在主机/AICPU 上运行，用于控制流和依赖分析 |
| `InCore` | 特定 AICore 上的子图 |

## 程序节点

包含多个函数的容器，具有确定性排序：

| 字段 | 类型 | 描述 |
|-------|------|-------------|
| `name_` | string | 程序名（IgnoreField） |
| `functions_` | map[GlobalVarPtr, FunctionPtr] | 函数的有序映射 |

```python
# 创建程序
func1 = ir.Function("add", params1, return_types1, body1, span)
func2 = ir.Function("multiply", params2, return_types2, body2, span)

program = ir.Program([func1, func2], "my_program", span)

# 访问函数
add_func = program.get_function("add")
add_gvar = program.get_global_var("add")

# 函数自动按名称排序
```

**关键特性：**
- 函数存储在有序映射中以确保确定性排序
- GlobalVar 名称必须与函数名匹配
- 通过 GlobalVar 实现程序内调用
- 确保一致的结构相等性和哈希

## 按类别的节点摘要

| 类别 | 数量 | 节点 |
|----------|-------|-------|
| **基类** | 4 | IRNode, Expr, Stmt, Type |
| **变量** | 2 | Var, IterArg |
| **常量** | 3 | ConstInt, ConstFloat, ConstBool |
| **二元操作** | 23 | Add, Sub, Mul, FloorDiv, FloorMod, FloatDiv, Min, Max, Pow, Eq, Ne, Lt, Le, Gt, Ge, And, Or, Xor, BitAnd, BitOr, BitXor, BitShiftLeft, BitShiftRight |
| **一元操作** | 5 | Abs, Neg, Not, BitNot, Cast |
| **调用/访问** | 3 | Call, MakeTuple, TupleGetItemExpr |
| **操作** | 2 | Op, GlobalVar |
| **语句** | 8 | AssignStmt, IfStmt, ForStmt, YieldStmt, ReturnStmt, EvalStmt, SeqStmts, OpStmts |
| **类型** | 7 | UnknownType, MemRefType, ScalarType, TensorType, TileType, TupleType, PipeType |
| **函数** | 2 | Function, Program |

## 相关文档

- [IR 概述](00-ir_overview_zh.md) - 核心概念和设计原则
- [IR 类型和示例](02-ir_types_examples_zh.md) - 类型系统详细信息和示例
- [结构比较](03-structural_comparison_zh.md) - 相等性和哈希
