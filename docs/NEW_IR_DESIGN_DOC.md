# PyPTO 新 IR 系统详细设计文档

> **版本**: 2.0
> **基于代码目录**: `/data/g00655722/new-ir/pypto_yhz`
> **命名空间**: `pypto::ir`
> **适用对象**: 新入职开发人员、算子开发人员、框架维护者

---

## 目录

1. [系统概述与架构](#1-系统概述与架构)
2. [核心基础设施](#2-核心基础设施)
3. [IR 节点体系](#3-ir-节点体系)
4. [类型系统](#4-类型系统)
5. [反射与字段注解系统](#5-反射与字段注解系统)
6. [类型分发系统 (ObjectKind / KindTrait)](#6-类型分发系统)
7. [IRBuilder 构建器](#7-irbuilder-构建器)
8. [算子注册与类型推导 (OpRegistry)](#8-算子注册与类型推导)
9. [IR 遍历框架 (Functor / Visitor / Mutator)](#9-ir-遍历框架)
10. [Pass 框架](#10-pass-框架)
11. [序列化系统](#11-序列化系统)
12. [验证系统](#12-验证系统)
13. [Python Printer](#13-python-printer)
14. [pybind11 绑定层](#14-pybind11-绑定层)
15. [Python 封装层](#15-python-封装层)
16. [算子开发指南](#16-算子开发指南)
17. [端到端工作流程](#17-端到端工作流程)
18. [测试策略](#18-测试策略)
19. [完整文件索引](#19-完整文件索引)

---

## 1. 系统概述与架构

### 1.1 项目定位

PyPTO 新 IR（Intermediate Representation）是面向华为昇腾 AI 处理器的编译器中间表示层。它承载从用户 Python 代码到硬件 CCE（Custom Compute Engine）二进制之间的所有分析和变换工作。

### 1.2 三层架构

```
┌──────────────────────────────────────────────────────────────────┐
│                    Python 封装层 (pypto.ir)                       │
│  IRBuilder 上下文管理器 │ 运算符重载 │ 类型补丁 │ 算子封装       │
├──────────────────────────────────────────────────────────────────┤
│                 pybind11 绑定层 (pypto_impl)                     │
│  ir.cpp (575 行) │ passes.cpp │ error.cpp │ core.cpp            │
├──────────────────────────────────────────────────────────────────┤
│                    C++ 核心层 (pypto::ir)                         │
│  IR 节点 │ 类型系统 │ OpRegistry │ Builder │ Pass │ 序列化       │
└──────────────────────────────────────────────────────────────────┘
```

**数据流方向**：用户 Python 代码 → Python 封装层（类型规范化、Span 捕获）→ pybind11 绑定 → C++ IRBuilder 构建不可变 IR 树 → Pass 流水线变换 → 代码生成 → 硬件执行。

### 1.3 核心设计原则

| 原则 | 说明 |
|------|------|
| **不可变性** | 所有 IR 节点构建后不可修改，通过 `shared_ptr<const T>` 持有，变换通过创建新节点实现（copy-on-write） |
| **零 RTTI** | 不使用 C++ RTTI（`dynamic_cast` 仅在极少数遗留路径），通过 `ObjectKind` 枚举 + `KindTrait` 模板实现高效类型分发 |
| **反射驱动** | 每个节点类通过 `GetFieldDescriptors()` 静态方法声明字段，序列化、比较、遍历等操作自动从字段描述符派生 |
| **SSA 风格** | 循环使用 `IterArg` + `YieldStmt` + `return_vars` 实现循环携带值，而非可变赋值 |
| **分层设计** | C++ 核心层纯逻辑、Python 封装层提供人体工程学便利（上下文管理器、运算符重载、自动 Span 捕获） |

### 1.4 目录结构

```
pypto_yhz/
├── framework/
│   ├── include/                         # C++ 头文件
│   │   ├── core/                        # 基础设施（DataType、Error、Logging）
│   │   │   ├── dtype.h                  # 数据类型定义
│   │   │   ├── error.h                  # 异常层次
│   │   │   ├── logging.h               # 日志系统
│   │   │   ├── common.h                # 版本常量、通用宏
│   │   │   └── any_cast.h              # 类型安全 std::any 转换
│   │   └── ir/                          # IR 核心
│   │       ├── core.h                   # IRNode 基类、ObjectKind 枚举
│   │       ├── type.h                   # 类型层次
│   │       ├── expr.h                   # 表达式节点（Var、Call、Op 等）
│   │       ├── scalar_expr.h            # 标量表达式（常量、二元/一元运算）
│   │       ├── stmt.h                   # 语句节点
│   │       ├── function.h               # 函数定义
│   │       ├── program.h                # 程序定义
│   │       ├── span.h                   # 源码位置
│   │       ├── memref.h                 # 内存空间枚举
│   │       ├── pipe.h                   # 流水线类型枚举
│   │       ├── kind_traits.h            # KindTrait 特化、IsA/As 函数
│   │       ├── builder.h               # C++ IRBuilder
│   │       ├── op_registry.h           # OpRegistry 算子注册
│   │       ├── type_inference.h        # 类型推导工具
│   │       ├── reflection/              # 反射系统
│   │       │   ├── field_traits.h      # 字段描述符（DefField/UsualField/IgnoreField）
│   │       │   └── field_visitor.h     # 字段迭代器
│   │       ├── serialization/           # 序列化系统
│   │       │   ├── serializer.h        # MessagePack 序列化器
│   │       │   ├── deserializer.h      # MessagePack 反序列化器
│   │       │   └── type_registry.h     # 反序列化类型注册表
│   │       └── transform/               # 变换框架
│   │           ├── passes.h             # Pass/PassImpl 接口
│   │           ├── printer.h            # Python Printer
│   │           ├── verifier.h           # 验证器
│   │           ├── verification_error.h # 验证错误类型
│   │           ├── structural_comparison.h # 结构比较
│   │           └── base/                # 遍历基类
│   │               ├── functor.h       # ExprFunctor/StmtFunctor/IRFunctor
│   │               ├── visitor.h       # IRVisitor
│   │               └── mutator.h       # IRMutator
│   ├── src/interface/ir/                # C++ 实现文件
│   │   ├── core.cpp                     # IRNode 相关实现
│   │   ├── type.cpp                     # 类型构建实现
│   │   ├── expr.cpp                     # 表达式构建实现
│   │   ├── builder.cpp                  # IRBuilder 实现
│   │   ├── op_registry.cpp             # OpRegistry 实现
│   │   ├── op/                          # 算子注册
│   │   │   ├── block_ops/              # Block 算子（37 个）
│   │   │   ├── tensor_ops/             # Tensor 算子（16 个）
│   │   │   ├── sync_ops/              # 同步算子（5 个）
│   │   │   └── type_inference.cpp     # 类型推导实现
│   │   ├── serialization/              # 序列化实现
│   │   │   ├── serializer.cpp
│   │   │   └── deserializer.cpp
│   │   └── transform/                  # 变换实现
│   │       ├── passes.cpp
│   │       ├── visitor.cpp
│   │       ├── mutator.cpp
│   │       ├── printer.cpp
│   │       └── verifier.cpp
│   └── tests/                          # C++ 测试（~357 个文件）
├── python/
│   ├── src/
│   │   ├── pybind11.cpp                # 模块入口
│   │   └── bindings/                   # pybind11 绑定
│   │       ├── bindings.h
│   │       ├── modules/
│   │       │   ├── ir.cpp             # IR 类型绑定（575 行）
│   │       │   ├── passes.cpp         # Pass 绑定
│   │       │   ├── core.cpp           # Core 桩
│   │       │   └── error.cpp          # 异常翻译
│   ├── pypto/ir/                       # Python 封装层
│   │   ├── __init__.py                # 模块入口
│   │   ├── builder.py                 # Python IRBuilder（~1004 行）
│   │   ├── operators.py               # 运算符重载
│   │   ├── type.py                    # 类型构造函数补丁
│   │   ├── utils.py                   # 工具函数
│   │   ├── printer.py                 # Print 分发器
│   │   ├── pass_manager.py            # Pass 编排
│   │   └── op/                        # 算子 Python 封装
│   │       ├── __init__.py
│   │       ├── block_ops.py           # Block 操作（~1001 行）
│   │       └── tensor_ops.py          # Tensor 操作（~477 行）
│   └── tests/                         # Python 测试
│       ├── ut/                        # 单元测试（~30 个文件）
│       └── st/                        # 系统测试（~90 个文件）
```

---

## 2. 核心基础设施

### 2.1 DataType — 数据类型

**文件**: `framework/include/core/dtype.h`（342 行）

`DataType` 封装所有受支持的数值类型，内部仅存储一个 `uint8_t code_`，支持 constexpr 构造。

**类型编码方案**（按范围分区，每区 16 个槽位预留扩展）：

| 范围 | 分类 | 类型 |
|------|------|------|
| `0x00` | 布尔 | BOOL |
| `0x10-0x1F` | 有符号整数 | INT4, INT8, INT16, INT32, INT64 |
| `0x20-0x2F` | 无符号整数 | UINT4, UINT8, UINT16, UINT32, UINT64 |
| `0x30-0x3F` | IEEE 浮点 | FP4, FP8E4M3FN, FP8E5M2, FP16, FP32 |
| `0x40-0x4F` | 脑浮点/海思浮点 | BF16, HF4, HF8 |

**关键方法**：

| 方法 | 说明 |
|------|------|
| `GetBit()` | 返回位宽（BOOL=1, INT4=4, FP32=32 等） |
| `ToString()` | 人类可读名（"int32", "fp16", "bfloat16"） |
| `ToCTypeString()` | C 类型名（"int32_t", "half", "float"）用于代码生成 |
| `IsFloat()` | 是否浮点（IEEE + 脑浮点） |
| `IsSignedInt()` | 是否有符号整数 |
| `IsUnsignedInt()` | 是否无符号整数 |
| `IsInt()` | 是否任意整数 |
| `operator==` / `operator!=` | constexpr 比较 |
| `Code()` | 获取底层 uint8_t 编码 |

**使用示例**：
```cpp
DataType dt = DataType::FP16;
if (dt.IsFloat()) {
    size_t bits = dt.GetBit();   // 16
    std::string s = dt.ToString(); // "fp16"
    std::string c = dt.ToCTypeString(); // "half"
}
```

### 2.2 Error — 异常层次

**文件**: `framework/include/core/error.h`

所有异常继承自 `Error`（自身继承 `std::runtime_error`），构造时自动捕获 C++ 调用栈。

```
std::runtime_error
└── Error（基类，自动附加 Backtrace）
    ├── ValueError     — 值不合法
    ├── TypeError      — 类型不匹配
    ├── RuntimeError   — 运行时错误
    ├── NotImplementedError — 功能未实现
    ├── IndexError     — 索引越界
    ├── AssertionError — 断言失败
    └── InternalError  — 内部错误（不应暴露给用户）
```

**辅助类**：

| 类 | 说明 |
|------|------|
| `StackFrame` | 单帧：文件名、函数名、行号 |
| `Backtrace` | 单例，通过 `Backtrace::instance().Capture()` 获取当前调用栈 |
| `Diagnostic` | 诊断消息：severity、rule_name、error_code、message、span |
| `VerificationError` | 携带 `vector<Diagnostic>` 的验证异常 |

### 2.3 Logging — 日志系统

**文件**: `framework/include/core/logging.h`

```cpp
enum class LogLevel { DEBUG = 0, INFO = 1, WARNING = 2, ERROR = 3, NONE = 4 };
```

**组件**：

| 组件 | 说明 |
|------|------|
| `StdLogger` | 输出到 stderr |
| `FileLogger` | 输出到文件 |
| `LineLogger` | RAII 行日志（析构时 flush） |
| `LoggerManager` | 单例，管理当前日志级别和日志器 |
| `FatalLogger<ExceptionType>` | 模板类，析构时抛出指定异常 |

**宏**：

| 宏 | 说明 |
|------|------|
| `LOG(level)` | 按级别记录日志 |
| `CHECK(cond)` | 条件检查，失败抛出 `RuntimeError` |
| `INTERNAL_CHECK(cond)` | 内部检查，失败抛出 `InternalError` |

### 2.4 Common — 通用定义

**文件**: `framework/include/core/common.h`

| 定义 | 说明 |
|------|------|
| `kDynamicDim = -1` | 动态维度标记 |
| `PYPTO_ALWAYS_INLINE` | 强制内联宏 |
| `PYPTO_UNUSED` | 抑制未使用警告宏 |
| 版本常量 | 框架版本信息 |

### 2.5 AnyCast — 类型安全 any 转换

**文件**: `framework/include/core/any_cast.h`

```cpp
template <typename T>
T AnyCast(const std::any& value, const std::string& context = "");

template <typename T>
const T& AnyCastRef(const std::any& value, const std::string& context = "");
```

失败时抛出 `TypeError`，附带 demangled 类型名和上下文信息，例如：
```
TypeError: AnyCast failed for kwarg key: target_memory.
Expected type: int, actual type: std::string
```

辅助函数 `DemangleTypeName(const std::type_info&)` 用于获取可读类型名。

---

## 3. IR 节点体系

### 3.1 整体层次

```
IRNode (基类, 持有 span_)
├── Expr (表达式基类, 持有 type_: TypePtr)
│   ├── Var (变量引用, name_ 为 IgnoreField)
│   │   ├── IterArg (循环迭代参数, 持有 initValue_)
│   │   └── MemRef (内存引用, 持有 memory_space_, addr_, size_, id_)
│   ├── Call (函数调用, 持有 op_, args_, kwargs_)
│   ├── MakeTuple (构造元组)
│   ├── TupleGetItemExpr (元组取元素)
│   ├── ConstInt (整数常量, value_: int64_t)
│   ├── ConstFloat (浮点常量, value_: double)
│   ├── ConstBool (布尔常量, value_: bool)
│   ├── BinaryExpr (二元运算基类, 持有 left_, right_)
│   │   ├── Add, Sub, Mul, FloorDiv, FloorMod, FloatDiv
│   │   ├── Min, Max, Pow
│   │   ├── Eq, Ne, Lt, Le, Gt, Ge (比较运算，结果类型为 BOOL)
│   │   ├── And, Or, Xor (逻辑运算)
│   │   └── BitAnd, BitOr, BitXor, BitShiftLeft, BitShiftRight (位运算)
│   └── UnaryExpr (一元运算基类, 持有 operand_)
│       ├── Abs, Neg, Not, BitNot, Cast
├── Stmt (语句基类)
│   ├── AssignStmt (赋值: var = value)
│   ├── IfStmt (条件: if/else, 持有 return_vars_)
│   ├── YieldStmt (产出: 循环/条件中传值)
│   ├── ReturnStmt (返回: 函数返回)
│   ├── ForStmt (循环: 持有 loop_var_, iter_args_, return_vars_)
│   ├── SeqStmts (语句序列)
│   ├── OpStmts (操作语句组，仅接受 AssignStmt/EvalStmt)
│   └── EvalStmt (求值: 执行表达式的副作用)
├── Function (函数: name_, params_, return_types_, body_, func_type_)
└── Program (程序: map<GlobalVar → Function>)

Op (算子基类, 非 IRNode, 持有 name_, attrs_, pipe_)
├── Op (具体算子, 通过 OpRegistry 全局注册)
└── GlobalVar (全局变量引用, 用于程序内跨函数调用)
```

**统计**: 37 种 Expr + 8 种 Stmt + Function + Program + 7 种 Type = **56 个 ObjectKind 值**。

### 3.2 IRNode 基类

**文件**: `framework/include/ir/core.h`

```cpp
class IRNode {
public:
    explicit IRNode(Span s);
    virtual ~IRNode() = default;

    // 禁止拷贝和移动，强制不可变性
    IRNode(IRNode&&) = delete;
    IRNode& operator=(IRNode&&) = delete;

    virtual ObjectKind GetKind() const = 0;
    virtual std::string TypeName() const;

    Span span_;  // 源码位置（IgnoreField，不参与比较和序列化）

    static constexpr auto GetFieldDescriptors() {
        return std::make_tuple(reflection::IgnoreField(&IRNode::span_, "span"));
    }
};
using IRNodePtr = std::shared_ptr<const IRNode>;
```

**关键设计点**：
- `shared_ptr<const T>` 保证不可变性，所有修改必须创建新节点
- `IRNodePtr` 的 `operator==` 是**引用相等**（比较指针地址），非结构相等
- `std::hash<IRNodePtr>` 也基于指针地址，可用于 `unordered_map`
- `span_` 标记为 `IgnoreField`，在结构比较和序列化中被忽略

### 3.3 Expr — 表达式基类

**文件**: `framework/include/ir/expr.h`

```cpp
class Expr : public IRNode {
protected:
    TypePtr type_;  // 表达式的结果类型
public:
    explicit Expr(Span s, TypePtr type = GetUnknownType());
    const TypePtr& GetType() const;

    static constexpr auto GetFieldDescriptors() {
        return std::tuple_cat(IRNode::GetFieldDescriptors(),
            std::make_tuple(reflection::UsualField(&Expr::type_, "type")));
    }
};
```

每个表达式都有一个 `type_` 字段，表示其计算结果的类型。默认为 `UnknownType`，可由 IRBuilder 或类型推导设置。

### 3.4 Var / IterArg / MemRef — 变量家族

#### Var — 变量引用

```cpp
class Var : public Expr {
public:
    std::string name_;  // IgnoreField，不参与结构比较
    Var(std::string name, TypePtr type, Span span);
};
```

`name_` 是 `IgnoreField` 意味着两个同类型但不同名的 `Var` 在结构比较中可能被认为相等（取决于 `enable_auto_mapping` 设置）。

#### IterArg — 循环迭代参数

```cpp
class IterArg : public Var {
public:
    ExprPtr initValue_;  // 首次迭代的初始值（UsualField）
    IterArg(std::string name, TypePtr type, ExprPtr initValue, Span span);
};
```

`IterArg` 实现 SSA 风格的循环携带值。它的作用域仅限于循环体内部，循环结束后通过 `ForStmt::return_vars_` 暴露最终值。

**使用模式**：
```
for i, (sum,) in pl.range(0, N, 1, init_values=[init_sum]):
    new_sum = sum + a[i]
    pl.yield_(new_sum)
final_sum = sum  # 通过 return_vars 访问
```

#### MemRef — 内存引用

```cpp
class MemRef : public Var {
public:
    MemorySpace memory_space_;  // DDR, UB, L1, L0A, L0B, L0C
    ExprPtr addr_;               // 起始地址表达式
    uint64_t size_;              // 大小（字节）
    uint64_t id_;                // 唯一标识符（用于生成变量名 "mem_123"）
};
```

### 3.5 Op / GlobalVar — 算子与全局变量

#### Op — 算子

```cpp
class Op {
public:
    std::string name_;  // 算子名，如 "block.add"

    template <typename T>
    void SetAttrType(const std::string& key) const;  // 注册 kwarg 类型
    std::type_index GetAttrType(const std::string& key) const;
    bool HasAttr(const std::string& key) const;
    std::vector<std::string> GetAttrKeys() const;
    const std::unordered_map<std::string, std::type_index>& GetAttrs() const;
    void SetPipe(PipeType pipe) const;
    std::optional<PipeType> GetPipe() const;

private:
    mutable std::unordered_map<std::string, std::type_index> attrs_;
    mutable std::optional<PipeType> pipe_;
};
```

`Op` **不是** `IRNode` 子类。它通过 `OpRegistry` 全局注册，通过 `Op::get("block.add")` 获取单例。kwarg 的值类型约束为：`bool`, `int`, `std::string`, `double`, `DataType`（编译时 static_assert 保证）。

#### GlobalVar — 全局变量引用

```cpp
class GlobalVar : public Op {
public:
    explicit GlobalVar(std::string name);
};

struct GlobalVarPtrLess {
    bool operator()(const GlobalVarPtr& lhs, const GlobalVarPtr& rhs) const;
};
```

`GlobalVar` 用于在 `Program` 中引用函数，也可作为 `Call` 的 `op_` 实现跨函数调用。`GlobalVarPtrLess` 按名称排序，保证 Program 中函数的确定性顺序。

### 3.6 Call — 函数调用

```cpp
class Call : public Expr {
public:
    OpPtr op_;                                              // 调用的算子
    std::vector<ExprPtr> args_;                             // 位置参数
    std::vector<std::pair<std::string, std::any>> kwargs_;  // 关键字参数

    // 四个构造函数：有/无 kwargs × 有/无显式 type
    Call(OpPtr op, std::vector<ExprPtr> args, Span span);
    Call(OpPtr op, std::vector<ExprPtr> args, TypePtr type, Span span);
    Call(OpPtr op, std::vector<ExprPtr> args,
         std::vector<std::pair<std::string, std::any>> kwargs, Span span);
    Call(OpPtr op, std::vector<ExprPtr> args,
         std::vector<std::pair<std::string, std::any>> kwargs, TypePtr type, Span span);

    template <typename T>
    T GetKwarg(const std::string& key, const T& default_value = T{}) const;
    bool HasKwarg(const std::string& key) const;
};
```

`Call` 是 IR 中表示算子调用的核心节点。例如 `block.add(a, b)` 对应：
```cpp
Call(Op::get("block.add"), {a_expr, b_expr}, span)
```

### 3.7 标量常量节点

| 类 | 值类型 | 构造函数 | 备注 |
|------|------|------|------|
| `ConstInt` | `int64_t value_` | `ConstInt(int64_t, DataType, Span)` | 类型为 `ScalarType(dtype)` |
| `ConstFloat` | `double value_` | `ConstFloat(double, DataType, Span)` | 类型为 `ScalarType(dtype)` |
| `ConstBool` | `bool value_` | `ConstBool(bool, Span)` | 类型固定为 `ScalarType(BOOL)` |

### 3.8 二元表达式 (BinaryExpr)

**基类**：
```cpp
class BinaryExpr : public Expr {
public:
    ExprPtr left_;
    ExprPtr right_;
    BinaryExpr(ExprPtr left, ExprPtr right, DataType dtype, Span span);
};
```

**22 个具体子类**（通过 `DEFINE_BINARY_EXPR_NODE` 宏生成）：

| 分类 | 算子 |
|------|------|
| 算术 | `Add`, `Sub`, `Mul`, `FloorDiv`, `FloorMod`, `FloatDiv`, `Pow` |
| 极值 | `Min`, `Max` |
| 比较 | `Eq`, `Ne`, `Lt`, `Le`, `Gt`, `Ge`（结果类型固定 BOOL） |
| 逻辑 | `And`, `Or`, `Xor` |
| 位运算 | `BitAnd`, `BitOr`, `BitXor`, `BitShiftLeft`, `BitShiftRight` |

**构造辅助函数**（带自动类型提升）：

```cpp
ExprPtr MakeAdd(const ExprPtr& left, const ExprPtr& right, const Span& span);
ExprPtr MakeSub(const ExprPtr& left, const ExprPtr& right, const Span& span);
ExprPtr MakeMul(const ExprPtr& left, const ExprPtr& right, const Span& span);
ExprPtr MakeFloatDiv(const ExprPtr& left, const ExprPtr& right, const Span& span);
ExprPtr MakeFloorDiv(const ExprPtr& left, const ExprPtr& right, const Span& span);
ExprPtr MakeFloorMod(const ExprPtr& left, const ExprPtr& right, const Span& span);
ExprPtr MakePow(const ExprPtr& left, const ExprPtr& right, const Span& span);
ExprPtr MakeEq(const ExprPtr& left, const ExprPtr& right, const Span& span);
ExprPtr MakeNe(const ExprPtr& left, const ExprPtr& right, const Span& span);
ExprPtr MakeLt(const ExprPtr& left, const ExprPtr& right, const Span& span);
ExprPtr MakeLe(const ExprPtr& left, const ExprPtr& right, const Span& span);
ExprPtr MakeGt(const ExprPtr& left, const ExprPtr& right, const Span& span);
ExprPtr MakeGe(const ExprPtr& left, const ExprPtr& right, const Span& span);
ExprPtr MakeBitAnd(const ExprPtr& left, const ExprPtr& right, const Span& span);
ExprPtr MakeBitOr(const ExprPtr& left, const ExprPtr& right, const Span& span);
ExprPtr MakeBitXor(const ExprPtr& left, const ExprPtr& right, const Span& span);
ExprPtr MakeBitShiftLeft(const ExprPtr& left, const ExprPtr& right, const Span& span);
ExprPtr MakeBitShiftRight(const ExprPtr& left, const ExprPtr& right, const Span& span);
```

**类型提升规则** (`PromoteBinaryOperands`)：
1. 提取双方的 `ScalarType::dtype_`
2. 检查是否同一数值类别（int/float），否则报 `TypeError`
3. 取较大位宽的类型作为结果类型
4. 如需要，自动插入 `Cast` 节点

**位运算类型检查** (`PromoteIntBinaryOperands`)：要求双方都是整数类型，否则报 `TypeError`。

### 3.9 一元表达式 (UnaryExpr)

**基类**：
```cpp
class UnaryExpr : public Expr {
public:
    ExprPtr operand_;
    UnaryExpr(ExprPtr operand, DataType dtype, Span span);
};
```

**5 个具体子类**（通过 `DEFINE_UNARY_EXPR_NODE` 宏生成）：

| 算子 | 说明 | 构造辅助函数 |
|------|------|------|
| `Abs` | 绝对值 | — |
| `Neg` | 取负 | `MakeNeg(operand, span)` |
| `Not` | 逻辑非 | — |
| `BitNot` | 位取反（仅整数） | `MakeBitNot(operand, span)` — 检查整数类型 |
| `Cast` | 类型转换 | `MakeCast(operand, dtype, span)` |

### 3.10 元组操作

```cpp
class MakeTuple : public Expr {
public:
    std::vector<ExprPtr> elements_;
    // 构造时自动推导 TupleType
};

class TupleGetItemExpr : public Expr {
public:
    ExprPtr tuple_;
    int index_;
    // 构造时从 TupleType 中提取对应元素类型
};
```

### 3.11 语句节点

#### AssignStmt — 赋值语句

```cpp
class AssignStmt : public Stmt {
public:
    VarPtr var_;     // DefField — 定义变量
    ExprPtr value_;  // UsualField — 值表达式
};
```

`var_` 标记为 `DefField` 意味着这是变量的定义点，在结构比较中用于建立变量映射。

#### ForStmt — 循环语句

```cpp
class ForStmt : public Stmt {
public:
    VarPtr loop_var_;                    // DefField — 循环变量
    ExprPtr start_, stop_, step_;        // UsualField — 范围
    std::vector<IterArgPtr> iter_args_;  // DefField — 循环携带值
    StmtPtr body_;                       // UsualField — 循环体
    std::vector<VarPtr> return_vars_;    // DefField — 结果变量
};
```

**关键关系**：
- `iter_args_` 的数量 = `return_vars_` 的数量
- 循环体中的 `YieldStmt` 产出值的数量 = `iter_args_` 的数量
- `IterArg` 仅在循环体内可见，`return_vars_` 在循环后可见

#### IfStmt — 条件语句

```cpp
class IfStmt : public Stmt {
public:
    ExprPtr condition_;
    StmtPtr then_body_;
    std::optional<StmtPtr> else_body_;
    std::vector<VarPtr> return_vars_;  // DefField
};
```

#### SeqStmts / OpStmts — 语句序列

```cpp
class SeqStmts : public Stmt {
public:
    std::vector<StmtPtr> stmts_;  // 可包含任意语句
};

class OpStmts : public Stmt {
public:
    std::vector<StmtPtr> stmts_;  // 仅接受 AssignStmt 和 EvalStmt
};
```

`OpStmts` 在构造时验证只包含 `AssignStmt` 和 `EvalStmt`，用于表示一组操作语句。

#### YieldStmt / ReturnStmt / EvalStmt

```cpp
class YieldStmt : public Stmt {
    std::vector<ExprPtr> value_;  // 产出的表达式列表
};

class ReturnStmt : public Stmt {
    std::vector<ExprPtr> value_;  // 返回的表达式列表
};

class EvalStmt : public Stmt {
    ExprPtr expr_;  // 为副作用执行的表达式（通常是 Call）
};
```

### 3.12 Function — 函数

```cpp
enum class FunctionType : uint8_t {
    Opaque = 0,         // 默认：未指定
    Orchestration = 1,  // 主机/AICPU 控制
    InCore = 2          // AICore 子图
};

class Function : public IRNode {
public:
    std::string name_;                   // IgnoreField
    FunctionType func_type_;             // UsualField
    std::vector<VarPtr> params_;         // DefField — 参数定义
    std::vector<TypePtr> return_types_;  // UsualField
    StmtPtr body_;                       // UsualField
};
```

`params_` 标记为 `DefField`，在结构比较中用于建立参数映射。

`FunctionType` 的三种值：
- `Opaque`：默认值，未指定执行上下文
- `Orchestration`：在 Host/AICPU 上运行，负责控制流和依赖分析
- `InCore`：在 AICore 上运行的子图

辅助函数 `FunctionTypeToString` / `StringToFunctionType` 提供枚举与字符串的双向转换。

### 3.13 Program — 程序

```cpp
class Program : public IRNode {
public:
    std::string name_;  // IgnoreField
    std::map<GlobalVarPtr, FunctionPtr, GlobalVarPtrLess> functions_;  // UsualField

    FunctionPtr GetFunction(const std::string& name) const;
    GlobalVarPtr GetGlobalVar(const std::string& name) const;
};
```

`functions_` 使用 `std::map` + `GlobalVarPtrLess` 按名称排序，保证确定性。便捷构造函数接受 `vector<FunctionPtr>`，自动为每个函数创建对应的 `GlobalVar`。

---

## 4. 类型系统

**文件**: `framework/include/ir/type.h`

类型层次**独立于** IRNode 层次（`Type` 不继承 `IRNode`）：

```
Type (基类)
├── UnknownType — 未知/未指定类型（单例，通过 GetUnknownType() 获取）
├── ScalarType — 标量类型（持有 dtype_: DataType）
├── ShapedType — 形状类型基类（持有 dtype_, shape_, memref_）
│   ├── TensorType — 张量类型（全局内存中的数据）
│   └── TileType — 片上类型（持有 tile_view_: optional<TileView>）
├── TupleType — 元组类型（持有 types_: vector<TypePtr>）
└── MemRefType — 内存引用类型（单例，通过 GetMemRefType() 获取）
```

### 4.1 各类型详细说明

| 类 | 字段 | 说明 |
|------|------|------|
| `UnknownType` | （无） | 默认类型，表达式类型未确定时使用。单例通过 `GetUnknownType()` 获取 |
| `ScalarType` | `dtype_: DataType` | 标量值类型，如 `ScalarType(DataType::INT32)` |
| `ShapedType` | `dtype_`, `shape_: vector<ExprPtr>`, `memref_: optional<MemRefPtr>` | 有形状的类型基类。shape 中的每个维度可以是 ConstInt（静态）或 Var（动态） |
| `TensorType` | （继承 ShapedType） | 全局内存中的张量。多个构造函数支持 ExprPtr shape、int64_t shape、带/不带 MemRef |
| `TileType` | `tile_view_: optional<TileView>` + 继承字段 | 片上 tile，用于硬件操作。代码生成当前仅支持 2D |
| `TupleType` | `types_: vector<TypePtr>` | 多值元组，用于多返回值和结构化数据 |
| `MemRefType` | （无） | MemRef 变量的类型标记。单例通过 `GetMemRefType()` 获取 |

### 4.2 TileView 结构

```cpp
struct TileView {
    std::vector<ExprPtr> valid_shape;  // 有效形状
    std::vector<ExprPtr> stride;       // 步长
    ExprPtr start_offset;              // 起始偏移

    static constexpr auto GetFieldDescriptors();  // 支持反射
};
```

`TileView` 用于跟踪 tile 在底层内存中的视图信息，类似于 NumPy 的 stride 概念。

### 4.3 ShapedType 的维度表示

形状维度 `shape_` 是 `vector<ExprPtr>`，每个维度可以是：
- `ConstInt(4, INT64, span)` — 静态维度
- `Var("n", ScalarType(INT64), span)` — 符号维度（动态）

这种设计允许同时表示静态和动态形状。

### 4.4 内存空间枚举

**文件**: `framework/include/ir/memref.h`

```cpp
enum class MemorySpace {
    DDR = 0,   // 外部存储
    UB = 1,    // 统一缓存 (Unified Buffer)
    L1 = 2,    // 一级缓存
    L0A = 3,   // L0 A 矩阵缓存（矩阵乘法左操作数）
    L0B = 4,   // L0 B 矩阵缓存（矩阵乘法右操作数）
    L0C = 5    // L0 C 累加器缓存（矩阵乘法结果）
};
```

### 4.5 流水线类型枚举

**文件**: `framework/include/ir/pipe.h`

```cpp
enum class PipeType {
    MTE1 = 0,  // 数据搬运引擎 1（L1 → L0A/L0B）
    MTE2 = 1,  // 数据搬运引擎 2（DDR → UB/L1）
    MTE3 = 2,  // 数据搬运引擎 3（L0C → UB）
    V = 3,     // 向量计算引擎 (Vector)
    M = 4,     // 矩阵计算引擎 (Matrix/Cube)
    ALL = 5    // 所有流水线（全局同步）
};

enum class CoreType {
    CUBE = 0,   // Cube 核心（矩阵运算）
    VECTOR = 1  // Vector 核心（向量运算）
};
```

### 4.6 Span — 源码位置

**文件**: `framework/include/ir/span.h`

```cpp
class Span {
public:
    std::string filename_;
    int begin_line_, begin_col_;
    int end_line_, end_col_;

    Span(std::string filename, int begin_line, int begin_col,
         int end_line = -1, int end_col = -1);
    static Span unknown();  // 返回未知位置（"unknown", -1, -1）
};
```

Python 封装层通过 `inspect.currentframe()` 自动捕获用户代码位置，附加到每个 IR 节点上，用于错误报告和调试。

---

## 5. 反射与字段注解系统

### 5.1 概述

反射系统是新 IR 的核心基础设施之一。每个 IR 节点类通过静态方法 `GetFieldDescriptors()` 声明其字段列表和字段类别。序列化、结构比较、visitor 遍历等操作都基于这些字段描述符自动生成。

**文件**: `framework/include/ir/reflection/field_traits.h`

### 5.2 字段标签

```cpp
namespace reflection {
    struct DefFieldTag {};     // 定义字段 — 标记变量定义点
    struct UsualFieldTag {};   // 普通字段 — 正常遍历/比较/序列化
    struct IgnoreFieldTag {};  // 忽略字段 — 跳过比较和序列化
}
```

### 5.3 FieldDescriptor

```cpp
template <typename NodeType, typename FieldType, typename KindTag>
struct FieldDescriptor {
    FieldType NodeType::*ptr;  // 成员指针
    const char* name;          // 字段名
};

// 工厂函数
template <typename NodeType, typename FieldType>
constexpr auto DefField(FieldType NodeType::*ptr, const char* name);

template <typename NodeType, typename FieldType>
constexpr auto UsualField(FieldType NodeType::*ptr, const char* name);

template <typename NodeType, typename FieldType>
constexpr auto IgnoreField(FieldType NodeType::*ptr, const char* name);
```

### 5.4 字段标签的语义

| 标签 | 序列化 | 结构比较 | 遍历 | 典型字段 |
|------|--------|---------|------|---------|
| `DefField` | ✓ | ✓（建立变量映射） | ✓ | `AssignStmt::var_`, `Function::params_`, `ForStmt::loop_var_`, `ForStmt::iter_args_`, `ForStmt::return_vars_`, `IfStmt::return_vars_` |
| `UsualField` | ✓ | ✓ | ✓ | `Call::args_`, `BinaryExpr::left_/right_`, `Expr::type_`, `IterArg::initValue_`, `ForStmt::start_/stop_/step_/body_`, `Function::return_types_/body_/func_type_`, `Program::functions_` 等大多数字段 |
| `IgnoreField` | ✗ | ✗ | ✗ | `IRNode::span_`, `Var::name_`, `Function::name_`, `Program::name_` |

**DefField 的特殊含义**：标记为 `DefField` 的变量是"定义点"。在结构比较中，两个 IR 树的 `DefField` 变量会自动建立一一映射（auto_mapping），从而正确比较包含不同变量实例但结构相同的 IR。

### 5.5 GetFieldDescriptors 继承链

字段描述符通过 `std::tuple_cat` 链式组合：

```cpp
// IRNode 只有 span（IgnoreField）
class IRNode {
    static constexpr auto GetFieldDescriptors() {
        return std::make_tuple(reflection::IgnoreField(&IRNode::span_, "span"));
    }
};

// Expr 继承 IRNode 的字段并添加 type
class Expr : public IRNode {
    static constexpr auto GetFieldDescriptors() {
        return std::tuple_cat(
            IRNode::GetFieldDescriptors(),
            std::make_tuple(reflection::UsualField(&Expr::type_, "type")));
    }
};

// Var 继承 Expr 的字段并添加 name（IgnoreField）
class Var : public Expr {
    static constexpr auto GetFieldDescriptors() {
        return std::tuple_cat(
            Expr::GetFieldDescriptors(),
            std::make_tuple(reflection::IgnoreField(&Var::name_, "name")));
    }
};

// IterArg 继承 Var 并添加 initValue（UsualField）
class IterArg : public Var {
    static constexpr auto GetFieldDescriptors() {
        return std::tuple_cat(
            Var::GetFieldDescriptors(),
            std::make_tuple(reflection::UsualField(&IterArg::initValue_, "initValue")));
    }
};

// AssignStmt 中 var 是 DefField，value 是 UsualField
class AssignStmt : public Stmt {
    static constexpr auto GetFieldDescriptors() {
        return std::tuple_cat(
            Stmt::GetFieldDescriptors(),
            std::make_tuple(
                reflection::DefField(&AssignStmt::var_, "var"),
                reflection::UsualField(&AssignStmt::value_, "value")));
    }
};
```

### 5.6 FieldIterator — 字段迭代器

**文件**: `framework/include/ir/reflection/field_visitor.h`

```cpp
class FieldIterator {
public:
    // 针对不同字段类型的回调
    virtual void VisitIRNodeField(const std::string& name, const IRNodePtr& node) = 0;
    virtual void VisitIRNodeVectorField(const std::string& name, const std::vector<...>& nodes) = 0;
    virtual void VisitIRNodeMapField(const std::string& name, const MapType& map) = 0;
    virtual void VisitLeafField(const std::string& name, const LeafType& value) = 0;
};
```

`FieldIterator` 使用编译时字段描述符遍历节点的所有字段，自动区分 IR 节点引用、向量、映射和叶子值。序列化器和结构比较器都基于此实现。

---

## 6. 类型分发系统

### 6.1 ObjectKind 枚举

**文件**: `framework/include/ir/core.h`

```cpp
enum class ObjectKind {
    // 抽象基类
    IRNode, Expr, Stmt, Type,
    // 表达式 (37 种)
    Var, IterArg, MemRef, Call, MakeTuple, TupleGetItemExpr,
    ConstInt, ConstFloat, ConstBool,
    Add, Sub, Mul, FloorDiv, FloorMod, FloatDiv, Min, Max, Pow,
    Eq, Ne, Lt, Le, Gt, Ge, And, Or, Xor,
    BitAnd, BitOr, BitXor, BitShiftLeft, BitShiftRight,
    Abs, Neg, Not, BitNot, Cast,
    // 语句 (8 种)
    AssignStmt, IfStmt, YieldStmt, ReturnStmt, ForStmt,
    SeqStmts, OpStmts, EvalStmt,
    // 类型 (7 种)
    UnknownType, MemRefType, ScalarType, ShapedType,
    TensorType, TileType, TupleType,
    // 其他
    Function, Program, Op, GlobalVar
};
```

### 6.2 KindTrait 特化

**文件**: `framework/include/ir/kind_traits.h`

每个具体类型有一个 `KindTrait` 特化，提供其 `ObjectKind` 值：

```cpp
// 具体类型 — 单一 kind 值
template <>
struct KindTrait<Var> {
    static constexpr ObjectKind kind = ObjectKind::Var;
};

// 抽象基类 — kinds 数组（列举所有子类）
template <>
struct KindTrait<Expr> {
    static constexpr size_t count = 37;
    static constexpr ObjectKind kinds[] = {
        ObjectKind::Var, ObjectKind::IterArg, ObjectKind::MemRef,
        ObjectKind::Call, ObjectKind::MakeTuple, ObjectKind::TupleGetItemExpr,
        ObjectKind::ConstInt, ObjectKind::ConstFloat, ObjectKind::ConstBool,
        ObjectKind::Add, ObjectKind::Sub, /* ... 所有 37 个 Expr 子类 */
    };
};
```

**SFINAE 辅助**：
- `detail::HasSingleKind<T>` — 检测 `KindTrait<T>::kind` 是否存在（具体类型）
- `detail::HasKindArray<T>` — 检测 `KindTrait<T>::kinds` 是否存在（抽象基类）
- `detail::IsKindInArray<T>(kind)` — 编译时遍历 `kinds[]` 查找匹配

### 6.3 IsA / As 函数

```cpp
// 类型检查
template <typename T>
bool IsA(const IRNodePtr& node);

// 类型转换（失败返回 nullptr）
template <typename T>
std::shared_ptr<const T> As(const IRNodePtr& node);
```

**实现原理**：
1. 对于具体类型（有 `kind` 成员）：直接比较 `node->GetKind() == KindTrait<T>::kind`，O(1)
2. 对于抽象基类（有 `kinds` 数组）：遍历 `kinds[]` 查找匹配，O(N) 但 N 很小

**这比 `dynamic_cast` 快得多**，无需 RTTI 支持，仅需一次或几次整数比较。

**使用示例**：
```cpp
if (IsA<Var>(expr)) {
    auto var = As<Var>(expr);
    LOG(INFO) << "变量名: " << var->name_;
}

if (IsA<BinaryExpr>(expr)) {
    auto bin = As<BinaryExpr>(expr);
    // 处理所有 22 种二元表达式
}
```

---

## 7. IRBuilder 构建器

### 7.1 C++ IRBuilder

**头文件**: `framework/include/ir/builder.h`
**实现**: `framework/src/interface/ir/builder.cpp`

IRBuilder 使用**上下文栈**模式，管理嵌套的构建作用域：

```cpp
class IRBuilder {
public:
    // ===== 上下文管理 =====
    // 函数上下文
    VarPtr BeginFunction(const std::string& name, FunctionType type = FunctionType::Opaque);
    FunctionPtr EndFunction();

    // For 循环上下文
    VarPtr BeginForLoop(const std::string& loop_var_name, ExprPtr start, ExprPtr stop, ExprPtr step);
    void EndForLoop();

    // If 条件上下文
    void BeginIf(ExprPtr condition);
    void BeginElse();
    void EndIf();

    // 程序上下文
    void BeginProgram(const std::string& name = "");
    ProgramPtr EndProgram();

    // ===== 语句操作 =====
    VarPtr Emit(ExprPtr expr, const std::string& name = "", TypePtr type = nullptr);
    void Assign(VarPtr var, ExprPtr value);
    VarPtr Var(const std::string& name, TypePtr type);
    void Return(std::vector<ExprPtr> values = {});
    void Yield(std::vector<ExprPtr> values = {});
    void EvalStmt(ExprPtr expr);

    // ===== 参数管理 =====
    VarPtr AddParam(const std::string& name, TypePtr type);
    void SetReturnType(TypePtr type);
    void SetReturnTypes(std::vector<TypePtr> types);

    // ===== IterArg 管理 =====
    IterArgPtr AddIterArg(const std::string& name, TypePtr type, ExprPtr init_value);
    VarPtr AddReturnVar(const std::string& name, TypePtr type);
};
```

### 7.2 上下文栈模式

内部使用 `context_stack_` 维护嵌套上下文：

```cpp
// 上下文类型
enum class ContextType { FUNCTION, FOR_LOOP, IF_STMT, PROGRAM };

struct BuildContext {
    ContextType type;
    std::vector<StmtPtr> stmts;  // 当前上下文的语句列表
};

struct FunctionContext : BuildContext {
    std::string name;
    FunctionType func_type;
    std::vector<VarPtr> params;
    std::vector<TypePtr> return_types;
};

struct ForLoopContext : BuildContext {
    VarPtr loop_var;
    ExprPtr start, stop, step;
    std::vector<IterArgPtr> iter_args;
    std::vector<VarPtr> return_vars;
};

struct IfStmtContext : BuildContext {
    ExprPtr condition;
    StmtPtr then_body;       // EndIf 时构建
    std::optional<StmtPtr> else_body;
    bool in_else = false;
};
```

**Emit 的工作流程**：
1. 创建 `VarPtr` 和 `AssignStmt`
2. 如果传入了 `type`，设置为 Var 类型；否则尝试类型推导
3. 将 `AssignStmt` 追加到当前上下文的语句列表
4. 返回 `VarPtr`

**EndFunction 的工作流程**：
1. 从栈顶弹出 `FunctionContext`
2. 将语句列表包装为 `SeqStmts`
3. 创建 `Function` 节点
4. 如果在 `PROGRAM` 上下文中，自动将函数注册到程序

### 7.3 Python IRBuilder

**文件**: `python/pypto/ir/builder.py`（~1004 行）

Python 封装层在 C++ IRBuilder 之上添加了：

1. **上下文管理器**：`with ib.function(...) as f:`、`with ib.for_loop(...):`、`with ib.if_stmt(...):`
2. **自动 Span 捕获**：通过 `inspect.currentframe()` 捕获调用者的 Python 源码位置
3. **表达式规范化**：Python `int` → `ConstInt`，`float` → `ConstFloat`

```python
ib = IRBuilder()

with ib.program("my_program"):
    with ib.function("kernel") as f:
        a = f.param("a", TensorType([4, 8], FP16))
        b = f.param("b", TensorType([4, 8], FP16))
        f.return_type(TensorType([4, 8], FP32))

        tile_a = ib.let(op.block.load(a, 0, 0, 4, 8))
        tile_b = ib.let(op.block.load(b, 0, 0, 4, 8))

        with ib.for_loop("i", 0, 10, 1) as loop:
            result = ib.let(tile_a + tile_b)  # 运算符重载
            ib.eval_stmt(op.block.store(result, a, 0, 0))

        ib.return_stmt([])

program = ib.get_program()
```

**上下文管理器辅助类**：

| 类 | 方法 | 说明 |
|------|------|------|
| `FunctionBuilder` | `param(name, type)`, `return_type(type)`, `get_result()` | 函数定义作用域 |
| `ForLoopBuilder` | `var` 属性, `add_iter_arg(name, type, init)`, `add_return_var(name, type)` | 循环作用域 |
| `IfStmtBuilder` | `else_branch()` 上下文管理器 | 条件作用域 |
| `ProgramBuilder` | `get_result()` | 程序作用域 |

**Span 捕获机制**：
```python
def _capture_call_span(self):
    frame = inspect.currentframe()
    if frame and frame.f_back:
        info = inspect.getframeinfo(frame.f_back)
        return ir.Span(info.filename, info.lineno, -1)
    return ir.Span.unknown()
```

---

## 8. 算子注册与类型推导

### 8.1 OpRegistry 架构

**头文件**: `framework/include/ir/op_registry.h`
**实现**: `framework/src/interface/ir/op_registry.cpp`

```cpp
class OpRegistryEntry {
public:
    // Fluent builder API
    OpRegistryEntry& set_description(const std::string& desc);
    OpRegistryEntry& set_op_category(const std::string& category);
    OpRegistryEntry& add_argument(const std::string& name, const std::string& type,
                                  const std::string& desc);
    OpRegistryEntry& set_attr(const std::string& key, const std::string& type);

    template <typename T>
    OpRegistryEntry& set_attr_type(const std::string& key);

    OpRegistryEntry& f_deduce_type(
        std::function<TypePtr(const std::vector<ExprPtr>&,
                              const std::vector<std::pair<std::string, std::any>>&)> func);
    OpRegistryEntry& set_pipe(PipeType pipe);
    OpRegistryEntry& f_codegen_cce(std::function<...> func);
};

class OpRegistry {
public:
    static OpRegistry& Global();  // 全局单例
    OpRegistryEntry& Register(const std::string& name);
    OpPtr GetOp(const std::string& name) const;
    bool HasOp(const std::string& name) const;
};

// 注册宏（在全局初始化期间执行）
#define REGISTER_OP(name) \
    static auto& __op_registry_##name = OpRegistry::Global().Register(name)
```

### 8.2 算子注册示例

```cpp
// framework/src/interface/ir/op/block_ops/elementwise.cpp
REGISTER_OP("block.add")
    .set_description("Block-level element-wise addition")
    .set_op_category("BlockOp")
    .add_argument("lhs", "Expr", "Left operand (tile)")
    .add_argument("rhs", "Expr", "Right operand (tile)")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) -> TypePtr {
        CHECK(args.size() == 2) << "block.add expects 2 arguments";
        auto result = BroadcastShapes(args[0], args[1]);
        auto dtype = PromoteDataTypes(args[0], args[1]);
        return std::make_shared<TileType>(result.shape, dtype);
    })
    .set_pipe(PipeType::V);  // 在向量引擎上执行
```

### 8.3 类型推导 (type_inference.h / type_inference.cpp)

**文件**: `framework/include/ir/type_inference.h`, `framework/src/interface/ir/op/type_inference.cpp`

```cpp
struct BroadcastResult {
    std::vector<ExprPtr> shape;
    bool did_broadcast;
};

BroadcastResult BroadcastShapes(const ExprPtr& a, const ExprPtr& b);
DataType PromoteDataTypes(const ExprPtr& a, const ExprPtr& b);
void CheckTypeCompatibility(const TypePtr& expected, const TypePtr& actual,
                            const std::string& context);
DataType ExtractDataType(const ExprPtr& expr);
std::vector<ExprPtr> ExtractShape(const ExprPtr& expr);
std::string FormatShape(const std::vector<ExprPtr>& shape);
```

### 8.4 算子清单

**共 69 个注册算子**：

| 文件 | 算子数 | 算子列表 |
|------|--------|---------|
| `block_ops/elementwise.cpp` | 16 | mul, add, div, sub, muls, adds, divs, subs, maximum, minimum, cmp, cmps, col_expand_add/sub/mul/div |
| `block_ops/memory.cpp` | 8 | get_block_idx, load, store, l0c_store, move, alloc, zeros, alloc_tensor |
| `block_ops/matmul.cpp` | 2 | matmul, matmul_acc |
| `block_ops/batch_matmul.cpp` | 1 | batch_matmul |
| `block_ops/reduction.cpp` | 6 | sum, max, min, row_max, row_sum, row_min |
| `block_ops/unary.cpp` | 9 | neg, exp, recip, sqrt, rsqrt, log, abs, relu, cast |
| `block_ops/transform.cpp` | 3 | view, reshape, transpose |
| `block_ops/broadcast.cpp` | 4 | row_expand_sub, row_expand_div, row_expand_mul, row_expand_add |
| `tensor_ops/elementwise.cpp` | 9 | tensor_add, tensor_sub, tensor_mul, tensor_div 等 |
| `tensor_ops/memory.cpp` | 3 | tensor_load, tensor_store, tensor_alloc |
| `tensor_ops/matmul.cpp` | 2 | tensor_matmul, tensor_matmul_acc |
| `tensor_ops/reduction.cpp` | 2 | tensor_sum, tensor_max |
| `sync_ops/sync.cpp` | 5 | sync_src, sync_dst, bar_v, bar_m, bar_all |

### 8.5 ValidateKwargs

```cpp
void ValidateKwargs(const OpPtr& op,
                    const std::vector<std::pair<std::string, std::any>>& kwargs);
```

在 `Call` 构建时调用，验证：
1. 每个 kwarg 的 key 在 Op 的 `attrs_` 中注册过
2. 每个 kwarg 的值类型与注册的类型匹配

---

## 9. IR 遍历框架

### 9.1 Functor — 双分派基类

**文件**: `framework/include/ir/transform/base/functor.h`（307 行）

```cpp
template <typename R, typename... Args>
class ExprFunctor {
public:
    R VisitExpr(const ExprPtr& expr, Args... args);  // 分发器
protected:
    // 37 个纯虚方法，每种 Expr 类型一个
    virtual R VisitExpr_(const VarPtr& op, Args... args) = 0;
    virtual R VisitExpr_(const AddPtr& op, Args... args) = 0;
    virtual R VisitExpr_(const CallPtr& op, Args... args) = 0;
    // ... 等
};

template <typename R, typename... Args>
class StmtFunctor {
public:
    R VisitStmt(const StmtPtr& stmt, Args... args);  // 分发器
protected:
    // 8 个纯虚方法，每种 Stmt 类型一个
    virtual R VisitStmt_(const AssignStmtPtr& op, Args... args) = 0;
    virtual R VisitStmt_(const ForStmtPtr& op, Args... args) = 0;
    // ... 等
};

template <typename R, typename... Args>
class IRFunctor : public ExprFunctor<R, Args...>,
                  public StmtFunctor<R, Args...> {
public:
    R VisitIRNode(const IRNodePtr& node, Args... args);
};
```

**分发机制**（VisitExpr 内部）：
```cpp
#define EXPR_FUNCTOR_DISPATCH(OpType) \
    if (auto op = As<OpType>(expr)) { \
        return VisitExpr_(op, std::forward<Args>(args)...); \
    }

R VisitExpr(const ExprPtr& expr, Args... args) {
    EXPR_FUNCTOR_DISPATCH(Add);
    EXPR_FUNCTOR_DISPATCH(Sub);
    // ... 37 个分支
    throw InternalError("Unknown expression kind");
}
```

`PYPTO_DECLARE_ALL_VISITOR_OVERRIDES` 宏用于在子类中一次性声明所有 override。

### 9.2 IRVisitor — 只读遍历

**文件**: `framework/include/ir/transform/base/visitor.h`（53 行）
**实现**: `framework/src/interface/ir/transform/visitor.cpp`

```cpp
class IRVisitor : public IRFunctor<void> {
protected:
    // 默认实现：递归访问所有子节点
    void VisitExpr_(const AddPtr& op) override {
        VisitExpr(op->left_);
        VisitExpr(op->right_);
    }
    void VisitExpr_(const CallPtr& op) override {
        for (const auto& arg : op->args_) VisitExpr(arg);
    }
    void VisitStmt_(const ForStmtPtr& op) override {
        VisitExpr(op->start_);
        VisitExpr(op->stop_);
        VisitExpr(op->step_);
        for (const auto& arg : op->iter_args_) VisitExpr(arg->initValue_);
        VisitStmt(op->body_);
    }
    // ... 所有节点类型的默认递归实现
};
```

**使用示例 — 收集所有变量**：
```cpp
class VarCollector : public IRVisitor {
public:
    std::vector<VarPtr> vars;
    void VisitExpr_(const VarPtr& var) override {
        vars.push_back(var);
    }
};

VarCollector collector;
collector.VisitStmt(function->body_);
// collector.vars 包含函数体中所有引用的变量
```

### 9.3 IRMutator — 写时复制变换

**文件**: `framework/include/ir/transform/base/mutator.h`（95 行）
**实现**: `framework/src/interface/ir/transform/mutator.cpp`

```cpp
class IRMutator : public ExprFunctor<ExprPtr>,
                  public StmtFunctor<StmtPtr> {
protected:
    // 默认：仅在子节点改变时重建
    ExprPtr VisitExpr_(const AddPtr& op) override {
        auto lhs = VisitExpr(op->left_);
        auto rhs = VisitExpr(op->right_);
        if (lhs == op->left_ && rhs == op->right_) {
            return op;  // 子节点未变，直接返回原节点（共享）
        }
        return MakeAdd(lhs, rhs, op->span_);  // 创建新节点
    }

    StmtPtr VisitStmt_(const AssignStmtPtr& op) override {
        auto value = VisitExpr(op->value_);
        if (value == op->value_) return op;
        return std::make_shared<AssignStmt>(op->var_, value, op->span_);
    }

    // 函数/程序级遍历
    FunctionPtr VisitFunction(const FunctionPtr& func);
    ProgramPtr VisitProgram(const ProgramPtr& program);
};
```

**核心原则**：指针相等检查（`lhs == op->left_`）利用了 `shared_ptr` 引用相等语义。如果子树未被修改，返回原指针可最大化共享，避免不必要的复制。

**使用示例 — 常量折叠**：
```cpp
class ConstantFolder : public IRMutator {
    ExprPtr VisitExpr_(const AddPtr& op) override {
        auto lhs = VisitExpr(op->left_);
        auto rhs = VisitExpr(op->right_);
        // 如果两边都是常量，直接计算
        if (auto l = As<ConstInt>(lhs)) {
            if (auto r = As<ConstInt>(rhs)) {
                return std::make_shared<ConstInt>(
                    l->value_ + r->value_,
                    l->dtype(), op->span_);
            }
        }
        if (lhs == op->left_ && rhs == op->right_) return op;
        return MakeAdd(lhs, rhs, op->span_);
    }
};
```

---

## 10. Pass 框架

### 10.1 Pass 接口

**文件**: `framework/include/ir/transform/passes.h`
**实现**: `framework/src/interface/ir/transform/passes.cpp`

```cpp
class PassImpl {
public:
    virtual ~PassImpl() = default;
    virtual ProgramPtr operator()(const ProgramPtr& program) = 0;
    virtual std::string GetName() const;
};

class Pass {
public:
    Pass(std::shared_ptr<PassImpl> impl);
    ProgramPtr operator()(const ProgramPtr& program);
    ProgramPtr run(const ProgramPtr& program);  // 别名
    std::string GetName() const;
private:
    std::shared_ptr<PassImpl> impl_;  // Pimpl 模式
};
```

### 10.2 工厂函数

```cpp
// 函数级 Pass：对程序中每个函数独立应用变换
Pass CreateFunctionPass(
    std::function<FunctionPtr(const FunctionPtr&)> transform,
    std::string name);

// 程序级 Pass：处理整个程序
Pass CreateProgramPass(
    std::function<ProgramPtr(const ProgramPtr&)> transform,
    std::string name);
```

`CreateFunctionPass` 内部创建 `FunctionPassImpl`，在 `operator()` 中遍历 `program->functions_`，对每个函数调用 `transform`，然后重组为新 Program。

### 10.3 创建自定义 Pass

```cpp
// 方式 1：Lambda + IRMutator
auto my_pass = CreateFunctionPass(
    [](const FunctionPtr& func) -> FunctionPtr {
        MyMutator mutator;
        return mutator.VisitFunction(func);
    },
    "my_pass"
);

// 方式 2：继承 PassImpl
class MyPassImpl : public PassImpl {
public:
    ProgramPtr operator()(const ProgramPtr& program) override {
        // 自定义程序级变换
        return program;
    }
    std::string GetName() const override { return "MyPass"; }
};
Pass my_pass(std::make_shared<MyPassImpl>());

// 执行
auto result = my_pass(program);
```

### 10.4 Python PassManager

**文件**: `python/pypto/ir/pass_manager.py`

```python
class OptimizationStrategy(Enum):
    Default = "Default"
    PTOAS = "PTOAS"

class PassManager:
    def __init__(self, strategy=OptimizationStrategy.Default):
        self.passes = self._get_passes(strategy)

    def run_passes(self, input_ir, dump_ir=False, output_dir=None, prefix="pl"):
        current = input_ir
        for pass_instance in self.passes:
            current = pass_instance(current)
            if dump_ir:
                self._dump(current, output_dir, prefix)
        return current
```

当前策略的 Pass 列表为空（具体功能 Pass 待实现），但基础设施已完备。

---

## 11. 序列化系统

### 11.1 概述

**文件**: `framework/include/ir/serialization/serializer.h`, `deserializer.h`, `type_registry.h`
**实现**: `framework/src/interface/ir/serialization/serializer.cpp`, `deserializer.cpp`

使用 **MessagePack** 格式实现 IR 树的序列化和反序列化。

### 11.2 序列化器

```cpp
class IRSerializer {
public:
    static std::vector<uint8_t> Serialize(const IRNodePtr& node);
    static std::vector<uint8_t> Serialize(const FunctionPtr& func);
    static std::vector<uint8_t> Serialize(const ProgramPtr& program);
};
```

**内部实现** (`FieldSerializerVisitor`)：
1. 使用反射系统遍历节点的所有 `UsualField` 和 `DefField`
2. 对每个 IR 节点指针，分配唯一 ID 并记录 `{id, type, fields}` 三元组
3. **指针去重**：如果同一节点已序列化过，仅写入 `{ref: id}` 引用
4. `IgnoreField`（如 `span_`、`name_`）不序列化
5. 叶子值（int、double、string、DataType 等）直接编码

### 11.3 反序列化器

```cpp
class IRDeserializer {
public:
    static IRNodePtr Deserialize(const std::vector<uint8_t>& data);
    static FunctionPtr DeserializeFunction(const std::vector<uint8_t>& data);
    static ProgramPtr DeserializeProgram(const std::vector<uint8_t>& data);
};
```

**内部实现**：
1. 读取 `{id, type, fields}` 三元组
2. 通过 `TypeRegistry` 查找 `type` 对应的构造函数
3. 递归构建子节点
4. 通过 `id_to_ptr_` 映射解析 `{ref: id}` 引用

### 11.4 TypeRegistry — 类型注册表

```cpp
class TypeRegistry {
public:
    static TypeRegistry& Global();

    template <typename T>
    void Register(const std::string& type_name, ConstructorFunc func);

    IRNodePtr Construct(const std::string& type_name, const Fields& fields) const;
};

// RAII 注册辅助
template <typename T>
class TypeRegistrar {
public:
    TypeRegistrar(const std::string& name) {
        TypeRegistry::Global().Register<T>(name, ...);
    }
};

// 注册宏
#define REGISTER_IR_TYPE(Type) \
    static TypeRegistrar<Type> __type_registrar_##Type(#Type)
```

**当前注册了 48 种类型**，覆盖所有 IR 节点和类型节点。

---

## 12. 验证系统

**文件**: `framework/include/ir/transform/verifier.h`, `verification_error.h`

```cpp
class VerifyRule {
public:
    virtual ~VerifyRule() = default;
    virtual std::string GetName() const = 0;
    virtual std::vector<Diagnostic> Verify(const FunctionPtr& func) const = 0;
};

class IRVerifier {
public:
    static std::unique_ptr<IRVerifier> CreateDefault();

    void AddRule(std::unique_ptr<VerifyRule> rule);
    void EnableRule(const std::string& name);
    void DisableRule(const std::string& name);

    std::vector<Diagnostic> Verify(const FunctionPtr& func);      // 非抛出
    void VerifyOrThrow(const FunctionPtr& func);                   // 出错抛出 VerificationError
    std::string GenerateReport(const std::vector<Diagnostic>&);    // 人类可读报告
};
```

**Diagnostic 结构**：
```cpp
struct Diagnostic {
    DiagnosticSeverity severity;  // Error | Warning
    std::string rule_name;
    int error_code;
    std::string message;
    Span span;
};
```

**验证错误类型**：
```cpp
namespace ssa {
    enum class ErrorType { UndefinedVar, DuplicateVar, ... };
}
namespace typecheck {
    enum class ErrorType { TypeMismatch, ShapeMismatch, ... };
}
```

### 12.1 结构比较

**文件**: `framework/include/ir/transform/structural_comparison.h`

```cpp
bool structural_equal(const IRNodePtr& a, const IRNodePtr& b,
                      bool enable_auto_mapping = true);
size_t structural_hash(const IRNodePtr& node);
void assert_structural_equal(const IRNodePtr& a, const IRNodePtr& b,
                              bool enable_auto_mapping = true);
```

- `enable_auto_mapping=true`（默认）：`DefField` 标记的变量按出现顺序自动建立一一映射，允许不同名但结构相同的 IR 被判定相等
- `enable_auto_mapping=false`：仅比较变量的类型（不建立映射），更宽松
- `assert_structural_equal` 失败时抛出异常，附带详细差异信息

---

## 13. Python Printer

**文件**: `framework/include/ir/transform/printer.h`
**实现**: `framework/src/interface/ir/transform/printer.cpp`（1034 行）

```cpp
std::string PythonPrint(const FunctionPtr& func, const std::string& prefix = "pl");
std::string PythonPrint(const ProgramPtr& program, const std::string& prefix = "pl");
std::string PythonPrintType(const TypePtr& type, const std::string& prefix = "pl");
```

**主要特性**：

1. **运算符优先级**：14 个优先级层次（kLowest → kHighest），正确最小化括号
2. **SSA 控制流**：`for i, (x,) in pl.range(...)` + `pl.yield_(...)`
3. **程序结构**：`@pl.program class ... :` + `@pl.function` 方法
4. **类型注解**：所有变量带完整类型签名
5. **拓扑排序**：函数按依赖关系排序（被调用者在前）
6. **跨函数调用**：`self.callee_name(args)` 语法

**输出示例**：
```python
import pypto.language as pl

@pl.program
class MyProgram:
    @pl.function
    def kernel(self, a: pl.Tensor[[4, 8], pl.FP16], b: pl.Tensor[[4, 8], pl.FP16]):
        t0: pl.Tile[[4, 8], pl.FP16, memref=MemRef(UB, 0, 256, 1)] = pl.op.block.load(a, 0, 0, 4, 8)
        t1: pl.Tile[[4, 8], pl.FP16, memref=MemRef(UB, 256, 256, 2)] = pl.op.block.load(b, 0, 0, 4, 8)
        t2: pl.Tile[[4, 8], pl.FP32] = t0 + t1
        pl.op.block.store(t2, a, 0, 0)
```

---

## 14. pybind11 绑定层

### 14.1 模块入口

**文件**: `python/src/pybind11.cpp`（~30 行）

```cpp
NB_MODULE(pypto_impl, m) {
    auto ir_m = m.def_submodule("ir", "IR module");
    BindErrors(m);      // 异常翻译（必须最先）
    BindCore(m);        // DataType 桥接
    BindIR(ir_m);       // 所有 IR 类型
    BindPasses(ir_m);   // Pass 框架
}
```

### 14.2 IR 绑定 (ir.cpp)

**文件**: `python/src/bindings/modules/ir.cpp`（575 行）

绑定所有 C++ IR 类型到 Python：

| Python 类 | C++ 类 | 关键绑定 |
|-----------|--------|---------|
| `ir.DataType` | `DataType` | 所有静态常量（FP16, FP32, INT32 等）、ToString、GetBit、IsFloat/IsInt 等 |
| `ir.Span` | `Span` | 构造函数、unknown() 静态方法、filename/line/col 属性 |
| `ir.Expr` | `Expr` | GetType、GetKind |
| `ir.Var` | `Var` | 构造函数、name/type 属性 |
| `ir.IterArg` | `IterArg` | 构造函数、initValue 属性 |
| `ir.MemRef` | `MemRef` | 构造函数、memory_space/addr/size/id 属性 |
| `ir.Op` | `Op` | `get(name)` 静态方法、name/pipe 属性 |
| `ir.GlobalVar` | `GlobalVar` | 构造函数 |
| `ir.Call` | `Call` | 四个构造函数、GetKwarg/HasKwarg、op/args/kwargs 属性 |
| `ir.ConstInt` | `ConstInt` | 构造函数、value/dtype 属性 |
| `ir.ConstFloat` | `ConstFloat` | 构造函数、value/dtype 属性 |
| `ir.ConstBool` | `ConstBool` | 构造函数、value 属性 |
| 所有 22 个 BinaryExpr | 对应 C++ 类 | 构造函数、left/right 属性 |
| 所有 5 个 UnaryExpr | 对应 C++ 类 | 构造函数、operand 属性 |
| `ir.MakeTuple` | `MakeTuple` | 构造函数、elements 属性 |
| `ir.TupleGetItemExpr` | `TupleGetItemExpr` | 构造函数、tuple/index 属性 |
| 所有 8 个 Stmt 类 | 对应 C++ 类 | 构造函数、所有字段属性 |
| `ir.Function` | `Function` | 构造函数、name/params/return_types/body/func_type 属性 |
| `ir.Program` | `Program` | 两个构造函数、GetFunction/GetGlobalVar/functions 属性 |
| 所有 7 个 Type 类 | 对应 C++ 类 | 构造函数、所有字段属性 |
| `ir.IRBuilder` | `IRBuilder` | 所有 Begin/End/Emit/Var/Return/Yield 等方法 |
| `ir.python_print` | `PythonPrint` | 函数/程序打印 |
| `ir.python_print_type` | `PythonPrintType` | 类型打印 |
| `ir.structural_equal` | `structural_equal` | 结构比较 |
| `ir.structural_hash` | `structural_hash` | 结构哈希 |
| `ir.assert_structural_equal` | `assert_structural_equal` | 结构断言 |
| Make* 函数 | 对应 C++ 函数 | MakeAdd、MakeSub 等所有构造辅助函数 |

### 14.3 Pass 绑定 (passes.cpp)

**文件**: `python/src/bindings/modules/passes.cpp`（81 行）

绑定 `Pass`、`CreateFunctionPass`、`CreateProgramPass`、`IRVerifier`、`Diagnostic`、`DiagnosticSeverity`。

### 14.4 异常翻译 (error.cpp)

**文件**: `python/src/bindings/modules/error.cpp`（60 行）

将 7 种 C++ 异常 + `VerificationError` 注册为 Python 异常并安装翻译器。**必须最先注册**。

---

## 15. Python 封装层

### 15.1 模块组织

**文件**: `python/pypto/ir/__init__.py`（74 行）

```python
from pypto.pypto_impl.ir import *         # 重新导出 C++ 绑定
from pypto.pypto_impl.ir import DataType  # 显式导出

from . import op, operators               # 运算符重载（导入时自动猴子补丁）
from .builder import IRBuilder            # Python IRBuilder
from .pass_manager import OptimizationStrategy, PassManager
from .printer import python_print
from .type import TensorType, TileType    # 增强的构造函数

# DataType 便捷别名
FP16 = DataType.FP16
FP32 = DataType.FP32
INT32 = DataType.INT32
# ... 等
```

### 15.2 工具函数 (utils.py)

**文件**: `python/pypto/ir/utils.py`（101 行）

| 函数 | 签名 | 说明 |
|------|------|------|
| `_normalize_expr` | `(value, span=None, int_dtype=INT64, float_dtype=FP32)` | int → ConstInt, float → ConstFloat, Expr 透传 |
| `_normalize_shape` | `(shape, span=None)` | `[4, 8]` → `[ConstInt(4, INT64), ConstInt(8, INT64)]` |
| `_get_span_or_capture` | `(span=None, frame_offset=2)` | 返回显式 Span 或通过 inspect 捕获调用栈位置 |

### 15.3 类型猴子补丁 (type.py)

**文件**: `python/pypto/ir/type.py`（73 行）

修补 `TensorType.__init__` 和 `TileType.__init__`，使其接受 Python 整数 shape：

```python
# 修补前：TensorType([ConstInt(4, INT64, span), ConstInt(8, INT64, span)], DataType.FP32)
# 修补后：TensorType([4, 8], DataType.FP32)
# 也支持：TensorType([4, 8], DataType.FP32, memref)
```

### 15.4 运算符重载 (operators.py)

**文件**: `python/pypto/ir/operators.py`（146 行）

为 `Expr` 类添加 30+ Python 运算符：

| 类别 | 运算符 |
|------|--------|
| 算术 | `+`, `-`, `*`, `/`, `//`, `%`, `**` |
| 比较 | `==`, `!=`, `<`, `<=`, `>`, `>=` |
| 位运算 | `&`, `\|`, `^`, `<<`, `>>` |
| 一元 | `-`（取负）, `~`（位取反） |
| 反向 | `__radd__`, `__rmul__`, `__rsub__` 等 |

每个封装器：(1) 捕获 Span (2) 规范化操作数 (3) 调用 C++ Make* 函数。

### 15.5 Block 操作 (op/block_ops.py)

**文件**: `python/pypto/ir/op/block_ops.py`（~1001 行）

所有 Block 算子的 Python 封装。每个函数负责：参数类型验证、默认值处理、Span 捕获、表达式规范化、构建 Call 节点。

**操作分类**：

| 分类 | 操作 |
|------|------|
| 内存 | `load`, `store`, `move`, `alloc`, `zeros`, `get_block_idx` |
| 逐元素 | `add`, `sub`, `mul`, `div`, `maximum`, `minimum` |
| 标量运算 | `add_scalar`, `mul_scalar`, `sub_scalar`, `div_scalar` |
| 一元 | `neg`, `exp`, `recip`, `sqrt`, `rsqrt`, `log`, `abs`, `relu`, `cast` |
| 矩阵 | `matmul`, `matmul_acc`, `batch_matmul` |
| 规约 | `reduce_sum`, `reduce_max`, `reduce_min`, `row_max`, `row_sum`, `row_min` |
| 变换 | `view`, `reshape`, `transpose` |
| 广播 | `row_expand_add/sub/mul/div` |

### 15.6 Tensor 操作 (op/tensor_ops.py)

**文件**: `python/pypto/ir/op/tensor_ops.py`（~477 行）

Tensor 级操作封装，部分函数会根据输入类型自动选择 block 或 tensor 变体。

### 15.7 Printer (printer.py)

**文件**: `python/pypto/ir/printer.py`（36 行）

```python
def python_print(node, prefix="pl"):
    if isinstance(node, Type):
        return ir.python_print_type(node, prefix)
    else:
        return ir.python_print(node, prefix)
```

---

## 16. 算子开发指南

### 16.1 添加新 Block 算子的完整步骤

以添加 `block.sigmoid` 算子为例：

#### 步骤 1：C++ 算子注册

在 `framework/src/interface/ir/op/block_ops/` 中新建或修改文件：

```cpp
REGISTER_OP("block.sigmoid")
    .set_description("Block-level sigmoid activation: 1 / (1 + exp(-x))")
    .set_op_category("BlockOp")
    .add_argument("input", "Expr", "Input tile expression")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) -> TypePtr {
        CHECK(args.size() == 1) << "block.sigmoid expects 1 argument, got " << args.size();
        return args[0]->GetType();  // 返回类型与输入相同
    })
    .set_pipe(PipeType::V);  // 向量引擎
```

#### 步骤 2：Python 封装

在 `python/pypto/ir/op/block_ops.py` 中添加：

```python
def sigmoid(input_tile, span=None):
    """Block 级 sigmoid 激活函数。

    Args:
        input_tile: 输入 tile 表达式
        span: 可选源码位置

    Returns:
        Call 表达式，表示 sigmoid 操作
    """
    actual_span = _get_span_or_capture(span)
    args = [input_tile]
    return Call(Op.get("block.sigmoid"), args, actual_span)
```

#### 步骤 3：使用

```python
from pypto.ir import IRBuilder, TensorType, FP16
from pypto.ir import op

ib = IRBuilder()
with ib.function("sigmoid_kernel") as f:
    a = f.param("a", TensorType([4, 8], FP16))
    f.return_type(TensorType([4, 8], FP16))

    tile = ib.let(op.block.load(a, 0, 0, 4, 8))
    result = ib.let(op.block.sigmoid(tile))
    ib.eval_stmt(op.block.store(result, a, 0, 0))
    ib.return_stmt([])

func = f.get_result()
print(python_print(func))
```

#### 步骤 4：测试

```python
def test_sigmoid_op_registered():
    op = ir.Op.get("block.sigmoid")
    assert op.name == "block.sigmoid"

def test_sigmoid_call():
    span = ir.Span.unknown()
    tile_type = ir.TileType([ir.ConstInt(4, ir.DataType.INT64, span)], ir.DataType.FP16)
    var = ir.Var("x", tile_type, span)
    result = ir.Call(ir.Op.get("block.sigmoid"), [var], span)
    assert result.op.name == "block.sigmoid"
    assert len(result.args) == 1
```

### 16.2 添加带 kwargs 的算子

以带 `axis` 参数的 `block.softmax` 为例：

```cpp
// C++ 注册
REGISTER_OP("block.softmax")
    .set_description("Block-level softmax")
    .set_op_category("BlockOp")
    .add_argument("input", "Expr", "Input tile")
    .set_attr_type<int>("axis")  // 注册 kwarg 类型
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) -> TypePtr {
        CHECK(args.size() == 1);
        return args[0]->GetType();
    })
    .set_pipe(PipeType::V);
```

```python
# Python 封装
def softmax(input_tile, axis=-1, span=None):
    actual_span = _get_span_or_capture(span)
    args = [input_tile]
    kwargs = {"axis": axis}
    return Call(Op.get("block.softmax"), args, kwargs, actual_span)
```

### 16.3 添加新 Pass

```cpp
// C++ 方式
auto constant_fold = CreateFunctionPass(
    [](const FunctionPtr& func) -> FunctionPtr {
        ConstantFolder folder;
        return folder.VisitFunction(func);
    },
    "constant_fold"
);

auto result = constant_fold(program);
```

```python
# Python 方式
def my_transform(func):
    # 使用 Python 构建新 IR
    return func

my_pass = ir.CreateFunctionPass(my_transform, "my_pass")
result = my_pass(program)
```

---

## 17. 端到端工作流程

```
1. 用户编写 Python 代码
┌──────────────────────────────────────────────────┐
│ ib = IRBuilder()                                  │
│ with ib.function("kernel") as f:                  │
│     x = f.param("x", TensorType([4,8], FP16))    │
│     tile = ib.let(op.block.load(x, 0, 0, 4, 8))  │
│     result = ib.let(tile + tile)                  │
│     ib.eval_stmt(op.block.store(result, x, 0, 0)) │
│     ib.return_stmt([])                            │
│ func = f.get_result()                             │
└──────────────────────────────────────────────────┘
                    │
                    v
2. Python 封装层规范化（自动发生）
   - int 0 → ConstInt(0, INT64, Span("user.py", 5, -1))
   - tile + tile → MakeAdd(tile, tile, span)
   - op.block.load(...) → Call(Op.get("block.load"), args, kwargs, span)
                    │
                    v
3. C++ IRBuilder 构建不可变 IR 树
   Function "kernel" {
     params: [x: TensorType([4,8], FP16)]
     return_types: []
     body: SeqStmts [
       AssignStmt(tile, Call("block.load", [x, 0, 0, 4, 8]))
       AssignStmt(result, Add(tile, tile))
       EvalStmt(Call("block.store", [result, x, 0, 0]))
       ReturnStmt([])
     ]
   }
                    │
                    v
4. PassManager 执行变换流水线
   ┌────────────────────────────────────────────────┐
   │ Pass 基础设施已就绪                             │
   │ 具体 Pass（SSA 转换、内存分配、同步插入）       │
   │ 待后续实现                                      │
   └────────────────────────────────────────────────┘
                    │
                    v
5. python_print(func) 输出可读 IR
   ┌────────────────────────────────────────────────┐
   │ import pypto.language as pl                     │
   │                                                 │
   │ @pl.function                                    │
   │ def kernel(x: pl.Tensor[[4, 8], pl.FP16]):     │
   │     tile = pl.op.block.load(x, 0, 0, 4, 8)    │
   │     result = tile + tile                        │
   │     pl.op.block.store(result, x, 0, 0)         │
   └────────────────────────────────────────────────┘
                    │
                    v
6. 代码生成 → CCE 二进制 → 硬件执行
```

---

## 18. 测试策略

### 18.1 测试组织

```
python/tests/
├── ut/                          # Python 单元测试（~30 个文件）
│   ├── ir/core/                 # IR 核心类型测试
│   │   ├── test_op.py          # Op 注册表测试
│   │   ├── test_span.py        # Span 测试
│   │   ├── test_tuple_type.py  # 复合类型测试
│   │   └── test_var.py         # 变量创建测试
│   ├── interface/               # 接口测试（16 个文件）
│   ├── operation/               # 操作测试
│   └── operator/                # 算子测试
└── st/                          # 系统测试（~90 个文件）
    ├── interface/               # 集成测试
    ├── operation/               # 端到端操作测试
    └── operator/                # 模型/算子测试

framework/tests/                 # C++ 测试（~357 个文件）
├── ut/                          # C++ 单元测试
└── st/                          # C++ 系统测试
```

### 18.2 测试示例

```python
# 单元测试 — 基础类型
def test_var_creation():
    span = ir.Span.unknown()
    var = ir.Var("x", ir.ScalarType(ir.DataType.INT64), span)
    assert var.name == "x"
    assert isinstance(var.type, ir.ScalarType)

# 单元测试 — 结构比较
def test_structural_equal():
    span = ir.Span.unknown()
    a = ir.ConstInt(42, ir.DataType.INT64, span)
    b = ir.ConstInt(42, ir.DataType.INT64, span)
    assert ir.structural_equal(a, b)

# 系统测试 — 端到端
def test_add_e2e():
    ib = IRBuilder()
    with ib.function("add_kernel") as f:
        a = f.param("a", TensorType([4, 8], FP16))
        b = f.param("b", TensorType([4, 8], FP16))
        tile_a = ib.let(op.block.load(a, 0, 0, 4, 8))
        tile_b = ib.let(op.block.load(b, 0, 0, 4, 8))
        result = ib.let(op.block.add(tile_a, tile_b))
        ib.eval_stmt(op.block.store(result, a, 0, 0))
        ib.return_stmt([])
    func = f.get_result()
    output = python_print(func)
    assert "block.add" in output
```

---

## 19. 完整文件索引

### C++ 核心头文件

| 文件 | 行数 | 用途 |
|------|------|------|
| `include/core/dtype.h` | 342 | DataType 类，20+ 类型常量 |
| `include/core/error.h` | ~200 | 7 种异常类型，Backtrace，Diagnostic |
| `include/core/logging.h` | ~300 | 日志系统，CHECK 宏 |
| `include/core/common.h` | ~50 | 版本常量，kDynamicDim，宏 |
| `include/core/any_cast.h` | ~80 | AnyCast/AnyCastRef，类型 demangling |

### IR 节点头文件

| 文件 | 行数 | 用途 |
|------|------|------|
| `include/ir/core.h` | 227 | IRNode 基类，ObjectKind (56 值)，KindTrait 前向声明 |
| `include/ir/type.h` | 427 | 7 种类型类，TileView 结构 |
| `include/ir/expr.h` | 550 | Expr/Op/GlobalVar/Var/IterArg/MemRef/Call/MakeTuple/TupleGetItemExpr |
| `include/ir/scalar_expr.h` | 516 | ScalarExpr，ConstInt/Float/Bool，22 BinaryExpr，5 UnaryExpr，Make* 函数 |
| `include/ir/stmt.h` | 423 | 8 种语句类（AssignStmt 到 EvalStmt） |
| `include/ir/function.h` | 138 | Function 类，FunctionType 枚举 |
| `include/ir/program.h` | 106 | Program 类 |
| `include/ir/span.h` | ~60 | Span 类 |
| `include/ir/memref.h` | ~80 | MemorySpace 枚举 (DDR/UB/L1/L0A/L0B/L0C) |
| `include/ir/pipe.h` | ~60 | PipeType (MTE1-3/V/M/ALL)，CoreType (CUBE/VECTOR) |
| `include/ir/kind_traits.h` | ~300 | KindTrait 特化（44 具体 + 抽象基类），IsA/As 函数 |

### 基础设施头文件

| 文件 | 行数 | 用途 |
|------|------|------|
| `include/ir/builder.h` | ~300 | C++ IRBuilder（上下文栈 API） |
| `include/ir/op_registry.h` | ~300 | OpRegistryEntry/OpRegistry，REGISTER_OP 宏 |
| `include/ir/type_inference.h` | ~100 | BroadcastShapes/PromoteDataTypes 等 |
| `include/ir/reflection/field_traits.h` | ~150 | DefField/UsualField/IgnoreField 字段描述符 |
| `include/ir/reflection/field_visitor.h` | ~200 | FieldIterator 字段迭代器 |
| `include/ir/serialization/serializer.h` | ~50 | IRSerializer（MessagePack 序列化） |
| `include/ir/serialization/deserializer.h` | ~50 | IRDeserializer（MessagePack 反序列化） |
| `include/ir/serialization/type_registry.h` | ~100 | TypeRegistry/REGISTER_IR_TYPE |
| `include/ir/transform/passes.h` | ~80 | PassImpl/Pass 接口，CreateFunctionPass/CreateProgramPass |
| `include/ir/transform/base/functor.h` | 307 | ExprFunctor/StmtFunctor/IRFunctor 双分派模板 |
| `include/ir/transform/base/visitor.h` | 53 | IRVisitor（只读递归遍历） |
| `include/ir/transform/base/mutator.h` | 95 | IRMutator（写时复制变换） |
| `include/ir/transform/printer.h` | ~50 | PythonPrint 函数声明 |
| `include/ir/transform/verifier.h` | ~100 | VerifyRule/IRVerifier |
| `include/ir/transform/verification_error.h` | ~50 | ssa::ErrorType/typecheck::ErrorType |
| `include/ir/transform/structural_comparison.h` | ~30 | structural_equal/hash/assert |

### C++ 实现文件

| 文件 | 行数 | 用途 |
|------|------|------|
| `src/ir/core.cpp` | ~50 | IRNode 实现 |
| `src/ir/type.cpp` | ~100 | ShapedType 常量 shape 构造 |
| `src/ir/expr.cpp` | ~100 | MakeTuple/TupleGetItemExpr 构造逻辑 |
| `src/ir/builder.cpp` | ~400 | IRBuilder 上下文栈实现 |
| `src/ir/op_registry.cpp` | ~200 | OpRegistry 单例、注册、验证 |
| `src/ir/op/type_inference.cpp` | ~150 | BroadcastShapes/PromoteDataTypes 实现 |
| `src/ir/op/block_ops/*.cpp` | ~1500 | 37 个 Block 算子注册（8 文件） |
| `src/ir/op/tensor_ops/*.cpp` | ~500 | 16 个 Tensor 算子注册（4 文件） |
| `src/ir/op/sync_ops/*.cpp` | ~100 | 5 个同步算子注册（1 文件） |
| `src/ir/serialization/serializer.cpp` | ~300 | FieldSerializerVisitor，指针去重 |
| `src/ir/serialization/deserializer.cpp` | ~300 | 引用解析，TypeRegistry 分发 |
| `src/ir/transform/passes.cpp` | ~100 | FunctionPassImpl/ProgramPassImpl |
| `src/ir/transform/visitor.cpp` | ~200 | IRVisitor 默认递归实现，宏生成 |
| `src/ir/transform/mutator.cpp` | ~300 | IRMutator copy-on-write 实现 |
| `src/ir/transform/printer.cpp` | 1034 | IRPythonPrinter，优先级，拓扑排序 |
| `src/ir/transform/verifier.cpp` | ~150 | IRVerifier 实现 |

### pybind11 绑定

| 文件 | 行数 | 用途 |
|------|------|------|
| `python/src/pybind11.cpp` | ~30 | 模块入口（pypto_impl） |
| `python/src/bindings/bindings.h` | ~30 | 前向声明 |
| `python/src/bindings/modules/ir.cpp` | 575 | 所有 IR 类型绑定到 Python |
| `python/src/bindings/modules/passes.cpp` | 81 | Pass/Verifier 绑定 |
| `python/src/bindings/modules/core.cpp` | 25 | Core 桩（DataType 桥接） |
| `python/src/bindings/modules/error.cpp` | 60 | 7 种异常翻译 |

### Python 封装层

| 文件 | 行数 | 用途 |
|------|------|------|
| `python/pypto/ir/__init__.py` | 74 | 模块重新导出 + DataType 别名 |
| `python/pypto/ir/builder.py` | 1004 | Python IRBuilder + 4 种上下文管理器 |
| `python/pypto/ir/operators.py` | 146 | 30+ 运算符重载（猴子补丁到 Expr） |
| `python/pypto/ir/type.py` | 73 | TensorType/TileType 构造函数补丁 |
| `python/pypto/ir/utils.py` | 101 | _normalize_expr/_normalize_shape/_get_span_or_capture |
| `python/pypto/ir/printer.py` | 36 | python_print 分发器（Type vs Node） |
| `python/pypto/ir/pass_manager.py` | 173 | PassManager + OptimizationStrategy |
| `python/pypto/ir/op/__init__.py` | ~20 | 算子模块入口 |
| `python/pypto/ir/op/block_ops.py` | 1001 | 37+ Block 操作 Python 封装 |
| `python/pypto/ir/op/tensor_ops.py` | 477 | 16+ Tensor 操作 Python 封装 |

### 测试

| 目录 | 数量 | 用途 |
|------|------|------|
| `python/tests/ut/` | ~30 | Python 单元测试 |
| `python/tests/st/` | ~90 | Python 系统测试（端到端 + 设备验证） |
| `framework/tests/` | ~357 | C++ 框架测试（编译、仿真、正确性） |

---

*本文档基于 `/data/g00655722/new-ir/pypto_yhz` 代码库分析生成。*
