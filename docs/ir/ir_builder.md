# IRBuilder

## OverView

`IRBuilder` 是 PTO-IR 中用于构建中间表示（IR）的核心工具类。它提供了类型安全、结构化的 API 来构建程序模块（ProgramModule）、函数（Function）、语句（Statement）和操作（Operation）。

IRBuilder 的使用流程：
```
IRBuilder
├── 创建模块和函数
│   ├── CreateFunction()
│   └── EnterFunctionBody()
├── 创建值对象
│   ├── CreateTensor()
│   ├── CreateScalar()
│   └── CreateConst()
├── 构建操作
│   └── CreateOp()
└── 构建语句
    ├── CreateOpStmt()
    ├── CreateForStmt()
    ├── CreateIfStmt()
    ├── CreateReturn()
    └── CreateYield()
```

示例：
```cpp
auto module = std::make_shared<ProgramModule>("main");
IRBuilder builder(module);

// 创建函数签名
FunctionSignature sig;
auto inputTensor = std::make_shared<Tensor>(tensorShape, DataType::FP32, "input");
sig.arguments = { inputTensor };

// 创建函数
auto func = builder.CreateFunction("test_value", FunctionKind::ControlFlow, sig, true);

{
    // 进入函数体作用域
    auto guard = builder.EnterFunctionBody(func);
    
    // 创建常量
    auto constant0 = builder.CreateConst(int64_t(0), "const_0");
    
    // 创建操作
    auto result = builder.CreateOp(
        Opcode::OP_MUL,
        { inputTensor, scale1 },
        nullptr,
        "result"
    )[0];
    
    // 创建返回语句
    builder.CreateReturn({ result });
}
```

## IRBuilder

IRBuilder 维护当前构建状态，通过 RAII 机制（ScopeGuard）管理嵌套作用域。

### Syntax
```cpp
IRBuilder builder(module);
auto func = builder.CreateFunction(name, kind, sig, setAsEntry);
auto guard = builder.EnterFunctionBody(func);
// ... 构建IR ...
// guard 析构时自动恢复之前的状态
```

### 数据结构
```cpp
class IRBuilder {
    std::shared_ptr<ProgramModule> module_;
    std::shared_ptr<Function> func_;
    CompoundStatementPtr compound_;
    OpStatementPtr opStmt_;
};
```

### 约束
- IRBuilder 必须关联一个 ProgramModule
- 所有操作必须在函数体作用域内进行
- 使用 ScopeGuard 确保作用域的正确嵌套和自动清理

## 模块和函数管理

### CreateFunction

创建新函数并添加到模块中。

#### Syntax
```cpp
auto func = builder.CreateFunction(
    "test_value",
    FunctionKind::ControlFlow,
    sig,
    /*setAsEntry=*/true
);
```

#### 数据结构
```cpp
std::shared_ptr<Function> CreateFunction(
    std::string name,
    FunctionKind kind,
    FunctionSignature sig,
    bool setAsEntry = false
);
```

**参数**：
- `name`: 函数名称
- `kind`: 函数类型（ControlFlow、DataFlow、Kernel）
- `sig`: 函数签名（参数和返回值）
- `setAsEntry`: 是否设置为程序入口点

**返回**：创建的 Function 对象

#### 约束
- 函数名在模块内必须唯一
- 函数签名中的参数类型必须是 Tensor、Tile 或 Scalar

### EnterFunctionBody

进入函数体作用域，返回 ScopeGuard 对象用于自动管理作用域。

#### Syntax
```cpp
{
    auto guard = builder.EnterFunctionBody(func);
    // 在函数体作用域内构建IR
    // guard 析构时自动退出函数体作用域
}
```

#### 数据结构
```cpp
std::shared_ptr<ScopeGuard> EnterFunctionBody(std::shared_ptr<Function> func);
```

#### 约束
- 必须使用 RAII 方式管理 ScopeGuard，确保作用域正确退出
- 在函数体作用域内才能创建操作和语句

## 值创建

### CreateTensor

创建 Tensor 值对象。

#### Syntax
```cpp
std::vector<Scalar> shape = { batch, Scalar(int64_t(128)) };
auto tensor = builder.CreateTensor(shape, DataType::FP32, "input");
```

#### 数据结构
```cpp
std::shared_ptr<Tensor> CreateTensor(
    const std::vector<Scalar>& shape, 
    DataType dt, 
    std::string name = ""
);
```

#### 约束
- shape 中的 Scalar 可以是常量或符号值
- 必须在函数体作用域内调用

### CreateScalar

创建符号标量（Symbolic Scalar）值对象。

#### Syntax
```cpp
auto scalar = builder.CreateScalar(DataType::FP32, "scale1");
auto iv = builder.CreateScalar(DataType::INT32, "i");
```

#### 数据结构
```cpp
std::shared_ptr<Scalar> CreateScalar(DataType dt, std::string name = "");
```

#### 约束
- 创建的标量是符号值，运行时确定
- 必须在函数体作用域内调用

### CreateConst

创建常量标量值对象。

#### Syntax
```cpp
auto const0 = builder.CreateConst(int64_t(0), "const_0");
auto const1 = builder.CreateConst(int64_t(1), "const_1");
auto pi = builder.CreateConst(3.14, "const_pi");
```

#### 数据结构
```cpp
std::shared_ptr<Scalar> CreateConst(int64_t v, std::string name = "");
std::shared_ptr<Scalar> CreateConst(double v, std::string name = "");
```

#### 约束
- 支持 int64_t 和 double 类型的常量
- 类型自动推断（int64_t → INT64，double → FP64）
- 必须在函数体作用域内调用

## 操作构建

### CreateOp

构建操作的核心接口，通过 Schema 系统统一处理各种操作的语义。

#### Syntax
```cpp
// 创建操作（无 payload）
auto result = builder.CreateOp(
    Opcode::OP_MUL,
    { input1, input2 },
    nullptr,
    "result"
)[0];

// 创建操作（带 payload）
ViewSpec viewSpec;
viewSpec.shape = { 1, 128 };
viewSpec.offset = { *constant0, *constant0 };
auto view = builder.CreateOp(
    Opcode::OP_VIEW,
    { inputTensor },
    std::make_shared<ViewPayload>(viewSpec),
    "loop_tile"
)[0];
```

#### 数据结构
```cpp
ValuePtrs CreateOp(
    Opcode opcode,
    ValuePtrs inputs,
    std::shared_ptr<OpPayload> payload = nullptr,
    std::string name = ""
);
```

**参数**：
- `opcode`: 操作码
- `inputs`: 输入值列表
- `payload`: 操作特定的负载数据（如 ViewSpec、AssembleSpec 等）
- `name`: 结果值的名称

**返回**：操作输出的值列表（ValuePtrs）

#### 约束
- 所有操作必须位于 OpStatement 中
- 输入值的类型必须符合操作的 Schema 定义
- 操作会自动添加到当前 OpStatement
- 必须在函数体作用域内调用

## 语句构建

### CreateOpStmt

创建操作语句，作为操作的容器。

#### Syntax
```cpp
auto opStmt = builder.CreateOpStmt();
// 后续的 CreateOp 调用会将操作添加到这个 OpStatement 中
```

#### 数据结构
```cpp
OpStatementPtr CreateOpStmt();
```

#### 约束
- 如果当前没有 OpStatement，CreateOp 会自动创建一个
- OpStatement 中可以包含多个操作，按顺序执行

### CreateForStmt

创建 for 循环语句。

#### Syntax
```cpp
auto i = builder.CreateScalar(DataType::INT32, "i");
auto constant0 = builder.CreateConst(int64_t(0), "const_0");
auto constant1 = builder.CreateConst(int64_t(1), "const_1");
auto batch = ...; // 循环上界

auto fs = builder.CreateForStmt(i, constant0, batch, constant1);
{
    auto fsGuard = builder.EnterForBody(fs);
    // 循环体内的操作
    // ...
    builder.ExitForStatement(fs);
}
```

#### 数据结构
```cpp
ForStatementPtr CreateForStmt(
    std::shared_ptr<Scalar> iv,
    std::shared_ptr<Scalar> start,
    std::shared_ptr<Scalar> end,
    std::shared_ptr<Scalar> step
);
```

**参数**：
- `iv`: 循环变量（induction variable）
- `start`: 起始值
- `end`: 结束值
- `step`: 步长

#### 约束
- 循环变量、起始值、结束值、步长都必须是 Scalar 类型
- 必须使用 `EnterForBody()` 进入循环体作用域
- 循环体末尾必须调用 `ExitForStatement()` 或自动添加 yield
- IRBuilder 会自动识别循环携带变量并创建 iter_args

### CreateIfStmt

创建 if 条件语句。

#### Syntax
```cpp
auto ifs = builder.CreateIfStmt("i");
ValuePtr resIfX, resIfY;
{
    auto ifThenGuard = builder.EnterIfThen(ifs);
    resIfX = builder.CreateOp(Opcode::OP_MUL, {resLoopX, scale1}, nullptr, "outputX")[0];
}
{
    auto ifElseGuard = builder.EnterIfElse(ifs);
    resIfY = builder.CreateOp(Opcode::OP_MUL, {resLoopY, scale2}, nullptr, "outputY")[0];
}
builder.ExitIfStatement(ifs);
```

#### 数据结构
```cpp
IfStatementPtr CreateIfStmt(std::string cond);
```

**参数**：
- `cond`: 条件表达式（字符串形式，如 "i"）

#### 约束
- 条件必须是布尔类型的 Scalar
- 必须使用 `EnterIfThen()` 和 `EnterIfElse()` 分别进入 then 和 else 分支
- 必须调用 `ExitIfStatement()` 完成 if 语句构建
- IRBuilder 会自动在 then/else 分支末尾添加 yield 语句
- 若 then 和 else 中修改的变量不同，builder 会自动补齐 yield

### CreateReturn

创建返回语句。

#### Syntax
```cpp
builder.CreateReturn({ result });
builder.CreateReturn(fs->Results());  // 返回循环结果
```

#### 数据结构
```cpp
ReturnStatementPtr CreateReturn(ValuePtrs values);
```

#### 约束
- 返回值必须是可通过寄存器传递的值（Scalar）
- 输出 tensor 通过参数方式传递，不在返回值中
- 目前 ReturnStatement 必须且仅能在函数尾出现

### CreateYield

创建 yield 语句（用于循环和条件分支的值传递）。

#### Syntax
```cpp
builder.CreateYield({ new_acc0, new_acc1 });
```

#### 数据结构
```cpp
YieldStatementPtr CreateYield(ValuePtrs values);
```

#### 约束
- 在 for 循环体中，yield 用于提供更新后的循环传递值
- 在 if 的 then/else 分支中，yield 用于将分支结果回传给 if 语句
- IRBuilder 通常会自动添加 yield，一般不需要手动调用

## 作用域管理

IRBuilder 使用嵌套的作用域结构来管理变量的可见性，通过 RAII 机制（ScopeGuard）自动管理作用域的生命周期。

### ScopeGuard

`ScopeGuard` 是 RAII 风格的辅助类，用于管理 IRBuilder 的作用域转换。

#### Syntax
```cpp
{
    auto guard = builder.EnterFunctionBody(func);
    // 在函数体作用域内
    {
        auto fsGuard = builder.EnterForBody(fs);
        // 在循环体作用域内
        // fsGuard 析构时自动退出循环体作用域
    }
    // guard 析构时自动退出函数体作用域
}
```

#### 作用域层次
1. **函数输入作用域（InputCompound）**：存储函数参数
2. **函数体作用域（Compound）**：函数体的主作用域，是输入作用域的子作用域
3. **嵌套作用域**：For、If 等语句创建嵌套的作用域

#### 约束
- 作用域之间通过 `GetAncestorValues()` 查找父作用域中的变量，实现词法作用域语义
- 必须使用 RAII 方式管理 ScopeGuard，确保作用域正确退出
- 子作用域中的变量对父作用域不可见

### 环境表（Environment Table）

每个 CompoundStatement 维护一个环境表（EnvTable），用于存储作用域内的变量绑定：

- Key: 变量的 SSA 名称
- Value: 变量的值对象

环境表支持：
- `SetEnvVar()`: 设置变量绑定
- `GetEnvVar()`: 获取变量值（在当前作用域）
- `GetAncestorValues()`: 获取所有祖先作用域中的变量

### 循环携带变量（Loop-Carried Variables）

在 `ExitForStatement()` 中，IRBuilder 会自动识别循环携带变量：

1. 查找在循环体中被修改的变量
2. 创建 `iter_arg` 结构
3. 在循环体内创建对应的值对象
4. 在循环体末尾添加 yield 语句
5. 构建循环结果并更新父作用域环境表

### 条件分支合并（If Statement Merging）

在 `ExitIfStatement()` 中，IRBuilder 会自动处理条件分支的变量合并：

1. 识别在 then/else 分支中被修改的变量
2. 在 then/else 分支末尾添加 yield 语句
3. 使用 `IfStatement::BuildResult()` 构建合并结果
4. 更新父作用域环境表
