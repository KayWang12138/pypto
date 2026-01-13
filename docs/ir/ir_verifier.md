# IR Verifier

## OverView

IR Verifier 是 PTO-IR 中用于验证中间表示（IR）正确性的工具模块。它提供了基于规则的验证系统，可以检查 IR 程序是否符合特定的语义规则和约束条件。

Verifier 的主要功能包括：
- **形状验证（Shape Verification）**：验证操作（Operation）的输入输出形状是否兼容
- **SSA 语义验证（SSA Semantics Verification）**：验证 TileValue 是否符合 SSA（Static Single Assignment）语义
- **可扩展的规则系统**：支持注册自定义验证规则

## 核心组件

### VerifyResult

验证操作的结果结构：

```cpp
struct VerifyResult {
    bool passed;           // 验证是否通过
    std::string errorMsg;  // 错误信息（如果验证失败）
};
```

### Verifier 类

`Verifier` 类提供了基于规则的验证框架：

```cpp
class Verifier {
public:
    using RuleFunc = std::function<VerifyResult(const TileValue &)>;
    
    // 验证单个规则
    VerifyResult VerifyRule(const std::string &ruleName, const TileValue &tile) const;
    
    // 验证所有已注册的规则
    VerifyResult VerifyAllRules(const TileValue &tile) const;
    
    // 获取所有规则名称
    std::vector<std::string> GetRuleNames() const;
    
    // 注册自定义规则
    void RegisterRule(const std::string &ruleName, RuleFunc ruleFunc);
};
```

## 内置验证规则

### 1. 形状验证（Shape Verification）

形状验证检查操作（Operation）的输入输出形状是否兼容。

**使用示例：**

```cpp
#include "ir/verifier/shape_verify.h"

// 创建程序模块
auto module = std::make_shared<ProgramModule>("main");
IRBuilder builder;
IRBuilderContext ctx;

// ... 构建 IR 程序 ...

// 执行形状验证
VerifyResult result = VerifyOpShape(module);

if (!result.passed) {
    std::cout << "形状验证失败: " << result.errorMsg << std::endl;
}
```

**验证失败示例：**

```cpp
// 错误：一元操作的输入输出形状不匹配
std::vector<int64_t> inputShape = {128, 64};
std::vector<int64_t> outputShape = {64, 128};  // 不匹配

auto inputTile = std::make_shared<TileValue>(inputShape, DataType::FP32, "input");
auto outputTile = std::make_shared<TileValue>(outputShape, DataType::FP32, "output");
auto unaryOp = builder.CreateUnaryOp(Opcode::OP_NEG, inputTile, outputTile);

// VerifyOpShape 会返回失败，错误信息类似：
// "UnaryOp 'OP_NEG': input shape [128, 64] != output shape [64, 128]"
```

**广播支持示例：**

```cpp
// 正确：支持广播
std::vector<int64_t> lhsShape = {128, 64};
std::vector<int64_t> rhsShape = {1, 64};      // 第一个维度为 1，可以广播
std::vector<int64_t> outputShape = {128, 64};

auto lhs = std::make_shared<TileValue>(lhsShape, DataType::FP32, "lhs");
auto rhs = std::make_shared<TileValue>(rhsShape, DataType::FP32, "rhs");
auto output = std::make_shared<TileValue>(outputShape, DataType::FP32, "output");
auto binaryOp = builder.CreateBinaryOp(Opcode::OP_ADD, lhs, rhs, output);

// 验证通过：rhs 的形状 [1, 64] 可以广播到 [128, 64]
```

### 2. SSA 语义验证（SSA Semantics Verification）

SSA 语义验证确保每个 TileValue 作为输入只被使用一次，符合 SSA（Static Single Assignment）形式的要求。

#### TileValueSSAVisitor

`TileValueSSAVisitor` 遍历 IR 并统计每个 TileValue 作为输入被使用的次数。

**验证规则：**
- 每个 TileValue 只能被一次定义和赋值
- 如果某个 TileValue 被多次赋值，则违反 SSA 语义

**使用示例：**

```cpp
#include "ir/verifier/ssa_verify.h"

// 创建程序模块
auto module = std::make_shared<ProgramModule>("main");
IRBuilder builder;
IRBuilderContext ctx;

// ... 构建 IR 程序 ...

// 执行 SSA 语义验证
VerifyResult result = VerifySSASingleInput(module);

if (!result.passed) {
    std::cout << "SSA 语义验证失败: " << result.errorMsg << std::endl;
}
```

**验证失败示例：**

```cpp
// 错误：同一个 TileValue 被多次使用
std::vector<int64_t> tileShape = {128, 64};
auto inputTile = std::make_shared<TileValue>(tileShape, DataType::FP32, "input");

// 第一次使用
auto output1 = builder.CreateTile(ctx, tileShape, DataType::FP32, "output1");
auto op1 = builder.CreateUnaryOp(Opcode::OP_NEG, inputTile, output1);
builder.Emit(ctx, op1);

// 第二次使用同一个 inputTile（违反 SSA）
auto output2 = builder.CreateTile(ctx, tileShape, DataType::FP32, "output2");
auto op2 = builder.CreateUnaryOp(Opcode::OP_ABS, inputTile, output2);
builder.Emit(ctx, op2);

// VerifySSASingleInput 会返回失败，错误信息类似：
// "SSA semantics violation - 1 TileValue(s) have incorrect input count:
//   1. TileValue 'input' (ID: 1) has 2 input(s), expected exactly 1"
```

## 自定义验证规则

可以通过 `Verifier::RegisterRule` 方法注册自定义验证规则：

```cpp
#include "ir/verifier/verifier.h"

// 创建 Verifier 实例
Verifier verifier;

// 定义自定义规则函数
auto customRule = [](const TileValue &tile) -> VerifyResult {
    // 检查 TileValue 的形状维度
    const auto &shape = tile.GetShape();
    if (shape.size() > 4) {
        return {false, "TileValue has more than 4 dimensions"};
    }
    return {true, ""};
};

// 注册规则
verifier.RegisterRule("MaxDimensions", customRule);

// 验证单个规则
TileValuePtr tile = /* ... */;
VerifyResult result = verifier.VerifyRule("MaxDimensions", *tile);

// 验证所有规则
VerifyResult allResults = verifier.VerifyAllRules(*tile);
```

## 完整使用示例

以下示例展示了如何同时使用多个验证规则：

```cpp
#include "ir/verifier/shape_verify.h"
#include "ir/verifier/ssa_verify.h"
#include "ir/builder/ir_builder.h"

void VerifyProgram(ProgramModulePtr module) {
    // 1. 形状验证
    VerifyResult shapeResult = VerifyOpShape(module);
    if (!shapeResult.passed) {
        std::cerr << "形状验证失败:\n" << shapeResult.errorMsg << std::endl;
        return;
    }
    
    // 2. SSA 语义验证
    VerifyResult ssaResult = VerifySSASingleInput(module);
    if (!ssaResult.passed) {
        std::cerr << "SSA 语义验证失败:\n" << ssaResult.errorMsg << std::endl;
        return;
    }
    
    std::cout << "所有验证通过！" << std::endl;
}

int main() {
    // 创建并构建 IR 程序
    auto module = std::make_shared<ProgramModule>("main");
    IRBuilder builder;
    IRBuilderContext ctx;
    
    // ... 构建 IR 程序 ...
    
    // 执行验证
    VerifyProgram(module);
    
    return 0;
}
```

## 验证流程

典型的验证流程如下：

```
ProgramModule
    ↓
VerifyOpShape / VerifySSASingleInput
    ↓
创建 Visitor (TileOpShapeVisitor / TileValueSSAVisitor)
    ↓
Visitor.VisitProgram(module)
    ↓
遍历所有 Function → Statement → Operation
    ↓
收集违反规则的信息
    ↓
返回 VerifyResult
```

## 注意事项

1. **验证时机**：建议在 IR 构建完成后、代码生成之前进行验证
2. **性能考虑**：验证会遍历整个 IR，对于大型程序可能有性能开销
3. **错误信息**：验证失败时，错误信息会包含详细的违反规则的位置和原因
4. **扩展性**：可以通过注册自定义规则来扩展验证功能

## API 参考

### VerifyOpShape

```cpp
VerifyResult VerifyOpShape(ProgramModulePtr program);
```

验证程序中所有操作的形状兼容性。

**参数：**
- `program`: 要验证的程序模块

**返回值：**
- `VerifyResult`: 验证结果，包含是否通过和错误信息

### VerifySSASingleInput

```cpp
VerifyResult VerifySSASingleInput(ProgramModulePtr program);
```

验证程序中所有 TileValue 是否符合 SSA 语义（每个 TileValue 作为输入只能使用一次）。

**参数：**
- `program`: 要验证的程序模块

**返回值：**
- `VerifyResult`: 验证结果，包含是否通过和错误信息

### Verifier::VerifyAllRules

```cpp
VerifyResult VerifyAllRules(const TileValue &tile) const;
```

对指定的 TileValue 执行所有已注册的验证规则，并打印验证结果表格。

**参数：**
- `tile`: 要验证的 TileValue

**返回值：**
- `VerifyResult`: 验证结果，如果所有规则都通过则 `passed` 为 `true`
