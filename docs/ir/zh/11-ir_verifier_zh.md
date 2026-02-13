# IR 验证器

通过可插拔规则、诊断报告和 Pass 集成，为验证 PyPTO IR 正确性提供可扩展的验证系统。

## 概述

| 组件 | 描述 |
|-----------|-------------|
| **VerifyRule (C++)** | 验证规则的基类 - 每个规则实现特定的 IR 检查 |
| **IRVerifier (C++)** | 管理规则集合并在 Program 上执行验证 |
| **Diagnostic** | 结构化的错误/警告报告，包含严重性、位置和消息 |
| **VerificationError** | 在抛出模式下验证失败时抛出的异常 |

### 主要特性

- **可插拔规则系统**：使用自定义验证规则进行扩展
- **选择性验证**：根据使用场景单独启用/禁用规则
- **双重验证模式**：收集诊断信息或在首次错误时抛出异常
- **Pass 集成**：在优化管道中作为 Pass 使用
- **全面的诊断**：收集所有问题及其源代码位置

### 使用场景

- **开发**：在构建期间尽早捕获 IR 错误
- **测试**：验证转换是否保持正确性
- **管道集成**：在优化 pass 之间插入验证
- **调试**：为格式错误的 IR 生成详细的诊断报告

## 架构

### 验证规则系统

验证器使用**插件架构**，其中每个验证规则都是独立的组件：

- **VerifyRule 基类**：定义所有规则必须实现的接口
- **规则注册**：规则在构造时或运行时添加到 IRVerifier
- **执行顺序**：规则按注册顺序在所有函数上运行
- **独立性**：每个规则独立运行 - 一个规则的失败不会影响其他规则

**设计原则**：规则应该是**可组合的**和**专注的** - 每个规则检查 IR 正确性的一个方面（SSA 形式、类型等）。

**启用/禁用机制**：可以选择性地禁用规则而不删除它们。这允许：
- 使用检查子集进行测试
- 在生产环境中禁用昂贵的检查
- 添加新规则时逐步迁移

### 验证模式

| 模式 | 方法 | 行为 | 使用时机 |
|------|--------|----------|----------|
| **诊断收集** | `Verify()` | 收集所有错误/警告，返回向量 | 需要完整错误列表、构建工具、报告 |
| **快速失败** | `VerifyOrThrow()` | 在首次错误时抛出 VerificationError | 管道验证、测试、开发 |

**模式选择指南**：
- 使用 `Verify()` 进行 IDE/工具集成 - 用户希望看到所有问题
- 在管道中使用 `VerifyOrThrow()` - 在无效 IR 上立即失败
- 在测试中使用 `VerifyOrThrow()` - 通过异常处理明确通过/失败

### 诊断系统

**诊断结构**：

| 字段 | 类型 | 用途 |
|-------|------|---------|
| `severity` | `DiagnosticSeverity` | 错误或警告 |
| `rule_name` | `string` | 检测到问题的规则 |
| `error_code` | `int` | 数字错误标识符 |
| `message` | `string` | 人类可读的描述 |
| `span` | `Span` | 源代码位置信息 |

**严重性级别**：
- `Error`：IR 无效，必须修复
- `Warning`：IR 有效但可能存在问题

**报告生成**：`GenerateReport()` 将诊断信息格式化为人类可读的报告，包含计数、分组和位置详情。

### 与 Pass 系统集成

验证器通过 `run_verifier()` 集成到 Pass 管道中：

- **返回**：一个 `Pass` 对象（Program → Program 转换）
- **行为**：验证程序，记录诊断信息，在错误时抛出异常
- **配置**：接受 `disabled_rules` 参数
- **管道位置**：通常在转换后插入以验证输出

**设计考虑**：验证器 Pass 是**透明的** - 如果有效，它返回未更改的输入程序，使其可以安全地插入管道的任何位置。

## C++ API 参考

**头文件**：`include/pypto/ir/transforms/verifier.h`

### VerifyRule 接口

实现自定义验证规则的基类。

| 方法 | 签名 | 描述 |
|--------|-----------|-------------|
| `GetName()` | `std::string GetName() const` | 返回唯一的规则标识符 |
| `Verify()` | `void Verify(const FunctionPtr&, std::vector<Diagnostic>&)` | 检查函数并追加诊断信息 |

**实现要求**：
- `GetName()` 必须返回唯一、稳定的标识符
- `Verify()` 应该追加到诊断信息，而不是抛出异常
- 规则应该是无状态的（或使用线程安全状态）

### IRVerifier 类

管理验证规则并执行验证。

#### 构造和配置

| 方法 | 描述 |
|--------|-------------|
| `IRVerifier()` | 构造没有规则的空验证器 |
| `static IRVerifier CreateDefault()` | 工厂方法 - 返回空验证器（可以通过 AddRule 添加规则） |
| `void AddRule(VerifyRulePtr rule)` | 注册验证规则（如果名称重复则忽略） |

#### 规则管理

| 方法 | 描述 |
|--------|-------------|
| `void EnableRule(const std::string& name)` | 启用先前禁用的规则（如果未找到则无操作） |
| `void DisableRule(const std::string& name)` | 按名称禁用规则 - 在验证期间将跳过它 |
| `bool IsRuleEnabled(const std::string& name) const` | 检查规则当前是否已启用 |

#### 验证执行

| 方法 | 返回 | 抛出 | 描述 |
|--------|--------|--------|-------------|
| `Verify(const ProgramPtr&)` | `std::vector<Diagnostic>` | 否 | 运行所有已启用的规则，收集所有诊断信息 |
| `VerifyOrThrow(const ProgramPtr&)` | `void` | `VerificationError` | 运行验证，如果发现任何错误则抛出异常 |

#### 报告

| 方法 | 描述 |
|--------|-------------|
| `static std::string GenerateReport(const std::vector<Diagnostic>&)` | 将诊断信息格式化为可读报告，包含计数和详情 |

**报告格式**：带有错误/警告计数的摘要行，后跟每个诊断的详细列表，包括规则名称、严重性、位置和消息。

## Python API 参考

**模块**：`pypto.pypto_core.passes`

### IRVerifier 类

C++ IRVerifier 的 Python 绑定，使用 snake_case 命名。

#### 工厂和构造

| 方法 | 描述 |
|--------|-------------|
| `IRVerifier()` | 创建空验证器（通常不直接使用） |
| `IRVerifier.create_default()` | 静态方法 - 返回空验证器（可以通过自定义 C++ 实现添加规则） |

#### 规则管理

| 方法 | 参数 | 描述 |
|--------|-----------|-------------|
| `enable_rule(name)` | `name: str` | 启用已禁用的规则 |
| `disable_rule(name)` | `name: str` | 按名称禁用规则 |
| `is_rule_enabled(name)` | `name: str` | 检查规则是否已启用（返回 `bool`） |

#### 验证

| 方法 | 参数 | 返回 | 抛出 | 描述 |
|--------|-----------|---------|--------|-------------|
| `verify(program)` | `program: Program` | `list[Diagnostic]` | 否 | 收集所有诊断信息 |
| `verify_or_throw(program)` | `program: Program` | `None` | Exception | 在错误时抛出异常 |

#### 报告

| 方法 | 参数 | 返回 | 描述 |
|--------|-----------|---------|-------------|
| `generate_report(diagnostics)` | `diagnostics: list[Diagnostic]` | `str` | 静态方法 - 格式化诊断信息 |

### Diagnostic 类型

表示单个验证问题的只读结构。

| 字段 | 类型 | 描述 |
|-------|------|-------------|
| `severity` | `DiagnosticSeverity` | `Error` 或 `Warning` |
| `rule_name` | `str` | 检测到问题的规则名称 |
| `error_code` | `int` | 数字标识符 |
| `message` | `str` | 人类可读的描述 |
| `span` | `Span` | 源代码位置 |

### DiagnosticSeverity 枚举

| 值 | 含义 |
|-------|---------|
| `DiagnosticSeverity.Error` | IR 无效 |
| `DiagnosticSeverity.Warning` | 可能存在问题但有效 |

## 使用示例

### 基本验证

```python
from pypto import ir
from pypto.pypto_core import passes

# 构建程序（假设 'program' 已构造）
verifier = passes.IRVerifier.create_default()

# 注意：默认验证器为空，必须通过 C++ 添加自定义规则
diagnostics = verifier.verify(program)

if diagnostics:
    report = passes.IRVerifier.generate_report(diagnostics)
    print(report)
```

### 使用异常进行错误处理

```python
verifier = passes.IRVerifier.create_default()

try:
    verifier.verify_or_throw(program)
    print("Program is valid")
except Exception as e:
    print(f"Verification failed: {e}")
```

### 检查诊断信息

```python
verifier = passes.IRVerifier.create_default()
diagnostics = verifier.verify(program)

for diag in diagnostics:
    if diag.severity == passes.DiagnosticSeverity.Error:
        print(f"ERROR in {diag.rule_name}: {diag.message}")
        print(f"  Location: {diag.span}")
```

## 添加自定义规则

可以添加自定义验证规则以使用特定领域的检查扩展验证器。**注意**：自定义规则只能在 **C++ 级别**注册。Python IRVerifier API 不公开 `add_rule()`，因此规则必须集成到默认验证器中或直接在 C++ 中实例化。

### 实现步骤

**1. 创建规则类** (C++)

从 `VerifyRule` 继承并实现所需的方法：

```cpp
// my_custom_rule.cpp
#include "pypto/ir/transforms/verifier.h"

namespace {
class MyCustomRule : public VerifyRule {
 public:
  std::string GetName() const override { return "MyCustom"; }

  void Verify(const FunctionPtr& func,
              std::vector<Diagnostic>& diagnostics) override {
    // 实现验证逻辑
    // 遍历 IR，检查条件，追加诊断信息
  }
};
}
```

**2. 创建工厂函数**

```cpp
VerifyRulePtr CreateMyCustomRule() {
  return std::make_shared<MyCustomRule>();
}
```

**3. 在 C++ 中注册规则**

添加到默认验证器（推荐用于项目范围的规则）：

```cpp
// 在 src/ir/transforms/verifier.cpp CreateDefault() 中：
IRVerifier IRVerifier::CreateDefault() {
  IRVerifier verifier;
  // 在此处添加您的自定义规则
  verifier.AddRule(CreateMyCustomRule());
  return verifier;
}
```

**替代方案**：在 C++ 代码中以编程方式使用：

```cpp
auto verifier = IRVerifier();
verifier.AddRule(CreateMyCustomRule());
verifier.Verify(program);
```

**Python 用法**：一旦添加到 `CreateDefault()`，规则将自动可用：

```python
from pypto.pypto_core import passes

# 自定义规则包含在默认验证器中
verifier = passes.IRVerifier.create_default()
verifier.disable_rule("MyCustom")  # 如果需要可以禁用
diagnostics = verifier.verify(program)
```

### 实现指南

**使用 IRVisitor**：利用访问者模式系统地遍历 IR 节点。

**创建描述性诊断信息**：
```cpp
Diagnostic diag;
diag.severity = DiagnosticSeverity::Error;
diag.rule_name = GetName();
diag.error_code = 1001;  // 此检查的唯一代码
diag.message = "描述性错误消息";
diag.span = problematic_node->span_;
diagnostics.push_back(diag);
```

**保持规则专注**：每个规则应该检查一类问题。多个小规则优于一个复杂规则。

**避免副作用**：规则应该只读取 IR 和写入诊断信息，不修改 IR 或在函数之间维护状态。

### 规则集成点

| 位置 | 用途 |
|----------|---------|
| `src/ir/transforms/your_rule.cpp` | 规则实现 |
| `src/ir/transforms/verifier.cpp` | 在 `CreateDefault()` 中注册以自动包含 |
| `tests/ut/ir/transforms/test_verifier.py` | 测试用例 |

**关于 Python 集成的说明**：当前的 Python 绑定不公开 `VerifyRule` 或 `IRVerifier.add_rule()`。自定义规则必须添加到 C++ 中的 `CreateDefault()` 才能从 Python 中使用。要使规则可以从 Python 动态添加，需要扩展绑定：
- `VerifyRule` 类绑定
- `IRVerifier.add_rule(rule)` 方法绑定

此限制确保规则在部署前经过适当的测试和验证。

## 设计原理

| 设计选择 | 原理 |
|--------------|-----------|
| **插件架构** | 可扩展性 - 项目可以添加特定领域的检查而无需修改核心验证器 |
| **规则启用/禁用** | 灵活性 - 昂贵或实验性规则可以根据使用场景切换 |
| **双重验证模式** | 可用性 - 工具需要所有错误（IDE），管道需要快速失败（CI） |
| **诊断收集** | 完整性 - 用户一次看到所有问题，而不是修复-重新运行循环 |
| **Pass 系统集成** | 一致性 - 验证使用与转换相同的接口 |
| **关注点分离** | 可维护性 - 每个规则都是独立的，规则之间互不了解 |
| **函数级验证** | 可扩展性 - 将来可以跨函数并行化验证 |
| **静态报告生成** | 实用性 - 诊断是数据，格式化是表示 |

**关键权衡**：收集所有诊断信息（与在首次错误时停止相比）需要更多内存和处理，但提供更好的用户体验。双模式 API 让用户根据上下文进行选择。

**为什么不内联验证？**：将验证与构造/转换分离可以：
- 可选验证（在受信任的管道中跳过）
- 选择性检查（仅在调试中进行昂贵的规则）
- 构造后验证（捕获转换中的错误）

## 总结

IR 验证器提供：

- **可扩展的验证框架**，用于 PyPTO IR 正确性
- **内置规则**，用于 SSA 形式和类型一致性
- **灵活配置**，通过选择性规则启用/禁用
- **双重使用模式**，适用于不同上下文（工具 vs. 管道）
- **Pass 集成**，用于无缝管道验证
- **全面的诊断**，具有结构化错误报告

### 何时使用

| 场景 | 方法 |
|----------|----------|
| **IR 构造后** | 使用 `verify_or_throw()` 捕获前端错误 |
| **转换之间** | 插入 `run_verifier()` pass 以验证每个步骤 |
| **在测试中** | 使用 `verify_or_throw()` 确保测试输入有效 |
| **IDE/工具集成** | 使用 `verify()` 收集所有问题以供显示 |
| **生产管道** | 可选择禁用昂贵的规则，仅在调试构建中使用 |

### 相关组件

- **Pass 系统** (`10-pass_manager.md`)：验证器作为 Pass 集成
- **IRBuilder** (`08-ir_builder.md`)：构造验证器验证的 IR
- **类型系统** (`02-ir_types_examples.md`)：TypeCheck 规则针对类型系统进行验证
- **错误处理** (`include/pypto/core/error.h`)：Diagnostic 和 VerificationError 定义

### 测试

`tests/ut/ir/transforms/test_verifier.py` 中的全面测试覆盖：
- 有效和无效程序验证
- 规则启用/禁用行为
- 异常 vs. 诊断收集模式
- Pass 集成
- 诊断字段访问
- 报告生成

验证器确保 PyPTO IR 在整个编译过程中保持正确性不变量，从而实现可靠的代码生成和优化。
