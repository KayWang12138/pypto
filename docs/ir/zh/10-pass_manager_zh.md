# Pass 和 PassManager

用于组织和执行 IR 转换 Pass 的框架，支持基于策略的优化管道（Default/PTOAS）。

## 概述

| 组件 | 描述 |
|-----------|-------------|
| **Pass (C++)** | 用于 Program → Program 转换的独立类 |
| **PassManager (Python)** | 管理 Pass 序列和执行策略 |
| **工厂函数** | 创建 Pass（例如 `pass::InitMemRef()`, `pass::BasicMemoryReuse()`） |

### 主要特性

- **仅 Program 接口**：所有 Pass 都执行 Program → Program 转换
- **不可变转换**：返回新的 IR 节点，不进行原地修改
- **基于策略的管道**：预配置的优化级别
- **工厂模式**：通过工厂函数创建 Pass，隐藏实现细节
- **统一头文件**：所有声明都在 `include/pypto/ir/transforms/passes.h` 中

## C++ Pass 基础设施

### Pass 基类

**头文件**：`include/pypto/ir/transforms/passes.h`

```cpp
class Pass {
 public:
  ProgramPtr operator()(const ProgramPtr& program) const;  // 执行 Pass
};

// 内置 Pass 的工厂函数
namespace pass {
  Pass ExamplePass();   // 示例转换 Pass
  // 更多 Pass 可用 - 参见实现
}
```

**要点**：Pimpl 模式隐藏实现；所有声明在单个头文件中；仅 Program → Program 转换。

### Pass 实现模式

### Pass 实现示例

```cpp
// 示例：带状态的复杂 Pass
namespace {
class ExamplePassImpl : public PassImpl {
 public:
  ProgramPtr operator()(const ProgramPtr& program) override {
    for (const auto& [name, func] : program->functions_)
      state_ += ComputeSomething(func);
    return program;
  }
  std::string GetName() const override { return "ExamplePass"; }
 private:
  int state_ = 0;
};
}
namespace pass {
Pass ExamplePass() { return Pass(std::make_shared<ExamplePassImpl>()); }
}
```

### Python 绑定

**文件**：`python/bindings/modules/passes.cpp`

```cpp
void BindPass(nb::module_& m) {
  nb::module_ passes = m.def_submodule("passes", "IR transformation passes");

  // 不透明的 Pass 对象
  nb::class_<Pass>(passes, "Pass")
      .def("__call__", &Pass::operator(), nb::arg("program"));

  // 工厂函数（snake_case）
  passes.def("example_pass", &pass::ExamplePass);
  // 更多 Pass 绑定可用
}
```

创建 `pypto.pypto_core.passes` 模块，包含不透明的 `Pass` 类和工厂函数。

## Python PassManager

**文件**：`python/pypto/ir/pass_manager.py`

### 优化策略

```python
class OptimizationStrategy(Enum):
    Default = "Default"      # 完整优化管道
    PTOAS = "PTOAS"         # PTO 汇编策略
```

### PassManager API

| 方法 | 描述 |
|--------|-------------|
| `get_strategy(strategy)` | 获取为策略配置的 PassManager |
| `run_passes(program)` | 在 Program 上顺序执行所有 Pass |
| `get_pass_names()` | 获取管理器中所有 Pass 的名称 |

### 策略配置

策略在 `_register_passes` 中配置，为每个优化级别提供适当的 Pass 序列。

## 使用示例

```python
from pypto import ir, DataType

# 创建包含多个函数的程序
span = ir.Span.unknown()
dtype = DataType.INT64
x1, y1 = ir.Var("x", ir.ScalarType(dtype), span), ir.Var("y", ir.ScalarType(dtype), span)
func1 = ir.Function("func1", [x1], [ir.ScalarType(dtype)], ir.AssignStmt(x1, y1, span), span)
x2, y2 = ir.Var("x", ir.ScalarType(dtype), span), ir.Var("y", ir.ScalarType(dtype), span)
func2 = ir.Function("func2", [x2], [ir.ScalarType(dtype)], ir.AssignStmt(x2, y2, span), span)
program = ir.Program([func1, func2], "test_program", span)

# 使用 PTOAS 策略运行 Pass
pm = ir.PassManager.get_strategy(ir.OptimizationStrategy.PTOAS)
result = pm.run_passes(program)
# 结果具有相同的函数名称；Pass 根据策略应用转换

# 单行简写
result = ir.PassManager.get_strategy(ir.OptimizationStrategy.PTOAS).run_passes(program)
```

## 实现细节

### Program 转换流程

```python
def run_passes(self, program: core_ir.Program) -> core_ir.Program:
    current = program
    for pass_instance in self.passes:
        current = pass_instance(current)  # Program → Program
    return current
```

管道组合：`Pass3(Pass2(Pass1(program)))` - 每个 Pass 接收并返回一个 Program。

### Pass 注册模式

- 每个策略映射到 `(name, factory)` 元组
- 工厂是创建新 Pass 实例的 lambda
- 支持独立的 PassManager 实例

## 测试

**位置**：`tests/ut/ir/transforms/test_pass_manager.py`

**示例**：策略执行保留函数名称：

```python
def test_run_passes_on_program_with_strategy(self):
    program = ir.Program([func1, func2], "test_program", span)
    pm = ir.PassManager.get_strategy(ir.OptimizationStrategy.PTOAS)
    result = pm.run_passes(program)
    func_names = [func.name for func in result.functions.values()]
    assert "func1" in func_names
    assert "func2" in func_names
```

## 添加新 Pass

1. **在 `passes.h` 中声明**：`Pass YourNewPass();`

2. **实现**（`src/ir/transforms/your_new_pass.cpp`）：
   ```cpp
   // 示例实现
   namespace pass {
   Pass YourNewPass() {
     return CreateFunctionPass([](const FunctionPtr& func) {
       // 转换函数
       return func;
     }, "YourNewPass");
   }
   }
   ```

3. **Python 绑定**（`python/bindings/modules/passes.cpp`）：
   ```cpp
   passes.def("your_new_pass", &pass::YourNewPass, "Description");
   ```

4. **在 PassManager 中注册**（`python/pypto/ir/pass_manager.py`）：
   ```python
   ("YourNewPass", lambda: passes.your_new_pass()),
   ```

5. **类型存根**（`python/pypto/pypto_core/passes.pyi`）：
   ```python
   def your_new_pass() -> Pass: """Description."""
   ```

6. **测试**（`tests/ut/ir/transforms/test_your_new_pass.py`）

## 设计原理

| 设计选择 | 原理 |
|---------------|-----------|
| **不可变转换** | 线程安全、调试（保留原始 IR）、易于回滚、函数式风格 |
| **基于策略的配置** | 易用性、一致性、集中维护、可扩展性 |
| **仅 Program 接口** | 统一 API、支持过程间优化、更简单的心智模型 |
| **单一头文件（`passes.h`）** | 减少膨胀、清晰发现、通过 pimpl 实现不透明实现 |

## 总结

Pass 和 PassManager 系统提供：
- **可扩展框架**：通过工厂函数轻松添加 Pass
- **基于策略的优化**：预配置级别（Default/PTOAS）
- **统一接口**：所有 Pass 都执行 Program → Program 转换
- **清晰 API**：带工厂函数的不透明 Pass 对象
- **良好测试**：全面的测试覆盖
- **不可变转换**：安全的函数式 IR 转换
- **有组织的结构**：所有声明在单个头文件中

此基础设施为在 PyPTO 中构建复杂的优化管道提供了基础。
