# PyPTO 前端机制详解

> **适用对象：** 想要深入理解PyPTO前端解析机制的开发者  
> **学习时间：** 45-60分钟  
> **前置知识：** 已阅读[核心概念](../02-core/01-concepts.md)、[Frontend 模块](../02-core/15-frontend.md)  
> **学习目标：** 理解AST解析、动态形状、形状推断等前端核心机制

**机制价值：**
- 🎯 **代码解析**：将Python代码转换为IR
- 🔍 **形状处理**：支持静态和动态形状
- ⚡ **类型推断**：自动推导张量形状和类型
- 🛠️ **错误诊断**：提供详细的错误信息

**相关主题：**
- 前端模块详解：见 [Frontend 模块](../02-core/15-frontend.md)
- 核心概念：见 [核心概念](../02-core/01-concepts.md)
- 控制流机制：见 [控制流编译机制](05-controlflow.md)

---

## 目录

1. [概述](#1-概述)
2. [AST解析机制](#2-ast解析机制)
3. [动态形状机制](#3-动态形状机制)
4. [形状推断机制](#4-形状推断机制)
5. [机制关系与协作](#5-机制关系与协作)
6. [使用示例](#6-使用示例)
7. [常见问题](#7-常见问题)

---

## 1. 概述

PyPTO前端机制负责将用户编写的Python代码转换为框架内部的中间表示（IR），主要包括三个核心机制：

### 1.1 机制分类

| 机制 | 职责 | 关键组件 |
|------|------|---------|
| **AST解析机制** | 将Python代码解析为AST并转换为IR | AST遍历、作用域管理、表达式求值 |
| **动态形状机制** | 支持运行时确定形状的计算 | SymbolicScalar、dynamic_axis、dynamic() |
| **形状推断机制** | 自动推导张量形状 | InferShapeRegistry |

### 1.2 机制流程

```
用户Python代码
    ↓
AST解析机制 → 生成IR节点
    ↓
动态形状机制 → 处理动态维度
    ↓
形状推断机制 → 推导完整形状
    ↓
生成Function IR
```

---

## 2. AST解析机制

### 2.1 定义与作用

**定义**：将Python代码解析为AST（抽象语法树）并转换为PyPTO IR的机制。

**作用**：
- 支持Python前端，实现JIT编译
- 提供静态分析能力
- 支持嵌套函数内联
- 改进的错误诊断

### 2.2 关键组件

#### 2.2.1 AST遍历机制

**实现方式**：使用Visitor模式遍历Python AST

**关键类**：
- `Parser`：核心解析器，实现AST Visitor
- `DocAST`：自定义doc AST节点定义
- `DocConverter`：Python AST ↔ doc AST转换器

**代码位置**：
- `python/pypto/frontend/parser/parser.py`
- `python/pypto/frontend/parser/doc_core.py`
- `python/pypto/frontend/parser/doc.py`

**工作流程**：
```python
# 1. Python代码 → Python AST
import ast
tree = ast.parse(source_code)

# 2. Python AST → doc AST
doc_tree = DocConverter.convert(tree)

# 3. doc AST → IR节点
parser = Parser()
function_ir = parser.parse(doc_tree)
```

#### 2.2.2 作用域管理机制

**定义**：管理变量作用域和生命周期的机制

**关键组件**：
- `Context`：作用域上下文管理器
- `Scope`：作用域定义（全局、局部、闭包）

**代码位置**：
- `python/pypto/frontend/parser/context.py`

**作用域类型**：
- **全局作用域**：模块级变量
- **局部作用域**：函数内变量
- **闭包作用域**：嵌套函数捕获的外部变量

#### 2.2.3 表达式求值机制

**定义**：在解析时求值常量表达式的机制

**关键组件**：
- `Evaluator`：表达式求值器
- 支持常量折叠、类型推断

**代码位置**：
- `python/pypto/frontend/parser/evaluator.py`

**求值范围**：
- 常量表达式（数字、字符串）
- 编译时已知的形状计算
- 类型推断

### 2.3 解析流程

```
1. 源码提取
   ↓
2. Python AST解析
   ↓
3. doc AST转换
   ↓
4. AST遍历（Visitor模式）
   ↓
5. IR节点生成
   ↓
6. 作用域分析
   ↓
7. 表达式求值
   ↓
8. Function IR构建
```

### 2.4 特性支持

| 特性 | 支持情况 | 说明 |
|------|---------|------|
| **控制流** | ✅ 完整支持 | if/for/while/break/continue |
| **嵌套函数** | ✅ 支持内联 | `@pypto.frontend.function` |
| **闭包捕获** | ✅ 支持 | 自动捕获外部变量 |
| **动态形状** | ✅ 完整支持 | `dynamic()`、`SymbolicScalar` |
| **错误诊断** | ✅ 源码位置 | 带行号、列号的错误信息 |

---

## 3. 动态形状机制

### 3.1 定义与作用

**定义**：支持运行时确定形状的机制，允许张量维度在运行时才确定。

**作用**：
- 支持动态输入形状的计算
- 处理批处理大小变化
- 支持变长序列处理

### 3.2 关键概念

#### 3.2.1 SymbolicScalar（符号化标量）

**定义**：表示在编译时未知、运行时确定的标量值。

**使用场景**：
- 动态维度大小
- 动态偏移量
- 动态有效形状

**代码位置**：
- `framework/src/interface/tensor/logical_tensor.h`

**示例**：
```python
import pypto.frontend as fe

@fe.jit
def dynamic_func(x: fe.Tensor[fe.dynamic(), 10]):
    # fe.dynamic() 创建 SymbolicScalar
    # 第一个维度在运行时确定
    return x + 1
```

#### 3.2.2 dynamic_axis（动态轴标记）

**定义**：标记张量中哪些轴是动态的。

**使用方式**：
```python
# 标记第一个轴为动态
x: fe.Tensor[fe.dynamic(), 10, 20]
```

#### 3.2.3 dynamic()（动态维度定义）

**定义**：定义动态维度的函数。

**API**：
```python
fe.dynamic()  # 创建动态维度
```

### 3.3 动态形状处理流程

```
1. 用户定义动态维度
   ↓
2. 创建SymbolicScalar
   ↓
3. 形状推断时保留符号
   ↓
4. 运行时绑定实际值
   ↓
5. 生成支持动态形状的代码
```

### 3.4 与静态形状的对比

| 特性 | 静态形状 | 动态形状 |
|------|---------|---------|
| **编译时确定** | ✅ | ❌ |
| **运行时确定** | ❌ | ✅ |
| **性能** | 更优 | 略低 |
| **灵活性** | 较低 | 更高 |
| **内存分配** | 编译时 | 运行时 |

---

## 4. 形状推断机制

### 4.1 定义与作用

**定义**：自动推导张量形状的机制，根据操作的输入形状推导输出形状。

**作用**：
- 验证形状一致性
- 优化内存分配
- 支持形状传播
- 提供编译时错误检查

### 4.2 关键组件

#### 4.2.1 InferShapeRegistry（形状推断注册表）

**定义**：形状推断函数的注册表系统（单例模式）。

**职责**：
- 注册操作的形状推断函数
- 查找操作的形状推断函数
- 管理形状推断函数映射

**代码位置**：
- `framework/src/interface/operation/op_infer_shape_impl.h`

**注册方式**：
```cpp
// 注册形状推断函数
REGISTER_INFER_SHAPE_FUNC(Opcode::OP_ADD, InferAddShape);

// 查找形状推断函数
auto infer_func = InferShapeRegistry::GetInstance()
    ->GetInferShapeFunc(Opcode::OP_ADD);
```

#### 4.2.2 形状推断函数

**定义**：根据输入形状推导输出形状的函数。

**函数签名**：
```cpp
using InferShapeFunc = std::function<void(
    const std::vector<LogicalTensor*>& inputs,
    std::vector<LogicalTensor*>& outputs
)>;
```

**推断规则**：
- 逐元素操作：输出形状 = 输入形状
- 广播操作：输出形状 = 广播后的形状
- 归约操作：输出形状 = 归约后的形状
- 矩阵乘法：输出形状 = 矩阵乘法规则

### 4.3 形状推断流程

```
1. 操作输入形状已知
   ↓
2. 查找形状推断函数
   ↓
3. 执行形状推断
   ↓
4. 推导输出形状
   ↓
5. 验证形状一致性
   ↓
6. 传播到下游操作
```

### 4.4 形状推断示例

**示例1：逐元素加法**
```python
# 输入：a.shape = [10, 20], b.shape = [10, 20]
# 输出：c.shape = [10, 20]
c = a + b
```

**示例2：矩阵乘法**
```python
# 输入：a.shape = [10, 5], b.shape = [5, 20]
# 输出：c.shape = [10, 20]
c = a @ b
```

**示例3：动态形状**
```python
# 输入：a.shape = [dynamic, 10], b.shape = [dynamic, 10]
# 输出：c.shape = [dynamic, 10]  (动态维度保留)
c = a + b
```

---

## 5. 机制关系与协作

### 5.1 机制协作流程

```
用户代码
    ↓
AST解析机制
    ├─→ 生成IR节点
    ├─→ 识别动态维度
    └─→ 提取形状信息
    ↓
动态形状机制
    ├─→ 创建SymbolicScalar
    ├─→ 标记动态轴
    └─→ 保留符号信息
    ↓
形状推断机制
    ├─→ 查找推断函数
    ├─→ 推导输出形状
    └─→ 验证形状一致性
    ↓
生成Function IR
```

### 5.2 机制依赖关系

| 机制 | 依赖 | 被依赖 |
|------|------|--------|
| **AST解析** | - | 动态形状、形状推断 |
| **动态形状** | AST解析 | 形状推断 |
| **形状推断** | AST解析、动态形状 | - |

---

## 6. 使用示例

### 6.1 基础AST解析

```python
import pypto.frontend as fe

@fe.jit
def add(x: fe.Tensor[10, 20], y: fe.Tensor[10, 20]):
    return x + y

# AST解析机制：
# 1. 解析函数定义
# 2. 识别操作（加法）
# 3. 生成IR节点
# 4. 构建Function IR
```

### 6.2 动态形状使用

```python
@fe.jit
def dynamic_add(x: fe.Tensor[fe.dynamic(), 10]):
    # 第一个维度在运行时确定
    return x + 1

# 动态形状机制：
# 1. 识别dynamic()标记
# 2. 创建SymbolicScalar
# 3. 保留符号信息
# 4. 运行时绑定实际值
```

### 6.3 形状推断验证

```python
@fe.jit
def matmul(x: fe.Tensor[10, 5], y: fe.Tensor[5, 20]):
    return x @ y  # 形状推断：输出 [10, 20]

# 形状推断机制：
# 1. 识别矩阵乘法操作
# 2. 查找推断函数
# 3. 推导输出形状 [10, 20]
# 4. 验证形状一致性
```

---

## 7. 常见问题

### 7.1 AST解析相关问题

**Q: 为什么需要自定义doc AST？**

A: doc AST提供了比Python AST更丰富的语义信息，支持PyPTO特定的概念（如Tensor、动态形状等），便于后续IR生成。

**Q: 如何处理嵌套函数？**

A: 使用`@pypto.frontend.function`装饰的嵌套函数会被自动内联到父函数中，避免函数调用开销。

### 7.2 动态形状相关问题

**Q: 动态形状会影响性能吗？**

A: 会有一定性能影响，因为需要在运行时处理形状信息。但对于批处理大小变化等场景，动态形状是必需的。

**Q: 如何判断是否应该使用动态形状？**

A: 如果输入形状在编译时已知且固定，使用静态形状；如果需要在运行时变化，使用动态形状。

### 7.3 形状推断相关问题

**Q: 形状推断失败怎么办？**

A: 检查操作是否已注册形状推断函数，验证输入形状是否正确，查看错误信息中的具体位置。

**Q: 如何添加自定义操作的形状推断？**

A: 使用`REGISTER_INFER_SHAPE_FUNC`宏注册形状推断函数，参考现有操作的实现。

---

## 相关文档

- [Frontend 模块](../02-core/15-frontend.md) - 前端模块详细文档
- [核心概念](../02-core/01-concepts.md) - 基础概念和术语
- [Operation 类技术文档](../02-core/06-operation.md) - 操作和形状推断
- [控制流编译机制](05-controlflow.md) - 控制流相关机制
- [关键机制列表](00-key-mechanisms-list.md) - 所有机制概览

---

## 总结

PyPTO前端机制通过AST解析、动态形状和形状推断三个核心机制的协同工作，实现了从Python代码到IR的完整转换流程：

1. **AST解析机制**：将Python代码转换为IR，支持静态分析和错误诊断
2. **动态形状机制**：支持运行时确定形状，提供灵活性
3. **形状推断机制**：自动推导形状，验证一致性，优化内存分配

理解这些机制有助于：
- 深入理解前端工作原理
- 正确使用动态形状功能
- 调试形状相关问题
- 扩展前端功能

