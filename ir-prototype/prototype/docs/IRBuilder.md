# IR Builder

## 1. 背景与设计目标

在本项目的 IR 体系中，IR 由 **Value / Operation / Statement / Scope / Function** 等结构化节点组成。

**IRBuilder 的设计目标是：**

> 成为一个 *结构性 IR 组装器*（Structural IR Assembler），而不是一个“语义 Builder”或“智能推导器”。

---

## 2. IRBuilder 的核心职责

### 2.1 插入点（Insertion Point）管理

IRBuilder 维护当前 IR 构造上下文，包括：

* 当前函数（`Func`）
* 当前作用域（`Scope`）

所有通过 IRBuilder 创建的 IR 节点，**都会隐式插入到当前插入点**。

插入点的切换通过 **RAII 风格的 Guard（如 `ScopeGuard`）** 完成，确保：

* 作用域进入 / 退出成对出现
* 构造过程异常安全
* IR 结构始终保持一致

---

### 2.2 Statement（结构性节点）的创建与插入

IRBuilder 负责创建并插入结构性 IR 节点，包括但不限于：

* `ForStatement`
* `IfStatement`
* `BlockStatement`

这些节点的作用是：

* 定义控制流结构
* 引入新的作用域边界

> IRBuilder **只负责创建和插入**，**不负责判断控制流是否合法**（如 dominance、CFG 正确性等）。

相关校验应由后续 IR Pass 或 Verify 阶段完成。

当前的 `BlockStatement` 由在创建一个 `op` 的时候懒生成。可以避免出现空 `BlockStatement`。

```C++
BlockStatement& IRBuilder::GetOrCreateActiveBlock() {
    if (!scope_) throw std::runtime_error("IRBuilder::GetOrCreateActiveBlock: scope is null");
    if (block_) return *block_;

    // Create a new block statement at current scope tail
    auto blk = std::make_shared<BlockStatement>();
    auto& ref = *blk;
    scope_->AddStatement(std::move(blk));
    block_ = &ref;
    return ref;
}
```
