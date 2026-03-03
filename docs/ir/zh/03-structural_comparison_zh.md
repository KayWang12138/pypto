# PyPTO 结构化比较

## 概述

PyPTO 提供了两个实用函数，用于按结构而非指针标识来比较 IR 节点：

```python
structural_equal(lhs, rhs, enable_auto_mapping=False) -> bool
structural_hash(node, enable_auto_mapping=False) -> int
```

**使用场景：** CSE（公共子表达式消除）、IR 优化、模式匹配、测试

**关键特性：** 两个函数都忽略 `Span`（源代码位置），只关注逻辑结构。

## 引用相等性 vs 结构化相等性

### 引用相等性（默认 `==`）

比较指针地址（O(1)，快速）：

```python
from pypto import DataType, ir

x1 = ir.Var("x", ir.ScalarType(DataType.INT64), ir.Span.unknown())
x2 = ir.Var("x", ir.ScalarType(DataType.INT64), ir.Span.unknown())
assert x1 != x2  # 不同的指针
```

### 结构化相等性

比较内容和结构：

```python
ir.assert_structural_equal(x1, x2, enable_auto_mapping=True)  # True
```

## 比较过程

`structural_equal` 函数遵循以下步骤：

1. **快速路径检查**
   - 引用相等性：如果是同一个指针，返回 `true`
   - 空值检查：如果任一为空，返回 `false`
   - 类型检查：比较 `TypeName()` - 必须完全匹配

2. **类型分发**
   - 变量获得特殊处理（自动映射）
   - 其他类型使用基于反射的字段比较

3. **基于字段的递归比较**
   - 通过 `GetFieldDescriptors()` 获取字段描述符
   - 使用反射遍历所有字段
   - 根据字段类型比较每个字段
   - 使用 AND 逻辑组合结果

## 反射和字段类型

反射系统定义了三种字段类型：

| 字段类型 | 自动映射 | 是否比较？ | 使用场景 | 效果 |
|------------|--------------|-----------|----------|--------|
| **IgnoreField** | 不适用 | ❌ 否 | 源代码位置（`Span`）、名称 | 始终视为相等 |
| **UsualField** | 遵循参数 | ✅ 是 | 操作数、表达式、类型 | 使用当前的 `enable_auto_mapping` 进行比较 |
| **DefField** | ✅ 始终启用 | ✅ 是 | 变量定义、参数 | 始终使用自动映射 |

### 字段定义示例

```cpp
class IRNode {
  Span span_;
  static constexpr auto GetFieldDescriptors() {
    return std::make_tuple(
      reflection::IgnoreField(&IRNode::span_, "span")
    );
  }
};

class BinaryExpr : public Expr {
  ExprPtr left_;
  ExprPtr right_;
  static constexpr auto GetFieldDescriptors() {
    return std::tuple_cat(
      Expr::GetFieldDescriptors(),
      std::make_tuple(
        reflection::UsualField(&BinaryExpr::left_, "left"),
        reflection::UsualField(&BinaryExpr::right_, "right")
      )
    );
  }
};

class AssignStmt : public Stmt {
  VarPtr var_;     // 定义
  ExprPtr value_;  // 使用
  static constexpr auto GetFieldDescriptors() {
    return std::tuple_cat(
      Stmt::GetFieldDescriptors(),
      std::make_tuple(
        reflection::DefField(&AssignStmt::var_, "var"),
        reflection::UsualField(&AssignStmt::value_, "value")
      )
    );
  }
};
```

### 为什么 DefField 很重要

DefField 表示变量定义。在比较定义时，我们关心的是结构位置，而不是标识：

```python
# 构建: x = y
x1 = ir.Var("x", ir.ScalarType(DataType.INT64), span)
y1 = ir.Var("y", ir.ScalarType(DataType.INT64), span)
stmt1 = ir.AssignStmt(x1, y1, span)

# 构建: a = b
a = ir.Var("a", ir.ScalarType(DataType.INT64), span)
b = ir.Var("b", ir.ScalarType(DataType.INT64), span)
stmt2 = ir.AssignStmt(a, b, span)

# var_ 是 DefField，所以 x1 和 a 会自动映射
ir.assert_structural_equal(stmt1, stmt2, enable_auto_mapping=True)
```

## structural_equal 函数

### 基本用法

```python
# 相同的值
c1 = ir.ConstInt(42, DataType.INT64, ir.Span.unknown())
c2 = ir.ConstInt(42, DataType.INT64, ir.Span.unknown())
ir.assert_structural_equal(c1, c2)  # True

# 不同的类型
var = ir.Var("x", ir.ScalarType(DataType.INT64), ir.Span.unknown())
const = ir.ConstInt(1, DataType.INT64, ir.Span.unknown())
assert not ir.structural_equal(var, const)  # False
```

### 自动映射行为

| 场景 | enable_auto_mapping=False | enable_auto_mapping=True |
|----------|---------------------------|--------------------------|
| 相同的变量指针 | ✅ 相等 | ✅ 相等 |
| 不同的变量指针 | ❌ 不相等 | ✅ 相等（如果类型匹配） |
| 一致的映射（`x + x` vs `y + y`） | ❌ 不相等 | ✅ 相等 |
| 不一致的映射（`x + x` vs `y + z`） | ❌ 不相等 | ❌ 不相等 |

### 何时启用自动映射

| 使用场景 | 设置 |
|----------|---------|
| 不考虑变量名的模式匹配 | `True` |
| 优化规则的模板匹配 | `True` |
| 使用相同变量的精确匹配 | `False` |
| CSE（公共子表达式消除） | `False` |

## structural_hash 函数

### 基本用法

```python
c1 = ir.ConstInt(42, DataType.INT64, ir.Span.unknown())
c2 = ir.ConstInt(42, DataType.INT64, ir.Span.unknown())
assert ir.structural_hash(c1) == ir.structural_hash(c2)
```

### 哈希一致性保证

**规则：** 如果 `structural_equal(a, b, mode)` 为 `True`，则 `structural_hash(a, mode) == structural_hash(b, mode)`

### 与容器一起使用

```python
class CSEPass:
    def __init__(self):
        self.expr_cache = {}

    def deduplicate(self, expr):
        hash_val = ir.structural_hash(expr, enable_auto_mapping=False)
        if hash_val in self.expr_cache:
            for cached_expr in self.expr_cache[hash_val]:
                if ir.structural_equal(expr, cached_expr, enable_auto_mapping=False):
                    return cached_expr
            self.expr_cache[hash_val].append(expr)
        else:
            self.expr_cache[hash_val] = [expr]
        return expr
```

## 自动映射算法

实现维护双向映射：

```cpp
class StructuralEqual {
  std::unordered_map<VarPtr, VarPtr> lhs_to_rhs_var_map_;
  std::unordered_map<VarPtr, VarPtr> rhs_to_lhs_var_map_;

  bool EqualVar(const VarPtr& lhs, const VarPtr& rhs) {
    if (!enable_auto_mapping_) {
      return lhs.get() == rhs.get();  // 严格的指针相等性
    }

    // 首先检查类型相等性
    if (!EqualType(lhs->GetType(), rhs->GetType())) return false;

    // 检查现有映射
    auto it = lhs_to_rhs_var_map_.find(lhs);
    if (it != lhs_to_rhs_var_map_.end()) {
      return it->second == rhs;  // 验证一致性
    }

    // 确保 rhs 尚未映射到不同的 lhs
    auto rhs_it = rhs_to_lhs_var_map_.find(rhs);
    if (rhs_it != rhs_to_lhs_var_map_.end() && rhs_it->second != lhs) {
      return false;
    }

    // 创建新映射
    lhs_to_rhs_var_map_[lhs] = rhs;
    rhs_to_lhs_var_map_[rhs] = lhs;
    return true;
  }
};
```

**关键点：**
- 不使用自动映射：严格的指针比较
- 使用自动映射：建立并强制执行一致的映射
- 在映射之前检查类型相等性
- 双向映射防止不一致的映射

## 实现细节

### 哈希组合算法

使用受 Boost 启发的算法：

```cpp
inline uint64_t hash_combine(uint64_t seed, uint64_t value) {
  return seed ^ (value + 0x9e3779b9 + (seed << 6) + (seed >> 2));
}
```

### 基于反射的字段访问器

无需类型特定代码的通用遍历：

```cpp
template <typename NodePtr>
bool EqualWithFields(const NodePtr& lhs_op, const NodePtr& rhs_op) {
  using NodeType = typename NodePtr::element_type;
  auto descriptors = NodeType::GetFieldDescriptors();
  return std::apply([&](auto&&... descs) {
    return reflection::FieldIterator<...>::Visit(
      *lhs_op, *rhs_op, *this, descs...);
  }, descriptors);
}
```

## 总结

**关键要点：**

1. **三种字段类型**：
   - `IgnoreField`：从不比较（Span、名称）
   - `UsualField`：使用用户的 `enable_auto_mapping` 进行比较
   - `DefField`：始终使用自动映射

2. **自动映射**：
   - 为模式匹配启用
   - 为精确 CSE 禁用
   - 始终保持一致：维护双射变量映射

3. **哈希一致性**：
   - 相等的节点 → 相等的哈希（保证）
   - 对两个函数使用相同的 `enable_auto_mapping`

有关 IR 节点类型和构造的信息,请参阅 [IR 概述](00-ir_overview_zh.md)。
