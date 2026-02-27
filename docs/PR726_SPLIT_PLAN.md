
# PR #726 拆分计划

## 背景

PR #726 包含 269 个文件改动（197 新增、22 修改、50 删除），共 61220 行新增、13702 行删除。
改动量过大不利于 review，需拆分为 6 个独立 PR，按依赖顺序逐一合入。

> 注意：删除旧 IR 已作为单独的 PR 在 `delete_origin_ir` 分支提交，本计划不再包含删除操作。

---

## 依赖关系概览

```
PR1 (Core基础库 + span.h)
  └── PR2 (IR核心类型: type/core/expr/stmt/function/program + kind_traits.h)
        └── PR3 (Op注册/Builder/算子实现)
              └── PR4 (Transform基础框架: visitor/mutator/printer/verifier等)
                    └── PR5 (Serialization序列化)
                          └── PR6 (Python层接口/绑定 + 文档 + 构建集成 + 外部引用适配)
```

> 严格线性链，每个 PR 只依赖前一个 PR。

---

## .h 与 .cpp 配对验证

**核心原则：每个 .h 必须和其对应的 .cpp 在同一个 PR 中，否则会出现链接错误。**

| .h 文件 | 对应 .cpp | 是否同 PR | 说明 |
|---------|----------|----------|------|
| `core/common.h` | (无) | - | header-only（宏定义） |
| `core/dtype.h` | (无) | - | header-only |
| `core/error.h` | `core/error.cpp`, `core/backtrace.cpp` | ✅ PR1 | |
| `core/logging.h` | (无) | - | header-only（内联/宏） |
| `core/any_cast.h` | (无) | - | header-only（模板） |
| `ir/span.h` | (无) | - | header-only |
| `ir/reflection/field_traits.h` | (无) | - | header-only（模板） |
| `ir/reflection/field_visitor.h` | (无) | - | header-only（模板） |
| `ir/pipe.h` | (无) | - | header-only（枚举） |
| `ir/memref.h` | `ir/memref.cpp` | ✅ PR2 | |
| `ir/core.h` | `ir/core.cpp` | ✅ PR2 | |
| `ir/expr.h` | `ir/expr.cpp` | ✅ PR2 | **`kind_traits.h` 已移入 PR2 解决此问题** |
| `ir/scalar_expr.h` | (无) | - | header-only（模板） |
| `ir/stmt.h` | `ir/stmt.cpp` | ✅ PR2 | |
| `ir/type.h` (M) | `ir/type.cpp` (M) | ✅ PR2 | |
| `ir/function.h` (M) | `ir/function.cpp` (D) | ⚠️ | 旧 .cpp 被删除，新 function.h 无需新 .cpp |
| `ir/program.h` (M) | `ir/program.cpp` (M) | ✅ PR2 | |
| `ir/kind_traits.h` | (无) | - | header-only（模板+宏），**已移入 PR2** |
| `ir/op_utils.h` | (无) | - | header-only |
| `ir/op_registry.h` | `ir/op_registry.cpp` | ✅ PR3 | |
| `ir/builder.h` | `ir/builder.cpp` | ✅ PR3 | |
| `ir/type_inference.h` | `ir/op/type_inference.cpp` | ✅ PR3 | |
| `ir/transform/base/functor.h` | (无) | - | header-only（模板） |
| `ir/transform/base/visitor.h` | `ir/transform/visitor.cpp` | ✅ PR4 | |
| `ir/transform/base/mutator.h` | `ir/transform/mutator.cpp` | ✅ PR4 | |
| `ir/transform/verification_error.h` | (无) | - | header-only |
| `ir/transform/dependency_graph.h` | (无) | - | header-only |
| `ir/transform/dependency_analyzer.h` | `ir/transform/dependency_analyzer.cpp` | ✅ PR4 | |
| `ir/transform/printer.h` | `ir/transform/printer.cpp` | ✅ PR4 | |
| `ir/transform/structural_comparison.h` | `structural_equal.cpp`, `structural_hash.cpp` | ✅ PR4 | |
| `ir/transform/verifier.h` | `ir/transform/verifier.cpp` | ✅ PR4 | |
| `ir/transform/passes.h` | `ir/transform/passes.cpp` | ✅ PR4 | |
| `ir/serialization/serializer.h` | `ir/serialization/serializer.cpp` | ✅ PR5 | |
| `ir/serialization/deserializer.h` | `ir/serialization/deserializer.cpp` | ✅ PR5 | |
| `ir/serialization/type_registry.h` | `type_registry.cpp`, `type_deserializers.cpp` | ✅ PR5 | |

---

## 依赖验证结果（头文件 + cpp 源文件 + 测试文件）

### 头文件依赖

| 头文件 | 本地 include | 所属 PR | 依赖 PR |
|--------|-------------|---------|---------|
| `core/common.h` | (无) | PR1 | 无 |
| `core/dtype.h` | (无) | PR1 | 无 |
| `core/error.h` | `core/common.h`, `ir/span.h` | PR1 | 无 |
| `core/logging.h` | `core/error.h` | PR1 | 无 |
| `core/any_cast.h` | `core/error.h` | PR1 | 无 |
| `ir/span.h` | (无) | PR1 | 无 |
| `ir/reflection/field_traits.h` | (无) | PR2 | 无 |
| `ir/reflection/field_visitor.h` | `core/logging.h`, `ir/reflection/field_traits.h` | PR2 | PR1 |
| `ir/pipe.h` | (无) | PR2 | 无 |
| `ir/memref.h` | `core/dtype.h`, `ir/reflection/field_traits.h` | PR2 | PR1 |
| `ir/core.h` | `ir/reflection/field_traits.h`, `ir/span.h` | PR2 | PR1 |
| `ir/type.h` (M) | `core/dtype.h`, `core/logging.h`, `ir/core.h`, `ir/memref.h`, `ir/reflection/field_traits.h` | PR2 | PR1 |
| `ir/expr.h` | `core/any_cast.h`, `core/dtype.h`, `core/error.h`, `ir/core.h`, `ir/memref.h`, `ir/pipe.h`, `ir/reflection/field_traits.h`, `ir/type.h` | PR2 | PR1 |
| `ir/scalar_expr.h` | `core/dtype.h`, `core/logging.h`, `ir/core.h`, `ir/expr.h`, `ir/reflection/field_traits.h`, `ir/type.h` | PR2 | PR1 |
| `ir/stmt.h` | `ir/core.h`, `ir/expr.h`, `ir/reflection/field_traits.h` | PR2 | PR1 |
| `ir/function.h` (M) | `ir/core.h`, `ir/expr.h`, `ir/reflection/field_traits.h`, `ir/stmt.h`, `ir/type.h` | PR2 | PR1 |
| `ir/program.h` (M) | `ir/core.h`, `ir/expr.h`, `ir/function.h`, `ir/reflection/field_traits.h` | PR2 | PR1 |
| **`ir/kind_traits.h`** | `ir/core.h`, `ir/expr.h`, `ir/function.h`, `ir/program.h`, `ir/scalar_expr.h`, `ir/stmt.h`, `ir/type.h` | **PR2** | PR1 (全部是 PR2 内部头文件) |
| `ir/op_utils.h` | `core/any_cast.h`, `core/error.h`, `core/logging.h`, `ir/kind_traits.h`, `ir/scalar_expr.h` | PR3 | PR2 |
| `ir/op_registry.h` | `core/common.h`, `core/logging.h`, `ir/core.h`, `ir/expr.h`, `ir/pipe.h`, `ir/type.h` | PR3 | PR2 |
| `ir/builder.h` | `ir/core.h`, `ir/expr.h`, `ir/function.h`, `ir/program.h`, `ir/stmt.h`, `ir/type.h` | PR3 | PR2 |
| `ir/type_inference.h` | `core/dtype.h`, `ir/expr.h`, `ir/type.h` | PR3 | PR2 |
| `ir/transform/base/functor.h` | `core/error.h`, `ir/kind_traits.h`, `ir/scalar_expr.h`, `ir/stmt.h` | PR4 | **PR2** |
| `ir/transform/base/visitor.h` | `ir/stmt.h`, `ir/transform/base/functor.h` | PR4 | PR2 |
| `ir/transform/base/mutator.h` | `ir/stmt.h`, `ir/transform/base/functor.h` | PR4 | PR2 |
| `ir/transform/verification_error.h` | `ir/span.h` | PR4 | PR1 |
| `ir/transform/dependency_graph.h` | `ir/expr.h`, `ir/stmt.h` | PR4 | PR2 |
| `ir/transform/dependency_analyzer.h` | `ir/function.h`, `ir/transform/base/mutator.h`, `ir/transform/dependency_graph.h` | PR4 | PR2 |
| `ir/transform/printer.h` | `ir/core.h`, `ir/expr.h`, `ir/type.h` | PR4 | PR2 |
| `ir/transform/structural_comparison.h` | `ir/core.h`, `ir/type.h` | PR4 | PR2 |
| `ir/transform/verifier.h` | `core/error.h`, `ir/function.h`, `ir/program.h` | PR4 | PR2 |
| `ir/transform/passes.h` | `ir/function.h`, `ir/program.h` | PR4 | PR2 |
| `ir/serialization/serializer.h` | `ir/core.h` | PR5 | PR2 |
| `ir/serialization/deserializer.h` | `ir/core.h` | PR5 | PR2 |
| `ir/serialization/type_registry.h` | `ir/core.h`, `ir/expr.h`, `ir/type.h` | PR5 | PR2 |

### cpp 源文件依赖

| cpp 文件 | 关键本地 include | 所属 PR | 实际依赖 PR |
|---------|-----------------|---------|------------|
| **PR1** | | | |
| `core/error.cpp` | `core/error.h` | PR1 | 无 ✅ |
| `core/backtrace.cpp` | `core/error.h` | PR1 | 无 ✅ |
| **PR2** | | | |
| `ir/core.cpp` | `ir/core.h` | PR2 | PR1 ✅ |
| `ir/expr.cpp` | `ir/kind_traits.h`, `ir/type.h` | PR2 | PR1 ✅ (**kind_traits 已在 PR2**) |
| `ir/memref.cpp` | `ir/expr.h`, `ir/type.h` | PR2 | PR1 ✅ |
| `ir/stmt.cpp` | `core/error.h`, `core/logging.h` | PR2 | PR1 ✅ |
| `ir/type.cpp` (M) | `ir/scalar_expr.h` | PR2 | PR1 ✅ |
| `ir/program.cpp` (M) | `ir/expr.h`, `ir/function.h` | PR2 | PR1 ✅ |
| **PR3** | | | |
| `ir/builder.cpp` | `core/error.h`, `core/logging.h` | PR3 | PR1 ✅ |
| `ir/op_registry.cpp` | `core/dtype.h`, `core/logging.h` | PR3 | PR1 ✅ |
| `ir/op/type_inference.cpp` | `ir/kind_traits.h`, `ir/scalar_expr.h`, `ir/type.h` | PR3 | PR2 ✅ |
| `ir/op/block_ops/*.cpp` | `ir/kind_traits.h`, `ir/op_registry.h`, `ir/op_utils.h`, `ir/type_inference.h` | PR3 | PR2 ✅ (全部 PR3 内部) |
| `ir/op/tensor_ops/*.cpp` | `ir/kind_traits.h`, `ir/op_registry.h`, `ir/op_utils.h` | PR3 | PR2 ✅ |
| `ir/op/sync_ops/sync.cpp` | `ir/expr.h`, `ir/op_registry.h`, `ir/pipe.h` | PR3 | PR2 ✅ |
| **PR4** | | | |
| `ir/transform/visitor.cpp` | `ir/kind_traits.h`, `ir/scalar_expr.h` | PR4 | PR2 ✅ |
| `ir/transform/mutator.cpp` | `ir/kind_traits.h`, `ir/scalar_expr.h` | PR4 | PR2 ✅ |
| `ir/transform/printer.cpp` | `ir/kind_traits.h`, `ir/transform/base/visitor.h` | PR4 | PR2 ✅ (visitor.h 同 PR4) |
| `ir/transform/structural_equal.cpp` | `ir/kind_traits.h`, `ir/reflection/field_visitor.h`, `ir/transform/printer.h` | PR4 | PR2 ✅ |
| `ir/transform/structural_hash.cpp` | `ir/kind_traits.h`, `ir/reflection/field_visitor.h` | PR4 | PR2 ✅ |
| `ir/transform/dependency_analyzer.cpp` | `ir/kind_traits.h`, `ir/transform/base/visitor.h` | PR4 | PR2 ✅ |
| `ir/transform/verifier.cpp` | `core/logging.h` | PR4 | PR1 ✅ |
| `ir/transform/passes.cpp` | `core/logging.h`, `ir/program.h` | PR4 | PR2 ✅ |
| **PR5** | | | |
| `ir/serialization/serializer.cpp` | `ir/kind_traits.h`, `ir/reflection/field_visitor.h` | PR5 | PR2 ✅ (**kind_traits 已在 PR2**) |
| `ir/serialization/deserializer.cpp` | `core/error.h`, `ir/serialization/type_registry.h` | PR5 | PR2 ✅ |
| `ir/serialization/type_registry.cpp` | `core/error.h` | PR5 | PR1 ✅ |
| `ir/serialization/type_deserializers.cpp` | `ir/kind_traits.h`, `ir/serialization/type_registry.h` | PR5 | PR2 ✅ (**kind_traits 已在 PR2**) |

### 测试文件依赖

| 测试文件 | 关键跨 PR 依赖 | 所属 PR | 实际依赖 PR |
|---------|---------------|---------|------------|
| **PR1 测试** | | | |
| `test_any_cast.cpp` | `core/any_cast.h` | PR1 | 无 ✅ |
| `test_backtrace.cpp` | `core/error.h` | PR1 | 无 ✅ |
| `test_dtype.cpp` | `core/dtype.h` | PR1 | 无 ✅ |
| `test_error.cpp` | `core/error.h` | PR1 | 无 ✅ |
| `test_logging.cpp` | `core/logging.h` | PR1 | 无 ✅ |
| **PR2 测试** | | | |
| `test_common.cpp` | `core/common.h` | PR2 | PR1 ✅ |
| `test_span_irnode.cpp` | `ir/core.h`, `ir/scalar_expr.h` | PR2 | PR1 ✅ |
| `test_core.cpp` | `ir/core.h` | PR2 | PR1 ✅ |
| `test_memref.cpp` | `ir/memref.h`, `ir/scalar_expr.h` | PR2 | PR1 ✅ |
| `test_expr.cpp` | `ir/kind_traits.h` | PR2 | PR1 ✅ (**kind_traits 已在 PR2**) |
| `test_expr_basic.cpp` | `ir/expr.h`, `ir/scalar_expr.h` | PR2 | PR1 ✅ |
| `test_scalar_expr_basic.cpp` | `ir/scalar_expr.h` | PR2 | PR1 ✅ |
| `test_stmt.cpp` | `ir/stmt.h`, `ir/expr.h` | PR2 | PR1 ✅ |
| `test_type.cpp` (M) | `ir/scalar_expr.h`, `ir/type.h` | PR2 | PR1 ✅ |
| `test_program.cpp` | `ir/program.h`, `ir/function.h` | PR2 | PR1 ✅ |
| `test_program_stmt_error.cpp` | `ir/kind_traits.h` | PR2 | PR1 ✅ (**kind_traits 已在 PR2**) |
| **PR3 测试** | | | |
| `test_builder.cpp` | `ir/builder.h` | PR3 | PR2 ✅ |
| `test_op_registry.cpp` | `ir/op_registry.h` | PR3 | PR2 ✅ |
| `test_op_registration.cpp` | `ir/op_registry.h`, `ir/type_inference.h` | PR3 | PR2 ✅ |
| `test_block_ops.cpp` | `ir/kind_traits.h`, `ir/op_registry.h` | PR3 | PR2 ✅ |
| `test_tensor_ops.cpp` | `ir/kind_traits.h`, `ir/op_registry.h` | PR3 | PR2 ✅ |
| `test_sync_ops.cpp` | `ir/op_registry.h` | PR3 | PR2 ✅ |
| `test_type_inference.cpp` | `ir/type_inference.h` | PR3 | PR2 ✅ |
| **PR4 测试** | | | |
| `test_visitor.cpp` | `ir/op_registry.h`, `ir/transform/base/visitor.h` | PR4 | PR3 ✅ |
| `test_mutator.cpp` | `ir/op_registry.h`, `ir/transform/base/mutator.h` | PR4 | PR3 ✅ |
| `test_printer.cpp` | `ir/op_registry.h`, `ir/transform/printer.h` | PR4 | PR3 ✅ |
| `test_structural_equal.cpp` | `ir/op_registry.h`, `ir/transform/structural_comparison.h` | PR4 | PR3 ✅ |
| `test_structural_hash.cpp` | `ir/op_registry.h`, `ir/transform/structural_comparison.h` | PR4 | PR3 ✅ |
| `test_dependency_analyzer.cpp` | `ir/transform/dependency_analyzer.h` | PR4 | PR2 ✅ |
| `test_verifier.cpp` | `ir/transform/verifier.h` | PR4 | PR2 ✅ |
| `test_passes.cpp` | `ir/transform/passes.h` | PR4 | PR2 ✅ |
| `test_transform.cpp` | `ir/builder.h`, `ir/transform/*` | PR4 | PR3 ✅ |
| **PR5 测试** | | | |
| `test_serialization.cpp` | `ir/op_registry.h`, `ir/transform/structural_comparison.h` | PR5 | PR3+PR4 ✅ (线性链中 PR5 在 PR4 之后) |

> **全部通过** ✅ 所有 .h/.cpp 配对在同一 PR，所有文件只依赖前序 PR。

---

## PR1: Core 基础库

**目标**: 新增基础设施头文件和实现，独立可编译。

### 新增文件 (13个)

**头文件 (6)**:
- `framework/include/core/common.h` — 通用宏定义（header-only）
- `framework/include/core/dtype.h` — 数据类型定义（header-only）
- `framework/include/core/error.h` — 错误处理
- `framework/include/core/logging.h` — 日志系统（header-only）
- `framework/include/core/any_cast.h` — 类型安全的 any_cast（header-only）
- `framework/include/ir/span.h` — 源码位置信息（header-only，`error.h` 依赖）

**源文件 (2)** — 与 `error.h` 配对:
- `framework/src/interface/core/error.cpp`
- `framework/src/interface/core/backtrace.cpp`

**C++ 测试 (5)**:
- `framework/tests/ut/interface/src/ir/test_any_cast.cpp`
- `framework/tests/ut/interface/src/ir/test_backtrace.cpp`
- `framework/tests/ut/interface/src/ir/test_dtype.cpp`
- `framework/tests/ut/interface/src/ir/test_error.cpp`
- `framework/tests/ut/interface/src/ir/test_logging.cpp`

### 内部依赖链
```
common.h ← error.h ← logging.h
                    ← any_cast.h
span.h   ← error.h
dtype.h    (独立)
```

### 依赖: 无 ✅

---

## PR2: IR 核心类型体系

**目标**: 建立 IR 核心类型系统 + kind_traits，保证所有 .h 与 .cpp 配对完整。

### 新增文件 (19个)

**头文件 (10)**:
- `framework/include/ir/reflection/field_traits.h` — 字段反射 traits（header-only）
- `framework/include/ir/reflection/field_visitor.h` — 字段访问器（header-only）
- `framework/include/ir/pipe.h` — Pipe 枚举（header-only）
- `framework/include/ir/memref.h` — 内存引用类型
- `framework/include/ir/core.h` — IRNode 核心基类
- `framework/include/ir/expr.h` — 表达式节点
- `framework/include/ir/scalar_expr.h` — 标量表达式（header-only）
- `framework/include/ir/stmt.h` — 语句节点
- `framework/include/ir/kind_traits.h` — IRNode 类型 traits（header-only，**从 PR3 移入**，只依赖 PR2 内部头文件，`expr.cpp` 需要它）

**源文件 (4)** — 与头文件配对:
- `framework/src/interface/ir/core.cpp` ← 配对 `core.h`
- `framework/src/interface/ir/expr.cpp` ← 配对 `expr.h`（依赖 `kind_traits.h`，故 kind_traits 必须同 PR）
- `framework/src/interface/ir/memref.cpp` ← 配对 `memref.h`
- `framework/src/interface/ir/stmt.cpp` ← 配对 `stmt.h`

**C++ 测试 (10)**:
- `framework/tests/ut/interface/src/ir/test_common.cpp`
- `framework/tests/ut/interface/src/ir/test_span_irnode.cpp`
- `framework/tests/ut/interface/src/ir/test_core.cpp`
- `framework/tests/ut/interface/src/ir/test_memref.cpp`
- `framework/tests/ut/interface/src/ir/test_expr.cpp` — 依赖 `kind_traits.h` ✅
- `framework/tests/ut/interface/src/ir/test_expr_basic.cpp`
- `framework/tests/ut/interface/src/ir/test_scalar_expr_basic.cpp`
- `framework/tests/ut/interface/src/ir/test_stmt.cpp`
- `framework/tests/ut/interface/src/ir/test_program.cpp`
- `framework/tests/ut/interface/src/ir/test_program_stmt_error.cpp` — 依赖 `kind_traits.h` ✅

### 修改文件 (6个) — .h 与 .cpp 配对:
- `framework/include/ir/type.h` + `framework/src/interface/ir/type.cpp` ← 配对 ✅
- `framework/include/ir/function.h` ← 旧 function.cpp 已删除，新 function.h 不需要新 .cpp
- `framework/include/ir/program.h` + `framework/src/interface/ir/program.cpp` ← 配对 ✅

### 修改测试 (1个)
- `framework/tests/ut/interface/src/ir/test_type.cpp`

### 依赖: PR1 ✅

---

## PR3: Op 注册 / Builder / 算子实现

**目标**: 提供算子注册机制、IR 构建器以及各类算子的具体实现。

### 新增文件 (29个)

**头文件 (4)**:
- `framework/include/ir/op_utils.h` — 算子工具函数（header-only）
- `framework/include/ir/op_registry.h` — 算子注册表
- `framework/include/ir/builder.h` — IR 构建器
- `framework/include/ir/type_inference.h` — 类型推断

**源文件 (18)** — 与头文件配对:
- `framework/src/interface/ir/op_registry.cpp` ← 配对 `op_registry.h`
- `framework/src/interface/ir/builder.cpp` ← 配对 `builder.h`
- `framework/src/interface/ir/op/type_inference.cpp` ← 配对 `type_inference.h`
- `framework/src/interface/ir/op/README.md`
- `framework/src/interface/ir/op/block_ops/batch_matmul.cpp`
- `framework/src/interface/ir/op/block_ops/broadcast.cpp`
- `framework/src/interface/ir/op/block_ops/elementwise.cpp`
- `framework/src/interface/ir/op/block_ops/matmul.cpp`
- `framework/src/interface/ir/op/block_ops/memory.cpp`
- `framework/src/interface/ir/op/block_ops/reduction.cpp`
- `framework/src/interface/ir/op/block_ops/transform.cpp`
- `framework/src/interface/ir/op/block_ops/unary.cpp`
- `framework/src/interface/ir/op/sync_ops/sync.cpp`
- `framework/src/interface/ir/op/tensor_ops/elementwise.cpp`
- `framework/src/interface/ir/op/tensor_ops/matmul.cpp`
- `framework/src/interface/ir/op/tensor_ops/memory.cpp`
- `framework/src/interface/ir/op/tensor_ops/reduction.cpp`
- `framework/src/interface/ir/op/tensor_ops/transform.cpp`
- `framework/src/interface/ir/op/tensor_ops/unary.cpp`

**C++ 测试 (7)**:
- `framework/tests/ut/interface/src/ir/test_builder.cpp`
- `framework/tests/ut/interface/src/ir/test_op_registry.cpp`
- `framework/tests/ut/interface/src/ir/test_op_registration.cpp`
- `framework/tests/ut/interface/src/ir/test_block_ops.cpp`
- `framework/tests/ut/interface/src/ir/test_tensor_ops.cpp`
- `framework/tests/ut/interface/src/ir/test_sync_ops.cpp`
- `framework/tests/ut/interface/src/ir/test_type_inference.cpp`

### 依赖: PR2 ✅

---

## PR4: Transform 基础框架 (Visitor/Mutator/Printer/Verifier)

**目标**: 提供 IR 遍历、变换、打印、校验等基础 pass 框架。

### 新增文件 (27个)

**头文件 (10)** — 代码层只依赖 PR2，测试依赖 PR3:
- `framework/include/ir/transform/base/functor.h` — 遍历函子（header-only，依赖 `kind_traits.h` → PR2）
- `framework/include/ir/transform/base/visitor.h` — 只读遍历器
- `framework/include/ir/transform/base/mutator.h` — 变换器
- `framework/include/ir/transform/verification_error.h` — 校验错误（header-only）
- `framework/include/ir/transform/dependency_graph.h` — 依赖图（header-only）
- `framework/include/ir/transform/dependency_analyzer.h` — 依赖分析
- `framework/include/ir/transform/printer.h` — IR 打印器
- `framework/include/ir/transform/structural_comparison.h` — 结构化比较（header-only）
- `framework/include/ir/transform/verifier.h` — IR 校验器
- `framework/include/ir/transform/passes.h` — Pass 管理

**源文件 (8)** — 与头文件配对:
- `framework/src/interface/ir/transform/visitor.cpp` ← 配对 `visitor.h`
- `framework/src/interface/ir/transform/mutator.cpp` ← 配对 `mutator.h`
- `framework/src/interface/ir/transform/printer.cpp` ← 配对 `printer.h`
- `framework/src/interface/ir/transform/structural_equal.cpp` ← 配对 `structural_comparison.h`
- `framework/src/interface/ir/transform/structural_hash.cpp` ← 配对 `structural_comparison.h`
- `framework/src/interface/ir/transform/dependency_analyzer.cpp` ← 配对 `dependency_analyzer.h`
- `framework/src/interface/ir/transform/verifier.cpp` ← 配对 `verifier.h`
- `framework/src/interface/ir/transform/passes.cpp` ← 配对 `passes.h`

**C++ 测试 (9)** — 6个测试依赖 `op_registry.h`(PR3)：
- `framework/tests/ut/interface/src/ir/test_visitor.cpp` — 依赖 `op_registry.h`
- `framework/tests/ut/interface/src/ir/test_mutator.cpp` — 依赖 `op_registry.h`
- `framework/tests/ut/interface/src/ir/test_printer.cpp` — 依赖 `op_registry.h`
- `framework/tests/ut/interface/src/ir/test_structural_equal.cpp` — 依赖 `op_registry.h`
- `framework/tests/ut/interface/src/ir/test_structural_hash.cpp` — 依赖 `op_registry.h`
- `framework/tests/ut/interface/src/ir/test_dependency_analyzer.cpp`
- `framework/tests/ut/interface/src/ir/test_verifier.cpp`
- `framework/tests/ut/interface/src/ir/test_passes.cpp`
- `framework/tests/ut/interface/src/ir/test_transform.cpp` — 依赖 `builder.h`

### 依赖: PR3 ✅ (头文件/源文件只依赖 PR2，但测试需要 PR3 的 `op_registry.h` 来构造测试数据)

---

## PR5: Serialization 序列化

**目标**: 提供 IR 的序列化/反序列化能力（基于 msgpack）。

### 新增文件 (9个)

**头文件 (3)**:
- `framework/include/ir/serialization/serializer.h`
- `framework/include/ir/serialization/deserializer.h`
- `framework/include/ir/serialization/type_registry.h`

**源文件 (4)** — 与头文件配对:
- `framework/src/interface/ir/serialization/serializer.cpp` ← 配对 `serializer.h`
- `framework/src/interface/ir/serialization/deserializer.cpp` ← 配对 `deserializer.h`
- `framework/src/interface/ir/serialization/type_registry.cpp` ← 配对 `type_registry.h`
- `framework/src/interface/ir/serialization/type_deserializers.cpp` ← 配对 `type_registry.h`

**三方库构建 (1)**:
- `cmake/third_party/msgpack/msgpack.cmake`

**C++ 测试 (1)**:
- `framework/tests/ut/interface/src/ir/test_serialization.cpp` — 依赖 `op_registry.h`(PR3) + `structural_comparison.h`(PR4) ✅

### 依赖: PR4 ✅ (头文件只依赖 PR2，源文件依赖 PR2，测试依赖 PR3+PR4)

---

## PR6: Python 绑定 / Python 接口 / 文档 / 构建集成

**目标**: 提供完整的 Python 层接口、pybind11 绑定、文档、以及对外部引用的适配修改。

### 新增文件 — Python 绑定 (7个)
- `python/src/bindings/modules/core.cpp`
- `python/src/bindings/modules/error.cpp`
- `python/src/bindings/modules/logging.cpp`
- `python/src/bindings/modules/ir.cpp`
- `python/src/bindings/modules/ir_builder.cpp`
- `python/src/bindings/modules/passes.cpp`
- `python/src/bindings/modules/testing.cpp`

### 新增文件 — Python IR 模块 (10个)
- `python/pypto/ir/__init__.py`
- `python/pypto/ir/builder.py`
- `python/pypto/ir/type.py`
- `python/pypto/ir/operators.py`
- `python/pypto/ir/pass_manager.py`
- `python/pypto/ir/printer.py`
- `python/pypto/ir/utils.py`
- `python/pypto/ir/op/__init__.py`
- `python/pypto/ir/op/block_ops.py`
- `python/pypto/ir/op/tensor_ops.py`

### 新增文件 — Python 测试 (共 50+ 个)
- `python/tests/ut/core/test_dtype.py`
- `python/tests/ut/core/test_error.py`
- `python/tests/ut/core/test_logging.py`
- `python/tests/ut/ir/__init__.py`
- `python/tests/ut/ir/core/` (5 文件)
- `python/tests/ut/ir/high_level/` (4 文件)
- `python/tests/ut/ir/memory/` (2 文件)
- `python/tests/ut/ir/operators/` (10 文件)
- `python/tests/ut/ir/printing/` (2 文件)
- `python/tests/ut/ir/serialization/` (12 文件)
- `python/tests/ut/ir/statements/` (8 文件)
- `python/tests/ut/ir/transforms/` (7 文件)

### 新增文件 — 文档 (24个)
- `docs/ir/en/` (12 个英文文档)
- `docs/ir/zh/` (12 个中文文档)

### 修改文件 — 构建集成与外部引用适配
- `CMakeLists.txt` — 顶层构建添加 core/新 IR 模块
- `framework/src/interface/CMakeLists.txt` — 添加新 IR 源文件
- `framework/tests/ut/interface/CMakeLists.txt` — 添加新 IR 测试
- `python/src/CMakeLists.txt` — 添加新绑定模块
- `python/src/bindings/bindings.h` — 声明新绑定函数
- `python/src/pybind11.cpp` — 注册新绑定模块
- `python/pypto/__init__.py` — 导入新 IR 模块
- `python/pypto/converter.py` — 移除旧 IR 引用

### 修改文件 — 外部代码适配 (8个)
- `framework/src/interface/cache/function_cache.cpp` — 适配新 function.h
- `framework/src/interface/cache/function_cache.h` — 适配新 function.h
- `framework/src/interface/function/function.h` — 移除旧 IR 引用
- `framework/src/machine/host/backend.cpp` — 适配新 IR 接口
- `framework/src/machine/utils/dynamic/dev_encode.cpp` — 适配新类型
- `framework/src/passes/block_graph_pass/dyn_attr_to_static.cpp` — 移除旧引用
- `framework/src/passes/tile_graph_pass/subgraph_to_function.cpp` — 适配新 IR
- `framework/src/passes/tile_graph_pass/subgraph_to_function.h` — 适配新 IR

### 依赖: PR5 ✅

---

## 统计汇总

| PR | 主题 | 新增文件数 | 修改文件数 | 估算行数 | 依赖 |
|-----|------|-----------|-----------|---------|------|
| PR1 | Core 基础库 | 13 | 0 | ~2,500 | 无 |
| PR2 | IR 核心类型体系 | 19 | 6 | ~9,000 | PR1 |
| PR3 | Op 注册/Builder/算子 | 29 | 0 | ~11,000 | PR2 |
| PR4 | Transform 框架 | 27 | 0 | ~12,000 | PR3 |
| PR5 | Serialization 序列化 | 9 | 0 | ~3,500 | PR4 |
| PR6 | Python/文档/集成 | 100+ | 8 | ~23,000 | PR5 |

## 合入顺序

```
PR1 → PR2 → PR3 → PR4 → PR5 → PR6
```

严格线性链，每个 PR 只依赖前一个。

## 关键调整说明

1. **`kind_traits.h` 从 PR3 移入 PR2**：它是 header-only（模板+宏），只依赖 PR2 内部头文件，而 `expr.cpp` 需要它 → 保证 `expr.h` 和 `expr.cpp` 在同一 PR
2. **改为严格线性链**：PR5 的 `test_serialization.cpp` 同时依赖 PR3(`op_registry.h`) 和 PR4(`structural_comparison.h`)，所以 PR5 必须在 PR4 之后
3. **每个 .h 与其 .cpp 都在同一 PR** — 已逐一验证，无遗漏
4. **PR1-PR5 均为 C++ 层**，Python 层全部集中在 PR6
5. **CMakeLists.txt 修改**需分散到各 PR 中，每个 PR 添加自身新增的源文件和测试
6. **删除旧 IR** 已在 `delete_origin_ir` 分支单独处理
7. PR6 体量仍然较大，如需进一步拆分可将「文档」和「Python 测试」单独拆出
