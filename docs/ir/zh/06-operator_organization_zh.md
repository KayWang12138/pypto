# 算子实现组织

PyPTO 将算子组织为三个类别（TensorOp、BlockOp、SyncOp），在 `src/ir/op/` 下使用模块化的源文件。有关注册详细信息，请参阅 [05-operator_registration_zh.md](05-operator_registration_zh.md)。

## 文件结构

| 目录/文件 | 内容 |
|----------------|----------|
| `src/ir/op/type_inference.cpp` | 共享的类型推断工具 |
| **张量操作** | |
| `tensor_ops/elementwise.cpp` | TensorOp: add, sub, mul, div, maximum（包含标量变体） |
| `tensor_ops/matmul.cpp` | TensorOp: matmul（支持转置、批量处理） |
| `tensor_ops/memory.cpp` | TensorOp: create, view, assemble |
| `tensor_ops/reduction.cpp` | TensorOp: row_max, row_sum |
| `tensor_ops/transform.cpp` | TensorOp: reshape, transpose |
| `tensor_ops/unary.cpp` | TensorOp: exp, cast |
| **块操作** | |
| `block_ops/memory.cpp` | BlockOp: load, store, l0c_store, move, alloc, zeros, get_block_idx |
| `block_ops/elementwise.cpp` | BlockOp: add, sub, mul, div, maximum, minimum, adds, subs, muls, divs, cmp, cmps, col_expand, col_expand_mul, col_expand_div, col_expand_sub, expands |
| `block_ops/matmul.cpp` | BlockOp: matmul, matmul_acc |
| `block_ops/batch_matmul.cpp` | BlockOp: batch_matmul |
| `block_ops/reduction.cpp` | BlockOp: sum, max, min, row_max, row_sum, row_min（支持 axis、keepdim） |
| `block_ops/unary.cpp` | BlockOp: neg, exp, recip, sqrt, rsqrt, cast, log, abs, relu |
| `block_ops/transform.cpp` | BlockOp: view, reshape, transpose |
| `block_ops/broadcast.cpp` | BlockOp: row_expand_add, row_expand_sub, row_expand_mul, row_expand_div |
| **同步操作** | |
| `sync_ops/sync.cpp` | SyncOp: sync_src, sync_dst, bar_v, bar_m, bar_all |
| **测试** | |
| `testing.cpp` | test.op（用于测试目的） |

## 算子类别

### TensorOp: N 维张量操作

**目的**: 支持完整广播的通用 N 维张量
**类型**: `TensorType`（任意维度） | **位置**: `src/ir/op/tensor_ops/` | **Python API**: `from pypto.ir.op import tensor`

**操作:**

| 类别 | 操作 | 描述 |
|----------|-----------|-------------|
| **逐元素** | `tensor.add`, `tensor.sub`, `tensor.mul`, `tensor.div` | 支持 N 维广播的二元操作 |
| | `tensor.add_scalar`, `tensor.sub_scalar`, `tensor.mul_scalar`, `tensor.div_scalar` | 标量变体 |
| | `tensor.maximum` | 逐元素最大值 |
| **矩阵** | `tensor.matmul` | 支持转置的矩阵乘法（a_trans, b_trans）、批量矩阵乘法 |
| **内存** | `tensor.create` | 创建指定形状和数据类型的张量 |
| | `tensor.view` | 创建具有新形状和偏移量的视图/切片 |
| | `tensor.assemble` | 在偏移位置写入/更新张量值 |
| **归约** | `tensor.row_max`, `tensor.row_sum` | 按行归约操作 |
| **变换** | `tensor.reshape`, `tensor.transpose` | 形状变换 |
| **一元** | `tensor.exp`, `tensor.cast` | 逐元素一元操作 |

**示例:**
```python
from pypto.ir.op import tensor

ib = IRBuilder()
with ib.function("tensor_example") as f:
    input_a = f.param("input_a", ir.TensorType([128, 64, 32], DataType.FP32))
    input_b = f.param("input_b", ir.TensorType([128, 64, 32], DataType.FP32))
    f.return_type(ir.TensorType([128, 64, 32], DataType.FP32))
    result = ib.let("result", tensor.add(input_a, input_b))
    ib.return_stmt(result)
```

**C++ 实现:**
```cpp
// src/ir/op/tensor_ops/elementwise.cpp
TypePtr DeduceTensorOpElementwiseBinaryType(args, kwargs, op_name) {
  auto tensor_type1 = cast<TensorType>(args[0]->GetType());
  auto tensor_type2 = cast<TensorType>(args[1]->GetType());
  auto result_dtype = PromoteDataTypes(tensor_type1->dtype_, tensor_type2->dtype_);
  auto broadcast_result = BroadcastShapes(tensor_type1->shape_, tensor_type2->shape_);
  return make_shared<TensorType>(broadcast_result.shape, *result_dtype);
}

REGISTER_OP("tensor.add")
    .set_op_category("TensorOp")
    .set_description("Element-wise addition with broadcasting")
    .add_argument("lhs", "Left tensor").add_argument("rhs", "Right tensor")
    .f_deduce_type(DeduceTensorOpElementwiseBinaryType);

// src/ir/op/tensor_ops/matmul.cpp
REGISTER_OP("tensor.matmul")
    .set_op_category("TensorOp")
    .set_description("Matrix multiplication of two tensors with optional transpose")
    .add_argument("lhs", "Left-hand side tensor (TensorType)")
    .add_argument("rhs", "Right-hand side tensor (TensorType)")
    .set_attr<DataType>("out_dtype")
    .set_attr<bool>("a_trans")
    .set_attr<bool>("b_trans")
    .set_attr<bool>("c_matrix_nz")
    .f_deduce_type(DeduceTensorMatMulType);

// src/ir/op/tensor_ops/memory.cpp
REGISTER_OP("tensor.create")
    .set_op_category("TensorOp")
    .set_description("Create a new tensor with specified shape and dtype")
    .add_argument("shape", "Shape dimensions (TupleType of ScalarType(UINT64))")
    .set_attr<DataType>("dtype")
    .f_deduce_type(DeduceTensorCreateType);
```

### BlockOp: 硬件优化的块操作

**目的**: 具有显式内存管理的硬件优化块操作
**类型**: `TileType`（统一缓冲区中的块）
**位置**: `src/ir/op/block_ops/`
**Python API**: `from pypto.ir.op import block`

**设计**: 使用 `TileType`（而非单独的 `BlockType`）以与现有基础设施保持一致。命名空间 `block.*` + `TileType` 清楚地表明硬件优化的块操作。

#### 操作

| 类别 | 操作 | 描述 |
|----------|-----------|-------------|
| **内存** | `block.get_block_idx` | 获取块索引（→ ScalarType） |
| | `block.load` | TensorType → TileType（DDR 到统一缓冲区） |
| | `block.store` | TileType → TensorType（统一缓冲区到 DDR） |
| | `block.l0c_store` | 存储到 L0C 内存 |
| | `block.move` | 在内存空间之间移动块 |
| | `block.alloc` | 在内存中分配块 |
| | `block.zeros` | 创建零初始化的块 |
| **逐元素** | `block.add`, `block.sub`, `block.mul`, `block.div` | 块-块操作 |
| | `block.adds`, `block.subs`, `block.muls`, `block.divs` | 块-标量操作 |
| | `block.maximum`, `block.minimum` | 逐元素最小/最大值 |
| **比较** | `block.cmp` | 块-块比较 |
| | `block.cmps` | 块-标量比较 |
| **列扩展** | `block.col_expand` | 扩展列向量 |
| | `block.col_expand_mul`, `block.col_expand_div`, `block.col_expand_sub` | 列扩展与操作 |
| | `block.expands` | 使用标量的通用扩展 |
| **矩阵** | `block.matmul` | 2D 块矩阵乘法 |
| | `block.matmul_acc` | 带累加的矩阵乘法 |
| | `block.batch_matmul` | 批量矩阵乘法 |
| **一元** | `block.neg`, `block.abs`, `block.relu` | 基本一元操作 |
| | `block.exp`, `block.log` | 指数和对数 |
| | `block.sqrt`, `block.rsqrt`, `block.recip` | 平方根和倒数 |
| | `block.cast` | 类型转换 |
| **归约** | `block.sum`, `block.max`, `block.min` | 完全归约（axis、keepdim） |
| | `block.row_sum`, `block.row_max`, `block.row_min` | 按行归约 |
| **变换** | `block.view`, `block.reshape`, `block.transpose` | 形状变换 |
| **广播** | `block.row_expand_add`, `block.row_expand_sub` | 行广播与操作 |
| | `block.row_expand_mul`, `block.row_expand_div` | 行广播与操作 |

**数据流:** `TensorType (DDR) → block.load → TileType (统一缓冲区) → block.{ops} → TileType → block.store → TensorType (DDR)`

#### 使用示例

**低级 API (IRBuilder):**
```python
from pypto.ir.op import block

ib = IRBuilder()
with ib.function("block_computation") as f:
    input_a = f.param("input_a", ir.TensorType([128, 128], DataType.FP32))
    input_b = f.param("input_b", ir.TensorType([128, 128], DataType.FP32))
    output = f.param("output", ir.TensorType([128, 1], DataType.FP32))
    f.return_type(ir.TensorType([128, 1], DataType.FP32))

    # 加载、计算、归约、存储
    tile_a = ib.let("tile_a", block.load(input_a, [0, 0], [32, 128]))
    tile_b = ib.let("tile_b", block.load(input_b, [0, 0], [32, 128]))
    tile_mul = ib.let("tile_mul", block.mul(tile_a, tile_b))
    tile_sqrt = ib.let("tile_sqrt", block.sqrt(tile_mul))
    tile_sum = ib.let("tile_sum", block.sum(tile_sqrt, axis=1, keepdim=True))
    result = ib.let("result", block.store(tile_sum, [0, 0], [32, 1], output))
    ib.return_stmt(result)
```

**高级 API (Language DSL):**
```python
import pypto.language as pl

@pl.program
class MyProgram:
    @pl.function
    def block_computation(
        self,
        input_a: pl.Tensor[[128, 128], pl.FP32],
        input_b: pl.Tensor[[128, 128], pl.FP32],
        output: pl.Tensor[[128, 1], pl.FP32],
    ) -> pl.Tensor[[128, 1], pl.FP32]:
        tile_a: pl.Tile[[32, 128], pl.FP32] = pl.load(input_a, [0, 0], [32, 128])
        tile_b: pl.Tile[[32, 128], pl.FP32] = pl.load(input_b, [0, 0], [32, 128])
        tile_mul: pl.Tile[[32, 128], pl.FP32] = pl.mul(tile_a, tile_b)
        tile_sqrt: pl.Tile[[32, 128], pl.FP32] = pl.sqrt(tile_mul)
        tile_sum: pl.Tile[[32, 1], pl.FP32] = pl.row_sum(tile_sqrt)
        result: pl.Tensor[[128, 1], pl.FP32] = pl.store(tile_sum, [0, 0], [32, 1], output)
        return result
```

#### C++ 实现模式

**内存** (`src/ir/op/block_ops/memory.cpp`): `DeduceBlockLoadType` 从加载参数中提取块形状，返回 `TileType`。其他操作包括 `l0c_store`、`move`、`alloc` 和 `zeros`，用于全面的内存管理。

**逐元素** (`src/ir/op/block_ops/elementwise.cpp`): `DeduceBlockOpElementwiseBinaryType` 处理块-块（广播）和块-标量（保留块形状）两种情况。包括比较操作（`cmp`、`cmps`）和列扩展操作。

**矩阵** (`src/ir/op/block_ops/matmul.cpp`, `batch_matmul.cpp`): `DeduceBlockMatMulType` 用于 2D 块矩阵乘法，`DeduceBlockMatMulAccType` 用于累加变体，`DeduceBlockBatchMatMulType` 用于批量操作。

**归约** (`src/ir/op/block_ops/reduction.cpp`): `DeduceBlockSumType` 根据 axis/keepdim 计算输出形状，返回 `TileType` 或 `ScalarType`。支持完全归约（`sum`、`max`、`min`）和按行归约（`row_sum`、`row_max`、`row_min`）。

**一元** (`src/ir/op/block_ops/unary.cpp`): 全面的一元操作集，包括 `neg`、`exp`、`recip`、`sqrt`、`rsqrt`、`cast`、`log`、`abs` 和 `relu`。

**变换** (`src/ir/op/block_ops/transform.cpp`): 形状操作，包括 `view`、`reshape` 和 `transpose`。

**广播** (`src/ir/op/block_ops/broadcast.cpp`): 带算术运算的行扩展操作（`row_expand_add`、`row_expand_sub`、`row_expand_mul`、`row_expand_div`）。

### SyncOp: 同步操作

**目的**: 硬件同步和屏障 | **类型**: `UnknownType`（无返回值），在 `EvalStmt` 中使用
**位置**: `src/ir/op/sync_ops/` | **Python API**: `from pypto.ir.op import system`

**操作:** `system.sync_src/sync_dst`（设置/等待标志）、`system.bar_v/bar_m/bar_all`（屏障）

**示例:**
```python
from pypto.ir.op import system

with ib.function("sync_example") as f:
    ib.emit(system.bar_all())  # 全局屏障
    ib.emit(system.sync_src(set_pipe=2, wait_pipe=4, event_id=0))
    ib.emit(system.sync_dst(set_pipe=2, wait_pipe=4, event_id=0))
```

**C++ 实现:**
```cpp
// src/ir/op/sync_ops/sync.cpp
REGISTER_OP("system.bar_all")
    .set_op_category("SyncOp").set_pipe(PipeType::S)
    .no_argument().f_deduce_type(DeduceUnknownType);
```

**注意:** 对于不返回值的操作，使用 `ib.emit()`。与 PipeType::S 关联。

## 类型系统

| 类型 | 维度 | 使用场景 | 内存 | 特殊字段 |
|------|-----------|----------|--------|----------------|
| **TensorType** | N 维 | 通用张量、函数参数/返回值 | DDR（可选 MemRef） | 无 |
| **TileType** | N 维 | 统一缓冲区中的硬件优化块 | 统一缓冲区（可选 MemRef） | 可选 TileView |
| **ScalarType** | 0 维 | 标量值 | 寄存器 | 仅 dtype |
| **UnknownType** | N/A | 无返回值（同步操作） | N/A | 无 |

**类型层次结构:**
```
Type (抽象)
├── UnknownType
├── ScalarType(dtype)
├── ShapedType(dtype, shape, memref?)
│   ├── TensorType(shape, dtype, memref?)
│   └── TileType(shape, dtype, memref?, tile_view?)
└── TupleType(types[])
```

**何时使用:**
- **TensorType**: N 维张量、DDR 存储、函数边界、灵活形状
- **TileType**: 统一缓冲区中的块、硬件优化计算、显式内存管理

## 组织优势

**之前的结构 (✗):** 所有算子在 1-2 个大文件中 → 重新编译开销大、难以导航

**新结构 (✓):** 按类别模块化文件

| 优势 | 描述 |
|---------|-------------|
| **模块化** | 自包含的算子类别 |
| **构建性能** | 对一个类别的更改不会重新构建其他类别 |
| **可维护性** | 易于定位和修改算子 |
| **可扩展性** | 添加新算子简单直接 |
| **注册** | 通过 `REGISTER_OP` 静态初始化自动注册 |

## 设计模式

**1. 基于类别的组织:** 对相关算子分组，共享类型推断辅助函数。

```cpp
// src/ir/op/block_ops/elementwise.cpp
TypePtr DeduceBlockOpElementwiseBinaryType(...) { /* 共享逻辑 */ }
REGISTER_OP("block.add").f_deduce_type(DeduceBlockOpElementwiseBinaryType);
REGISTER_OP("block.mul").f_deduce_type(DeduceBlockOpElementwiseBinaryType);
```

**2. 静态初始化:** `REGISTER_OP` 宏在 `main()` 之前自动注册算子。

**3. 类型推断辅助函数:** 每个类别的函数处理类型推断。

```cpp
DeduceTensorOpElementwiseBinaryType(...)  // Tensor: 完整的 N 维广播
DeduceBlockOpElementwiseBinaryType(...)   // Block: 块 + 标量支持
DeduceBlockSumType(...)                   // Block: 带 axis/keepdim 的归约
```

## 实现指南

**添加算子:** 参见 [src/ir/op/README.md](../../src/ir/op/README.md)

**未来扩展:**
- 根据需要添加额外的张量操作
- 更多用于硬件优化的专用块操作
- 扩展的同步原语

## 测试与构建

**测试:** `tests/ut/ir/test_op_registry.py`、`test_tensor_ops.py`、`test_block_ops.py`

**CMakeLists.txt:**
```cmake
set(PYPTO_SOURCES
    src/ir/op_registry.cpp src/ir/op/type_inference.cpp
    src/ir/op/tensor_ops/elementwise.cpp
    src/ir/op/block_ops/memory.cpp src/ir/op/block_ops/elementwise.cpp
    src/ir/op/block_ops/reduction.cpp src/ir/op/block_ops/unary.cpp
    src/ir/op/sync_ops/sync.cpp  # 在此添加新文件
)
```

## 相关文档

- [05-operator_registration_zh.md](05-operator_registration_zh.md) - 算子注册系统详细信息
- [08-ir_builder_zh.md](08-ir_builder_zh.md) - 使用 IRBuilder 构建 IR
- [07-python_syntax_zh.md](07-python_syntax_zh.md) - Python IR 语法规范

## 统一语言 API (`pl.*`)

在语言层面，**统一命名空间**根据第一个参数的类型（`Tensor` vs `Tile`）自动分派张量和块操作。显式的 `pl.tensor.*` 和 `pl.block.*` 命名空间仍然可用。

### 分派规则

| 第一个参数类型 | `pl.add(a, b)` 分派到 | 标量右操作数处理 |
|----------------|----------------------------------|---------------------|
| `Tensor` | `tensor.add` | 由 `tensor.add` 内部处理 |
| `Tile` + Tile 右操作数 | `block.add` | N/A |
| `Tile` + 标量右操作数 | `block.adds` | 自动选择标量变体 |

### 统一操作

| 类别 | 操作 |
|----------|-----------|
| **二元算术** | `add`, `sub`, `mul`, `div`（Tile 的标量自动分派） |
| **逐元素** | `maximum`, `exp` |
| **形状** | `reshape`, `transpose`, `view` |
| **矩阵** | `matmul`（Tensor 路径接受额外的 kwargs） |
| **归约** | `row_max`, `row_sum` |
| **仅张量** | `cast`, `create`, `assemble` |

### 提升操作（仅单模块）

仅块操作如 `load`、`store`、`neg`、`sqrt` 等被提升到 `pl.*` 以方便使用。标量特定操作（`adds`、`subs`、`muls`、`divs`）**不会**被提升 — 请改用 `pl.add(tile, scalar)`。

### 示例

```python
import pypto.language as pl

@pl.program
class Example:
    @pl.function
    def compute(
        self,
        a: pl.Tensor[[64, 64], pl.FP32],
        b: pl.Tensor[[64, 64], pl.FP32],
        out: pl.Tensor[[64, 64], pl.FP32],
    ) -> pl.Tensor[[64, 64], pl.FP32]:
        # 统一 API — 分派到 tensor.add
        c: pl.Tensor[[64, 64], pl.FP32] = pl.add(a, b)

        # 块路径 — 统一 API 分派到 block.add
        tile_a: pl.Tile[[64, 64], pl.FP32] = pl.load(a, [0, 0], [64, 64])
        tile_b: pl.Tile[[64, 64], pl.FP32] = pl.load(b, [0, 0], [64, 64])
        tile_c: pl.Tile[[64, 64], pl.FP32] = pl.add(tile_a, tile_b)

        # 标量自动分派 — 分派到 block.muls
        tile_d: pl.Tile[[64, 64], pl.FP32] = pl.mul(tile_c, 2.0)

        result: pl.Tensor[[64, 64], pl.FP32] = pl.store(tile_d, [0, 0], [64, 64], out)
        return result
```
