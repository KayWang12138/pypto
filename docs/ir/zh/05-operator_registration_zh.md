# 算子注册系统

使用流式 API 实现类型安全的算子定义和自动类型推导。

## 算子类别

| 类别 | 类型 | 使用场景 | 参见 |
|----------|-------|----------|----------|
| **TensorOp** | TensorType | 支持广播的 N 维张量操作 | [06-operator_organization_zh.md](06-operator_organization_zh.md) |
| **BlockOp** | TileType | 硬件优化的块操作 | [06-operator_organization_zh.md](06-operator_organization_zh.md) |
| **SyncOp** | UnknownType/PipeType | 流水线屏障和同步 | [SyncOp 操作](#syncop-操作) |

**核心特性**：流式 API、自动类型推导、用于元数据的 kwargs、NumPy 风格的广播、类型提升、动态维度（`kDynamicDim`）

## 类型系统

```cpp
// TensorType: N 维张量
TensorType(DataType::FP32, {dim1, dim2, dim3, ...})

// TileType: 硬件优化的块
TileType(DataType::FP16, {dim1, dim2})
TileType(DataType::FP16, {dim1})
TileType(DataType::FP16, {d1, d2, d3})  // N-D 也可以

// 动态维度 (pypto/core/common.h)
constexpr int64_t kDynamicDim = -1;
auto dynamic_dim = make_int(kDynamicDim);
```

## REGISTER_OP 流式 API

| 方法 | 用途 | 示例 |
|--------|---------|---------|
| `set_op_category(str)` | 算子类别 | `.set_op_category("TensorOp")` |
| `set_description(str)` | 人类可读的描述 | `.set_description("Element-wise add")` |
| `add_argument(name, desc)` | 位置参数 Expr | `.add_argument("lhs", "Left tensor")` |
| `no_argument()` | 无参数（同步操作） | `.no_argument()` |
| `set_attr<T>(name)` | Kwarg 模式（T: bool, int, DataType 等） | `.set_attr<bool>("a_trans")` |
| `set_pipe(PipeType)` | 硬件流水线类型 | `.set_pipe(PipeType::S)` |
| `f_deduce_type(fn)` | 类型推导函数 | `.f_deduce_type(DeduceAddType)` |

**类型推导函数签名：**
```cpp
std::function<TypePtr(const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs)>
```

## C++ 注册示例

### 简单的逐元素算子

```cpp
// src/ir/op/tensor_ops/elementwise.cpp
REGISTER_OP("tensor.add")
    .set_op_category("TensorOp")
    .add_argument("lhs", "Left tensor")
    .add_argument("rhs", "Right tensor")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      CHECK(args.size() == 2);
      auto t1 = std::dynamic_pointer_cast<const TensorType>(args[0]->GetType());
      auto t2 = std::dynamic_pointer_cast<const TensorType>(args[1]->GetType());
      auto dtype = PromoteDataTypes(t1->dtype_, t2->dtype_);
      auto shape = BroadcastShapes(t1->shape_, t2->shape_);
      return std::make_shared<TensorType>(shape.shape, *dtype);
    });
```

### 带 Kwargs 的算子

```cpp
// src/ir/op/tensor_ops/matmul.cpp
TypePtr DeduceMatMul(const std::vector<ExprPtr>& args,
                     const std::vector<std::pair<std::string, std::any>>& kwargs) {
  auto lhs = std::dynamic_pointer_cast<const TensorType>(args[0]->GetType());
  auto rhs = std::dynamic_pointer_cast<const TensorType>(args[1]->GetType());

  auto get = [&](const std::string& k, bool d) {
    auto it = kwargs.find(k);
    return (it != kwargs.end()) ? std::any_cast<bool>(it->second) : d;
  };

  auto it = kwargs.find("out_dtype");
  DataType dtype = (it != kwargs.end()) ? static_cast<DataType>(std::any_cast<int>(it->second))
                                        : *PromoteDataTypes(lhs->dtype_, rhs->dtype_);

  bool a_t = get("a_trans", false), b_t = get("b_trans", false);
  ExprPtr m = a_t ? lhs->shape_[1] : lhs->shape_[0];
  ExprPtr n = b_t ? rhs->shape_[0] : rhs->shape_[1];
  return std::make_shared<TensorType>(std::vector<ExprPtr>{m, n}, dtype);
}

REGISTER_OP("tensor.matmul")
    .set_op_category("TensorOp")
    .add_argument("lhs", "Left matrix")
    .add_argument("rhs", "Right matrix")
    .set_attr<DataType>("out_dtype")
    .set_attr<bool>("a_trans")
    .set_attr<bool>("b_trans")
    .f_deduce_type(DeduceMatMul);
```

## Python 使用

### 创建和使用算子

```python
from pypto.pypto_core import DataType, ir
from pypto.ir import op

span = ir.Span.unknown()
dim4, dim8 = ir.ConstInt(4, DataType.INT32, span), ir.ConstInt(8, DataType.INT32, span)

# 创建张量
tensor_a = ir.Var("a", ir.TensorType([dim4, dim8], DataType.FP32), span)
tensor_b = ir.Var("b", ir.TensorType([dim8], DataType.FP32), span)

# 简单算子
result = op.tensor.add(tensor_a, tensor_b)  # 广播: [4,8] + [8] → [4,8]

# 带 kwargs 的算子
a = ir.Var("a", ir.TensorType([dim64, dim128], DataType.FP16), span)
b = ir.Var("b", ir.TensorType([dim128, dim64], DataType.FP16), span)
matmul = op.tensor.matmul(a, b, out_dtype=DataType.FP32, a_trans=True)
# 打印: tensor.matmul(a, b, a_trans=True, out_dtype=51)

# 查询注册表
assert ir.is_op_registered("tensor.add")
op_instance = ir.get_op("tensor.add")
```

## Kwargs（关键字参数）

调用表达式使用 kwargs 将 Expr 参数与元数据参数分离。

### Kwargs vs Args vs Attributes

| | **Args** | **Kwargs** | **Op Attributes** |
|---|----------|------------|-------------------|
| **类型** | `ExprPtr` | `std::any` | 类型擦除 |
| **作用域** | 每次调用 | 每次调用 | 全局 |
| **用途** | 张量、维度、偏移 | `out_dtype`、标志、模式 | 设备、类别 |
| **访问** | `call.args_` | `call.kwargs_` | `op.get_attr()` |

### C++ - 读取 Kwargs

```cpp
TypePtr DeduceCastType(const std::vector<ExprPtr>& args,
                       const std::vector<std::pair<std::string, std::any>>& kwargs) {
  auto input = std::dynamic_pointer_cast<const TensorType>(args[0]->GetType());

  // 必需的 kwarg
  auto it = kwargs.find("target_type");
  CHECK(it != kwargs.end()) << "tensor.cast requires 'target_type'";
  DataType target = static_cast<DataType>(std::any_cast<int>(it->second));

  // 可选的，带默认值
  int mode = 0;
  auto mode_it = kwargs.find("mode");
  if (mode_it != kwargs.end()) mode = std::any_cast<int>(mode_it->second);

  return std::make_shared<TensorType>(input->shape_, target);
}
```

### Python - 使用 Kwargs

```python
result = op.tensor.matmul(a, b, out_dtype=DataType.FP32, a_trans=True)
print(result.kwargs)  # {'out_dtype': 51, 'a_trans': True}
```

### 定义 Kwarg 模式

使用 `set_attr<T>()` 定义允许的 kwargs。类型：`bool`、`int`、`std::string`、`double`、`DataType`。

```cpp
REGISTER_OP("tensor.matmul")
    .set_attr<DataType>("out_dtype")
    .set_attr<bool>("a_trans")
    .f_deduce_type(DeduceMatMulType);

// Python: 查询模式
op = ir.get_op("tensor.matmul")
keys = op.get_attr_keys()  # ['out_dtype', 'a_trans']
```

## 广播和类型提升

### NumPy 风格的广播

维度从右到左对齐：
```
[4, 8] + [4, 8] → [4, 8]  # 完全匹配
[4, 8] + [8]    → [4, 8]  # 缺失的左侧维度 = 1
[4, 1] + [8]    → [4, 8]  # 大小为 1 的维度广播
[1, 8] + [4, 8] → [4, 8]  # 大小为 1 的维度广播
[4, 8] + [5]    → 错误   # 8 ≠ 5
```

### 类型提升

标准数值规则：浮点 > 整数，更大 > 更小，有符号 > 无符号（相同大小）。

```
INT32 + INT32 → INT32
INT32 + FP32  → FP32   (浮点优先)
INT32 + INT64 → INT64  (更大的大小)
UINT32 + INT32 → INT32 (有符号优先)
```

## SyncOp 操作

硬件同步和屏障。返回 `UnknownType`。与 `ib.emit()` 一起使用。

| 操作 | 描述 | Kwargs |
|-----------|-------------|--------|
| `system.bar_all` | 全局屏障 | 无 |
| `system.bar_v` | 向量屏障 | 无 |
| `system.bar_m` | 矩阵屏障 | 无 |
| `system.sync_src` | 设置同步标志 | `set_pipe`, `wait_pipe`, `event_id` |
| `system.sync_dst` | 等待同步标志 | `set_pipe`, `wait_pipe`, `event_id` |

**Python 示例：**
```python
from pypto.ir.op import system
ib.emit(system.bar_all())
ib.emit(system.sync_src(set_pipe=2, wait_pipe=4, event_id=0))
```

**C++ 注册（`src/ir/op/sync_ops/sync.cpp`）：**
```cpp
REGISTER_OP("system.bar_all")
    .set_op_category("SyncOp")
    .set_pipe(PipeType::S)
    .no_argument()
    .f_deduce_type(DeduceUnknownType);

REGISTER_OP("system.sync_src")
    .set_op_category("SyncOp")
    .set_pipe(PipeType::S)
    .set_attr<int>("set_pipe")
    .set_attr<int>("wait_pipe")
    .set_attr<int>("event_id")
    .no_argument()
    .f_deduce_type(DeduceUnknownType);
```

## 添加新操作

1. **选择类别文件**：`src/ir/op/tensor_ops/elementwise.cpp`、`matmul.cpp`、`reduction.cpp`，或 `src/ir/op/block_ops/memory.cpp`、`unary.cpp`

2. **实现类型推导**：
   ```cpp
   TypePtr DeduceType(const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
     CHECK(args.size() == 2) << "op requires 2 arguments";
     // 验证类型、读取 kwargs、计算输出类型
     return result_type;
   }
   ```

3. **注册**：
   ```cpp
   REGISTER_OP("tensor.matmul")
       .set_op_category("TensorOp")
       .add_argument("lhs", "Left tensor")
       .add_argument("rhs", "Right tensor")
       .set_attr<DataType>("out_dtype")
       .f_deduce_type(DeduceType);
   ```

4. **Python 包装器**（`python/pypto/ir/op/tensor_ops.py`）：
   ```python
   def matmul(lhs: Expr, rhs: Expr, out_dtype=None, a_trans=False) -> Call:
       kwargs = {}
       if out_dtype: kwargs["out_dtype"] = out_dtype.code() if isinstance(out_dtype, DataType) else out_dtype
       if a_trans: kwargs["a_trans"] = a_trans
       return _ir_core.create_op_call("tensor.matmul", [lhs, rhs], kwargs, Span.unknown())
   ```

5. **添加测试**，在 `tests/ut/ir/` 中，如果需要更新 `CMakeLists.txt`

## 参考

- 通用常量：`include/pypto/core/common.h`
- 类型定义：`include/pypto/ir/type.h`
- 算子注册表：`include/pypto/ir/op_registry.h`
- 类型推断工具：`include/pypto/ir/type_inference.h`
- 类型推断实现：`src/ir/op/type_inference.cpp`
- 算子注册表实现：`src/ir/op_registry.cpp`
- 张量算子实现：`src/ir/op/tensor_ops/`
- 块算子实现：`src/ir/op/block_ops/`
- 同步算子实现：`src/ir/op/sync_ops/`
