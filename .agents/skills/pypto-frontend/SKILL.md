---
name: pypto-frontend
description: 处理 PyPTO 前端相关问题和开发，包括 @pypto.frontend.jit、@pypto.frontend.function、pypto.frontend.dynamic、解析流水线、类型注解、控制流、嵌套函数等。适用于 frontend 目录、parser、entry、JitCallableWrapper 或 examples 中的前端用法。
---

# PyPTO 前端 Skill

## 主入口

**`@pypto.frontend.jit`** — 核心装饰器，将 Python 函数编译为 PTO IR，可用 torch.Tensor 调用。

```python
@pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.NPU})
def my_kernel(x: pypto.Tensor(shape, pypto.DT_FP32)) -> pypto.Tensor(shape, pypto.DT_FP32):
    pypto.set_vec_tile_shapes(8, 8)
    return x + x
```

支持 `@jit` 与 `@jit()` 两种写法。

## 目录结构

```
python/pypto/frontend/
├── __init__.py         # jit, function, dynamic
├── developer_doc.md    # 开发文档
└── parser/
    ├── entry.py        # jit(), function(), JitCallableWrapper, parse()
    ├── parser.py       # Parser, _visit_*, _try_nested_call
    ├── context.py      # Context, ContextFrame
    ├── evaluator.py    # ExprEvaluator
    ├── doc.py          # Python AST ↔ Doc AST
    ├── doc_core.py     # Doc AST 节点
    ├── diagnostics.py # 错误报告
    ├── error.py        # ParserError
    └── liveness.py     # LivenessAnalyzer
```

## 解析流水线（6 阶段）

1. **Source** — `diagnostics.Source` 从函数提取源码
2. **Python AST** — `ast.parse()`
3. **Doc AST** — `doc.to_doc()`（与 Python 版本解耦）
4. **Liveness** — `LivenessAnalyzer` 计算变量最后使用点
5. **IR** — `Parser` 访问者模式生成 PTO IR
6. **Lazy** — `JitCallableWrapper` 首次调用时解析

## 公共 API

| API | 作用 |
|-----|------|
| `@pypto.frontend.jit` | JIT 编译，用 torch tensor 调用 |
| `@pypto.frontend.function` | 嵌套函数，在 JIT kernel 中被内联 |
| `pypto.frontend.dynamic("N")` | 创建符号维度，返回 SymbolicScalar |

## jit 装饰器参数

```python
@pypto.frontend.jit(
    runtime_options={"run_mode": mode},  # NPU 或 SIM
    codegen_options={...},
    host_options={...},
    pass_options={...},
    verify_options={...},
    debug_options={...},
    use_cache=True,
)
```

- `runtime_options.run_mode`：`pypto.RunMode.NPU` 或 `pypto.RunMode.SIM`
- 无 NPU 时默认 SIM；有 `ASCEND_HOME_PATH` 时默认 NPU

## 类型注解要求

- 入参：`pypto.Tensor(shape, dtype)`，shape 可为常量或 `dynamic("N")`
- 返回值：`pypto.Tensor(...)` 或 `(tensor1, tensor2)`
- 示例：`examples/01_beginner/basic/basic_ops.py`

## 动态 shape

```python
N = pypto.frontend.dynamic("N")
@pypto.frontend.jit()
def kernel(x: pypto.Tensor((N, 128), pypto.DT_FP32)) -> pypto.Tensor((N, 128), pypto.DT_FP32):
    return x
```

首次调用时根据输入 shape 绑定 N，参考 `examples/02_intermediate/controflow/others/dynamic.py`、`examples/02_intermediate/controflow/loop/loop.py`。

## 循环

```python
for idx in pypto.loop(5):           # 0..5
for idx in pypto.loop(0, n, 1):     # start, stop, step
```

循环变量 `idx` 为 SymbolicScalar。参考 `examples/02_intermediate/controflow/loop/loop.py`。

## 条件分支

- **静态**：`if flag:`（flag 为编译时常量）
- **动态**：`if pypto.cond(expr):` 或 `if pypto.is_loop_begin(idx):`（解析器会包装 cond）

```python
for idx in pypto.loop(b_loop):
    if pypto.is_loop_begin(idx):
        output[...] = t3_sub + val
    elif pypto.is_loop_end(idx):
        output[...] = t3_sub + val + 1
    else:
        output[...] = t3_sub
```

参考 `examples/02_intermediate/controflow/condition/add_scalar_loop_dyn_axis_dyn_loop_cond.py`。

## 嵌套函数（@pypto.frontend.function）

```python
@pypto.frontend.function
def layernorm_core(x, gamma, beta, eps=1e-6):
    # 非 JIT，在 JIT kernel 中被内联
    ...
    return normalized * gamma + beta

@pypto.frontend.jit()
def layer_norm_kernel(x, gamma, beta):
    pypto.set_vec_tile_shapes(64, 128)
    out = layernorm_core(x, gamma, beta)  # 内联
    return out
```

参考 `examples/03_advanced/patterns/function/function.py`。

## 调用约定

- **Tensor 参数**：前 N 个位置参数必须为 `torch.Tensor`，contiguous、同一 device
- **非 Tensor 参数**：支持 `args` 或 `kwargs` 传入（如 `tiling`），必须位于 tensor 参数之后
- 返回 `torch.Tensor` 或 `tuple[torch.Tensor, ...]`

```python
# 非 tensor 参数示例
@pypto.frontend.jit()
def add(a: pypto.Tensor((32,32), pypto.DT_FP32), b: pypto.Tensor((32,32), pypto.DT_FP32), tiling=None):
    pypto.set_vec_tile_shapes(tiling, tiling)
    return a + b
add(a_t, b_t, tiling=32)   # kwargs
add(a_t, b_t, 32)          # args
```

## 示例索引

| 示例 | 路径 | 内容 |
|------|------|------|
| Hello World | `examples/00_hello_world/hello_world.py` | 最简加法 |
| 基础运算 | `examples/01_beginner/basic/basic_ops.py` | add/mul/matmul、dynamic |
| 循环 | `examples/02_intermediate/controflow/loop/loop.py` | pypto.loop、dynamic |
| 条件 | `examples/02_intermediate/controflow/condition/` | if/cond、is_loop_begin/end |
| 多函数 | `examples/03_advanced/patterns/function/function.py` | @function 内联、组合 |
| Tiling | `examples/01_beginner/tiling/tiling_config.py` | set_vec_tile_shapes、set_cube_tile_shapes |

## 解析器要点

- Visitor：`_visit_{snake_case}` 对应 AST 节点
- `_parse_arguments_with_specs`：解析参数，区分 tensor 与非 tensor；非 tensor 必须在 tensor 之后
- `_visit_arg`：tensor 返回 `pypto.Tensor`，非 tensor 返回 `(name, default_value)`
- `ParamSpec = (name, is_tensor, value)`：参数规格
- `_try_nested_call`：处理 `@function` 内联
- `Context`：变量作用域；`_create_parser` 会将 `self.kwargs` 注入 `captured_vars`，供解析时解析 tiling 等变量
- `ExprEvaluator`：注解与常量求值
- `ParserError`：带 AST 节点的错误；`PTO_BACKTRACE=1` 显示完整堆栈

## 常见调试

- **Unknown keyword argument**：传了非函数签名的 kwargs，或函数无 non-tensor 参数时传了 kwargs

## 参考

- `python/pypto/frontend/developer_doc.md` — 前端开发文档
- `examples/` — 按难度分级的示例
