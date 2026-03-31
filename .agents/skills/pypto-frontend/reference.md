# PyPTO 前端扩展参考

## JitCallableWrapper 调用流程

1. 用户调用 `wrapped(*args, **kwargs)`，前 N 个 args 为 tensor，其余为 non-tensor
2. `_parse_call_args` → `get_signature_high_performance` 得到 `(input_defs, output_defs, non_tensor_param_names)`
3. `_merge_non_tensor_params` 合并 kwargs、args、默认值 → `non_tensor_values`
4. `_get_or_create_kmodule`：`self.kwargs = non_tensor_values`，按 cache key（含 non_tensor_values）获取或创建 kmodule
5. `_allocate_output_tensors` → `_resolve_output_shape` 解析动态维度
6. `_execute_kernel` → `compile` → `_create_parser`（`captured_vars.update(self.kwargs)`）→ Parser 解析
7. 实际执行：LaunchKernel 或 _run_with_cpu

## entry.py 关键函数

- `jit(fn=None, **opts)` — 装饰器入口，返回 `JitCallableWrapper`
- `function(fn)` — 注册嵌套函数，供 parser 内联
- `parse(wrapper, *args)` — 绑定动态 shape、执行 6 阶段解析、返回 IR
- `get_signature_high_performance(func)` — 返回 `(input_tensor_defs, output_tensor_defs, non_tensor_param_names)`
- `__call__` 拆分为：`_parse_call_args`、`_merge_non_tensor_params`、`_get_or_create_kmodule`、`_resolve_device`、`_allocate_output_tensors`、`_resolve_output_shape`、`_convert_tensors_with_metadata`、`_execute_kernel`

## parser.py 参数解析

- `_parse_arguments_with_specs`：解析函数参数，返回 `(tensor_args, param_specs)`
- `_visit_arg`：tensor 返回 `pypto.Tensor`，非 tensor 返回 `(name, default_value)`
- 非 tensor 参数必须位于 tensor 之后；tensor 不能有默认值
- 辅助方法：`_validate_arguments_node`、`_build_arg_default_values`、`_raise_if_tensor_after_non_tensor`、`_raise_if_tensor_has_default`

## 解析阶段与 Liveness

- `LivenessAnalyzer` 在 IR 生成前运行，产出 `delete_after`
- `Parser` 在 `_visit_assign` 等中根据 `delete_after` 插入 `delete` 指令

## 嵌套函数内联

- `_try_nested_call(node)` 检测 `@function` 修饰的调用
- `param_specs` 包含 tensor 与非 tensor 参数，`expected_arg_count = len(param_specs)`
- 若匹配：解析被调用函数体，将结果内联到当前 IR
- 非 `@function` 的普通 Python 函数不能用于 JIT kernel

## 缓存

- `_get_compilation_cache_key(non_tensor_values)`：key 含 `(source_code, options_hash, captured_locals_hash, non_tensor_hash)`
- 不同 non_tensor_values（如 tiling=32 vs 64）生成不同 kernel

## 与 pypto-project 的关系

- `pypto-project` 覆盖整体架构、backend、troubleshooting
- `pypto-frontend` 专注 frontend、jit、parser、examples 用法
