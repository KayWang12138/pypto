# PyPTO 不支持场景清单

> **用途**: 开发前必读，了解已知限制和规避方案
> **更新时间**: 2026-04-06
> **数据来源**: /data/s00454010/issues/archive/pypto_issues/ + /mnt/workspace/gitCode/cann/package/issues_0/ (Issue #600~#650)

---

## 明确的不支持场景

本部分列出 PyPTO 明确不支持的场景及规避方案。只包含有明确触发条件和规避方案的高质量条目。

### 1. 内存复用 - view + assemble 场景

**不支持场景**: view + assemble 连用场景下 inplace pass 推导有误导致精度问题
- **触发条件**:
  - tensor 先经过 view 操作
  - 再被 assemble 使用
  - 不添加中间 add 操作(如 `x + 0.0`)
- **规避方案**: 使用最新版本(已修复),无需手动插入 add 算子
- **状态**: 已修复
- **来源**: Issue #27 (closed, resolved)

### 2. 内存连续性 - 合轴优化

**不支持场景**: 对内存不连续的 tensor 开启合轴优化
- **触发条件**:
  - tensor 是 slice/view 后的列向量(如从 [128,2] 中取出 [128,1])
  - tensor 在 GM 上内存不连续
  - 调用 `pypto.experimental.set_operation_config(combine_axis=True)` 或 `force_combine_axis=True`
- **规避方案**: 不对内存不连续的 tensor 开启合轴优化,或使用尾轴 reduce 的结果(保证连续)
- **状态**: 仍需规避
- **来源**: Issue #108 (closed)

### 3. Tensor 初始化位置 - full 在循环外

**不支持场景**: pypto.full 在循环外初始化,在循环内使用导致精度问题
- **触发条件**:
  - 在循环外创建 `zeros = pypto.full(...)`
  - 在循环内使用该 tensor 做 assemble 操作
- **规避方案**: 在循环内初始化 pypto.full,或使用最新版本(已修复)
- **状态**: 已修复
- **来源**: Issue #340 (closed, resolved)

### 4. 线程调度 - CPU 核数大于 64

**不支持场景**: CPU 核数大于 64 时,泳道图导出卡死
- **触发条件**:
  - CPU 核数 > 64(如 320 核服务器)
  - 使用 `@pypto.jit(runtime_options={"run_mode": 1})` 或 debug 模式导出泳道图
- **规避方案**: 使用 `@pypto.jit(debug_options={"runtime_debug_mode": 1})` 导出泳道图
- **状态**: 已修复
- **来源**: Issue #2 (closed, resolved)

### 5. 卷积算子 - Conv

**不支持场景**: 不支持 conv 算子(Conv2D)
- **触发条件**:
  - 需要使用卷积操作
- **规避方案**: 等待功能支持或自定义实现
- **状态**: 功能缺失
- **来源**: Issue #9 (open)

### 6. Inplace 操作 - assemble 后使用

**不支持场景**: assemble 到 tensor 后,该 tensor 后续有计算操作(非 SSA 模式)
- **触发条件**:
  - 使用非 SSA 模式:`pypto.assemble(a, offsets, c)` 后 `t = c + 1`
  - 对从外部传入的 tensor 进行 inplace assemble 修改
- **规避方案**: 使用 SSA 模式:`pypto.assemble([(a, offsets)], c)` 后 `t = c + 1`
- **状态**: 仍需规避
- **来源**: Issue #187 (closed)

### 7. 表达式处理 - 加 0.0 操作

**不支持场景**: 在表达式中添加加 0.0 操作导致精度异常
- **触发条件**:
  - 代码中添加形如 `x + 0.0` 的操作
- **规避方案**: 避免不必要的加 0.0 操作
- **状态**: 仍需规避
- **来源**: Issue #137 (closed)

### 8. 布尔掩码索引不支持 (新增 #600~#650)

**不支持场景**: 使用 Python 原生 `list[bool]` 或布尔表达式结果直接作为 `pypto.Tensor` 的掩码下标
- **触发条件**:
  - 使用 `tensor[bool_mask] = value` 或 `tensor[condition]` 形式的布尔掩码索引
  - 例如 `expert_ids[x_active_mask == 0] = -1`
- **现象**: 报错 `AttributeError: 'int' object has no attribute 'shape'`，错误信息不具指导性
- **规避方案**: 使用 `pypto.where(condition, true_value, false_value)` 替代布尔掩码赋值
- **状态**: 仍需规避
- **来源**: Issue #650 (closed)

### 9. Loop 内不支持 `a = a + 1` 写法 (新增 #600~#650)

**不支持场景**: 在 `pypto.loop` 循环体内使用 `a = a + 1` 形式的递增赋值
- **触发条件**:
  - 在 `pypto.loop` 或 `pypto.loop_unroll` 循环体内使用 `a = a + 1`
  - trace 过程中 `=` 号前后的 `a` 是同一个对象
- **现象**: TensorGraph 阶段精度异常
- **规避方案**: 改写为 `a[:] = a + 1`
- **状态**: 仍需规避
- **来源**: Issue #626 (closed)

### 10. `for in range` 内含 Symbolic Scalar 条件导致路径爆炸 (新增 #600~#650)

**不支持场景**: 使用 `for in range(N)` 写法且循环体内 `if` 条件包含 symbolic scalar
- **触发条件**:
  - 使用 `for i in range(N)`（N 为固定值）写法
  - 循环体内 `if` 判断条件包含 symbolic scalar（编译期无法确定 true/false 的变量，如动态轴推导出的值）
- **现象**: 编译卡住无响应，前端展开产生 `2^N` 个 dynamic_loop path，编译 function 数量指数级增长
- **规避方案**: 使用 `pypto.loop()` 替代 `for in range()`，或在循环外确保条件为编译期常量
- **状态**: 仍需规避
- **来源**: Issue #619 (closed)

### 11. Tensor 切片索引仅支持 INT32 (新增 #600~#650)

**不支持场景**: 对 `pypto.Tensor` 使用非 INT32 类型的索引进行切片
- **触发条件**:
  - 使用非 INT32 类型的索引进行切片，如 `tensor[int64_idx]`
- **现象**: 报错 `AssertionError: tensor dtype must be DT_INT32`
- **规避方案**: 将索引 cast 为 INT32 后再使用：`idx_int32 = pypto.cast(idx, pypto.DT_INT32)`
- **状态**: 仍需规避
- **来源**: Issue #642 (closed)

### 12. 比较操作符返回值非标准 0/1 (新增 #600~#650)

**不支持场景**: 比较操作符（ge、le）返回的 BOOL 类型 cast 为浮点后无法得到正确的 0/1
- **触发条件**:
  - 使用 `pypto.ge`、`pypto.le` 等比较操作符
  - 对返回的 BOOL 类型执行 `pypto.cast` 转换为浮点类型
- **现象**: cast 后无法得到正确的 0/1 结果，true 值表现为极小浮点数（如 `5.96e-08`）
- **规避方案**: 使用 `pypto.where(condition, one, zero)` 替代 `pypto.cast(condition, float_dtype)`
- **状态**: 已修复（cast 不支持 bool 类型问题已合入解决）
- **来源**: Issue #602 (closed, resolved)

### 13. `pypto.tensor()` 创建的局部临时变量缺乏看护 (新增 #600~#650)

**不支持场景**: 在 kernel 函数内部使用 `pypto.tensor()` 创建局部临时变量，在多个 loop 或 assemble 操作间共享
- **触发条件**:
  - 在 kernel 函数内部使用 `pypto.tensor([shape], dtype, "name")` 创建局部临时变量
  - 在多个 loop 或 assemble 操作间共享该临时变量
- **现象**: 添加额外 loop 或 assemble 后，其他 outcast 出现精度异常，疑似内存踩踏；精度工具 TensorGraph 和所有 Pass 校验均通过，但上板精度错误
- **规避方案**: 将临时变量作为 kernel 函数的参数输入（通过 `pypto.from_torch` 传入），而非在函数内部创建；或使用 `pypto.frontend.jit` 替代旧版 `pypto.jit`
- **状态**: 仍需规避（open）
- **来源**: Issue #608 (open)

### 14. 次尾轴 Reduce 场景不支持合轴（combine_axis）(新增 #600~#650)

**不支持场景**: 开启 combine_axis 后对次尾轴执行 Reduce 操作
- **触发条件**:
  - 开启 `pypto.experimental.set_operation_options(combine_axis=True)`
  - 对次尾轴执行 `pypto.sum(..., dim=-2, keepdim=True)` 操作
- **现象**: 合轴属性未正确传递至关联的 PAIRSUM op，导致输出精度异常
- **规避方案**: 关闭 combine_axis 开关，或避免在次尾轴 Reduce 场景使用合轴优化
- **状态**: 已修复
- **来源**: Issue #645 (closed, resolved)

### 15. index_add 算子 srcInput 轴维度大于 selfInput 时切分异常 (新增 #600~#650)

**不支持场景**: index_add 算子 srcInput 的 axis 轴维度大于 selfInput 时产生不合理切分
- **触发条件**:
  - `pypto.index_add` 算子中，srcInput 的 axis 轴维度大于 selfInput 的对应维度
- **现象**: 切分逻辑不合理，导致精度问题（ExpandFunction Pass 阶段首次出现差异）
- **规避方案**: 确保 srcInput 和 selfInput 在 axis 轴上的维度匹配
- **状态**: 已修复
- **来源**: Issue #600 (closed, resolved)

---

## 使用建议

### 开发前检查
1. ✅ 检查是否使用内存不连续的 tensor 开启合轴优化
2. ✅ 检查 assemble 操作是否使用 SSA 模式
3. ✅ 检查是否在循环外初始化 pypto.full
4. ✅ 检查 CPU 核数配置(仅影响泳道图导出)
5. ✅ 检查是否使用布尔掩码索引（应改用 where）
6. ✅ 检查 loop 内是否使用 `a = a + 1` 写法（应改用 `a[:] = a + 1`）
7. ✅ 检查是否使用 `for in range` 且内含 symbolic scalar 条件（应改用 pypto.loop）
8. ✅ 检查 tensor 切片索引是否为 INT32 类型
9. ✅ 检查是否使用 `pypto.tensor()` 创建局部临时变量（应改为函数参数传入）

### 遇到精度问题时
1. ✅ 检查是否添加了不必要的 `+ 0.0` 操作
2. ✅ 检查 view + assemble 连用场景
3. ✅ 检查 tensor 内存连续性配置
4. ✅ 检查 compare 操作后是否使用 cast 转换 BOOL（应改用 where）
5. ✅ 检查 index_add 算子 srcInput/selfInput 维度匹配

### 功能缺失时
1. ✅ 卷积算子暂不支持,需等待或自定义实现

---

**文档版本**: v2.0  
**更新日期**: 2026-04-06  
**维护者**: PyPTO Team
