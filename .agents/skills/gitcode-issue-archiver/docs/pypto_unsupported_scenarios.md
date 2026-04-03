# PyPTO 不支持场景清单

> **用途**: 开发前必读，了解已知限制和规避方案
> **更新时间**: 2026-04-02
> **数据来源**: /data/s00454010/issues/archive/pypto_issues/

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

---

## 使用建议

### 开发前检查
1. ✅ 检查是否使用内存不连续的 tensor 开启合轴优化
2. ✅ 检查 assemble 操作是否使用 SSA 模式
3. ✅ 检查是否在循环外初始化 pypto.full
4. ✅ 检查 CPU 核数配置(仅影响泳道图导出)

### 遇到精度问题时
1. ✅ 检查是否添加了不必要的 `+ 0.0` 操作
2. ✅ 检查 view + assemble 连用场景
3. ✅ 检查 tensor 内存连续性配置

### 功能缺失时
1. ✅ 卷积算子暂不支持,需等待或自定义实现

---

**文档版本**: v1.0  
**更新日期**: 2026-04-02  
**维护者**: PyPTO Team
