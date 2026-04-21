# Issue: Agent skills 使用已废弃的 pypto.jit API

**标题**: `[Documentation|文档反馈]: Agent skills 使用已废弃的 pypto.jit API`

---

### Document Link（文档链接）

- https://github.com/cann/pypto/tree/main/.agents/skills/pypto-op-perf-tune/tune-swimlane/SKILL.md
- https://github.com/cann/pypto/tree/main/.agents/skills/pypto-op-perf-tune/perf-analyzer/scripts/analyze_perf.py

---

### Issues Section（问题文档片段）

**tune-swimlane/SKILL.md 第463行**:
```python
@pypto.jit(runtime_options={"device_sched_mode": 1})
```

**tune-swimlane/SKILL.md 第522行**:
```python
@pypto.jit(runtime_options={"device_sched_mode": 1})
```

**perf-analyzer/scripts/analyze_perf.py 第279行**:
```python
'code': '@pypto.jit(runtime_options={"device_sched_mode": 1})',
```

---

### Existing Issues（存在的问题）

根据 `pypto-precision-debug` skill 的明确说明：

> **强烈建议使用 `pypto.frontend.jit` 而非 `pypto.jit`！**
> 新前端（`pypto.frontend.jit`）是 PyPTO 推荐的写法，旧前端（`pypto.jit`）已不再维护。使用新前端可避免许多已知问题。

当前 Agent skills 中的示例代码使用了已废弃的 `pypto.jit` API，这会导致：

1. 用户按照 skill 指导生成的代码使用废弃 API
2. 代码可能无法正常运行或遇到已知问题
3. 与 PyPTO 官方推荐的写法不一致

---

### Suggested Fix / 修正建议 (Optional / 选填)

将所有 Agent skills 中的 `@pypto.jit` 替换为 `@pypto.frontend.jit`：

**修正前**:
```python
@pypto.jit(runtime_options={"device_sched_mode": 1})
```

**修正后**:
```python
@pypto.frontend.jit(runtime_options={"device_sched_mode": 1})
```

需要修改的文件：

1. `.agents/skills/pypto-op-perf-tune/tune-swimlane/SKILL.md` (第463、522行)
2. `.agents/skills/pypto-op-perf-tune/perf-analyzer/scripts/analyze_perf.py` (第279行)

建议同时检查其他 agent skills 是否有类似问题。

---

### Special notes for this issue/备注 (Optional / 选填)

此问题在用户开发 `quant_grouped_matmul_inplace_add_mx` 算子过程中发现，当前生成的代码使用了 `@pypto.frontend.jit`（正确），但 agent skills 文档中的示例仍使用废弃的 `@pypto.jit`。

---

## 手动创建说明

请在 GitCode 平台手动创建 Issue：

1. 打开 https://gitcode.com/cann/pypto/issues/new
2. 标题填写: `[Documentation|文档反馈]: Agent skills 使用已废弃的 pypto.jit API`
3. 内容复制上述 markdown 内容（不包括本说明）
4. 提交 Issue