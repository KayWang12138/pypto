---
name: pypto-kernel-phase4-phase5
description: Phase 4（渐进式集成 — 逐个组装模块，最早失败边界规则）和 Phase 5（PyPTO 特定的结构规则 — loop、状态、广播、对齐、tile、写回）。
---

# PyPTO 复杂 Kernel — Phase 4–5：集成与结构规则

## Phase 4：渐进式集成

目标：组装最终 kernel，而不必等到最后才发现集成错误。

### 必选集成顺序

不要等到所有模块独立构建完成后再做一次最终合并。

而是（在暂存文件中体现 — 参见 `skills/lead-orchestrator/references/rules.md` 规则 14）：
1. 在 `<op>_module1.py` 中验证 M1
2. 在 `<op>_module12.py` 中集成 M1+M2，验证边界
3. 在 `<op>_module123.py` 中集成 M1+M2+M3，验证边界
4. 继续直到 `<op>_module1…N.py` 成为完整 kernel

### 最早失败边界规则

当集成行为失败时：
- 定位最早的不匹配，
- 仅修复该不匹配之前的模块或边界，
- 不要修补整个计算图，
- 没有证据时不得编辑上游已冻结的模块。

### 生产架构目标

目标是：
- 一个生产级 `@pypto.frontend.jit` kernel，
- 内部循环以 `pypto.loop` 表示，
- 语义模块集成为 kernel 结构。

如果因框架/工具链限制无法实现：
- 记录失败原因，
- 产出最小可靠的暂存回退方案，
- 解释融合失败的确切原因。

---

## Phase 5：PyPTO 特定结构规则

这些规则来自观察到的失败模式。`validate_kernel_structure` 工具可自动检测其中几项（标记 ✦）。

**交叉参考：** `skills/debugging/DEBUG.md §9` 对许多规则提供了具体的错误消息、根因和修复模式。特别是：§9.2（动态形状 / `pypto.loop`）、§9.4（`pypto.view` 维度）、§9.15（TileShape 配置）、§9.19（matmul API、reduction 对齐、assemble 形状）。

### 5.1 Loop 规则

对于目标 kernel 内的算法循环：
- 在 kernel 设计中使用 `pypto.loop`，
- 不要用宿主脚本中的 Python `for` 循环替代。

### 5.2 循环状态规则

对于循环缓冲区/状态：
- 切片语义正确的状态，
- 不要仅仅因为 PyTorch 允许就任意将高秩 Tensor `view` 成方便的秩。

### 5.3 广播规则

如果需要广播：
- 优先使用逐轴广播形状，
- 避免隐藏的双轴隐式扩展，
- 使行/列缩放 Tensor 显式化。

### 5.4 对齐规则

对于向量敏感路径：
- 遵守最后一维的对齐约束，
- 如果窄逻辑 Tensor 有问题，仅在对齐友好的表示能显式映射回语义时才使用。

需要时获取对齐规则：
```
retrieve_docs(query="<op_name> 32-byte alignment last dimension constraint", chunk_type="api_doc")
```

### 5.4b `set_vec_tile_shapes` — 有效 tile 尺寸

**背景：** Vector tile shape 错误很常见；在 kernel 编码和调试期间标准化 `set_vec_tile_shapes`。

**规则：**
1. 按你的 PyPTO 版本的 `docs/api/config/pypto-set_vec_tile_shapes.md` 要求传入正整数 tile 尺寸。
2. 如果使用非标准 tile 尺寸，在计划中记录使用的尺寸和理由。

### 5.5 Pad/concat/layout 规则

不要假设 PyTorch 风格的 padding 语义可直接映射。如果 layout/pad 操作不稳定：
- 优先使用显式的 reshape/concat/对齐形式，
- 保持 golden 和 kernel 映射清晰。

### 5.6 SIM vs NPU 规则

不得仅凭 SIM 声称数值正确性。

SIM 用于：编译、图结构、早期验证。
NPU 用于：真实 Tensor 对比、最终正确性判定。

### 5.7 写回规则 ✦

始终使用 `output[:] = expr` 或 `pypto.assemble(expr, offset, output)`。不要使用 `output = expr`。

`validate_kernel_structure` 会自动检测此项 — 每个模块编写后都运行它。

### 5.8 Tile 配置规则 ✦

- Vector 操作需要 `pypto.set_vec_tile_shapes(...)` 在第一个 vector 操作之前 — 使用文档要求的 tile 尺寸（参见 5.4b）。
- `pypto.matmul` 需要在调用前配置 `pypto.set_cube_tile_shapes(...)`。
- Loop 的 `idx_name` 值在函数内必须唯一。

`validate_kernel_structure` 会自动检测全部三项。

---

## Sub-skill 委派：集成期间的调试

当 Phase 4 中的集成测试失败或 Phase 5 中的结构检查失败，且标准调试工作流（`skills/debugging/SKILL.md` + `DEBUG.md §9`）无法解决问题时，升级到以下 sub-skill：

- **精度规避方案：** 阅读 `skills/pypto-precision-debug/SKILL.md`
- **精度二分法：** 阅读 `skills/pypto-precision-compare/SKILL.md`
- **内存重叠：** 阅读 `skills/pypto-memory-overlap-detector/SKILL.md`
- **AICore 错误：** 阅读 `skills/pypto-aicore-error-locator/SKILL.md`
- **宿主崩溃：** 阅读 `skills/pypto-host-stacktrace-analyzer/SKILL.md`
- **MACHINE workspace：** 阅读 `skills/pypto-machine-workspace/SKILL.md`

完整优先级顺序参见 `skills/phase2-phase3-construction/SKILL.md` → "Sub-skill 委派：调试升级"。
