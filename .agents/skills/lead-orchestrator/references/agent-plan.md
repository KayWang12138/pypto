# Agent 执行计划 — PyPTO Kernel 开发

**阅读顺序：** `principles.md` → 本文件 → `rules.md` → 然后按阶段逐项执行本检查清单。使用 `catalog.yaml` 定位 skill 分类。

**不要提前阅读所有 skill。** 仅在本检查清单指示时才阅读对应的 skill — 不要提前阅读。

将所有 `<op>` 替换为实际算子名称。

---

## Skill 目录

Skills 按 6 个类别组织在 `skills/` 下。阅读 `catalog.yaml` 获取完整索引。阅读某个类别的 `_category.yaml` 以找到合适的 skill。仅在指示时才阅读 skill。

| 类别 | 路径 | Skills | 使用时机 |
|------|------|--------|----------|
| **workflow** | `skills/` | Phase 0-6 skills、模板、验证 | 每个阶段 |
| **development** | `skills/` | 意图理解、API、golden、设计、实现、环境 | Phase 0-3、后续 |
| **debugging** | `skills/` | 精度、aicore、崩溃、内存 | Phase 3+（出错时） |
| **performance** | `skills/` | 3 阶段调优、泳道图、自动调优 | Phase 6 |
| **ci-and-pr** | `skills/` | 布局检查、PR、Issue、评审 | Phase 3+、后续 |
| **pass** | `skills/` | Pass 分析、错误、UT、性能 | 参考 |

---

## 禁止事项（违反需从失败的 Gate 重新开始）

1. **测试未通过不得推进。** 在当前模块的 `detailed_tensor_compare` 对所有输出返回 `all_close: true` 之前，不得编写下一个模块或进行完整集成。
2. **没有 PyPTO 友好的 golden 不得开始 kernel 实现。** 在 Phase 1 完成之前不得进入 Phase 3。
3. **golden 中不得遗留 `.T` / `.t()` 或隐式转置形式。** PyPTO 不支持 `.T`（`skills/debugging/DEBUG.md §9.19`）。将 golden 规范化为 `torch.transpose(t, dim0, dim1)` 并记录转置意图，用于 matmul（例如注释 `# a @ b^T` → `pypto.matmul(a, b, dtype, a_trans=False, b_trans=True)`）。
4. **不得在未交叉检查 golden 操作列表与 PyPTO kernel 的情况下声称完成。** 精度错误通常是遗漏了操作。
5. **不得一次性实现所有模块。** 每次只实现一个模块。

---

## 检查清单

在 **⛔ GATE** 通过之前，不得继续推进。在 `custom/plan/<op>.md` 中记录证据。

---

### Phase 0：准备

**现在阅读：** `skills/phase0-phase1-planning/SKILL.md`、`skills/plan-template/SKILL.md`、`skills/kernel-code-format/SKILL.md`

- [ ] **0.1** 通过复制 `skills/plan-template/plan.template.md` 创建 `custom/plan/<op>.md`
- [ ] **0.2** 列出目标公式中的所有操作（add、matmul、softmax、transpose 等）
- [ ] **0.3** 确认每个操作在 PyPTO 中存在 → 在计划中记录 **API 映射**（`skills/phase0-phase1-planning/SKILL.md` Phase 0）
  ```
  query_op(names=["matmul", "softmax", "exp", ...])
  ```
- [ ] **0.4** 搜索类似的 kernel 示例 → 在计划中记录
  ```
  retrieve_docs(query="<kernel 类型> example", chunk_type="example")
  ```

⛔ **GATE 0：** 计划中存在 API 映射；零个 `unsupported` 行（或每个都有文档化的替代方案）。

---

### Phase 1：构建 PyPTO 友好的 golden

**现在阅读（如尚未阅读）：** `skills/phase0-phase1-planning/SKILL.md`（Phase 1 部分）、`skills/kernel-code-format/pypto-kernel-design-format.md` §11（shape 注解）

- [ ] **1.1** 确定主要参考实现
- [ ] **1.2** 编写 **PyPTO 友好的 golden** 并应用以下**所有**规则：

| 规则 | 理由 |
|------|------|
| 替换 `.T` / `.t()` → `torch.transpose(t, dim0, dim1)` | PyPTO 不支持 `.T`（`skills/debugging/DEBUG.md §9.19`） |
| `.sum(dim)` 可以保留；注意 PyPTO 中的 32 字节对齐问题 | `skills/debugging/DEBUG.md §9.19` |
| 不允许隐式广播 → 显式 `reshape` 后操作 | `skills/phase4-phase5-integration/SKILL.md` Phase 5.3 |
| 为每个中间结果命名（`scores`、`weights`、…） | 便于调试 |
| 每个中间结果添加 shape 注释 `# [B, H, T, K]` | `rules.md` 规则 6；`skills/kernel-code-format/pypto-kernel-design-format.md` §11 |
| 用 `# --- Module M1: ... ---` 标记语义边界 | 为 Phase 2 拆分做准备 |

- [ ] **1.3** 构建 **golden 操作清单**：

```
列出 golden 中的每个操作（每行一个操作）：
  1. torch.matmul(q, k^T)     # scores  [B,H,T,T]
  2. torch.softmax(scores, -1) # weights [B,H,T,T]
  3. torch.matmul(weights, v)  # out     [B,H,T,V]
  ...
```

→ 在 `custom/plan/<op>.md` 的 **Golden function inventory** 部分记录此列表。

- [ ] **1.4** 比较规范化 golden 与原始 golden
  ```python
  assert torch.allclose(original_out, normalized_out, rtol=1e-5, atol=1e-5)
  ```
- [ ] **1.5** 冻结 golden → 在计划中设置 `correctness: golden_ok`

⛔ **GATE 1：** (a) golden 中零个 `.T`/`.t()` (b) 所有中间结果有 shape 注释 (c) 计划中存在 Golden function inventory (d) 原始与规范化 golden 对比通过。**所有条件满足之前不得进入 Phase 2。**

---

### Phase 2：模块分解

**现在阅读：** `skills/phase2-phase3-construction/SKILL.md`（Phase 2 部分）、`skills/debugging/DEBUG.md §9`（编写前检查清单 — 查阅与你的 kernel 操作匹配的 §9 子节）

- [ ] **2.1** 沿 Phase 1 中的 `# --- Module M1 ---` 边界定义模块
  - 按**语义**边界拆分（matmul / norm / 递归，…）。禁止等大小拆分。
  - 定义每个模块的输入/输出 tensor 名称、shape、dtype
  - → 在计划中记录 **模块分解** + **模块契约**（`skills/phase2-phase3-construction/SKILL.md` Phase 2）

- [ ] **2.2** 确定分阶段文件序列 → 在计划中填写 **分阶段模块文件** 表
  ```
  <op>_module1.py     → 仅 M1
  <op>_module12.py    → M1 + M2
  ...
  <op>_module1…N.py   → 所有模块 = 完整 kernel
  ```

- [ ] **2.3** 在计划中完成 `skills/debugging/DEBUG.md §9` 编写前检查清单（`skills/plan-template/plan.template.md` 对应部分）

⛔ **GATE 2：** 计划中存在模块分解 + 契约 + 分阶段文件表。

---

### Phase 3：实现模块（对每个 M_k 重复）

**现在阅读：** `skills/phase2-phase3-construction/SKILL.md`（Phase 3 部分 + DEBUG §9 查找表）、`skills/validation-and-deliverables/SKILL.md`

**每次只实现一个模块。当前模块通过之前不得推进。**

- [ ] **3.1** 阅读相关的 `skills/debugging/DEBUG.md §9` 子节（`skills/phase2-phase3-construction/SKILL.md` Phase 3 表格）
- [ ] **3.2** 从 `skills/kernel-code-format/pypto_kernel_template.py` 创建 `custom/<op>/<op>_module<suffix>.py`
  - 此累计范围的 golden
  - PyPTO：**一个** `@pypto.frontend.jit`；后续模块使用 golden 提供的 tensor 作为桩
  - Runner：`if __name__ == "__main__":` + `detailed_tensor_compare`
- [ ] **3.3** 运行 AST 检查
  ```
  validate_kernel_structure(source_code=<完整源代码>)
  ```
  → 修复直到零错误
- [ ] **3.4** **交叉检查 Golden function inventory 与 PyPTO**

  ```
  Golden function inventory（来自 Phase 1.3）：
    1. matmul(q, k^T)        → ✅ pypto.matmul(q, k, dtype, b_trans=True)  L.42
    2. softmax(scores, -1)    → ✅ pypto.softmax(scores, dim=-1)           L.45
    3. matmul(weights, v)     → ❌ 未实现 — 跳过会导致精度错误
  ```

  **在 M_k 范围内的每个操作都有 ✅ 之前，不要运行测试。**

- [ ] **3.5** 运行测试
  ```bash
  PYTHONPATH=.agents/skills/validation-and-deliverables python custom/<op>/<op>_module<suffix>.py
  ```
- [ ] **3.6** 布局检查
  ```bash
  bash .agents/skills/ci-and-layout-check/run_validate_layout.sh
  ```

⛔ **GATE 3 (M_k)：** (a) `detailed_tensor_compare` 对**所有**输出返回 `all_close: true` (b) Golden inventory：M_k 范围内所有操作 ✅ (c) 布局检查 exit 0 (d) 每模块验证日志已更新。**所有条件满足之前不得开始 M_{k+1}。**

- [ ] **3.7** 冻结 M_k → 更新计划：追加到 `modules_pypto_verified`，设置 `active_module: M_{k+1}`

**GATE 3 之后：** 返回 3.1 处理下一个模块，或在所有模块完成后进入 Phase 4。

---

### Phase 4：集成与端到端验证

**现在阅读：** `skills/phase4-phase5-integration/SKILL.md`（Phase 4 部分）、`skills/validation-and-deliverables/SKILL.md`

- [ ] **4.1** 确认最后一个分阶段文件（`<op>_module1…N.py`）集成了所有模块
- [ ] **4.2** 创建 `test_<op>.py`（`skills/validation-and-deliverables/SKILL.md` — 验证运行器）
  - 使用 `detailed_tensor_compare` 比较**每个**输出 tensor（禁止仅比较单一输出）
- [ ] **4.3** **最终 Golden function inventory 交叉检查**

  逐行对比：golden 操作列表与最终 PyPTO kernel。  
  **如果仍有任何 ❌，不要运行端到端测试 — 先实现。**

- [ ] **4.4** 运行端到端测试
  ```bash
  PYTHONPATH=.agents/skills/validation-and-deliverables python custom/<op>/test_<op>.py
  ```
- [ ] **4.5** 布局检查
  ```bash
  bash .agents/skills/ci-and-layout-check/run_validate_layout.sh
  ```

⛔ **GATE 4：** (a) 端到端测试所有输出 `all_close: true` (b) Golden inventory 100% ✅ (c) 布局检查 exit 0。

---

### Phase 5：最终结构规则

**现在阅读：** `skills/phase4-phase5-integration/SKILL.md`（Phase 5 部分）、`skills/debugging/DEBUG.md §9`

- [ ] `validate_kernel_structure` → 零错误
- [ ] `pypto_function` 内部无 `for ... in range(...)`（布局检查）
- [ ] `set_vec_tile_shapes` 配置了符合文档的有效 tile 维度
- [ ] matmul 前设置了 `set_cube_tile_shapes`
- [ ] 通过 `output[:] = expr` 或 `pypto.assemble(...)` 写回

（详情：`skills/phase4-phase5-integration/SKILL.md` Phase 5 + `skills/debugging/DEBUG.md §9`）

---

### Phase 6：优化（仅在正确性确认后）

**现在阅读：** `skills/phase6-optimization/SKILL.md`

按照该 skill 执行。如果正确性回退，立即回滚。

---

## 精度错误出现时的处理顺序（必须按序执行）

**现在阅读：** `skills/debugging/SKILL.md`（逐操作协议）、`skills/debugging/DEBUG.md §9.11`（错误表）

1. 打开 **Golden function inventory**
2. 逐行对比 PyPTO kernel → 找出遗漏或错误的操作
3. 对 kernel 运行 `extract_pypto_calls.py` 并与 golden 对照
4. 如果仍不清楚 → `skills/debugging/DEBUG.md` §9.11 错误表 → 逐操作协议（`skills/debugging/SKILL.md`）

---

## 计划必要章节（`skills/plan-template/plan.template.md`）

| 章节 | 更新时机 |
|------|----------|
| API 映射 | Phase 0 |
| **Golden function inventory** | Phase 1（创建）；Phase 3/4（追加交叉检查结果） |
| 模块分解 + 理由 | Phase 2 |
| 模块契约 | Phase 2 |
| 分阶段模块文件表 | Phase 2（创建）→ Phase 3（每个里程碑检查） |
| 每模块验证日志 | Phase 3 中每次 GATE 3 通过时 |
| `skills/debugging/DEBUG.md §9` 编写前检查清单 | Phase 3 之前 |
| 开发与调试日志 | 每个错误和修复 |


---

## 开发后（正确性确认后）

Phase 6 完成（或跳过）后，按需执行以下子技能：

- **模型集成**：阅读 `skills/pypto-fused-op-integration/SKILL.md` 将融合算子集成到模型中，替代小算子组合。
- **PR 创建**：阅读 `skills/pypto-pr-creator/SKILL.md` 创建符合规范的 PR 到 cann/pypto 仓库。
- **PR CI 修复**：阅读 `skills/pypto-pr-fixer/SKILL.md` 修复 CodeCheck CI 失败或处理已有 PR 的评审意见。
- **Issue 创建**：阅读 `skills/pypto-issue-creator/SKILL.md` 为开发过程中发现的 bug、功能或任务创建 GitCode Issue。
- **断裂点检测**：阅读 `skills/pypto-fracture-point-detector/SKILL.md` 识别本次 session 中遇到的框架或文档缺口。
