# 计划：`<operator_name>`

拷贝到：`custom/plan/<operator_name>.md`，每轮保持**机器可读**字段为最新状态。

**Agent：** **不得**跳过 **`skills/lead-orchestrator/references/rules.md`** 或 sub-skill 义务（暂存文件、全输出对比、计划日志）。参见 **`.agents/skills/lead-orchestrator/references/rules.md`** — *零容忍*。**代码布局：**每个暂存文件和完整 kernel **必须**遵循 **`.agents/skills/kernel-code-format/pypto_kernel_template.py`**（A–L 层）；在此处记录例外情况。在 **`custom/`** 变更后，运行 **`bash .agents/skills/ci-and-layout-check/run_validate_layout.sh`**（参见 **`skills/ci-and-layout-check/CI.md`**）。

## Agent 状态（精简版）

```yaml
phase: 0|1|2|3|4|5|6
active_module: M1   # 或 M2, …, none
current_staged_file: custom/<operator_name>/<operator_name>_module1.py   # 每个里程碑更新
modules_pypto_verified:
  - id: M1
    evidence: "<命令或模块验证日志行的指针>"
    detailed_tensor_compare_ok: true   # 在边界通过之前为 false
next_mandatory_step: "<一个具体步骤>"
correctness: not_started | golden_ok | sim_ok | npu_ok
optimization: not_started | … | complete | skipped_user_request
blockers: []
```

## 任务摘要

- **算子：**
- **I/O 形状（符号化）：**
- **参考 / golden 路径：**

## 验证（必填）

- **运行器：** `custom/<operator_name>/test_<operator_name>.py`
- **命令（仓库根目录）：** `PYTHONPATH=.agents/skills/validation-and-deliverables python custom/<operator_name>/test_<operator_name>.py`
- **对比工具：** `from detailed_tensor_compare import detailed_tensor_compare`（内置：`.agents/skills/validation-and-deliverables/detailed_tensor_compare.py`）。**不要**将 `pytest` 作为 golden 与 PyPTO 对比的默认工具，除非在 **blockers** 中记录为例外情况。
- **全部输出：**运行器必须对**每一个**叶子输出 Tensor（tuple/list/dict/nested 结构 — **不仅**是 `outputs[0]`）调用 **`detailed_tensor_compare`**。如果任何输出被有意跳过，须在 **blockers** 中记录并说明理由。

## 模块分解（必填）

记录 kernel **如何**按语义拆分为模块以及**为什么**（而非"等大小的分块"）。保持与下方 **模块契约** 对齐。

### 模块（概览）

| ID | 一句话职责 | 边界 Tensor（golden 检查点名称） | 依赖 |
|----|-----------|----------------------------------|------|
| M1 | … | … | — |
| M2 | … | … | M1 |

### 理由

- **语义边界：**（例如 matmul vs norm vs recurrence — 每个块承载的含义）
- **为什么是这个顺序：**（依赖关系 / 可调试性）
- **考虑过但拒绝的替代方案：**（可选；例如"单融合块方案在边界稳定之前暂不采纳"）

## 暂存模块文件（必填）

开发进度以 **Python 文件**形式**物化**于 **`custom/<operator_name>/`** 下。`_module` 后缀为**拼接的模块索引**（`1` → 仅 M1，`12` → M1+M2 在一个 JIT 中，`1234` → M1–M4 对应 N=4）。每个文件在**一个** `@jit` 中包含该累积范围的 **golden + PyPTO**，且必须在创建下一个文件之前通过**所有**输出的 **`detailed_tensor_compare`**。**最后一行**的文件即为**完整**端到端 PyPTO kernel。

| 暂存文件 | 一个 `@jit` 中的模块 | Golden + PyPTO 已验证（全部输出） |
|----------|----------------------|----------------------------------|
| `<operator_name>_module1.py` | M1 | ☐ |
| `<operator_name>_module12.py` | M1, M2 | ☐ |
| `<operator_name>_module123.py` | M1, M2, M3 | ☐ |
| `<operator_name>_module1234.py` | M1–M4（示例 N=4） | ☐ — **完整 kernel** |

*根据 **N** 个模块增减行数。某阶段的命令示例：*
`PYTHONPATH=.agents/skills/validation-and-deliverables python custom/<operator_name>/<operator_name>_module12.py`

## 模块验证日志（必填）

每次**模块边界**验证通过时（Phase 3：该检查点的 golden vs PyPTO），**追加**一行。对比**必须**使用 **`detailed_tensor_compare`**（与端到端相同的辅助函数）。记录足够信息使审阅者无需重新运行即可看到 **pass/fail**。

| 时间 | 模块 | 暂存文件（如 `<op>_module12.py`） | 对比的 Tensor（名称/角色） | rtol | atol | `all_close` | 关键统计（如返回字典中的 `out_of_tolerance_ratio`、`max_diff`） | 命令或脚本 |
|------|------|-------------------------------------|---------------------------|------|------|-------------|----------------------------------------------------------------|-----------|
| | M1 | `<operator_name>_module1.py` | | 1e-3 | 1e-3 | | | |

*失败时：*添加 **开发与调试日志** 条目，保留失败的行或在修复后添加后续行。

## Vector tile 配置

- **`pypto.set_vec_tile_shapes`：**按 **`docs/api/config/pypto-set_vec_tile_shapes.md`** 要求使用 tile 尺寸 — 参见 **`skills/phase4-phase5-integration/SKILL.md`** **5.4b**。

## API 映射（Phase 0）

| 公式步骤 | PyPTO | 支持 / 替代 / 不支持 |

## Golden 函数清单（Phase 1 — 交叉检查）

列出 PyPTO 友好 golden 中的**每一项操作**。在 Phase 3/4 中逐行与 PyPTO 实现交叉检查并标记 ✅/❌。**任何 ❌ 存在时不得运行测试或推进模块。**

| # | Golden 操作 | 形状变换 | PyPTO 实现 | 行号 | 状态 |
|---|------------|---------|-----------|------|------|
| 1 | `torch.matmul(q, k^T)` | `[B,H,T,K]@[B,H,K,T]->[B,H,T,T]` | `pypto.matmul(q, k, dtype, b_trans=True)` | L.42 | ✅ |
| 2 | `torch.softmax(scores, -1)` | `[B,H,T,T]->[B,H,T,T]` | | | ❌ |
| … | | | | | |

**何时交叉检查：**
- Phase 3（GATE 3）：当前模块范围内的每一行必须为 ✅
- Phase 4（GATE 4）：每一行必须为 ✅

## 模块契约（Phase 2）

| 模块 | 输入 | 输出 | PyPTO API | 已验证？ |

## 设计格式合规性

- **模板：** `.agents/skills/kernel-code-format/pypto_kernel_template.py` — 每个 `*_module*.py` 和集成 kernel 的**必填**骨架（相同层级；仅在有充分理由时裁剪未使用的章节）。
- 来自 `docs/pypto-kernel-design-format.md` 的 A–L 层（本 bundle 中的副本：`.agents/skills/kernel-code-format/pypto-kernel-design-format.md`）：哪些适用、哪些省略、原因。

## PyPTO 调用点检查清单（调试时使用）

粘贴以下命令的输出：

`python3 .agents/skills/ci-and-layout-check/scripts/extract_pypto_calls.py custom/<operator_name>/<operator_name>_module1234.py` *（或当前暂存/最终 kernel 文件）*

| # | 行号 | 调用 | 文档合规？ | 备注 |

## `skills/debugging/DEBUG.md` §9 — 编写前检查清单

在编写每个模块的 PyPTO 代码之前，查阅 **`.agents/skills/debugging/DEBUG.md` §9** 中的对应小节。阅读后勾选。完整查找表参见 **`skills/phase2-phase3-construction/SKILL.md`** → **Phase 3 → 编写 PyPTO 代码之前**。

| 小节 | 适用于此 kernel？ | 已查阅？ |
|------|-------------------|---------|
| §9.1 JIT 签名（`from __future__` 禁令） | ☐ 是 / ☐ 否 | ☐ |
| §9.2 动态形状、符号化循环边界 | ☐ 是 / ☐ 否 | ☐ |
| §9.4 `pypto.view` / `pypto.assemble` 指南 | ☐ 是 / ☐ 否 | ☐ |
| §9.13 Tensor 形状规格（`[]` vs `DYNAMIC`） | ☐ 是 / ☐ 否 | ☐ |
| §9.14 JIT 内的 Python 运算符 | ☐ 是 / ☐ 否 | ☐ |
| §9.15 TileShape 配置 | ☐ 是 / ☐ 否 | ☐ |
| §9.19 matmul API / reduction / assemble | ☐ 是 / ☐ 否 | ☐ |
| §9.11 常见错误速查表 | ☐ 是 / ☐ 否 | ☐ |

## 不透明错误码（FFFFF, UNKNOWN, Errcode: F…！）

**不要**仅因此类错误就放弃运行。遵循 **`.agents/skills/debugging/DEBUG.md`** §1–§8，捕获完整日志并迭代（token 预算不是限制）。调试时同时参阅 **§9.11**（常见错误 → 原因 → 修复方案速查表）。在下方记录每次尝试。

## 开发与调试日志

| 时间 | 操作 | 结果 | 下一步 |

## 人工审阅里程碑（可选）

MS1 … MS7 如完整工作流中定义 — 或链接到 `skills/` 下相关 sub-skill 中的里程碑。
