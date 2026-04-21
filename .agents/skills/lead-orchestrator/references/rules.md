# PyPTO kernel — 强制规则

> **导航：** 你被 `agent-plan.md` 引导至此。阅读完本文件后，返回 `agent-plan.md` 并按阶段逐步执行——它会告诉你每一步需要阅读哪个 skill。

## 零容忍 — 不得跳过，不得走捷径（请先阅读全文）

**这些不是"尽力而为"。** 为节省时间、token 或上下文而违反这些规则是**被禁止的**。在缺少所需工件和命令的情况下声称某个步骤已完成是**不合规的**。

| 禁止行为 | 应当执行的操作 |
|-----------|----------------|
| **跳过**你判断为"可选"的本文档集章节 | 对当前任务端到端遵循 **`rules.md`**、调试时遵循 **`skills/debugging/DEBUG.md`**、遵循 **`skills/plan-template/plan.template.md`** 的字段要求，以及分阶段文件/验证/逐模块日志规则。 |
| **跳过**分阶段链（`_module1.py` → `_module12.py` → …） | 在进入下一阶段**之前**，创建并通过每个分阶段文件；不得直接跳到"仅完整 kernel"。 |
| **省略** `detailed_tensor_compare` 或仅比较**一个**输出 | 使用随附的辅助工具；在每个阶段和 **`test_<op>.py`** 中比较**每个**叶子输出。 |
| **跳过**运行后的计划更新（`custom/plan/<op>.md`） | 按照下方 **计划文件（每轮）** 的要求**每轮**更新。 |
| **用**口头"应该通过"/"对齐"**替代**真实运行 | 运行命令；将证据粘贴到计划或日志中。 |
| 修改 `custom/` 后**跳过****布局检查** | 从仓库根目录运行 **`bash .agents/skills/ci-and-layout-check/run_validate_layout.sh`**（或参见 **`skills/ci-and-layout-check/CI.md`**）；在声称布局完成前修复 **exit 1**。 |
| `set_vec_tile_shapes` **传入无效维度** | 传入正数的 tile 大小（必要时使用 **`1`**）；维度要求参见对应版本的 **`docs/api/config/pypto-set_vec_tile_shapes.md`**。 |
| 以**节省 token**为由不阅读适用的文档/skill | 阅读 **`skills/`** 下的相关 skill。token 消耗**不是**省略步骤的理由。 |
| **临时性的 kernel 文件布局**（无 A–L 层级、随意命名函数） | 使用 **`skills/kernel-code-format/pypto_kernel_template.py`** 作为**每个**分阶段文件**和**完整 kernel 的**强制骨架** —— 参见 **`skills/kernel-code-format/pypto-kernel-design-format.md`**。在 **`custom/plan/<op>.md`** 中记录任何有意的偏差。 |
| `pypto_function` 内使用 **`for ... in range(...)`**（宿主 Python 循环遍历 tile/batch/seq） | 在 **`_your_op_kernel_impl`** / **`your_op_kernel_npu`** 中使用 **`pypto.loop`**（+ `pypto.view`）表达算法迭代。`pypto_function` 仅用于 I/O 打包/解包 —— 参见下方**禁止项 B**。CI：**`skills/ci-and-layout-check/scripts/validate_custom_kernel_layout.py`** 会检测此模式。 |

如果无法完成某个步骤，**在计划中记录阻塞原因** —— 不得静默跳过。

## 不可协商

1. **PyPTO 中一次一个模块** —— 一次只能有一个语义模块的真实 `pypto` 逻辑处于未冻结状态；后续阶段必须**存根化**或使用 **golden 边界张量**，直到当前模块通过验证。
2. 在逐模块边界检查通过之前，**不得**一次性完成完整融合的 `@jit`。
3. **宿主 Python `for` 不是 kernel tile 循环** —— 算法 tiling 应在 `pypto.loop` + `view` + `assemble`（需要时）中实现。
4. **单一生产 `@jit`** —— 除非文档记录了分阶段回退方案，否则每个模块不应有独立的 JIT。
5. Golden 在 PyPTO 实现之前**冻结**；**不得**在没有证据和计划日志的情况下修改它。
6. kernel 代码中张量行需要**形状注释**（参见 `skills/kernel-code-format/pypto-kernel-design-format.md` 中的**形状注释规范**）。
7. **遇到 PyPTO 错误卡住时** —— 阅读 **`skills/debugging/DEBUG.md`**，然后运行 `skills/ci-and-layout-check/scripts/extract_pypto_calls.py`，然后按 **`skills/debugging/SKILL.md`** 中的**逐算子协议**操作。
7b. **在编写 PyPTO 代码之前** —— 查阅 **`skills/debugging/DEBUG.md` §9** 中与你要编写的内容匹配的小节（JIT 签名 §9.1、`pypto.view` §9.4、`matmul` §9.19、reduction §9.19、动态 Shape §9.2、张量类型提示 §9.13、JIT 内的 Python 操作 §9.14、tile 配置 §9.15）。完整查找表见 **`skills/phase2-phase3-construction/SKILL.md`** → **Phase 3 → 编写 PyPTO 代码之前**。跳过此步骤是不合规的。
8. **端到端验证运行器** —— **`custom/<operator_name>/test_<operator_name>.py`**（不是以 `pytest` 作为默认驱动）。从仓库根目录：**`PYTHONPATH=.agents/skills/validation-and-deliverables python custom/<operator_name>/test_<operator_name>.py`**（参见 **`skills/validation-and-deliverables/SKILL.md`**）。
9. **Golden 与 PyPTO 对比** —— 使用 **`.agents/skills/validation-and-deliverables/detailed_tensor_compare.py`** 中的 **`detailed_tensor_compare`**（`from detailed_tensor_compare import detailed_tensor_compare`）；不得用不同实现替代主要报告。
10. **每个输出** —— **`test_<operator_name>.py`** 必须比较**所有** kernel 输出（tuple/list/dict/nested → 每个叶子张量）。**禁止：** kernel 返回多个输出时仅验证其中一个。例外情况仅在 **`custom/plan/<operator_name>.md`** → **blockers** 中注明并附理由。
11. **计划中的模块分解** —— **`custom/plan/<operator_name>.md`** 必须记录语义模块**如何**拆分以及**为什么**这样拆分（理由：边界、可检查性、顺序 —— 而非"平衡复杂度"）。参见 **`skills/plan-template/plan.template.md`** → **Module decomposition**。
12. **逐模块验证日志** —— 对于**每个**模块边界检查（golden vs PyPTO），使用 **`detailed_tensor_compare`** 结果（`all_close` 和返回字典中的关键字段）在计划的**逐模块验证日志**中追加一行。端到端和逐模块检查使用**同一个**随附辅助工具。
13. **不要仅因晦涩错误就停止** —— `FFFFF`、`UNKNOWN`、`0x3FFFF` 或其他不透明的 **`Errcode: F…!`** 行**不是**放弃任务的理由。遵循 **`skills/debugging/DEBUG.md`**，收集日志，应用 **`skills/ci-and-layout-check/scripts/extract_pypto_calls.py`** + 逐算子协议，然后迭代。token/轮次消耗不是限制因素。仅当出现真正的阻塞原因时才停止（参见下方**停止条件**）。
14. **分阶段模块 Python 文件** —— 在 **`custom/<operator_name>/`** 下，创建 **`<operator_name>_module1.py`**，然后 **`<operator_name>_module12.py`**、**`…_module123.py`**、…、**`…_module1…N.py`**（后缀 = 数字 **1**、**12**、**123**、… = 累计 M1..Mk）。每个文件：**golden + 一个 `@jit`** 用于该范围；**所有**输出的 **`detailed_tensor_compare`** 必须**在**下一个分阶段文件存在**之前**通过。最终的 **`…_module1…N.py`** = 完整的端到端 kernel。
15. **自动化布局检查** —— 在 **`custom/`** 下进行有意义的编辑后，从**仓库根目录**运行 **`bash .agents/skills/ci-and-layout-check/run_validate_layout.sh`**（参见 **`skills/ci-and-layout-check/CI.md`**）。当此命令返回 **1** 时，**不得**声称完成。逻辑与 CI/pre-commit 一致。
16. **`set_vec_tile_shapes` — 有效的 tile 维度** —— 编码或调试时，按照你的 PyPTO 版本对应的 **`docs/api/config/pypto-set_vec_tile_shapes.md`** 要求传入正数 tile 参数。参见 **`skills/phase4-phase5-integration/SKILL.md`** → **5.4b**。
17. **`skills/kernel-code-format/pypto_kernel_template.py` — 强制代码骨架** —— 使用 **`skills/kernel-code-format/pypto_kernel_template.py`** 中的 **A–L** 层级结构来组织**每个**交付物（`<op>_module1.py` … `<op>_module1…N.py` 和集成 kernel）。**不得**因为调试困难而丢弃模板。参见 **`skills/kernel-code-format/pypto-kernel-design-format.md`**。
18. **`pypto_function` 内禁止 `for ... in range(...)`** —— 宿主包装器 **`pypto_function`** **不得**使用 Python `for` + `range` 实现 kernel tile/batch/sequence 循环。将这些循环放在 **`_your_op_kernel_impl`** / JIT 入口中使用 **`pypto.loop`**。**`validate_custom_kernel_layout.py`** 会拒绝 **`custom/<op>/`** 下的此模式。
19. **Golden 中禁止 `.T` / `.t()`** —— PyPTO 友好的 golden 必须使用 `torch.transpose(t, dim0, dim1)` 代替 `.T`/`.t()`。PyPTO 张量不支持 `.T`（`skills/debugging/DEBUG.md §9.19`）。对于 `matmul a @ b.T`，应写为 `torch.matmul(a, b.transpose(-2, -1))` 并注释转置意图。
20. **Golden 函数清单** —— 编写 PyPTO 友好的 golden 后，在 **`custom/plan/<op>.md` → Golden function inventory** 中列出每个数学操作（每行一个）。在 Phase 3/4 中，逐行与 PyPTO 实现交叉检查：用 ✅ 标记并附 pypto 调用 + 行号，或用 ❌ 标记缺失项。**当范围内存在任何 ❌ 时，不得运行测试或推进模块。** 精度错误最常由从未实现的操作引起。

---

## 逐模块强制执行

**问题：** 一个将所有语义阶段组合在一起的大型 `@jit` 会触发复合失败（tiling、`view`/`assemble`、写回、dtype、图限制）。如果 agent 一次实现所有模块，错误将无法定位。

| 规则 | 要求 |
| --- | --- |
| **一个活跃模块** | 一次最多只能有一个语义模块的 PyPTO 逻辑是新的或未冻结的。后续阶段必须被存根化（恒等、零或来自 golden 的透传张量）。 |
| **先通过边界再进入下一步** | 在当前模块的输出与 golden 匹配之前，不要在 `kernel_impl` 中添加下一个模块的真实操作。 |
| **计划文件** | 在 `custom/plan/<operator_name>.md` 中保持 `active_module: Mk` 和 `modules_pypto_verified: [M1, …]`。仅在记录 `detailed_tensor_compare` 证据后才更改 `active_module`。 |
| **用户提示的默认含义** | "实现 kernel" = 仅实现下一个未验证的模块，除非用户明确要求完整集成。 |
| **存根必须显式** | 每个存根都加注释：`# STUB: until M2 verified; golden-fed tensor`。 |

**合规模式：** 仅实现 M1 → 验证 → 冻结 → 设置 `active_module: M2` → 重复。

---

## 三大架构禁止项

### 禁止项 A：不得一次性实现

不要从参考代码直接跳到一个集成的 PyPTO kernel。先标准化 golden → 按语义边界拆分 → 验证每个边界 → 渐进集成。

### 禁止项 B：不得使用 Python 宿主循环实现 kernel tiling 逻辑

不要使用 Python `for` 循环模拟 kernel 的算法化 tile 执行。算法循环必须使用 `pypto.loop` 或显式的语义分阶段。允许的宿主循环：遍历测试用例、候选配置、用于记录的模块，或宿主侧验证输入。

### 禁止项 C：不得采用每模块一个 JIT 的生产架构

模块是语义块，不是独立的生产 JIT 入口。组装为一个生产 `@pypto.frontend.jit` kernel。分阶段多 kernel 回退方案仅在被框架限制阻塞融合时允许 —— 必须明确标记为回退方案。

---

## 停止条件

仅在以下情况之一为真时暂停：
- 参考代码缺失，
- 标准化后的 golden 无法与原始版本等价，
- 框架从根本上阻止了所需的集成形式，
- 缺少必要的用户运行时日志，
- 继续推进将是盲目猜测。

**以下不是**有效的暂停理由：单个晦涩错误码、害怕使用更多 token、或不愿意尝试另一个文档记录的策略 —— 使用 `skills/debugging/DEBUG.md` 并继续迭代。

---

## 计划文件（每轮）

更新 `custom/plan/<operator_name>.md`：

- `active_module`、`modules_pypto_verified`、**`current_staged_file`**（例如 `custom/<op>/<op>_module12.py`）
- **模块分解**（概览 + 理由），当拆分方案已知或变更时
- **Golden 函数清单** —— 每个模块实现后更新 ✅/❌ 状态（规则 20）
- **分阶段模块文件**表 —— 完成阶段时打勾/填写文件名
- **逐模块验证日志**，每次边界运行后（附 **`detailed_tensor_compare`** 证据）
- `next_mandatory_step`
- **开发与调试日志**条目，每次运行或编辑后
- **`custom/`** 变更后：运行 **`skills/ci-and-layout-check/run_validate_layout.sh`**（或修复直到 exit 0）

---

## Skill 库规则

21. **Skill 优先级** —— 当类别 skill 的指令与这些规则冲突时，以本 rules.md 为准。特别是：逐模块强制执行、分阶段文件链、`detailed_tensor_compare` 对所有输出强制使用、Golden 函数清单交叉检查、以及 A-L 层级模板结构。
22. **Skill 读取时机** —— 仅当 `agent-plan.md` 或 Phase SKILL.md 明确指示时才读取某个 skill 的 SKILL.md。使用 `catalog.yaml` → `_category.yaml` 定位 skill。不要预读所有 skill。token 消耗不是跳过被指示读取 skill 的理由。
23. **Skill 文件只读** —— 不要编辑 `skills/` 下的文件。如果某个 skill 需要适配，在调用的 Phase SKILL.md 或 `custom/plan/<operator_name>.md` 中添加覆盖。
