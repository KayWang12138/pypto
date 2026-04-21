# Agent 团队 —— 活跃 Skill 分配

本文件定义了在本 skill 库之上执行 PyPTO kernel 开发的多 Agent 团队。它强制执行两个硬性约束：

1. **每个 Agent 同时加载 2–5 个活跃 skill。** 超过 5 个会导致"35-skill cliff"效应，路由准确率急剧下降。
2. **休眠 skill 仅通过路由 skill 按需加载。** 路由 skill 知道针对特定失败或阶段应拉取**哪个**子 skill，且从不急切地加载全部。

> **新 Agent 阅读顺序：** `principles.md` → 本文件 → `agent-plan.md` → `rules.md`。然后遵循 Phase 检查清单，仅在 `agent-plan.md` 指示时读取 skill。

---

## 在本库中什么算作"skill"

两种类型的文档被视为 skill 并计入 2–5 的活跃预算：

- **根控制文档** —— `.agents/` 下除 `skills/` 和 `README.md` 之外的每个文件。这些是 Lead Agent 的操作手册。
- **`skills/<category>/<skill-id>/` 下的 skill** —— 在 `catalog.yaml` 中编目的 39 个程序/领域 skill。

根控制文档**仅对 Lead Agent 活跃**。其他 Agent 不会将其加载到活跃集中；Lead Agent 将相关约束（规则、Gate、路由决策）作为每次任务分发的组成部分传递下去。

## 团队名册

| # | Agent | 负责阶段 | 活跃 skill 数 | 路由 / 枢纽 skill |
|---|-------|---------|-------------:|-------------------|
| 1 | **Lead**（编排者） | 所有阶段 —— Gate 强制执行 | 5 | `agent-plan.md` + `agents.md` |
| 2 | **Planning** | Phase 0 | 4 | `phase0-phase1-planning` |
| 3 | **Algorithm**（数学家） | Phase 1 | 3 | — |
| 4 | **Architecture** | Phase 1–2 边界 | 3 | — |
| 5 | **Design** | Phase 2 | 4 | `phase2-phase3-construction` |
| 6 | **Coding** | Phase 3–5 | 4 | `phase2-phase3-construction` |
| 7 | **Verification** | Phase 3–5 Gate 判定 + Phase 6 回归 | 2 | —（仅判定，从不加载 `debugging/*`）|
| 8 | **Debug**（专家） | Phase 3–5 失败调查 | 3 | `debugging` |
| 9 | **Optimization** | Phase 6（正确性优先）| 3 | `pypto-op-perf-tune` |

**活跃 skill 占用：** Lead 加载 5 个根控制文档；8 个子 Agent 共在 `skills/` 中的 13 个独立 skill 上占用 26 个槽位。剩余 skill 保持休眠，通过上述路由 skill 按需加载。

**Phase 3 Gate 失败中的关注点分离：** Verification = 判定者（通过或返回 `failure_category`）；Debug = 专家调查者（加载匹配的 `debugging/*` 子 skill，提出补丁方案）；Coding = 唯一编写生产 kernel 代码的 Agent。Lead 驱动循环 `verification → debug → coding → verification`。参见 `agent-plan.md` Phase 3 内循环和本文档 §8。

---

## 1. Lead Agent（编排者）

**职责：** 逐阶段驱动 `agent-plan.md`。强制执行 Gate。分发子 Agent。**不**直接执行领域工作。

**活跃 skill（5 个根控制文档）：**

| # | 文件 | 角色 |
|--:|------|------|
| 1 | `principles.md` | 4 条行为准则（Think、Simplify、Surgical、Goal-Driven）。Lead 在每次子 Agent 分发时强制执行的质量/伦理契约。 |
| 2 | `agent-plan.md` | Phase 0–6 检查清单、Gate、调试协议、完成标准。主编排脚本。 |
| 3 | `agents.md` | 团队名册、每个 Agent 的活跃 skill 分配、路由策略、子 Agent 分发规则。 |
| 4 | `rules.md` | 23 条强制规则、逐模块强制执行、3 条禁令。每个子 Agent 的输出都据此审计。 |
| 5 | `catalog.yaml` | 始终加载的 skill 路由索引（6 个类别）。Lead 用于定位分发目标 skill 的 Tier-1 发现层。 |

这 5 个文件是 Lead 独占的——其他 Agent 作为分发载荷的一部分接收相关约束，而不是自己加载这些文档。

**已分发 skill（Lead 将这些路由给子 Agent，不自己运行）：**
`plan-template`（Planning/Design/Verification）、`workflow/validation-and-deliverables`（Verification）、`pypto-op-workflow`（参考），以及通过以下路由策略分发 `skills/` 中的每个 skill。

**Lead 直接加载的休眠 skill（仅在开发后条件触发时）：**

| 条件 | Lead 加载的 skill |
|------|------------------|
| Phase 0 之前的环境/构建失败 | `pypto-environment-setup` |
| GATE 4 通过，请求创建 PR | `pypto-pr-creator` |
| PR CI 失败或评审评论 | `pypto-pr-fixer` |
| 发现 Bug / 功能 / 文档缺口 | `pypto-issue-creator` |
| 本 session 中的框架或文档缺口 | `pypto-fracture-point-detector` |
| 请求 skill 审计 | `pypto-skill-reviewer` |
| 正确性验证后的模型集成 | `pypto-fused-op-integration` |

开发后 skill 在 `skills/_category.yaml` 中标记为 `scope: post-dev`；只有 `ci-and-layout-check`（标记为 `scope: in-phase`）在阶段内可见，且归 Verification Agent 所有，而非 Lead。

**保持 5 个的交换策略。** 当 Lead 进入开发后模式时，它将 `catalog.yaml` 从活跃集中换出（剩余的分发目标是单个开发后 skill，因此不再需要 Tier-1 路由索引），并加载恰好一个开发后 skill 替代。交换使 Lead 的活跃计数在所有条件下保持 **5**。同一时间只有一个开发后 skill 活跃。

---

## 2. Planning Agent

**职责：** 产出 `SPEC.md`、`API_REPORT.md`，并用 API 映射填充 `custom/plan/<op>.md`。仅负责 Phase 0。

| 角色 | Skill | 输出 |
|------|-------|------|
| 路由 | `phase0-phase1-planning` | Phase 0 部分 |
| 活跃 | `pypto-intent-understand` | `SPEC.md` |
| 活跃 | `pypto-api-explore` | `API_REPORT.md` |
| 活跃 | `plan-template` | 计划骨架 + API 映射部分 |

**退出标准：** `agent-plan.md` 中的 GATE 0 —— API 映射中零个 `unsupported` 行（或每个都有文档化的变通方案）。

---

## 3. Algorithm Agent（数学家）

**职责：** 产出 PyPTO 友好的 `golden.py` 和计划中的 **Golden 函数清单**。仅负责 Phase 1。

| 角色 | Skill | 备注 |
|------|-------|------|
| 活跃 | `pypto-golden-generate` | 主要 —— 带置信度分数的 PyTorch/NumPy golden |
| 活跃 | `phase0-phase1-planning` | Phase 1 归一化规则（无 `.T`、显式 `reshape`、shape 注释）|
| 活跃 | `kernel-code-format` | §11 shape 注解约定 |

**退出标准：** GATE 1 —— golden 中零个 `.T`/`.t()`、所有中间变量有 shape 注释、清单已记录、`allclose` 通过与原始参考的对比。

**休眠升级：** `debugging`（用于 `DEBUG.md` §9.19 归约对齐和 matmul 注意事项 —— 仅在 golden 在预检中触发 PyPTO 约束时加载）。

---

## 4. Architecture Agent

**职责：** 产出 `DESIGN.md` 和性能目标表。**不**执行优化。

| 角色 | Skill | 备注 |
|------|-------|------|
| 活跃 | `pypto-op-design` | `DESIGN.md`：API 映射、tiling 策略、loop 结构 |
| 活跃 | `kernel-code-format` | Layers A–L 设计格式、命名约定 |
| 仅参考 | `pypto-op-perf-tune` | **仅读取目标指标结构。** 不要加载 `tune-frontend`/`tune-swimlane`/`tune-incore`——那些属于 Optimization Agent。 |

**退出标准：** `DESIGN.md` 存在、Layers A–L 已填充、性能目标表示为具体数字（基线、目标时间、所需加速比）。

---

## 5. Design Agent

**职责：** 将 kernel 拆分为语义模块，定义模块契约，规划 staged 文件。负责 Phase 2。

| 角色 | Skill | 备注 |
|------|-------|------|
| 路由 | `phase2-phase3-construction` | Phase 2 模块分解部分 |
| 活跃 | `pypto-op-design` | 将 `DESIGN.md` 的 tiling/loop 决策下放到逐模块级别 |
| 活跃 | `kernel-code-format` | `pypto_kernel_template.py` 骨架 |
| 活跃 | `plan-template` | 填写**模块分解**、**模块契约**、**Staged 模块文件** |

**退出标准：** GATE 2 —— 模块分解、契约和 staged 文件表都在计划中存在。

---

## 6. Coding Agent

**职责：** 逐模块实现 `custom/<op>/<op>_module<suffix>.py`。负责 Phase 3–5 实现。

| 角色 | Skill | 备注 |
|------|-------|------|
| 活跃 | `pypto-op-develop` | 生成 `impl.py`、`test.py`、`README` |
| 路由 | `phase2-phase3-construction` | Phase 3 部分 + DEBUG §9 查找表 |
| 活跃 | `phase4-phase5-integration` | Phase 4 集成、Phase 5 结构规则 |
| 活跃 | `kernel-code-format` | 模板、回写模式、tile 配置规则 |

**直接使用的工具（非 skill）：** MCP `query_op`、`list_ops`、`retrieve_docs`、`validate_kernel_structure`。

**失败时的交接：** 在任何测试/layout 失败时，将失败的模块和日志交接给 Verification Agent。**不要**自己加载调试子 skill。

---

## 7. Verification Agent（仅判定）

**职责：** 运行 `detailed_tensor_compare` 和 layout 检查。为每个 Gate 产出通过/失败的裁定。失败时，将失败分类为 `failure_category` 以便 Lead 分发 Debug Agent。**不**调查、二分或修复——那是 Debug Agent 的工作。

| 角色 | Skill | 备注 |
|------|-------|------|
| 活跃 | `validation-and-deliverables` | `detailed_tensor_compare` 运行器、成功标准 |
| 活跃 | `ci-and-layout-check` | `extract_pypto_calls.py`、`run_validate_layout.sh` |

活跃 skill 数：**2**。Verification 必须保持精简，因为它在每个模块补丁周期中被重新调用，且每次必须从头重新运行完整的 GATE 3 检查清单。

**裁定格式——始终为以下之一：**

```
GATE N passed for <scope>. Evidence: <plan-file row pointer>.
```

或

```
GATE N FAILED for <scope>. failure_category: <cat>.
Failing file: <path>. Evidence: <plan-file row + log excerpt>.
Dispatch @debug.
```

`failure_category` 值：`precision`、`aicore`、`host_crash`、`workspace_overlap`、`oom`、`structure`、`layout`、`other`。

**禁止：** 加载 `debugging/*` skill、编辑 kernel 代码、二分查找、在未经先前 @debug→@coding 循环的情况下重试检查。

---

## 8. Debug Agent（专家调查者）

**职责：** 调查特定 staged 文件上某个 GATE 失败的根本原因，提出具体补丁方案，并将方案交回给 Lead 以便 @coding 应用。每次调用恰好加载一个 `debugging/*` 子 skill，匹配 Verification 报告的 `failure_category`。

| 角色 | Skill | 备注 |
|------|-------|------|
| 路由 | `debugging` | 决策树 → 恰好加载一个子 skill |
| 活跃 | `debugging/DEBUG.md` § 查找 | §9 常见 PyPTO 陷阱速查表 |
| 活跃（按需）| 一个 `debugging/<sub>` | 按 `failure_category` 加载，切换前卸载 |

**路由分发表**（以 Verification 的 `failure_category` 为键）：

| `failure_category` | 按需加载的子 skill |
|---|---|
| `precision` | `pypto-precision-debug` → 升级到 `pypto-precision-compare` 进行二分 |
| `aicore` | `pypto-aicore-error-locator` |
| `host_crash` | `pypto-host-stacktrace-analyzer` |
| `workspace_overlap` | `pypto-memory-overlap-detector` |
| `oom` | `pypto-machine-workspace` |
| `tile_shape` | `pypto-tile-shape-debug`（L0/L1 超限、tile 对齐、`set_cube_tile_shapes` 误用、`enable_split_k`）|
| `structure` / `layout` | `debugging` + 仅 `DEBUG.md` §9 |
| `other` | `debugging` → 按 `SKILL.md` 决策树升级 |

每次失败只加载一个子 skill；在继续之前卸载。最坏情况活跃 skill 数：**3**。

**每次调用的交付物：** 一个补丁方案记录到 `custom/plan/<op>.md` → Development & debug log，包含 (a) 文件 + 行范围，(b) 当前代码片段，(c) 建议代码片段，(d) 对失败 Verification 检查的预期效果。Debug Agent **不**直接修改生产 kernel 代码；它可以在 `custom/<op>/_debug/` 下创建诊断 scratch 文件。

**迭代上限：** 每个模块 3 次完整修复/重新验证循环。如果同一模块第 4 次失败，停止并将阻塞项连同所有证据提交给 Lead。

---

## 9. Optimization Agent

**职责：** 在正确性冻结后运行 Phase 6 优化。与 Verification Agent 紧密协调以确保回归安全。

| 角色 | Skill | 备注 |
|------|-------|------|
| 活跃 | `phase6-optimization` | Phase 6 流程、回滚规则 |
| 路由 | `pypto-op-perf-tune` | 3 阶段编排器，带阶段 Gate |
| 活跃 | `perf-analyzer` | 指标提取、瓶颈识别 |

**路由分发表**（参见 `performance/pypto-op-perf-tune/SKILL.md`"Stage Gating (Router Policy)"）：

| 进入的阶段 | 按需加载的子 skill | 卸载时机 |
|-----------|-------------------|---------|
| Stage 1（frontend）| `tune-frontend` | Stage 1 退出 |
| Stage 2（swimlane）| `tune-swimlane` | Stage 2 退出 |
| Stage 3（incore）| `tune-incore` | Stage 3 退出 |
| 任何需要自动化的阶段 | `pypto-operator-auto-tuner` | 自动化任务完成 |

Stage N+1 必须在 Stage N 干净退出后才能进入。最坏情况活跃 skill 数：**4**（同一时间只有一个 `tune-*` 子 skill 活跃）。

**激活前置条件：** GATE 4 通过（E2E `all_close: true`，layout 检查 exit 0）。在此之前 Optimization Agent 处于休眠状态。

**与 Verification Agent 的回归循环：**

```
Optimization Agent: 应用变更 N
  ↓
Verification Agent:
  (1) detailed_tensor_compare → all_close?
  (2) layout check            → exit 0?
  (3) perf-analyzer           → 与基线的 delta
  → 写入验证报告
  ↓
Optimization Agent:
  - 回归       → 回滚，记录到计划中，尝试下一个方案
  - 无收益     → 记录，尝试下一个方案
  - 有收益且无回归 → 采纳，继续
  - 达到目标   → 停止，交回给 Lead
```

---

## Skill 到 Agent 的反向索引

用于快速检查"2–5 活跃"不变量。

**根控制文档（仅 Lead Agent）：**

| 文档 | 用途 |
|------|------|
| `principles.md` | 行为准则 |
| `agent-plan.md` | Phase 检查清单 / Gate |
| `agents.md` | 团队名册 / 分发 |
| `rules.md` | 强制规则 |
| `catalog.yaml` | Tier-1 路由索引 |

**`skills/` 下的 skill 及其活跃分配：**

| Skill | 活跃 Agent |
|-------|-----------|
| `plan-template` | Planning、Design |
| `kernel-code-format` | Algorithm、Architecture、Design、Coding |
| `validation-and-deliverables` | Verification |
| `phase0-phase1-planning` | Planning、Algorithm |
| `phase2-phase3-construction` | Design、Coding |
| `phase4-phase5-integration` | Coding |
| `phase6-optimization` | Optimization |
| `pypto-intent-understand` | Planning |
| `pypto-api-explore` | Planning |
| `pypto-golden-generate` | Algorithm |
| `pypto-op-design` | Architecture、Design |
| `pypto-op-develop` | Coding |
| `debugging` | Debug（路由）|
| `pypto-op-perf-tune` | Architecture（仅参考）、Optimization（路由）|
| `perf-analyzer` | Optimization |
| `ci-and-layout-check` | Verification |

`pypto-op-workflow` 被 Lead 作为分发指南引用，但**不**加载到 Lead 的活跃集中；它位于 `skills/` 下，按需查阅。库中所有其他 skill 默认**休眠**，仅通过上述路由 skill 或每个 Agent 表中的显式条件加载。

---

## 反模式

1. **不要将调试子 skill 直接交给 Coding Agent。** 通过 Verification（判定者）→ Debug（调查者）→ Coding（应用者）路由。Coding Agent 的工作是编写代码，不是诊断失败。
2. **不要让 Verification 调查或修复。** Verification 是判定者。一旦需要 `debugging/*` skill，Lead 必须分发 Debug Agent。
3. **不要让 Debug 修改生产 kernel 代码。** Debug 将补丁方案写入计划文件；Coding 应用它们。这保留了审计轨迹，并防止 Debug 意外推进模块。
4. **不要在 GATE 4 之前加载 Optimization skill。** Phase 6 以正确性为门控；预加载 `tune-*` skill 会导致过早优化和静默的正确性回归。
5. **不要将任何 Agent 扩展到超过 5 个活跃 skill。** 如果新工作流需要更多，引入新的路由 skill 或拆分为两个 Agent。
6. **不要在 Lead Agent 上预加载所有开发后 `ci-and-pr/*` skill。** 使用上面的休眠分发表——每个开发后 skill 在特定条件（请求 PR、CI 失败等）下激活。
7. **不要跳过共享计划文件。** `custom/plan/<op>.md` 是 Agent 之间的唯一事实来源；每次交接是计划更新，不是直接消息。
