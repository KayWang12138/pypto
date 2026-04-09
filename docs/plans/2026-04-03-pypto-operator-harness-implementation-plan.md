# PyPTO Operator Harness Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 先在 `.agents` 与 `.opencode` 范围内，为 PyPTO 算子 harness 能力建立统一的 skill / agent / 模板契约，使 Stage 5 的测试入口从“脚本式产物”升级为“显式 harness 产物”。

**Architecture:** 先改编排与工件契约，再改模板与说明文档。首阶段不展开代码实现层，只解决三类问题：Stage 5 要产出什么、Stage 5/6 如何识别 harness、失败后如何升级到 precision compare。

**Tech Stack:** Markdown、YAML 模板、`.agents/skills/` 文档契约、`.opencode/agents/` 状态机契约。

---

## 文件结构规划

### 预计新增文件

- `.agents/skills/pypto-op-develop/references/cases-template.yaml`

### 预计修改文件

- `.agents/skills/pypto-op-develop/SKILL.md`
- `.agents/skills/pypto-op-develop/references/test-template.py`
- `.agents/skills/pypto-op-workflow/SKILL.md`
- `.agents/skills/pypto-golden-generate/SKILL.md`
- `.agents/skills/pypto-precision-compare/SKILL.md`
- `.agents/README.md`
- `.opencode/agents/pypto-op-orchestrator.md`
- `.opencode/agents/pypto-op-developer.md`

### 职责划分

- `pypto-op-develop/SKILL.md`：定义 Stage 5 输出契约与 harness 默认行为
- `references/test-template.py`：定义 `test_{op}.py` 的 harness 入口模板
- `references/cases-template.yaml`：定义结构化 case 元数据模板
- `pypto-op-workflow/SKILL.md`：定义 harness 在全流程中的位置
- `pypto-golden-generate/SKILL.md`：把典型配置 / 泛化 case 语义对齐到 harness case 分类
- `pypto-precision-compare/SKILL.md`：定义 harness 失败后的升级路径
- `pypto-op-orchestrator.md`：定义全局门禁、工件契约与阶段流转
- `pypto-op-developer.md`：定义 Stage 5/6 局部预检与结果摘要

---

## Chunk 1: 固化 Stage 5 的 Harness 输出契约

### Task 1: 为 `pypto-op-develop` 增加 harness 工件定义

**Files:**
- Modify: `.agents/skills/pypto-op-develop/SKILL.md`
- Create: `.agents/skills/pypto-op-develop/references/cases-template.yaml`

- [ ] **Step 1: 更新 Stage 5 输出清单**

把 Stage 5 产物从：

- `{op}_impl.py`
- `test_{op}.py`
- `README.md`

升级为：

- `{op}_impl.py`
- `{op}_cases.yaml`
- `test_{op}.py`
- `README.md`

- [ ] **Step 2: 在 `pypto-op-develop/SKILL.md` 中新增 harness 段落**

至少补充：

1. `{op}_cases.yaml` 是 Stage 5 必选工件
2. `test_{op}.py` 必须消费 `{op}_cases.yaml`
3. 默认 case 分类为 smoke / typical / edge
4. 默认 compare 为 allclose 语义

- [ ] **Step 3: 新增 `cases-template.yaml`**

模板至少包含：

1. case name
2. case category
3. 输入参数说明
4. compare 配置
5. 优先级

- [ ] **Step 4: 回读文档确认术语统一**

确认 `pypto-op-develop/SKILL.md` 内部以下术语一致：

1. test 入口
2. harness
3. case 元数据
4. 默认 compare

---

### Task 2: 升级 `test-template.py` 的模板职责

**Files:**
- Modify: `.agents/skills/pypto-op-develop/references/test-template.py`

- [ ] **Step 1: 去掉“只依赖硬编码 level case”的模板叙述**

把模板职责明确改为：

1. `test_{op}.py` 是 harness 入口
2. 支持 `--list`
3. 支持按 case 执行
4. case 来源是 `{op}_cases.yaml`

- [ ] **Step 2: 保留现有三态判定要求**

模板叙述中必须继续强调：

1. `[PRECISION_PASS]`
2. `[PRECISION_FAIL]`
3. 运行失败

不得因为引入 harness 而弱化三态契约。

- [ ] **Step 3: 增加失败升级提示要求**

模板说明中明确：

1. 默认路径只负责常规精度校验
2. 复杂失败场景升级到 `pypto-precision-compare`

---

## Chunk 2: 对齐 Workflow 与 Agent 的编排认知

### Task 3: 在 workflow skill 中引入 harness 概念

**Files:**
- Modify: `.agents/skills/pypto-op-workflow/SKILL.md`

- [ ] **Step 1: 在 Stage 5 描述中加入 harness 工件说明**

明确写出：

1. Stage 5 生成的不只是测试脚本，还有 harness 元数据
2. Stage 6 / Stage 7 继续依赖同一 harness 入口复验

- [ ] **Step 2: 在工件依赖关系中加入 `{op}_cases.yaml`**

要求：

1. 不改变原有 7 阶段顺序
2. 只补充 Stage 5 之后的工件语义

---

### Task 4: 更新 Orchestrator 的标准工件契约与门禁

**Files:**
- Modify: `.opencode/agents/pypto-op-orchestrator.md`

- [ ] **Step 1: 在标准工件契约中加入 `{op}_cases.yaml`**

至少说明：

1. Owner 是 Stage 5
2. 消费者至少包含 Stage 5 / 6 / 7
3. 用途是描述 harness case 与默认 compare 规则

- [ ] **Step 2: 在 Stage 5 门禁中加入 harness 预检**

补充如下门禁要求：

1. `test_{op}.py` 存在
2. `{op}_cases.yaml` 存在
3. 测试入口与 case 工件语义一致

- [ ] **Step 3: 在 Stage 6 / 7 进入条件描述中强调 harness 连续性**

要求：

1. 精度修复与性能调优不得绕过 harness 入口
2. 不允许通过删除 case 或弱化 compare 规则取得假通过

---

### Task 5: 更新 Developer Subagent 的局部预检与返回摘要

**Files:**
- Modify: `.opencode/agents/pypto-op-developer.md`

- [ ] **Step 1: 在 Stage 5 首跑前预检中加入 harness 检查**

增加检查项：

1. `{op}_cases.yaml` 是否存在
2. `test_{op}.py` 是否能识别 case 元数据
3. harness 入口是否仍输出三态标记

- [ ] **Step 2: 在 Stage 6 约束中加入“保持 harness 契约稳定”**

明确禁止：

1. 删除 case
2. 跳过 harness 入口直接宣称 PASS
3. 通过降低 compare 规则绕过失败

- [ ] **Step 3: 扩充结构化返回摘要**

返回结果中补充：

1. harness 工件是否完整
2. 使用了哪些 case 分类
3. 是否触发 precision compare 升级建议

---

## Chunk 3: 对齐上下游 Skill 语义与仓库说明

### Task 6: 对齐 golden skill 的 case 语义

**Files:**
- Modify: `.agents/skills/pypto-golden-generate/SKILL.md`

- [ ] **Step 1: 把“典型配置 / 泛化 case”语义映射到 harness case 分类**

至少补充：

1. 典型配置可直接成为 harness typical case 来源
2. 泛化采样可作为 dynamic / edge case 来源

- [ ] **Step 2: 强调 golden 不负责 harness 编排**

明确边界：

1. golden 负责参考实现
2. harness 负责 case 与执行入口

---

### Task 7: 对齐 precision compare skill 的升级语义

**Files:**
- Modify: `.agents/skills/pypto-precision-compare/SKILL.md`

- [ ] **Step 1: 增加“何时从 harness 默认路径升级”说明**

至少说明：

1. 默认 harness 路径适合首轮校验
2. 需要定位具体误差来源时再升级到 precision compare

- [ ] **Step 2: 强调升级不是默认主路径**

避免 skill 文案让人误以为所有测试都要先走 compare 保存链路。

---

### Task 8: 更新 `.agents/README.md`

**Files:**
- Modify: `.agents/README.md`

- [ ] **Step 1: 在算子开发与编排说明中加入 harness 描述**

至少补充：

1. harness 在 Stage 5 的位置
2. 与 `{op}_golden.py`、`{op}_impl.py`、`test_{op}.py` 的关系
3. 新增 `{op}_cases.yaml` 的角色

- [ ] **Step 2: 保持 README 与 workflow/orchestrator 术语一致**

确保以下词在 README 中表达一致：

1. 测试入口
2. harness
3. case 元数据
4. 三态判定

---

## P0 / P1 / P2 优先级

### P0（必须先完成）

1. `pypto-op-develop/SKILL.md` 更新
2. `test-template.py` 职责升级
3. `cases-template.yaml` 新增
4. `pypto-op-workflow/SKILL.md` 增补 harness 语义
5. `pypto-op-orchestrator.md` 增补工件与门禁
6. `pypto-op-developer.md` 增补预检与摘要

### P1（P0 收敛后推进）

1. `pypto-golden-generate/SKILL.md` 术语对齐
2. `pypto-precision-compare/SKILL.md` 升级路径对齐
3. `.agents/README.md` 统一说明

### P2（后续单独立项）

1. 基于已稳定契约产出代码级实现计划
2. 把 harness 契约落到真实运行时承接层

---

## 验收标准

本计划完成后，应至少满足：

1. `.agents` / `.opencode` 中对 Stage 5 harness 的描述一致
2. `{op}_cases.yaml` 被明确纳入 Stage 5 输出契约
3. `test_{op}.py` 被明确界定为 harness 入口，而不只是普通测试脚本
4. Stage 5 / 6 都明确要求保持 harness 契约稳定
5. 默认校验路径与 precision compare 升级路径边界清晰

---

## 验证方式

由于本计划当前只涉及 `.agents` / `.opencode` 范围内的文档与模板契约改写，验收以回读检查为主：

1. 回读相关 skill / agent 文档
2. 检查术语一致性
3. 检查工件清单一致性
4. 检查门禁与摘要格式是否同步更新

---

计划已收敛为一份仅面向 `.agents` 与 `.opencode` 契约改造的实施计划。下一步可按本计划推进相关文档与模板修改。
