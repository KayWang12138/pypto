# PyPTO 算子 Harness 能力设计

## 1. 背景

基于 `.agents/README.md`、`.agents/skills/` 与 `.opencode/agents/` 中的现有定义，可以确认当前 PyPTO 算子开发流程已经具备以下稳定事实：

1. `.agents/README.md` 将完整流程定义为：需求理解 → API 探索 → Golden 生成 → 设计方案 → 代码实现 → 精度验证 → 性能调优 → PR 提交。
2. `.opencode/agents/pypto-op-orchestrator.md` 将流程固化为 7 阶段状态机，并明确 Stage 5 的输出为 `{op}_impl.py`、`test_{op}.py`、`README.md`。
3. `.agents/skills/pypto-golden-generate/SKILL.md` 将 `{op}_golden.py` 定义为独立工件，并强调 golden 与测试入口应职责分离。
4. `.agents/skills/pypto-op-develop/SKILL.md` 明确 `test_{op}.py` 是测试入口，当前承担数据准备、CLI 调用与精度判定职责。
5. `.opencode/agents/pypto-op-developer.md` 要求 Stage 5 / Stage 6 必须基于真实首跑结果做三态判定。
6. `.agents/skills/pypto-precision-compare/SKILL.md` 已经定义了两类精度升级路径：文件保存法与二分对比法。

因此，当前缺的不是新的阶段，也不是新的工件大类，而是：

> 在现有工件契约之间，缺少一层专门承载“结构化 case、统一测试入口、默认校验策略、失败升级路径”的 harness 契约。

---

## 2. 目标与非目标

### 2.1 目标

在不突破现有 `.agents` / `.opencode` 编排边界的前提下，为 PyPTO 算子开发流程补齐 harness 能力，使其成为 Stage 5 测试入口的显式组成部分：

1. 让 `test_{op}.py` 从“脚本式测试入口”升级为“metadata-driven harness 入口”。
2. 在 Stage 5 输出中新增结构化 case 工件，例如 `{op}_cases.yaml`。
3. 为 Stage 5 / Stage 6 明确默认校验策略与失败升级策略。
4. 让 Orchestrator、Developer Subagent 与相关 Skills 对 harness 工件有一致认知。

### 2.2 非目标

本设计不做以下事情：

1. 不新增新的流程阶段。
2. 不改变 7 阶段状态机顺序。
3. 不把精度深调试能力变成所有算子的默认主路径。
4. 不打破 `{op}_golden.py`、`{op}_impl.py`、`test_{op}.py` 三文件分离原则。

---

## 3. 在当前工作流中的 Harness 定义

在当前 `.agents` / `.opencode` 语境下，Harness 应定义为：

> Stage 5 产生的一组测试入口契约，用来把 `{op}_golden.py`、`{op}_impl.py`、结构化 case 描述、默认 compare 规则和失败升级路径绑定在一起，并向 Orchestrator 提供可复验的三态判定出口。

它不是新的独立阶段，也不是新的独立主工件族；它是对以下既有工件关系的补强：

- `{op}_golden.py`：参考实现
- `{op}_impl.py`：实现工件
- `test_{op}.py`：测试入口
- `README.md`：使用说明

新增的核心补件建议是：

- `{op}_cases.yaml`：结构化 case 元数据

---

## 4. 推荐的契约拆分

### 4.1 编排契约层

来源：`.opencode/agents/pypto-op-orchestrator.md`、`.opencode/agents/pypto-op-developer.md`

职责：

1. 明确 Stage 5 的输出物中包含 harness 工件。
2. 明确 Stage 5 首跑前预检要检查 harness 元数据是否存在且可消费。
3. 明确 Stage 6 的精度修复应保留 harness 入口与 case 契约稳定。

### 4.2 工件契约层

来源：`.agents/skills/pypto-op-workflow/SKILL.md`、`.agents/skills/pypto-op-develop/SKILL.md`

职责：

1. 维持 `{op}_golden.py` / `{op}_impl.py` / `test_{op}.py` 三文件分离。
2. 在此基础上新增 `{op}_cases.yaml`，由 Stage 5 一并生成。
3. 规定 `test_{op}.py` 必须消费 `{op}_cases.yaml`，而不是只依赖硬编码 case。

### 4.3 Harness 元数据层

来源：`.agents/skills/pypto-golden-generate/SKILL.md`

职责：

1. 承接 golden skill 已有的“典型配置”“泛化 case”“优先级”概念。
2. 把这些概念收敛成结构化 case 描述。
3. 为 `test_{op}.py` 提供 smoke / typical / edge / dynamic 等 case 分类。

### 4.4 校验与升级层

来源：`.agents/skills/pypto-op-develop/SKILL.md`、`.agents/skills/pypto-precision-compare/SKILL.md`

职责：

1. 定义默认 compare 规则。
2. 定义三态判定与 case 级别判定之间的关系。
3. 定义失败后升级到 precision compare 的路径，但不把它变成默认执行流。

---

## 5. 推荐新增的 Harness 能力事项

### 5.1 新增 `{op}_cases.yaml` 工件契约

让 Stage 5 除了生成 `{op}_impl.py`、`test_{op}.py`、`README.md` 之外，再显式生成 `{op}_cases.yaml`。

作用：

1. 承载 case 名称
2. 承载输入形状 / dtype / 参数
3. 承载 case 分类与优先级
4. 承载 compare 策略与容差

### 5.2 升级 `test_{op}.py` 为 metadata-driven 入口

现有 `test_{op}.py` 已有 CLI 和三态输出基础。建议在 skill 契约中将其升级为：

1. 支持列出 case
2. 支持按 case 执行
3. 从 `{op}_cases.yaml` 读取 case，而不是只写死 `level0/level1`

### 5.3 明确默认 case 分类

从 golden skill 已有概念出发，统一为：

1. smoke
2. typical
3. edge
4. dynamic（可选）

这样 harness 与 golden 的“典型配置 / 泛化 case”语义能够对齐。

### 5.4 明确默认 compare 契约

在 op-develop skill 中补充：

1. 默认 compare 仍以 `assert_allclose` 语义为主
2. compare 配置进入 `{op}_cases.yaml`
3. 复杂场景允许标记为升级到 precision compare

### 5.5 为 Stage 5 增加 harness 预检

在 developer subagent 契约中增加：

1. `{op}_cases.yaml` 存在
2. `test_{op}.py` 与 `{op}_cases.yaml` 的 case 命名一致
3. `{op}_golden.py`、`{op}_impl.py` 的函数签名与 case 描述一致

### 5.6 为 Stage 6 增加“保持 harness 契约稳定”要求

Stage 6 修复精度时，不应绕过 harness 入口，也不应通过删除 case、弱化 compare 或移除三态标记来“伪通过”。

### 5.7 在 workflow skill 中显式加入 Harness 概念

`pypto-op-workflow` 目前强调工件依赖，但没有把 harness 作为显式概念。建议新增说明：

- Stage 5 输出的不只是测试脚本，还有 harness 契约
- Stage 6、Stage 7 都应在该 harness 契约上继续验证

### 5.8 在 README 中增加 Harness 使用说明模板

`README.md` 当前被定义为交付物，但没有明确 harness 入口说明。建议固定补充：

1. case 文件说明
2. 如何列出 case
3. 如何执行单 case
4. 如何在失败后升级到 precision compare

---

## 6. 分阶段规划

### 6.1 近期：先补齐 Skill / Agent / 模板契约

目标：让 harness 能力先在 `.agents` / `.opencode` 范围内定义完整。

建议完成：

1. 在 `pypto-op-develop` 中增加 `{op}_cases.yaml` 输出约定
2. 升级 `test-template.py` 的模板职责
3. 在 `pypto-op-workflow` 中增加 harness 概念与流转说明
4. 在 orchestrator / developer agent 中补充 harness 门禁与预检规则

交付信号：

1. Stage 5 输出契约包含 harness 元数据
2. Stage 5 / 6 对 harness 有显式要求
3. README 与 workflow 对 harness 术语一致

### 6.2 中期：把 Harness 与精度升级路径对齐

目标：让 harness 失败后的处理与现有精度调试 skill 对齐。

建议完成：

1. 在 `pypto-precision-compare` 中增加面向 harness 的升级说明
2. 在 `pypto-op-develop` 中定义 case 级失败如何触发升级
3. 在 developer agent 中补充失败分类摘要格式

交付信号：

1. harness 默认路径与升级路径边界清晰
2. 不再需要临时解释“什么时候进入 precision compare”

### 6.3 远期：再进入代码实现计划

目标：在 skill / agent 契约稳定后，再单独产出代码级计划，实现真正的运行时承接层。

这一阶段不在本文中展开，只保留原则：

1. 代码实现必须服从已定好的 skill / agent 契约
2. 不能先写运行时，再倒推文档契约

---

## 7. 风险与反模式

### 7.1 把 Harness 误写成新的阶段

当前状态机已经完整，harness 应该内化在 Stage 5 / Stage 6 工件与门禁中，而不是新增 Stage 5.5 一类中间层。

### 7.2 破坏三文件分离

`.opencode/agents/pypto-op-orchestrator.md` 和 `pypto-op-develop` 已经明确三文件分离。harness 不能把 golden、impl、test 再混到一起。

### 7.3 只改模板，不改编排契约

如果只升级 `test-template.py`，但不更新 orchestrator / developer / workflow 对 harness 的认知，后续阶段仍会把 harness 当成“脚本细节”，无法稳定推进。

### 7.4 把 precision compare 默认化

`pypto-precision-compare` 是升级路径。若默认启用，会让 Stage 5 首跑成本过高，并模糊三态判定的基线含义。

### 7.5 让 case 契约脱离 golden skill 的“典型配置”语义

golden skill 已有典型配置与泛化 case 的概念。如果 harness 自行发明另一套分类命名，会造成上下游术语漂移。

---

## 8. 设计层验收标准

当以下条件满足时，可认为 harness 设计已经在 `.agents` / `.opencode` 范围内收敛：

1. Stage 5 输出契约明确包含 `{op}_cases.yaml`
2. `test_{op}.py` 的 harness 角色被显式定义
3. workflow / orchestrator / developer / golden / precision compare 五处文档术语一致
4. 三态判定与 harness case 执行关系清晰
5. 默认路径与升级路径边界清晰

---

## 9. 推荐下一步

下一步应进入一份**只面向 `.agents` / `.opencode` 改造**的实施计划，先完成：

1. Skill 模板与输出契约改写
2. Agent 门禁与摘要格式改写
3. README / workflow 术语统一

待这些契约稳定后，再单独产出代码级实现计划。
