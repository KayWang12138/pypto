---
name: pypto-machine-pr-review
description: 仅用于 cann/pypto 的 Machine 代码 PR 评审与行级评论回写。覆盖 machine 变更分析、风险分级、position 计算、行级评论发布与结果核验。用户提到“Review Machine PR”“Machine代码评审”“machine模块PR审查”“提交machine review意见”时使用。强制约束：所有 review 意见必须先与用户逐条确认后，才允许提交到 PR。
---

# PyPTO PR Reviewer

仅面向 `cann/pypto` **Machine 相关代码**的标准化 PR 审查与回写流程。

## 适用范围（强制）

- 仅适用于 `framework/src/machine/**` 及与 machine 强相关的接口/测试路径（如 `framework/src/interface/machine/**`、`framework/tests/**/machine/**`）。
- 若 PR 不属于 machine 范围，必须先告知用户“该 skill 不适用”，然后建议切换到通用 review 流程/其他 skill。

## 强制约束（必须遵守）

1. **先审后发**：先完成本地审查结论，再决定是否回写评论。
2. **先确认后提交**：任何评论在发布到 PR 前，必须先给用户展示“拟提交清单”，并得到用户明确确认（如“确认提交”“同意提交”）。
3. **无确认禁止提交**：未获得明确确认时，只能输出草稿，禁止调用远程评论接口。
4. **行级优先**：高风险问题必须回写到对应代码行（`path + position`），不得只发总评。
5. **基于证据**：结论必须可追溯到具体 diff 片段，区分“已确认事实 / 推断 / 建议动作”。
6. **不编造执行结果**：远程接口失败、权限不足、行号无效必须原样反馈。

## 参考文件

- [references/review-checklist.md](references/review-checklist.md)：review 规则与提交前确认模板

## 工作流

### 阶段 1：获取 PR 上下文

1. 获取 PR 元信息（标题、分支、状态、作者）。
2. 获取 commits 与 files 列表。
3. 拉取完整 diff（或按文件 diff）。

### 阶段 2：风险导向审查

按优先级检查：

1. 正确性与边界条件。
2. 并发/时序/内存资源风险。
3. 兼容性与行为变化风险。
4. 测试覆盖是否匹配行为变化。

输出本地 findings，按严重级别排序（High > Medium > Low）。

### 阶段 3：定位行级评论位置

对每条要回写的意见计算：

- `path`：PR 中的相对文件路径。
- `position`：diff 相对位置（不是源码绝对行号）。

`position` 计算规则：

1. 遍历该文件 patch；
2. `@@ ... @@` 行不计数；
3. 其余行（`+`/`-`/空格上下文）都计数；
4. 目标语句所在 patch 行的计数值即 `position`。

### 阶段 4：提交前确认门禁（强制）

向用户展示“拟提交评论清单”，每条至少包含：

- 严重级别
- `path`
- `position`
- 评论正文（可精简，但要保留风险与建议）
- 是否 `need_to_resolve`

然后询问用户是否提交。**只有用户明确同意后**，才能进入下一阶段。

### 阶段 5：发布评论到 PR

逐条发布 inline comment。若任一条失败：

- 记录失败项（path/position/body/错误信息）；
- 不伪造“已提交成功”。

### 阶段 6：发布后核验

1. 拉取评论列表验证刚发布的评论存在。
2. 回报用户：
   - 成功条数/失败条数
   - 每条评论的标识（如 note_id）
   - 失败原因与建议重试方式

## 输出格式

### A. 本地审查结论（未提交前）

1. Findings（按严重级别）
2. Open questions / assumptions
3. 拟提交评论清单（path + position + body）
4. 明确询问：“是否确认提交这些评论到 PR？”

### B. 已提交结果

- 已提交评论列表（含 path、position、note_id）
- 未提交/失败列表（含原因）
- 下一步建议（如作者修复后复审）
