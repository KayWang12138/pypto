---
name: pypto-op-autodev
description: |
  自动开发 PyPTO 算子并挖掘框架断裂点。每次执行仅开发 1 个算子，增量推进。
  触发词：自动开发算子、开发一个算子、批量开发算子、autodev、/loop 定时触发。
  也适用于：查询开发进度、重试失败算子、手动指定开发目标算子。
---

# PyPTO 自动算子开发

每次执行仅开发 1 个算子，完成后退出。所有 CSV 操作必须通过脚本完成，LLM 不直接操作 CSV。

## 工作目录

所有脚本的相对路径基准：**项目根目录**（即 `pypto/`）。
所有脚本位于：`.agents/skills/pypto-op-autodev/scripts/`

## 执行流程

### Step 1：前置检查（并发控制 + 中断恢复）

```bash
python .agents/skills/pypto-op-autodev/scripts/update_op.py \
  --csv autodev/scan_results.csv \
  --reset-stale --timeout-hours 6
```

**处理退出码**：
- exit 0（无活跃 in_progress）→ 继续 Step 2
- exit 1（有算子正在开发中，start_time ≤ 6h）→ 输出"算子 {active_op} 正在开发中（{start_time}），跳过本次执行"，退出
- exit 2+（脚本错误）→ 输出 stderr 内容，终止执行

### Step 2a：检查用户新需求

读取 `autodev/sources/requirements.md`，提取每个需求的 `op_name`。
对每个 `op_name`，调用：

```bash
python .agents/skills/pypto-op-autodev/scripts/add_op.py \
  --csv autodev/scan_results.csv \
  --op {op_name} --source manual \
  --complexity {推断的复杂度} --category {推断的类别}
```

- 若 `is_new: false`，说明已在 CSV 中，跳过。
- 若脚本非 0 退出：记录警告，跳过该需求，继续处理其他需求。
- 复杂度/类别推断规则参见 `.agents/skills/pypto-op-discover/references/`。

### Step 2b：选择下一个算子

```bash
python .agents/skills/pypto-op-autodev/scripts/select_next_op.py \
  --csv autodev/scan_results.csv
```

- exit 0 → 解析 JSON，获取 `selected.op_name`，进入 Step 3
- exit 1 → 进入 Step 2c（需要 discover）
- exit 2+ → 输出 stderr，终止执行

### Step 2c：按需补充候选（discover）

若 Step 2b 返回 exit 1，调用 `pypto-op-discover` Skill（通过 OpenCode Agent `pypto-op-discover`），补充 1 个新算子。
补充完成后，**再次执行 Step 2b**。
若仍返回 exit 1 → 输出"当前无合适候选算子，已触发 discover 但仍无满足条件的算子"，退出。

### Step 3：开始开发

标记开始：

```bash
python .agents/skills/pypto-op-autodev/scripts/update_op.py \
  --csv autodev/scan_results.csv \
  --op {op_name} --status in_progress
```

然后调用 `pypto-op-orchestrator` Agent 开发算子：
- 工作目录：`autodev/custom/{op_name}/`
- orchestrator 超时上限：**1 小时**
- 成功工件：`spec.md`, `design.md`, `{op}_impl.py`, `test_{op}.py`, `README.md`
- 无论成功或失败，记录 `dev_result`（SUCCESS / TIMEOUT / BLOCKED_API / BLOCKED_ENV / PRECISION / NOT_IMPLEMENTABLE）

### Step 4：断裂点检测 + 归因分析

**无论 Step 3 成功或失败，均执行此步。**

调用 `pypto-fracture-point-detector` Skill（增强版，含归因分析）：
- 调用前先切换当前工作目录到：`autodev/custom/{op_name}/`
- 输入：本次 Session 上下文 + 当前工作目录中的工件和日志
- 输出：当前工作目录下的 `fracture-point-{timestamp}.md`（含致命断裂点列表 + 每个的归因）
- 从报告提取：`fps_total`（致命断裂点总数），`fps_confirmed`（confidence=high 的数量）

> 成功的算子也可能暴露框架文档问题（如 API 行为与文档不一致但最终 workaround 成功），因此始终执行检测。

### Step 5：Issue 生成（仅 fps_total > 0 时执行）

若 `fps_total > 0`，调用 `pypto-issue-creator` Skill：
- 调用前先切换当前工作目录到：`autodev/custom/{op_name}/`
- 必须使用 `autodev/local-only` 模式：仅生成本地文件 `issue-templates/issue-*.md`
- 不做远程去重、不请求用户确认、不创建在线 Issue

若 `fps_total = 0`，跳过此步。

### Step 6：更新最终状态

```bash
python .agents/skills/pypto-op-autodev/scripts/update_op.py \
  --csv autodev/scan_results.csv \
  --op {op_name} \
  --status {completed 或 failed} \
  --dev-result {dev_result} \
  --fps-total {fps_total} \
  --fps-confirmed {fps_confirmed}
```

若脚本非 0 退出：输出 stderr，并在 `autodev/custom/{op_name}/autodev-status-error.log` 中写入失败时间、命令参数与 stderr 内容。
**不因此丢弃已产出的算子工件。**

---

## 特殊指令处理

### 查询进度

用户说"当前开发进度如何"、"查看进度"等：

```bash
python .agents/skills/pypto-op-autodev/scripts/get_progress.py \
  --csv autodev/scan_results.csv
```

格式化输出统计信息，不启动算子开发流程。

### 重试失败算子

用户说"重试 {op_name}"：

```bash
python .agents/skills/pypto-op-autodev/scripts/update_op.py \
  --csv autodev/scan_results.csv \
  --op {op_name} --reset
```

重置后重新执行 Step 2b 开始选择。该算子因 `fail_count` 保留会降分，但不会因为工作目录已存在而被直接排除。

### 手动指定算子

用户说"帮我开发 {op_name} 算子"：
1. 调用 `add_op.py` 添加（若已存在则跳过）
2. 从 Step 2b 开始（该算子刚创建，分数高，会被选中）

---

## 参考文档

- CSV 字段定义：`references/csv-fields.md`
- 类别推断规则：`.agents/skills/pypto-op-discover/references/category-rules.md`
- 复杂度判定规则：`.agents/skills/pypto-op-discover/references/complexity-rules.md`
- 设计方案：`docs/plans/2026-03-26-fracture-point-scanner-design.md`
