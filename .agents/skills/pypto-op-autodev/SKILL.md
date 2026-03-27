---
name: pypto-op-autodev
description: |
  自动开发 PyPTO 算子并挖掘框架断裂点。每次执行仅开发 1 个算子，增量推进。
  触发词：自动开发算子、开发一个算子、批量开发算子、autodev、/loop 定时触发。
  也适用于：查询开发进度、重试失败算子、手动指定开发目标算子。
---

# PyPTO 自动算子开发

每次执行开发 1 个算子，完成后退出。所有 CSV 操作必须通过脚本完成，LLM 不直接操作 CSV。

> **前置条件**：需要启用 Agent Teams 实验性功能（Claude Code v2.1.32+）：
> 在 settings.json 中设置 `"env": {"CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS": "1"}`

autodev 使用 **Agent Team** 机制调度 orchestrator：
- **autodev（team lead）**：选择算子、spawn orchestrator teammate、收割结果、验证、断裂点检测、状态管理
- **orchestrator（teammate）**：独立 Claude Code 实例，7 阶段算子开发，可自由 spawn subagent，autodev 不修改 orchestrator

## 工作目录

所有脚本的相对路径基准：**项目根目录**（即 `pypto/`）。
所有脚本位于：`autodev/scripts/`
算子工件目录：`autodev/custom/{op_name}/`

## 执行流程

### Step 1：前置检查（环境预检 + 并发控制 + 中断恢复）

#### 1a. 环境预检

```bash
python autodev/scripts/check_env.py
```

**处理退出码**：
- exit 0（环境正常）→ 继续 Step 1b
- exit 1（NPU 不可用，可降级）→ 记录警告"NPU 不可用，将以降级模式运行"，继续 Step 1b
- exit 2（PyPTO/torch 不可用）→ 输出"环境检查失败：PyPTO 或 torch 不可用，请先运行 pypto-environment-setup"，终止执行

#### 1b. 残留 team 清理

清理残留 team，避免后续 `TeamCreate` 失败：
- 读取 `autodev/.stale_teams`（若存在），对其中每个 team 名尝试 `TeamDelete`
- 检查 `~/.claude/teams/` 下是否存在 `autodev-*` 目录，若对应算子不在 `in_progress` 状态则清理
- 清理完成后删除 `autodev/.stale_teams`

#### 1c. 并发控制与中断恢复

```bash
python autodev/scripts/update_op.py \
  --csv autodev/scan_results.csv \
  --reset-stale --timeout-hours 3
```

**处理退出码**：
- exit 0（无活跃 in_progress）→ 继续 Step 2
- exit 1（有算子正在开发中，start_time ≤ 3h）→ 输出"算子 {active_op} 正在开发中（{start_time}），跳过本次执行"，退出
- exit 2+（脚本错误）→ 输出 stderr 内容，终止执行

### Step 2a：检查用户新需求

读取 `autodev/sources/requirements.md`，提取每个需求的 `op_name`。
对每个 `op_name`，调用：

```bash
python autodev/scripts/add_op.py \
  --csv autodev/scan_results.csv \
  --op {op_name} --source manual \
  --complexity {推断的复杂度} --category {推断的类别}
```

- 若 `is_new: false`，说明已在 CSV 中，跳过。
- 若脚本非 0 退出：记录警告，跳过该需求，继续处理其他需求。
- 复杂度/类别推断规则参见 `autodev/references/`。

### Step 2b：选择下一个算子

```bash
python autodev/scripts/select_next_op.py \
  --csv autodev/scan_results.csv
```

- exit 0 → 解析 JSON，获取 `selected.op_name`、`selected.resume_from_stage`、`selected.existing_artifacts`、`retry_strategy`，进入 Step 3
- exit 1 → 进入 Step 2c（需要 discover）
- exit 2+ → 输出 stderr，终止执行

`resume_from_stage` > 1 表示断点续跑。`retry_strategy` 仅 failed 算子有，脚本已自动处理（如 PRECISION 自动设为 stage 6）。

### Step 2c：按需补充候选（discover）

若 Step 2b 返回 exit 1，检查 `autodev/.discovery_fail_count` 文件：
- 若文件存在且计数 ≥ 3 → 输出"连续 discover 失败已达上限（3 次），暂停自动开发。请手动添加算子或检查 discover 配置"，退出
- 否则：调用 `pypto-op-discover` Skill（通过 OpenCode Agent `pypto-op-discover`），补充 1 个新算子

补充完成后，**再次执行 Step 2b**。
- 若成功选中 → 重置 `.discovery_fail_count` 为 0，进入 Step 3
- 若仍返回 exit 1 → 递增 `.discovery_fail_count`，输出"当前无合适候选算子"，退出

### Step 3：通过 Agent Team 调度 orchestrator 开发

标记开始：

```bash
python autodev/scripts/update_op.py \
  --csv autodev/scan_results.csv \
  --op {op_name} --status in_progress
```

将开发任务写入 `autodev/custom/{op_name}/autodev-task.md`：
```markdown
# 开发任务
- op_name: {op_name}
- working_dir: autodev/custom/{op_name}/
- resume_from_stage: {resume_from_stage}
- existing_artifacts: {existing_artifacts}
- complexity: {complexity}
- category: {category}
- known_limitations: autodev/references/known-limitations.md
```

创建 Agent Team 并 spawn orchestrator 作为 teammate。`{recommended_device_id}` 从 Step 1a check_env.py 输出中获取：

```
1. TeamCreate(team_name="autodev-{op_name}", description="开发 {op_name} 算子")

2. Agent(
     name="orchestrator",
     team_name="autodev-{op_name}",
     prompt="你是 pypto-op-orchestrator。读取 .opencode/agents/pypto-op-orchestrator.md 获取完整指令。
       读取 autodev/custom/{op_name}/autodev-task.md 获取任务参数。
       工作目录：autodev/custom/{op_name}/
       超时上限：1 小时。
       运行时设置 TILE_FWK_DEVICE_ID={recommended_device_id}。
       开发过程中持续追加写入 dev-log.md（报错、方案变更、API 不一致、workaround）。
       完成后必须发送完成消息，不得静默退出。"
   )
```

**等待策略：产物轮询 + 超时兜底**

等待 orchestrator 完成通知，同时每 5 分钟检查一次 `autodev/custom/{op_name}/.orchestrator_state.json`：
- 若 `terminal_state` 非空（SUCCESS/BLOCKED_*）→ orchestrator 已完成但未发通知，直接进入 Step 4
- 若 `current_stage` 与上次检查相同且已持续 30 分钟无变化 → 标记 `dev_result=TIMEOUT`，进入 Step 6
- 90 分钟硬超时兜底 → 标记 `dev_result=TIMEOUT`，进入 Step 6

### Step 4：独立验证

```bash
python autodev/scripts/verify_op.py \
  --csv autodev/scan_results.csv \
  --op {op_name} --run-test
```

- `verify_status=VERIFIED`：精度验证通过 → `dev_result=SUCCESS`
- `verify_status=TEST_PASS_NO_PRECISION`：测试通过但无精度标记 → `dev_result=SUCCESS`
- `verify_status=TEST_FAIL`：测试失败 → `dev_result=PRECISION`
- `verify_status=INCOMPLETE`：产物不完整 → `dev_result=TIMEOUT`

### Step 5：后台收尾（断裂点检测 + 健康检查 + Issue 生成）

启动后台 Agent 执行所有收尾分析，避免分析结果膨胀主 context：

```
Agent(
  description="收尾分析 {op_name}",
  run_in_background=true,
  prompt="对算子 {op_name} 执行收尾分析，工作目录 autodev/custom/{op_name}/：

    1. 断裂点检测：
       调用 Skill pypto-fracture-point-detector，
       输入 dev-log.md + 工件，重点关注报错、workaround 和 API 不一致

    2. Issue 生成（仅断裂点 > 0 时）：
       调用 Skill pypto-issue-creator，仅生成本地 issue 模板
       （autodev/custom/{op_name}/issue-templates/issue-*.md，不创建在线 Issue）

    3. 写入 autodev/custom/{op_name}/fracture-summary.json：
       {\"fps_total\": N, \"fps_confirmed\": N}

    4. 健康检查（最后执行，确保断裂点产物已生成）：
       python autodev/scripts/health_check.py \
         --op {op_name} --op-dir autodev/custom/{op_name} \
         --status {completed 或 failed} --start-time {start_time_iso}
       若 exit 1（needs_review: true），将 issues 和简要分析追加写入 autodev/self-review.md"
)
```

不等待后台 Agent 完成，直接进入 Step 6。

### Step 6：更新最终状态

读取 `autodev/custom/{op_name}/fracture-summary.json`（若已生成）获取 `fps_total`、`fps_confirmed`，否则默认 0。

```bash
python autodev/scripts/update_op.py \
  --csv autodev/scan_results.csv \
  --op {op_name} \
  --status {completed 或 failed} \
  --dev-result {dev_result} \
  --fps-total {fps_total} \
  --fps-confirmed {fps_confirmed} \
  --blocked-stage {从 .orchestrator_state.json 提取的当前阶段，仅失败时传}
```

若脚本非 0 退出：在 `autodev/custom/{op_name}/autodev-status-error.log` 中记录，不丢弃工件。

### Step 7：清理 team + 生成进度报告

关闭 orchestrator teammate 并清理 team 资源（指数退避重试）：

1. 发送 shutdown 请求，等待 3s，尝试 TeamDelete
2. 若失败 → 再次 shutdown，等待 10s，尝试 TeamDelete
3. 若仍失败 → 等待 30s，尝试 TeamDelete
4. 若 3 次均失败 → 跳过清理，将 team 名称追加到 `autodev/.stale_teams`，由下一轮 Step 1b 处理

生成进度报告：
```bash
python autodev/scripts/get_progress.py \
  --csv autodev/scan_results.csv \
  --format markdown --output autodev/PROGRESS.md
```

---

## 特殊指令处理

查询进度、重试失败算子、手动指定算子、验证、产物检查、依赖管理等操作参见 `references/autodev-commands.md`。

---

## 参考文档

- CSV 字段定义：`references/csv-fields.md`
- 类别推断规则：`autodev/references/category-rules.md`
- 复杂度判定规则：`autodev/references/complexity-rules.md`
- 设计方案：`docs/plans/2026-03-26-fracture-point-scanner-design.md`
