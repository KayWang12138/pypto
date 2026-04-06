---
name: pypto-op-autodev
description: |
  自动开发 PyPTO 算子并挖掘框架断裂点。每次仅开发 1 个算子，增量推进。
  当用户提到以下任何场景时使用此 skill：自动开发算子、批量开发算子、autodev、
  /loop 定时触发、查询开发进度、查看进度、重试失败算子、重试某个算子、
  手动指定开发目标算子、帮我开发一批算子、算子自动化、挖掘断裂点。
  即使用户只是模糊地提到"自动开发"或"批量跑算子"，也应触发此 skill。
---

# PyPTO 自动算子开发

每次执行开发 1 个算子，完成后退出。**所有 CSV 操作必须通过脚本完成，LLM 不直接操作 CSV。**

设计文档：`docs/agents/autodev.md`

## 首次使用

Step 0 会自动初始化（创建 CSV、目录、复制种子知识）。但 CSV 为空时调度器无算子可选，需先添加：
- **手动添加**：`python {scripts}/add_op.py --csv {csv_path} --op {名称} --source manual --complexity easy --category elementwise`
- **自动发现**：`scheduler.conf.json` 中设 `"enable_discover": true`

## 运行时变量

执行前先确定以下变量：

| 变量 | 解析方式 | 典型值 |
|------|---------|--------|
| `{project_root}` | git 仓库根目录 | `/workspace/projects/code/pypto` |
| `{skill_dir}` | 本 SKILL.md 所在目录 | `{project_root}/.agents/skills/pypto-op-autodev` |
| `{scripts}` | `{skill_dir}/scripts/` | — |
| `{csv_path}` | `{project_root}/autodev/scan_results.csv` | — |
| `{work_dir}` | `{project_root}/autodev/custom` | — |

以下所有脚本命令中 `python {scripts}/xxx.py` 均指该绝对路径。

---

## 执行流程

### Step 0：环境预检 + 初始化

```bash
python {scripts}/check_env.py --csv {csv_path} --work-dir {work_dir}
```

- exit 0 → 继续
- exit 1（NPU 不可用）→ 降级继续，记录警告
- exit 2（PyPTO/torch 不可用）→ 终止

该脚本同时负责初始化：CSV 不存在时创建 header，目录不存在时创建，`autodev/known-limitations.md` 不存在时从种子文件复制。

### Step 1：并发控制

```bash
python {scripts}/update_op.py --csv {csv_path} --reset-stale --timeout-hours 3
```

- exit 0 → 继续
- exit 1（活跃任务未超时）→ 输出提示，退出
- exit 2+ → 终止

### Step 2：选择目标算子

**2a. 检查用户需求**（可选）：若 `{project_root}/autodev/sources/requirements.md` 存在，读取需求，对每个算子：

```bash
python {scripts}/add_op.py --csv {csv_path} --op {op_name} --source manual \
  --complexity {推断} --category {推断}
```

复杂度/类别按 `references/complexity-rules.md` 和 `references/category-rules.md` 推断。`is_new: false` 跳过，脚本非 0 记录警告后继续。

**2b. 选择下一个算子**：

```bash
python {scripts}/select_next_op.py --csv {csv_path}
```

- exit 0 → 解析 JSON 获取 `selected.op_name`、`resume_from_stage`、`complexity`、`category`、`existing_artifacts`、`retry_strategy`。（`resume_from_stage` > 1 为断点续跑）
- exit 1 → 进入 2c
- exit 2+ → 终止

**2c. 按需 discover**：仅 `enable_discover == true` 时执行。

检查 `{work_dir}/.discovery_fail_count`：
- 计数 ≥ 3 → 输出"discover 失败达上限"，退出
- 否则 → 调用 `pypto-op-discover`（调度方式见 `references/platform-dispatch.md`），再次执行 2b
  - 成功 → 重置计数，继续
  - 仍 exit 1 → 递增计数，退出

`enable_discover == false` 且 exit 1 → 直接退出。

### Step 3：标记开始 + 准备任务

```bash
python {scripts}/update_op.py --csv {csv_path} --op {op_name} --status in_progress
```

确定开发策略 `dev_strategy`：
1. 用户显式指定 → 用那个
2. 配置文件中 `dev_strategy` 为固定值 → 用那个
3. `dev_strategy: "auto"` → 该算子上次用 orchestrator 失败 → workflow；complexity == easy → workflow；否则 → orchestrator

创建 `{work_dir}/{op_name}/autodev-task.md`（格式见 `references/contracts.md`）。

### Step 4：调度算子开发

根据 `dev_strategy` 调用执行层。调度方式见 `references/platform-dispatch.md`。

**调用时必须传递**：
- 任务参数（op_name、working_dir、resume_from_stage、complexity、category）
- known-limitations 内容（读取 `{project_root}/autodev/known-limitations.md`）
- 输出要求：完成后写 `.dev_result.json`（schema 见 `references/contracts.md`）
- 约束：持续写入 `dev-log.md`

**结果判定**（读取 `.dev_result.json`）：

| 情况 | 处理 |
|------|------|
| `status == SUCCESS` | 进入 Step 5（验证） |
| `status == BLOCKED` | `dev_result = blocked_reason` → 跳到 Step 6 |
| `status == FAILED` | `dev_result` 从 notes 提取 → 跳到 Step 6 |
| 文件不存在（超时/崩溃） | `dev_result = TIMEOUT` → 跳到 Step 6 |

### Step 5：独立验证

仅 Step 4 返回 SUCCESS 时执行。TIMEOUT / BLOCKED / FAILED 跳过此步。

```bash
python {scripts}/verify_op.py --csv {csv_path} --op {op_name} --run-test
```

| verify_status | dev_result |
|---|---|
| VERIFIED / TEST_PASS_NO_PRECISION | SUCCESS |
| TEST_FAIL | PRECISION |
| INCOMPLETE | TIMEOUT |

验证失败覆盖 Step 4 的 SUCCESS 判定。

### Step 6：知识提取

非致命步骤，任何子步骤失败不影响主流程。

**6a. 断裂点检测**（`enable_fracture == true` 且 `dev-log.md` 存在时）：

调用 `pypto-fracture-point-detector` skill（调度方式见 `references/platform-dispatch.md`），产出 `fracture-summary.json`。

**6b. 更新 known-limitations**（`fracture-summary.json` 存在时）：

```bash
python {scripts}/update_known_limitations.py \
  --known-limitations {project_root}/autodev/known-limitations.md \
  --fracture-summary {work_dir}/{op_name}/fracture-summary.json
```

**6c. 生成 issue 模板**（`fps_total > 0` 时）：

调用 `pypto-issue-creator` skill（本地模式），产出 `{work_dir}/{op_name}/issue-templates/`。

### Step 7：更新最终状态

读取 `fracture-summary.json`（若有）获取 `fps_total`、`fps_confirmed`，否则默认 0。

```bash
python {scripts}/update_op.py --csv {csv_path} --op {op_name} \
  --status {completed|failed} --dev-result {dev_result} \
  --fps-total {fps_total} --fps-confirmed {fps_confirmed} \
  --blocked-stage {仅失败时传} --last-strategy {orchestrator|workflow}
```

脚本失败时写入 `{work_dir}/{op_name}/autodev-status-error.log`，保留所有工件。

### Step 8：健康检查

```bash
python {scripts}/health_check.py --op {op_name} --op-dir {work_dir}/{op_name} \
  --status {completed|failed} --start-time {start_time_iso}
```

exit 1 → 将 issues 追加写入 `{project_root}/autodev/self-review.md`。

---

## 完成标准

Step 7 状态更新完成即视为本次执行完成。因业务原因提前退出（Step 1 活跃任务、Step 2 无候选）亦为正常完成。

## 特殊指令

查询进度、重试、手动指定、验证、产物检查、依赖管理见 `references/autodev-commands.md`。

## 参考文档

按需读取，不要一次性全部加载。

| 文件 | 何时读取 |
|------|---------|
| `references/contracts.md` | Step 4 调用执行层前，查看 .dev_result.json schema 和 prompt 模板 |
| `references/platform-dispatch.md` | Step 4/2c/6a 需要调度 agent/skill 时，查看当前平台的调度语法 |
| `references/category-rules.md` | Step 2a 推断算子类别时 |
| `references/complexity-rules.md` | Step 2a 推断算子复杂度时 |
| `references/csv-fields.md` | 需要理解 CSV 字段含义时（调试、特殊指令） |
| `references/autodev-commands.md` | 用户请求特殊指令时（查询进度、重试、手动指定等） |
| `references/known-limitations-seed.md` | 仅 Step 0 初始化时用于复制，正常流程不读此文件 |
