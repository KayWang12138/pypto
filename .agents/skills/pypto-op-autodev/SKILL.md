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

## 前置准备

### 运行时变量

执行前先确定以下变量：

| 变量 | 解析方式 | 典型值 |
|------|---------|--------|
| `{project_root}` | git 仓库根目录 | `/workspace/projects/code/pypto` |
| `{skill_dir}` | 本 SKILL.md 所在目录 | `{project_root}/.agents/skills/pypto-op-autodev` |
| `{scripts}` | `{skill_dir}/scripts/` | — |
| `{csv_path}` | `{project_root}/autodev/scan_results.csv` | — |
| `{work_dir}` | `{project_root}/autodev/custom` | — |

以下所有脚本命令中 `python {scripts}/xxx.py` 均指该绝对路径。

### 调度模式

执行前必须确定 `{dispatch_mode}`，后续调度步骤依此选择语法：

| 检测方式 | `{dispatch_mode}` | 调度语法 |
|---------|-------------------|---------|
| 拥有 `Agent` 工具（支持 `subagent_type`）和 `Skill` 工具 | `native` | `Agent(subagent_type=..., prompt=...)` / `Skill(...)` |
| 上条不满足，但 `which opencode` 返回有效路径 | `cli` | `Bash: opencode run --agent {agent} "{prompt}"` |

两种模式都不满足 → 终止并报错。

### 首次使用

Step 0 会自动初始化（创建 CSV、目录、复制种子知识）。CSV 为空时无算子可选，需先添加：

```bash
python {scripts}/add_op.py --csv {csv_path} --op {名称} --source manual \
  --complexity easy --category elementwise \
  --description "一句话需求描述" \
  [--requirement sources/{名称}.md] \
  [--reference "torch.nn.functional.xxx"]
```

`--description` 为必填项；`--requirement` 指向 `{project_root}/autodev/sources/` 下的需求文件（可选）；`--reference` 为参考实现位置（可选）。

---

## 执行流程

### Step 0：环境预检

```bash
python {scripts}/check_env.py --csv {csv_path} --work-dir {work_dir}
```

| exit code | 含义 | 处理 |
|-----------|------|------|
| 0 | 环境正常 | 继续 |
| 1 | NPU 不可用 | 降级继续，记录警告 |
| 2 | PyPTO/torch 不可用 | 终止 |

脚本同时负责初始化：CSV 不存在时创建 header，目录不存在时创建，`autodev/known-limitations.md` 不存在时从种子文件复制。

### Step 1：并发控制

```bash
python {scripts}/update_op.py --csv {csv_path} --reset-stale --timeout-hours 3
```

| exit code | 含义 | 处理 |
|-----------|------|------|
| 0 | 无活跃任务 | 继续 |
| 1 | 有活跃任务未超时 | 输出提示，退出 |
| 2+ | 脚本错误 | 终止 |

### Step 2：选择目标算子

#### 2a. 导入用户需求（可选）

若 `{project_root}/autodev/sources/requirements.md` 存在，读取需求，对每个算子执行：

```bash
python {scripts}/add_op.py --csv {csv_path} --op {op_name} --source manual \
  --complexity {推断} --category {推断} --description {描述} \
  [--requirement {需求文件}] [--reference {参考实现}]
```

复杂度和类别分别按 `references/complexity-rules.md` 和 `references/category-rules.md` 推断。若 requirements.md 中提供了 reference/description 信息，一并传入。`action: skipped` 表示已存在，跳过；脚本非 0 记录警告后继续。

#### 2b. 选择下一个算子

```bash
python {scripts}/select_next_op.py --csv {csv_path}
```

- exit 0 → 解析 JSON 获取 `selected.op_name`、`resume_from_stage`、`complexity`、`category`、`existing_artifacts`、`retry_strategy`（`resume_from_stage > 1` 为断点续跑）
- exit 1 → 无候选，进入 2c
- exit 2+ → 终止

#### 2c. 按需发现新算子

仅 `enable_discover == true` 时执行。

检查 `{work_dir}/.discovery_fail_count`：
- 计数 ≥ 3 → 输出"discover 连续失败达上限"，退出
- 否则 → 调度 discover agent，完成后重新执行 2b
  - 成功 → 重置计数，继续
  - 仍 exit 1 → 递增计数，退出

discover 调度：
- `native`：`Agent(subagent_type="pypto-op-discover", prompt="扫描 PyPTO 项目，发现 1 个新的待实现算子，更新 {csv_path}")`
- `cli`：`Bash: opencode run --agent pypto-op-discover "扫描 PyPTO 项目，发现 1 个新的待实现算子，更新 {csv_path}"`

`enable_discover == false` 且 exit 1 → 直接退出。

### Step 3：标记开始

```bash
python {scripts}/update_op.py --csv {csv_path} --op {op_name} --status in_progress
```

确定开发策略 `dev_strategy`：

| 优先级 | 条件 | 策略 |
|--------|------|------|
| 1 | 用户显式指定 | 按指定值 |
| 2 | 配置文件中 `dev_strategy` 为固定值 | 按配置值 |
| 3 | `dev_strategy == "auto"` 且上次用 orchestrator 失败 | `workflow` |
| 4 | `dev_strategy == "auto"` 且 `complexity == easy` | `workflow` |
| 5 | `dev_strategy == "auto"`，其余情况 | `orchestrator` |

创建 `{work_dir}/{op_name}/autodev-task.md`（格式见 `references/contracts.md`）。

### Step 4：调度算子开发

根据 `dev_strategy` 选择目标 agent，根据 `{dispatch_mode}` 选择调度语法。

#### agent 选择

| `dev_strategy` | agent |
|----------------|-------|
| `orchestrator` | `pypto-op-orchestrator` |
| `workflow` | `pypto-op-workflow` |

#### 调度语法

- `native`：`Agent(subagent_type="{agent}", prompt="{dev_prompt}")`
- `cli`：`Bash: opencode run --agent {agent} "{dev_prompt}"`

#### dev_prompt 构造

```
开发算子 {op_name}。
工作目录: {work_dir}/{op_name}/
从 Stage {resume_from_stage} 开始。
复杂度: {complexity}，类别: {category}。
{若 reference 非空: 参考实现: {reference}}
{若 description 非空: 需求描述: {description}}
{若 requirement 非空: 需求文件内容:
{读取 {project_root}/autodev/{requirement} 的内容}}

已知框架限制（开发时注意规避）:
{autodev/known-limitations.md 的内容}

完成后在工作目录写 .dev_result.json（schema 见 references/contracts.md）。
开发过程持续追加写入 dev-log.md。
```

#### 结果判定

读取 `{work_dir}/{op_name}/.dev_result.json`：

| 情况 | 处理 |
|------|------|
| `status == SUCCESS` | 进入 Step 5 |
| `status == BLOCKED` | `dev_result = blocked_reason` → 跳到 Step 6 |
| `status == FAILED` | `dev_result` 从 notes 提取 → 跳到 Step 6 |
| 文件不存在（超时/崩溃） | `dev_result = TIMEOUT` → 跳到 Step 6 |

### Step 5：独立验证

仅 Step 4 返回 SUCCESS 时执行，其余跳过。

```bash
python {scripts}/verify_op.py --csv {csv_path} --op {op_name} --run-test
```

| verify_status | dev_result 覆盖 |
|---|---|
| VERIFIED / TEST_PASS_NO_PRECISION | 维持 SUCCESS |
| TEST_FAIL | PRECISION |
| INCOMPLETE | TIMEOUT |

验证失败时覆盖 Step 4 的 SUCCESS 判定。

### Step 6：知识提取

非致命步骤，任何子步骤失败不影响主流程。

#### 6a. 断裂点检测

条件：`enable_fracture == true` 且 `dev-log.md` 存在。

- `native`：`Skill("pypto-fracture-point-detector")`
- `cli`：`Bash: opencode run "调用 pypto-fracture-point-detector skill，分析 {work_dir}/{op_name}/dev-log.md"`

产出 `{work_dir}/{op_name}/fracture-summary.json`。

#### 6b. 更新 known-limitations

条件：`fracture-summary.json` 存在。

```bash
python {scripts}/update_known_limitations.py \
  --known-limitations {project_root}/autodev/known-limitations.md \
  --fracture-summary {work_dir}/{op_name}/fracture-summary.json
```

#### 6c. 生成 issue 模板

条件：`fracture-summary.json` 中 `fps_total > 0`。

- `native`：`Skill("pypto-issue-creator")`
- `cli`：`Bash: opencode run "调用 pypto-issue-creator skill，为 {work_dir}/{op_name} 生成 issue 模板"`

产出 `{work_dir}/{op_name}/issue-templates/`。

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
  --status {completed|failed} --start-time {start_time_iso} \
  --strategy {orchestrator|workflow} [--enable-fracture]
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
| `references/contracts.md` | Step 3-4，查看 autodev-task.md 格式和 .dev_result.json schema |
| `references/platform-dispatch.md` | 需要查看 `cli` 模式完整调度语法时 |
| `references/category-rules.md` | Step 2a 推断算子类别时 |
| `references/complexity-rules.md` | Step 2a 推断算子复杂度时 |
| `references/csv-fields.md` | 需要理解 CSV 字段含义时（调试、特殊指令） |
| `references/autodev-commands.md` | 用户请求特殊指令时（查询进度、重试、手动指定等） |
| `references/known-limitations-seed.md` | 仅 Step 0 初始化时用于复制，正常流程不读此文件 |
