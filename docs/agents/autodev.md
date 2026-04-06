# 自动算子开发方案（autodev）

本文档描述 pypto-op-autodev 的设计：批量调度算子开发、管理生命周期、积累知识。

前置阅读：[README.md](README.md)（设计原则和分层架构）。

执行层设计见 [operator-development.md](operator-development.md)，autodev 不负责执行层的内部实现。

---

## 一、职责定义

autodev 是算子开发的**调度层**。它不直接开发算子——它决定开发哪个、用什么策略、开发完怎么处理。

**autodev 做的事**：

| 职责 | 说明 |
|------|------|
| 队列管理 | CSV 维护、优先级评分、选择下一个算子 |
| 并发控制 | 防止多个实例同时开发 |
| 策略选择 | 决定用 orchestrator 还是 workflow |
| 任务准备 | 组装参数、创建工作目录 |
| 调度执行 | 调用执行层，传入参数，等待结果 |
| 独立验证 | 不信任执行层的自我报告，独立跑测试确认 |
| 知识提取 | 从开发日志中提取框架限制，更新 known-limitations |
| 状态更新 | 根据结果更新 CSV |
| 健康检查 | 自检流程健康度 |

**autodev 不做的事**：
- 不实现阶段逻辑（那是执行层的事）
- 不解析执行层内部状态（只读 `.dev_result.json`）
- 不直接调用 skill 做开发工作
- 不感知执行层（orchestrator/workflow）的内部流程

---

## 二、触发方式

autodev 的流程定义在 SKILL.md 中。两种方式触发同一个流程：

| 方式 | 入口 | 脚本执行 | agent 调度 |
|------|------|---------|-----------|
| scheduler.py | `python scheduler.py` | subprocess 调用 | `opencode run --agent {name} "{prompt}"` |
| LLM 执行 SKILL.md | 用户说 "autodev" / "自动开发算子" | LLM 用 Bash 工具 | `Agent()` 工具（native 模式）或 `opencode run`（cli 模式） |

scheduler.py 是 SKILL.md 的机械执行器——按照 SKILL.md 的逻辑写成 Python 代码，不包含独立业务逻辑。两者行为一致。

### 调度模式

LLM 执行 SKILL.md 时，通过检测自身可用工具确定 `dispatch_mode`：

| 检测方式 | `dispatch_mode` | 调度语法 |
|---------|-----------------|---------|
| 拥有 `Agent` 工具（支持 `subagent_type`）和 `Skill` 工具 | `native` | `Agent(subagent_type=..., prompt=...)` / `Skill(...)` |
| 上条不满足，但 `which opencode` 返回有效路径 | `cli` | `Bash: opencode run --agent {agent} "{prompt}"` |

scheduler.py 始终使用 `opencode run` CLI。

### 配置

`scheduler.conf.json`（scheduler.py 使用，LLM 模式下由变量解析替代）：

```json
{
  "scripts_dir": ".agents/skills/pypto-op-autodev/scripts",
  "csv_path": "autodev/scan_results.csv",
  "custom_dir": "autodev/custom",
  "log_file": "autodev/logs/scheduler.log",
  "pid_file": "autodev/run/scheduler.pid",

  "stale_timeout_hours": 6,
  "dev_timeout_sec": 5400,
  "script_timeout_sec": 300,
  "verify_test_timeout_sec": 600,

  "opencode_bin": "opencode",
  "orchestrator_agent": "pypto-op-orchestrator",
  "workflow_agent": "",
  "discover_agent": "pypto-op-discover",
  "fracture_agent": "",

  "enable_env_check": true,
  "enable_verify_test": true,
  "dev_strategy": "auto",
  "enable_discover": false,
  "enable_fracture": true
}
```

---

## 三、执行流程

每次执行开发 1 个算子，完成后退出。所有 CSV 操作必须通过脚本完成，LLM 不直接操作 CSV。

### 变量解析

| 变量 | 含义 | 典型值 |
|------|------|--------|
| `{project_root}` | git 仓库根目录 | `/workspace/projects/code/pypto` |
| `{scripts}` | 脚本目录 | `{project_root}/.agents/skills/pypto-op-autodev/scripts` |
| `{csv_path}` | CSV 文件 | `{project_root}/autodev/scan_results.csv` |
| `{work_dir}` | 算子工件根目录 | `{project_root}/autodev/custom` |

### Step 0：环境预检 + 初始化

```bash
python {scripts}/check_env.py --csv {csv_path} --work-dir {work_dir}
```

职责：
1. 检查 PyPTO / torch 可用性
2. 检查 NPU 可用性（不可用则降级）
3. 若 `{csv_path}` 不存在 → 创建目录 + 写入 CSV header
4. 若 `{work_dir}` 不存在 → 创建目录
5. 若 `{project_root}/autodev/known-limitations.md` 不存在 → 从种子文件复制

退出码：0 正常 | 1 NPU 降级 | 2 PyPTO/torch 不可用，终止

### Step 1：并发控制

```bash
python {scripts}/update_op.py --csv {csv_path} --reset-stale --timeout-hours 3
```

退出码：0 继续 | 1 活跃任务未超时，退出 | 2+ 错误，终止

### Step 2：选择目标算子

**2a**（可选）：若 `{project_root}/autodev/sources/requirements.md` 存在，读取需求：

```bash
python {scripts}/add_op.py --csv {csv_path} \
  --op {op_name} --source manual --complexity {推断} --category {推断} \
  --description {描述} [--requirement {需求文件}] [--reference {参考实现}]
```

`--description` 为必填项。推断规则见 `references/complexity-rules.md` 和 `references/category-rules.md`。`action: skipped` 表示已存在，跳过；脚本非 0 记录警告后继续。

**2b**：选择下一个算子：

```bash
python {scripts}/select_next_op.py --csv {csv_path}
```

- exit 0 → 解析 JSON（含 op_name, resume_from_stage, complexity, category），继续
- exit 1 → 无候选，进入 2c
- exit 2+ → 终止

**2c**：按需发现新算子（`enable_discover == true` 时）：
- 检查 `{work_dir}/.discovery_fail_count`，≥ 3 → 退出
- 调用 `pypto-op-discover` agent 补充 1 个算子
- 再次执行 2b。成功→继续 | 仍无候选→递增计数，退出

### Step 3：标记开始 + 准备任务

```bash
python {scripts}/update_op.py --csv {csv_path} --op {op_name} --status in_progress
```

确定开发策略 `dev_strategy`（见 §四）。

创建 `{work_dir}/{op_name}/autodev-task.md`：
```markdown
# 开发任务
- op_name: {op_name}
- working_dir: {work_dir}/{op_name}/
- resume_from_stage: {resume_from_stage}
- complexity: {complexity}
- category: {category}
- reference: {参考实现位置，可为空}
- description: {一句话需求描述}
- requirement: {需求文件路径，可为空}
- start_time: {ISO 8601}
```

### Step 4：调度算子开发

根据 `dev_strategy` 选择目标 agent，根据 `dispatch_mode` 选择调度语法：

| `dev_strategy` | agent |
|----------------|-------|
| `orchestrator` | `pypto-op-orchestrator` |
| `workflow` | `pypto-op-workflow` |

调度语法：
- `native`：`Agent(subagent_type="{agent}", prompt="{dev_prompt}")`
- `cli`：`Bash: opencode run --agent {agent} "{dev_prompt}"`

dev_prompt 中必须包含：
- 任务参数（op_name、working_dir、resume_from_stage、complexity、category）
- reference / description / requirement 内容（若非空）
- known-limitations 内容（`{project_root}/autodev/known-limitations.md`）
- 输出要求：完成后写 `.dev_result.json`（schema 见 §五）
- 约束：持续写入 `dev-log.md`

**结果判定**：

| 情况 | 处理 |
|------|------|
| `.dev_result.json` 存在且 `status == SUCCESS` | 进入 Step 5 |
| `.dev_result.json` 存在且 `status == BLOCKED` | `dev_result = blocked_reason` → 跳到 Step 6 |
| `.dev_result.json` 存在且 `status == FAILED` | `dev_result` 从 notes 提取 → 跳到 Step 6 |
| `.dev_result.json` 不存在（超时/崩溃） | `dev_result = TIMEOUT` → 跳到 Step 6 |

### Step 5：独立验证

仅在 Step 4 返回 SUCCESS 时执行。

```bash
python {scripts}/verify_op.py --csv {csv_path} --op {op_name} --run-test
```

| verify_status | dev_result 覆盖 |
|---|---|
| VERIFIED / TEST_PASS_NO_PRECISION | 维持 SUCCESS |
| TEST_FAIL | PRECISION |
| INCOMPLETE | TIMEOUT |

验证失败覆盖 Step 4 的 SUCCESS。

### Step 6：知识提取

非致命步骤，失败不影响主流程。

**6a 断裂点检测**（`enable_fracture == true` 且 `dev-log.md` 存在）：
- `native`：`Skill("pypto-fracture-point-detector")`
- `cli`：`Bash: opencode run "调用 pypto-fracture-point-detector skill，分析 {work_dir}/{op_name}/dev-log.md"`
- 产出 `fracture-summary.json`

**6b 更新 known-limitations**（`fracture-summary.json` 存在时）：

```bash
python {scripts}/update_known_limitations.py \
  --known-limitations {project_root}/autodev/known-limitations.md \
  --fracture-summary {work_dir}/{op_name}/fracture-summary.json
```

逻辑：提取 confidence=high 的断裂点 → 与已有条目去重 → 追加新条目。

**6c 生成 issue 模板**（`fps_total > 0` 时）：
- `native`：`Skill("pypto-issue-creator")`
- `cli`：`Bash: opencode run "调用 pypto-issue-creator skill，为 {work_dir}/{op_name} 生成 issue 模板"`
- 产出 `{work_dir}/{op_name}/issue-templates/`

### Step 7：更新最终状态

```bash
python {scripts}/update_op.py --csv {csv_path} --op {op_name} \
  --status {completed|failed} --dev-result {dev_result} \
  --fps-total {fps_total} --fps-confirmed {fps_confirmed} \
  --blocked-stage {仅失败时} --last-strategy {orchestrator|workflow}
```

脚本失败 → 写入 `autodev-status-error.log`，保留工件。

### Step 8：健康检查

```bash
python {scripts}/health_check.py --op {op_name} \
  --op-dir {work_dir}/{op_name} \
  --status {completed|failed} --start-time {start_time} \
  --strategy {orchestrator|workflow} [--enable-fracture]
```

exit 1（needs_review）→ 将 issues 追加到 `{project_root}/autodev/self-review.md`。

### 完成标准

Step 7 状态更新完成即视为本次执行完成。因业务原因提前退出（Step 1 活跃任务、Step 2 无候选）亦为正常完成。

---

## 四、开发策略选择

### 决策优先级

| 优先级 | 条件 | 策略 |
|--------|------|------|
| 1 | 用户显式指定 | 按指定值 |
| 2 | 配置文件中 `dev_strategy` 为固定值 | 按配置值 |
| 3 | `dev_strategy == "auto"` 且上次用 orchestrator 失败 | `workflow` |
| 4 | `dev_strategy == "auto"` 且 `complexity == easy` | `workflow` |
| 5 | `dev_strategy == "auto"`，其余情况 | `orchestrator` |

### 设计理由

- orchestrator 的优势是状态管理和 subagent 分工，适合复杂算子
- workflow 简单直接，不依赖模型的复杂指令遵循能力
- "失败后换策略"是自适应机制——如果 orchestrator 模式的模型不能正确调度 subagent，下次自动切到 workflow

---

## 五、Contracts

本节定义 autodev 层的调用规范。这些是 autodev 对被调用方的要求，被调用方自身的定义中不预置这些概念。

### 5.1 .dev_result.json

autodev 在调用执行层时，通过 prompt 要求写入此文件。

```json
{
  "status": "SUCCESS | FAILED | BLOCKED",
  "blocked_reason": "BLOCKED_IMPL | BLOCKED_API | BLOCKED_GOLDEN | BLOCKED_DESIGN | BLOCKED_ACCURACY | BLOCKED_ENVIRONMENT | null",
  "precision_result": "PASS | FAIL | null",
  "completed_stages": [1, 2, 3, 4, 5],
  "artifacts": ["spec.md", "relu_impl.py", "test_relu.py"],
  "notes": ""
}
```

autodev 的成功检测逻辑：
- 文件存在且 `status == "SUCCESS"` → 成功
- 文件存在且 `status != "SUCCESS"` → 失败
- 文件不存在 → 超时/崩溃，视为 TIMEOUT

### 5.2 状态机

```
pending ──→ in_progress ──→ completed
                 │
                 ├──→ failed
                 │
                 └──→ pending (--reset-stale 超时重置 / --reset 手动重置)
```

非法转换（脚本拒绝）：`pending→completed`、`completed→in_progress`、`failed→in_progress`。

### 5.3 CSV 字段

详见 `references/csv-fields.md`。关键字段：

| 字段 | 说明 |
|------|------|
| `requirement` | 需求文件路径（如 `sources/gated_delta_net.md`） |
| `reference` | 参考实现位置（如 `torch.nn.functional.softmax` 或 `transformers/models/xxx.py::ClassName`） |
| `description` | 一句话需求描述（必填） |
| `verify_status` | 独立验证结果（Step 5 写入） |
| `last_strategy` | 上次使用的开发策略（`orchestrator` / `workflow`） |
| `blocked_stage` | 阻塞发生的阶段号（仅失败时） |

### 5.4 add_op.py 输出

```json
{"action": "created", "op_name": "tanh_linear"}
{"action": "skipped", "op_name": "softmax", "reason": "already_in_csv"}
{"action": "skipped", "op_name": "relu",    "reason": "complete_artifacts_exist"}
```

### 5.5 退出码约定

所有脚本统一：0 成功 | 1 业务级拒绝（非错误）| 2+ 脚本级错误。

### 5.6 特殊指令

查询进度、重试失败算子、手动指定算子、验证、产物检查等操作见 `references/autodev-commands.md`。

---

## 六、Memory 机制

### 6.1 分层

autodev 的运行时数据按生命周期分层：

| 层级 | 内容 | 位置 |
|------|------|------|
| Long-term | known-limitations.md | `autodev/` |
| Medium-term | scan_results.csv, self-review.md | `autodev/` |
| Short-term | dev-log.md, .dev_result.json, fracture-summary.json | `autodev/custom/{op}/` |
| Artifacts | spec.md, impl.py, test.py 等 | `autodev/custom/{op}/` |
| Requirements | 需求文件 | `autodev/sources/{op_name}.md` |

### 6.2 种子机制

Long-term memory 支持种子初始化：
- 种子文件：`{skill_dir}/references/known-limitations-seed.md`（随代码仓库管理）
- 运行时文件：`autodev/known-limitations.md`（生成目录中，持续积累）
- Step 0 中，运行时文件不存在时从种子复制

### 6.3 知识反馈闭环

```
开发算子 → dev-log.md
    → Step 6a: fracture-point-detector → fracture-summary.json
    → Step 6b: update_known_limitations.py → autodev/known-limitations.md 追加新发现
    → 下次 Step 4 调用执行层时，通过 prompt 传入 known-limitations
    → api-explorer / developer skill 参考已知限制 → 规避已知问题
```

### 6.4 需求上下文链

```
模型分析报告 → 提取算子 + 来源上下文
    → autodev/sources/{op_name}.md（需求文件，详细上下文）
    → scan_results.csv（requirement 字段指向需求文件，reference/description 为轻量摘要）
    → Step 4 dev_prompt 中传入 reference + description + 需求文件内容
    → intent-understanding 有充分上下文，不再靠名字猜测
```

---

## 七、文件组织

### 代码（随 git 管理）

```
.agents/skills/pypto-op-autodev/
├── SKILL.md                              ← 流程定义（唯一权威）
├── scripts/                              ← 可执行代码
│   ├── scheduler.py                      ← 触发层 runner（SKILL.md 的机械执行器）
│   ├── scheduler.conf.json               ← runner 配置
│   ├── data/csv_ops.py                   ← CSV 读写公共库
│   ├── check_env.py                      ← Step 0
│   ├── add_op.py                         ← Step 2a
│   ├── select_next_op.py                 ← Step 2b
│   ├── update_op.py                      ← Step 1/3/7
│   ├── verify_op.py                      ← Step 5
│   ├── update_known_limitations.py       ← Step 6b
│   ├── check_artifacts.py                ← 特殊指令
│   ├── get_progress.py                   ← 特殊指令
│   ├── health_check.py                   ← Step 8
│   └── tests/                            ← 单元测试（80 tests）
└── references/                           ← 按需加载的文档
    ├── contracts.md                      ← 调用规范快速参考
    ├── platform-dispatch.md              ← 平台调度语法参考
    ├── csv-fields.md                     ← CSV 字段定义
    ├── autodev-commands.md               ← 特殊指令
    ├── category-rules.md                 ← 分类推断规则
    ├── complexity-rules.md               ← 复杂度推断规则
    └── known-limitations-seed.md         ← 种子知识
```

### 运行时数据（生成目录，可安全删除重建）

```
autodev/
├── scan_results.csv                      ← medium-term memory
├── known-limitations.md                  ← long-term memory（从种子初始化）
├── self-review.md                        ← medium-term memory
├── sources/                              ← 需求文件（可选）
│   ├── gated_delta_net.md
│   └── multi_head_latent_attention.md
├── custom/{op_name}/                     ← 算子工件 + short-term memory
├── logs/
└── run/
```

`autodev/` 是纯运行时数据目录，不含代码。删除后由 Step 0 自动重建（sources/ 除外，需用户提供）。
