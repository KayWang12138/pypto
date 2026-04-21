# PyPTO 9-Agent 系统（Claude Code 版）

本目录包含从 `.opencode/agents/` 移植的 **8 个 sub-agent 定义**。团队结构和阶段归属与 opencode 设置一致，有两处不同：**所有 kernel 执行在远程 NPU 服务器上运行**，且 **Lead 位于 `CLAUDE.md`**（项目根目录的主 agent 文件），因为只有主 Claude Code agent 能通过 Task tool 派遣 sub-agent。

## Agent 名册

| Agent | 位置 | 阶段 | 职责 |
|---|---|---|---|
| **Lead**（主 agent） | `/CLAUDE.md` | 0–6 | **主编排器。** 通过 Task tool 派遣 sub-agent，执行 Gate 门禁。从不编写代码。 |
| `planning` | `.claude/agents/planning.md` | 0 | 将用户请求 → SPEC.md、API_REPORT.md、计划种子。 |
| `algorithm` | `.claude/agents/algorithm.md` | 1 | 用 PyTorch/NumPy 编写 PyPTO 友好的 golden.py。零隐式转置。 |
| `architecture` | `.claude/agents/architecture.md` | 1–2 边界 | DESIGN.md：tiling、loop 结构、内存计划、性能目标。 |
| `design` | `.claude/agents/design.md` | 2 | 模块分解、契约、暂存文件布局。 |
| `coding` | `.claude/agents/coding.md` | 3–5 | 每次派遣实现**一个**暂存文件。从不调试。 |
| `verification` | `.claude/agents/verification.md` | 2（模块化 golden + 对抗测试套件）、3–5 gate、6 回归 | 仅裁判角色。构建 `<op>_golden_modular.py` 和 `adversarial_runner.py`（支持前缀评估 `--up-to-module k`），在 NPU 上运行检查，输出 pass/fail + 失败类别。 |
| `debug` | `.claude/agents/debug.md` | GATE 失败时 | 仅调查角色。提出补丁建议。从不编辑生产代码。 |
| `optimization` | `.claude/agents/optimization.md` | 6 | GATE 4 之后的性能调优。3 个阶段：前端 → 泳道 → 核内。 |

### 为什么 Lead 位于 `CLAUDE.md`（而非 `.claude/agents/lead.md`）

Claude Code 的 Task tool — 派遣 sub-agent 的唯一方式 — **仅对主 agent 可用**。Sub-agent 不能进一步派遣 sub-agent。因为 Lead 的全部职责是编排（在各阶段和 gate 间调度其他 8 个 sub-agent），Lead **必须**是主 agent。

`CLAUDE.md` 在会话启动时被 Claude Code 自动加载，因此本项目中每个会话都以主 Claude agent 已配置为 Lead 开始。不存在 `lead.md` sub-agent 文件 — 创建一个会导致 Lead 无法正常工作（没有 Task tool）。

## 执行模型（与 opencode 不同）

所有 opencode agent 假定本地执行。**这些 Claude Code agent 将每个 kernel 运行通过 NPU 服务器执行。**

- **Mac 端（Claude Code）：** 代码生成、编辑、静态检查（`validate_kernel_structure`、`extract_pypto_calls.py`）、计划编排、日志分析。
- **NPU 端（远程服务器）：** kernel 执行、`detailed_tensor_compare`、布局检查、精度测试、性能测量。

### 主要机制：`Run <file> on npu:<N>` 提示

Claude Code 内置 NPU 执行机制。当 agent 需要运行 kernel 文件时，发出自然语言提示：

```
Run <file> on npu:<N>
```

示例：
- `Run custom/relu/relu_module1.py on npu:8`
- `Run custom/matmul/test_e2e.py on npu:0`

Claude Code 自动上传项目状态、在指定 NPU 设备上运行文件，并内联返回 stdout/stderr。**这是所有 agent 执行单文件 kernel 的主要机制。**

每个算子固定绑定一个 NPU 设备编号，记录在 `custom/plan/<op>.md` 的 `execution.npu_device` 字段中。Lead 在首次用户交互时询问此信息。

### 备选方案：批处理脚本

对于多文件测试运行（跨目录的 pytest）、布局检查、性能分析或不适合单文件运行的日志聚合，提供了辅助脚本：

| 脚本 | 用途 |
|---|---|
| `./scripts/npu_sync.sh` | 手动 rsync 项目到 NPU |
| `./scripts/npu_test.sh <op>` | 同步 + 对算子目录运行 pytest + 布局检查 + 拉取日志 |
| `./scripts/npu_run.sh "<cmd>"` | 在 NPU 上执行任意远程命令 |
| `./scripts/npu_shell.sh` | NPU 交互式 shell，进入项目目录 |

Golden 脚本（纯 PyTorch/NumPy）仍在 Mac 本地运行 — 它们不是 NPU kernel。

## 一次性设置（使用这些 agent 之前）

1. **SSH 配置：** 在 `~/.ssh/config` 中添加条目：
   ```
   Host npu
       HostName 1.95.147.197
       User y00960395
       ServerAliveInterval 60
       IdentityFile ~/.ssh/id_ed25519
   ```
2. **SSH 密钥认证：** `ssh-copy-id npu`（使 Claude Code 能非交互式运行 ssh/rsync）
3. **测试连通性：** `ssh npu "hostname"` 应返回 NPU 主机名且不提示输入密码。
4. **创建辅助脚本：** 参见 `scripts/README.md`（或让 Lead Agent 在首次运行时生成）。
5. **验证项目在 NPU 上存在：** 运行 `./scripts/npu_sync.sh` 后执行 `ssh npu "ls ~/pypto-multi/"`。

如果缺少任何步骤，Lead 将暂停并要求完成设置。

## 派遣流程（Phase 3 示例）

```
User → lead（询问算子名称、形状、性能目标、NPU 设备）
     → planning（Phase 0）→ lead
     → algorithm（Phase 1）→ lead
     → architecture（Phase 1–2）→ lead
     → design（Phase 2 分解 + module_interfaces.yaml）→ lead
     → verification（Phase 2 脚手架：构建 <op>_golden_modular.py +
                     adversarial_suite.json + adversarial_runner.py；
                     运行 GATE A.5 组合验证 + GATE B --self-test）→ lead
     → 对每个模块 M_k：
         coding（编写 M_k 文件）→ lead
         verification（Run <file> on npu:<N>
                      + Run adversarial_runner.py --up-to-module k on npu:<N>）→ lead
             如果通过 → 提交，进入下一模块
             如果失败 → debug（提出补丁；使用 evaluation_report.json
                              中的 failing_module_boundary）→ lead
                       coding（应用补丁）→ lead
                       verification（在 NPU 上重新运行）→ ...
     → optimization（Phase 6，NPU 性能调优 + 完整对抗扫描回归）
```

### 模块化 golden + 前缀评估（Phase 2/3）

Verification 产出两个支持逐模块调试的制品（移植自 Joshua evaluator 设计）：

1. **模块化 torch golden**（`custom/<op>/eval/<op>_golden_modular.py`）— 从 `module_interfaces.yaml` 派生的纯 torch 逐模块参考。`golden_composed(*primary_inputs)` 必须在 `composition_verification`（GATE A.5，GATE 2 的子 gate）的每个 `(seed, shape)` 组合上与 `<op>_golden(*primary_inputs)` 在容差范围内匹配。
2. **对抗测试套件 + 前缀评估运行器**（`adversarial_suite.json`、`test_inputs.py`、`adversarial_runner.py`）— 每个级别 L1（结构）/ L2（精度）/ L3（边界）/ L4（回归）/ L5（对抗）≥ 2 个用例。运行器的 `--up-to-module k` 标志将实现中的模块 [1..k] 与模块化 golden 中的模块 [k+1..N] 组合，为 Phase 3 提供逐模块正确性信号，即使完整 kernel 尚未实现（GATE B，GATE 2 的子 gate）。

每次 Phase 3 verification 派遣除了模块自身的入口测试外，还会运行前缀评估。`evaluation_report.json` 中的 `status`、`first_failure.failing_module_boundary` 和 `first_failure.failure_category` 字段直接传递给 Lead 转发给 @debug — 失败模块边界缩小了修复范围，并区分了模块内 bug 和输出契约不匹配。

### 信息隔离

Verification 返回给 Lead 的报告在离开评估工作区前经过 `_sanitize` 处理 — 它们不包含 golden Tensor 值、golden 源码片段或原始输入 Tensor 内容。@debug 和 @coding 从 `SPEC.md`、`module_interfaces.yaml` 和用户提供的 golden 独立推导正确性。

## Claude Code 如何调用这些 agent

在 Claude Code 会话中，用户直接与主 agent = Lead 对话。Lead 通过 Task tool 派遣 sub-agent，使用 `subagent_type` 指定 sub-agent 名称：

```
Task(subagent_type="verification", prompt="Run GATE 3 checks on custom/relu/relu_module1.py using npu:8")
```

Claude Code 读取 sub-agent 的前置信息，加载其系统提示，赋予声明的工具，并以隔离上下文运行。

## 与 .opencode/agents/ 的差异

- 移除了 `mode: subagent`（Claude Code 原生支持主 agent + sub-agent）
- **Lead 现在是主 agent**，其系统提示位于 `/CLAUDE.md`（会话启动时自动加载）。其他 8 个 agent 是本目录下的 sub-agent。
- `tools:` 从布尔对象转换为 Claude Code 工具名称（Read、Write、Edit、Bash 等）。Lead 额外拥有 `Task`（仅主 agent 可用）。
- 在 Lead（CLAUDE.md 中）以及 `verification`、`debug`、`optimization`、`coding` 中添加了显式的 NPU 执行规则
- 主要执行机制是 `Run <file> on npu:<N>` 提示（Claude Code 内置），脚本作为批处理备选
- 每个算子在其计划文件中固定绑定一个 NPU 设备编号（`execution.npu_device`）
- `coding` 和 `debug` 现在显式禁止本地 kernel 执行
- `verification` 新增了 `infra` 失败类别，用于 SSH/rsync 问题
