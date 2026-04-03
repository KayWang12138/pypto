# PyPTO 算子开发流程 Lint/Hook 增强方案

> 版本: v2.1 | 日期: 2026-04-03 | 状态: 已实现 (MVP)

---

## 一、背景与需求

### 1.1 问题陈述

PyPTO 算子开发流程以 `pypto-op-orchestrator`（OpenCode，7 阶段状态机）和 `pypto-op-workflow`（Claude Code，手动串联 Skill）为入口，涵盖需求理解、API 探索、Golden 生成、设计、实现、精度修复、性能调优七个阶段。

当前流程中所有质量保障手段均为 **prompt 级约束**（写在 agent/skill markdown 中），无任何确定性验证机制。核心问题：

1. **LLM 遗忘约束**: 长上下文中 LLM 容易遗忘 `execution-constraints.md` 中的框架规则，导致生成代码缺少 `@pypto.frontend.jit`、缺少 `[:]` 写回等
2. **测试结果误判**: 三态判定（`[PRECISION_PASS]`/`[PRECISION_FAIL]`/运行失败）由 LLM 解读 stdout/stderr，可能误判导致错误的阶段流转
3. **门禁可跳过**: 阶段门禁（工件完整性检查）写在 orchestrator prompt 中，LLM 可能跳过检查直接推进
4. **备份易遗忘**: Stage 6 精度修复要求修改前备份，LLM 可能忘记导致无法回滚

### 1.2 目标

- 将关键约束从 "prompt 建议" 升级为 "确定性脚本验证"
- 通过 Claude Code hooks 和 OpenCode plugin hooks 实现自动触发
- 两平台共享同一套验证逻辑，hook 层只做平台适配
- 对齐仓内已有的 `pypto-skill-reviewer` 三件套模式（rules.json + validate + score）
- 新增规则仅需注册，无需修改 hook 配置

### 1.3 约束

- 仅覆盖算子开发流程（orchestrator/workflow 入口），不碰仓级 pre-commit
- 验证逻辑必须独立于平台，核心检查为纯 Python 脚本
- 不增加外部依赖（仅用 Python 标准库 + ast 模块）

---

## 二、现状分析

### 2.1 当前架构

| 层级 | 组件 | 位置 |
|------|------|------|
| 编排层 | `pypto-op-orchestrator` (7 阶段状态机) | `.opencode/agents/` |
| 子 Agent | analyst / developer / perftuner / code-merge | `.opencode/agents/` |
| Skill 层 | 26 个专业 Skill (op-develop, precision-debugger 等) | `.agents/skills/` |
| 约束层 | `execution-constraints.md` (8 项检查清单 + API 约束) | Skill references |
| 质量审查 | `pypto-skill-reviewer` (48 规则 9 维度) | `.agents/skills/` |

### 2.2 现有质量保障手段（全部为 prompt 级）

| 保障点 | 机制 | 位置 | 问题 |
|--------|------|------|------|
| **阶段门禁** | orchestrator prompt 描述门禁条件 | `pypto-op-orchestrator.md` L139-162 | LLM 判断工件是否完整，无确定性校验 |
| **三态判定** | developer prompt 规定判定规则 | `pypto-op-developer.md` L84-91 | LLM 解析 stdout/stderr，可能误判 |
| **执行约束** | 8 项检查清单写在 references | `execution-constraints.md` L150-159 | 靠 LLM "写代码前检查"，长上下文下易遗忘 |
| **首跑前预检** | developer prompt 描述 3 项预检 | `pypto-op-developer.md` L66-72 | LLM 可能跳过预检直接跑 |
| **回滚规则** | Stage 6 备份/回滚规则写在 prompt | `pypto-op-developer.md` L134-141 | 无强制约束，LLM 可能忘记备份 |

### 2.3 已有模式参考：`pypto-skill-reviewer`

仓内已有的 `pypto-skill-reviewer` 提供了成熟的质量检查模式：

```
pypto-skill-reviewer/
├── references/
│   ├── rules.json          # 48 规则注册表 (9 维度, 4 严重级)
│   └── scoring-spec.md     # 评分算法规范
├── scripts/
│   ├── validate_skill.py   # 26 条静态规则检查器
│   └── score_findings.py   # 通用评分引擎
└── templates/
    └── report-template.md
```

核心设计模式：
- **规则注册表**: JSON 定义规则 ID、严重级、维度、检查策略
- **Finding 协议**: `{rule_id, severity, dimension, status, message, evidence}`
- **评分引擎**: 维度加权 + S0 一票否决 + 等级映射 (A/B/C/D/F)
- **状态优先级**: FAIL(4) > WARN(3) > PASS(2) > SKIP(1)

本方案完全对齐此模式。

---

## 三、双平台 Hook 机制调研

### 3.1 Claude Code Hooks

Claude Code 提供 24 种 hook 事件，与算子开发流程相关的：

| 事件 | 触发时机 | 能力 | 适用场景 |
|------|---------|------|---------|
| `PreToolUse` | 工具调用前 | allow/deny/ask + 修改输入 (`updatedInput`) | 拦截危险操作、门禁检查 |
| `PostToolUse` | 工具调用后 | block + 注入上下文 (`additionalContext`) | 代码 lint、测试结果解析 |
| `Stop` | agent 准备结束 | block 阻止结束 | 交付物完整性检查 |
| `UserPromptSubmit` | 用户输入时 | 注入上下文 | 约束注入 |

**Hook 类型**: `command`（Shell 脚本）、`http`、`prompt`（LLM 判断）、`agent`（完整 agent）

**配置位置**: `.claude/settings.json` 或 `.agents/settings.json`

**决策协议**:
- 退出码 `0` = 放行，`2` = 拦截
- JSON stdout 返回 `hookSpecificOutput` 结构体，支持 `permissionDecision`、`additionalContext`、`updatedInput`

### 3.2 OpenCode Plugin Hooks

OpenCode 通过 `@opencode-ai/plugin` SDK 提供 TypeScript plugin 接口：

| Hook | 触发时机 | 能力 | 对标 Claude |
|------|---------|------|------------|
| `tool.execute.before` | 工具调用前 | 修改 `output.args` | `PreToolUse` |
| `tool.execute.after` | 工具调用后 | 修改 `output.output` | `PostToolUse` |
| `permission.ask` | 权限请求时 | 设置 `output.status` = deny | `PermissionRequest` |
| `chat.message` | 消息接收时 | 修改消息内容 | `UserPromptSubmit` |
| `experimental.chat.system.transform` | 系统提示构建时 | 修改 system prompt | 约束注入 |

**关键发现**: OpenCode plugin SDK 的 hook 能力与 Claude Code 是**对等的**，可以实现真正的双平台统一方案，而非 "一个用 hook 一个靠 prompt"。

### 3.3 双平台 Hook 能力对照

| 能力 | Claude Code | OpenCode Plugin | 兼容方案 |
|------|-------------|-----------------|---------|
| 工具执行前拦截 | `PreToolUse` (exit 2 deny) | `tool.execute.before` (改 args) | 统一验证脚本，各自调用 |
| 工具执行后验证 | `PostToolUse` (additionalContext) | `tool.execute.after` (改 output) | 统一验证脚本，各自注入 |
| 权限拦截 | `PreToolUse` (deny) | `permission.ask` (deny) | 统一验证脚本判定 |
| 结束门禁 | `Stop` hook (block) | **无直接对应** | OpenCode 用 prompt fallback |
| Shell 环境 | 无 | `shell.env` | OpenCode 独有 |

**核心策略**: 验证逻辑封装为 `pypto_op_lint.py` 统一 CLI，两平台 hook 层各自调用同一个 CLI，只做输入输出格式适配。

---

## 四、方案设计

### 4.1 整体架构

```
                    ┌─────────────────────────────────────────┐
                    │        rules.json (Single Source)        │
                    │  规则注册表: ID, 阶段, 严重级, 检查函数    │
                    └───────────────┬─────────────────────────┘
                                    │
                    ┌───────────────▼─────────────────────────┐
                    │       pypto_op_lint.py (统一 CLI)         │
                    │  子命令:                                  │
                    │    lint-impl   → 检查 {op}_impl.py       │
                    │    lint-golden → 检查 {op}_golden.py     │
                    │    lint-test   → 检查 test_{op}.py       │
                    │    check-gate  → 检查阶段门禁             │
                    │    parse-result→ 解析测试输出              │
                    │    check-backup→ 检查 Stage6 备份         │
                    │    report      → 全量检查 + 评分报告       │
                    └───────┬───────────────┬─────────────────┘
                            │               │
             ┌──────────────┘               └──────────────┐
             ▼                                              ▼
  ┌─────────────────────┐                    ┌─────────────────────┐
  │   Claude Code        │                    │   OpenCode           │
  │   settings.json      │                    │   plugin.ts          │
  │   hooks:             │                    │   hooks:             │
  │   - PreToolUse       │                    │   - tool.execute     │
  │   - PostToolUse      │                    │     .before/.after   │
  │   - Stop             │                    │   - permission.ask   │
  └─────────────────────┘                    └─────────────────────┘
        ↕ 调用同一个                               ↕ 调用同一个
   pypto_op_lint.py                          pypto_op_lint.py
```

### 4.2 命名规范

| 层级 | 命名规则 | 示例 |
|------|---------|------|
| Skill 目录 | `pypto-op-lint` | `.agents/skills/pypto-op-lint/` |
| 规则 ID | `OL{NN}` (Op Lint) | `OL01`, `OL02`, ... |
| 维度 ID | `D1`-`D5` (5 个维度) | `D1: 框架约束合规`, `D2: 工件完整性`, ... |
| 严重级 | `S0`/`S1`/`S2`/`S3` (复用现有体系) | `S0` = 一票否决 |
| CLI 子命令 | `kebab-case` 动词 | `lint-impl`, `check-gate`, `parse-result` |
| Hook 脚本 | `hook_{platform}_{event}.sh` | `hook_claude_post_edit.sh` |
| 检查函数 | `check_ol{nn}` | `check_ol01(ctx)` |

---

## 五、规则注册表设计

### 5.1 维度定义

| 维度 | 名称 | 权重 | 覆盖范围 |
|------|------|------|---------|
| `D1` | 框架约束合规 | 35% | `@jit`、写回方式、kernel return、Tile 配置等 |
| `D2` | 工件完整性 | 25% | 各阶段门禁、文件存在、内容结构 |
| `D3` | 三文件分离 | 15% | golden/impl/test 职责隔离 |
| `D4` | 测试规范 | 15% | 三态标记、assert_allclose、环境设置 |
| `D5` | 流程合规 | 10% | 备份、状态文件、覆盖策略 |

### 5.2 完整规则表

#### D1: 框架约束合规（impl 文件）

| 规则 | 严重级 | 适用阶段 | 检查目标 | 检查策略 |
|------|--------|---------|---------|---------|
| `OL01` | S0 | 5,6,7 | kernel 函数必须有 `@pypto.frontend.jit` 装饰器 | AST: 遍历 FunctionDef，检查 decorator 含 `pypto.frontend.jit` |
| `OL02` | S0 | 5,6,7 | 输出写回必须用 `[:]`/`move()`/`assemble()`，禁止 `out = expr` 替代写回 | AST: kernel 内 Assign 目标为参数名时报违规 |
| `OL03` | S0 | 5,6,7 | kernel 函数不能有 `return` 语句 | AST: jit 函数体内无 Return 节点 |
| `OL04` | S1 | 5,6,7 | 必须调用 `set_vec_tile_shapes` 或 `set_cube_tile_shapes` | AST: kernel 内存在对应 Call 节点 |
| `OL05` | S1 | 5,6,7 | kernel Tensor 参数必须有 `pypto.Tensor(...)` 类型注解 | AST: 检查 FunctionDef.args.annotation |
| `OL06` | S2 | 5,6,7 | kernel 内禁用 Python 原生 `min()`/`max()` | AST: Call.func 为 Name("min"/"max") 时报违规 |
| `OL07` | S2 | 5,6,7 | 必须 `import pypto` | AST: 模块级 Import 检查 |
| `OL08` | S2 | 5,6,7 | wrapper 函数必须导出且以 `_wrapper` 结尾 | AST: 模块级函数名匹配 `*_wrapper` |

#### D2: 工件完整性（阶段门禁）

| 规则 | 严重级 | 适用阶段 | 检查目标 | 检查策略 |
|------|--------|---------|---------|---------|
| `OL09` | S0 | 2 | 进入 Stage 2 需 `spec.md` 存在且含算子名 | 文件存在 + 内容 grep |
| `OL10` | S0 | 3 | 进入 Stage 3 需 `api_report.md` 存在 | 文件存在 |
| `OL11` | S0 | 4 | 进入 Stage 4 需 `{op}_golden.py` 可导入 | `python -c "import ..."` 执行 |
| `OL12` | S0 | 5 | 进入 Stage 5 需 `design.md` 含 API 映射章节 | 文件存在 + heading grep |
| `OL13` | S1 | 5 | 生成文件三件套完整: `{op}_impl.py`, `test_{op}.py`, `README.md` | 文件存在 |
| `OL14` | S1 | 7 | 进入 Stage 7 需精度通过 | `.orchestrator_state.json` 状态检查 |

#### D3: 三文件分离

| 规则 | 严重级 | 适用阶段 | 检查目标 | 检查策略 |
|------|--------|---------|---------|---------|
| `OL15` | S0 | 3 | golden 文件禁止 `import pypto` | AST: Import 检查 |
| `OL16` | S1 | 5 | impl 文件禁止包含 golden 计算逻辑（不导入 golden） | AST: 无 `from {op}_golden import` |
| `OL17` | S1 | 5 | test 文件只做 import+调用+对比，不含 kernel 实现 | AST: 无 `@pypto.frontend.jit` 装饰器 |
| `OL18` | S2 | 5 | test 文件必须从 impl 和 golden 分别导入 | AST: 检查 ImportFrom |

#### D4: 测试规范

| 规则 | 严重级 | 适用阶段 | 检查目标 | 检查策略 |
|------|--------|---------|---------|---------|
| `OL19` | S0 | 5 | test 必须使用 `assert_allclose`，禁止 `assert max_diff <` | AST: Call 检查 + 反模式检测 |
| `OL20` | S1 | 5 | test 必须有 `TILE_FWK_DEVICE_ID` 环境变量处理 | grep `TILE_FWK_DEVICE_ID` |
| `OL21` | S1 | 5 | test 必须有 Level 0 和 Level 1 两级测试 | AST: FunctionDef 名称匹配 |
| `OL22` | S2 | 5 | test 应设置 `torch.manual_seed` 保证可复现 | grep `manual_seed` |

#### D5: 流程合规

| 规则 | 严重级 | 适用阶段 | 检查目标 | 检查策略 |
|------|--------|---------|---------|---------|
| `OL23` | S1 | 6 | 修改 impl 前 `history_version/` 下必须有对应备份 | 文件存在性检查 |
| `OL24` | S1 | * | `.orchestrator_state.json` 结构合法 | JSON schema 校验 |
| `OL25` | S2 | 6 | 备份文件命名符合 `{op}_impl_s6_attempt{N}.py` | glob 模式匹配 |

### 5.3 rules.json 完整定义

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "version": "1.0.0",
  "meta": {
    "title": "PyPTO Op Lint Rules",
    "description": "算子开发流程确定性检查规则注册表，与 pypto-skill-reviewer 同构",
    "sources": [
      {
        "id": "C1",
        "title": "execution-constraints.md",
        "path": "../pypto-op-develop/references/execution-constraints.md"
      },
      {
        "id": "C2",
        "title": "impl-template.py",
        "path": "../pypto-op-develop/references/impl-template.py"
      },
      {
        "id": "C3",
        "title": "test-template.py",
        "path": "../pypto-op-develop/references/test-template.py"
      },
      {
        "id": "C4",
        "title": "pypto-op-orchestrator.md",
        "path": "../../.opencode/agents/pypto-op-orchestrator.md"
      },
      {
        "id": "C5",
        "title": "pypto-op-developer.md",
        "path": "../../.opencode/agents/pypto-op-developer.md"
      }
    ]
  },
  "dimensions": {
    "D1": { "name": "框架约束合规", "weight": 35 },
    "D2": { "name": "工件完整性", "weight": 25 },
    "D3": { "name": "三文件分离", "weight": 15 },
    "D4": { "name": "测试规范", "weight": 15 },
    "D5": { "name": "流程合规", "weight": 10 }
  },
  "severity_deductions": {
    "S0": 20,
    "S1": 10,
    "S2": 5,
    "S3": 2
  },
  "rules": [
    {
      "id": "OL01",
      "severity": "S0",
      "type": "static",
      "dimension": "D1",
      "stages": [5, 6, 7],
      "target": "impl",
      "rule": "kernel 函数必须有 @pypto.frontend.jit 装饰器",
      "source": ["C1"],
      "check_function": "check_ol01",
      "check_strategy": "AST: 遍历所有 FunctionDef，至少一个 decorator 含 pypto.frontend.jit"
    },
    {
      "id": "OL02",
      "severity": "S0",
      "type": "static",
      "dimension": "D1",
      "stages": [5, 6, 7],
      "target": "impl",
      "rule": "输出写回必须用 [:]/move()/assemble()，禁止 out = expr 替代写回",
      "source": ["C1"],
      "check_function": "check_ol02",
      "check_strategy": "AST: 在 jit kernel 内，检查对输出参数的赋值是否使用 Subscript ([:]) 或 Call (move/assemble)"
    },
    {
      "id": "OL03",
      "severity": "S0",
      "type": "static",
      "dimension": "D1",
      "stages": [5, 6, 7],
      "target": "impl",
      "rule": "kernel 函数不能有 return 语句",
      "source": ["C1"],
      "check_function": "check_ol03",
      "check_strategy": "AST: 在 jit 装饰的 FunctionDef 体内，不存在 Return 节点"
    },
    {
      "id": "OL04",
      "severity": "S1",
      "type": "static",
      "dimension": "D1",
      "stages": [5, 6, 7],
      "target": "impl",
      "rule": "必须调用 set_vec_tile_shapes 或 set_cube_tile_shapes",
      "source": ["C1"],
      "check_function": "check_ol04",
      "check_strategy": "AST: kernel 函数体内存在 Call 节点，func 属性匹配 pypto.set_vec_tile_shapes 或 pypto.set_cube_tile_shapes"
    },
    {
      "id": "OL05",
      "severity": "S1",
      "type": "static",
      "dimension": "D1",
      "stages": [5, 6, 7],
      "target": "impl",
      "rule": "kernel Tensor 参数必须有 pypto.Tensor(...) 类型注解",
      "source": ["C1"],
      "check_function": "check_ol05",
      "check_strategy": "AST: jit FunctionDef 的参数 annotation 中包含 pypto.Tensor"
    },
    {
      "id": "OL06",
      "severity": "S2",
      "type": "static",
      "dimension": "D1",
      "stages": [5, 6, 7],
      "target": "impl",
      "rule": "kernel 内禁用 Python 原生 min()/max()",
      "source": ["C1"],
      "check_function": "check_ol06",
      "check_strategy": "AST: jit 函数体内 Call.func 为 Name('min') 或 Name('max') 时报违规"
    },
    {
      "id": "OL07",
      "severity": "S2",
      "type": "static",
      "dimension": "D1",
      "stages": [5, 6, 7],
      "target": "impl",
      "rule": "必须 import pypto",
      "source": ["C1"],
      "check_function": "check_ol07",
      "check_strategy": "AST: 模块级 Import 或 ImportFrom 含 pypto"
    },
    {
      "id": "OL08",
      "severity": "S2",
      "type": "static",
      "dimension": "D1",
      "stages": [5, 6, 7],
      "target": "impl",
      "rule": "wrapper 函数必须导出且以 _wrapper 结尾",
      "source": ["C2"],
      "check_function": "check_ol08",
      "check_strategy": "AST: 模块级存在至少一个 FunctionDef，名称以 _wrapper 结尾"
    },
    {
      "id": "OL09",
      "severity": "S0",
      "type": "static",
      "dimension": "D2",
      "stages": [2],
      "target": "gate",
      "rule": "进入 Stage 2 需 spec.md 存在且含算子名",
      "source": ["C4"],
      "check_function": "check_ol09",
      "check_strategy": "文件存在性 + 内容 grep 算子名关键字"
    },
    {
      "id": "OL10",
      "severity": "S0",
      "type": "static",
      "dimension": "D2",
      "stages": [3],
      "target": "gate",
      "rule": "进入 Stage 3 需 api_report.md 存在",
      "source": ["C4"],
      "check_function": "check_ol10",
      "check_strategy": "文件存在性检查"
    },
    {
      "id": "OL11",
      "severity": "S0",
      "type": "static",
      "dimension": "D2",
      "stages": [4],
      "target": "gate",
      "rule": "进入 Stage 4 需 {op}_golden.py 可导入",
      "source": ["C4"],
      "check_function": "check_ol11",
      "check_strategy": "python -c 'from {op}_golden import {op}_golden' 执行成功"
    },
    {
      "id": "OL12",
      "severity": "S0",
      "type": "static",
      "dimension": "D2",
      "stages": [5],
      "target": "gate",
      "rule": "进入 Stage 5 需 design.md 含 API 映射章节",
      "source": ["C4"],
      "check_function": "check_ol12",
      "check_strategy": "文件存在 + 内容 grep 'API' 或 'api_report' 相关 heading"
    },
    {
      "id": "OL13",
      "severity": "S1",
      "type": "static",
      "dimension": "D2",
      "stages": [5],
      "target": "gate",
      "rule": "生成文件三件套完整: {op}_impl.py, test_{op}.py, README.md",
      "source": ["C5"],
      "check_function": "check_ol13",
      "check_strategy": "三个文件存在性检查"
    },
    {
      "id": "OL14",
      "severity": "S1",
      "type": "static",
      "dimension": "D2",
      "stages": [7],
      "target": "gate",
      "rule": "进入 Stage 7 需精度通过",
      "source": ["C4"],
      "check_function": "check_ol14",
      "check_strategy": ".orchestrator_state.json 中 stage_status['5'] 或 stage_status['6'] 为 completed"
    },
    {
      "id": "OL15",
      "severity": "S0",
      "type": "static",
      "dimension": "D3",
      "stages": [3],
      "target": "golden",
      "rule": "golden 文件禁止 import pypto",
      "source": ["C4"],
      "check_function": "check_ol15",
      "check_strategy": "AST: Import/ImportFrom 中不含 pypto"
    },
    {
      "id": "OL16",
      "severity": "S1",
      "type": "static",
      "dimension": "D3",
      "stages": [5],
      "target": "impl",
      "rule": "impl 文件不应导入 golden 模块",
      "source": ["C4"],
      "check_function": "check_ol16",
      "check_strategy": "AST: ImportFrom 中不含 {op}_golden"
    },
    {
      "id": "OL17",
      "severity": "S1",
      "type": "static",
      "dimension": "D3",
      "stages": [5],
      "target": "test",
      "rule": "test 文件不应包含 kernel 实现代码",
      "source": ["C3"],
      "check_function": "check_ol17",
      "check_strategy": "AST: 不存在 @pypto.frontend.jit 装饰器"
    },
    {
      "id": "OL18",
      "severity": "S2",
      "type": "static",
      "dimension": "D3",
      "stages": [5],
      "target": "test",
      "rule": "test 文件必须从 impl 和 golden 分别导入",
      "source": ["C3"],
      "check_function": "check_ol18",
      "check_strategy": "AST: 存在 from {op}_impl import 和 from {op}_golden import"
    },
    {
      "id": "OL19",
      "severity": "S0",
      "type": "static",
      "dimension": "D4",
      "stages": [5],
      "target": "test",
      "rule": "test 必须使用 assert_allclose，禁止手写 assert max_diff",
      "source": ["C3"],
      "check_function": "check_ol19",
      "check_strategy": "AST: 存在 assert_allclose Call + 不存在 assert ... max_diff 模式"
    },
    {
      "id": "OL20",
      "severity": "S1",
      "type": "static",
      "dimension": "D4",
      "stages": [5],
      "target": "test",
      "rule": "test 必须处理 TILE_FWK_DEVICE_ID 环境变量",
      "source": ["C3"],
      "check_function": "check_ol20",
      "check_strategy": "源码 grep 包含 TILE_FWK_DEVICE_ID"
    },
    {
      "id": "OL21",
      "severity": "S1",
      "type": "static",
      "dimension": "D4",
      "stages": [5],
      "target": "test",
      "rule": "test 必须有 Level 0 和 Level 1 两级测试函数",
      "source": ["C3"],
      "check_function": "check_ol21",
      "check_strategy": "AST: FunctionDef 名称含 level0 和 level1（大小写不敏感）"
    },
    {
      "id": "OL22",
      "severity": "S2",
      "type": "static",
      "dimension": "D4",
      "stages": [5],
      "target": "test",
      "rule": "test 应设置 torch.manual_seed 保证可复现",
      "source": ["C3"],
      "check_function": "check_ol22",
      "check_strategy": "源码 grep 包含 manual_seed"
    },
    {
      "id": "OL23",
      "severity": "S1",
      "type": "static",
      "dimension": "D5",
      "stages": [6],
      "target": "flow",
      "rule": "修改 impl 前 history_version/ 下必须有对应备份",
      "source": ["C5"],
      "check_function": "check_ol23",
      "check_strategy": "检查 history_version/ 目录存在且含 {op}_impl_s6_attempt*.py 文件"
    },
    {
      "id": "OL24",
      "severity": "S1",
      "type": "static",
      "dimension": "D5",
      "stages": [1, 2, 3, 4, 5, 6, 7],
      "target": "flow",
      "rule": ".orchestrator_state.json 结构合法",
      "source": ["C4"],
      "check_function": "check_ol24",
      "check_strategy": "JSON 可解析 + 必需字段存在 (operator_name, current_stage, stage_status)"
    },
    {
      "id": "OL25",
      "severity": "S2",
      "type": "static",
      "dimension": "D5",
      "stages": [6],
      "target": "flow",
      "rule": "备份文件命名符合 {op}_impl_s6_attempt{N}.py",
      "source": ["C5"],
      "check_function": "check_ol25",
      "check_strategy": "glob 匹配 history_version/{op}_impl_s6_attempt*.py"
    }
  ]
}
```

---

## 六、统一 CLI 设计

### 6.1 接口定义

```
pypto_op_lint.py <子命令> [选项]

子命令:
  lint-impl      检查 {op}_impl.py 的框架约束 (D1)
  lint-golden    检查 {op}_golden.py 的纯净性 (D3)
  lint-test      检查 test_{op}.py 的测试规范 (D3+D4)
  check-gate     检查阶段门禁 (D2)
  check-backup   检查 Stage 6 备份 (D5)
  parse-result   确定性解析测试输出 (D4)
  report         全量检查 + 评分报告

公共选项:
  --op-dir PATH        算子工作目录 (必需, 如 custom/softmax/)
  --stage N            当前阶段 (1-7, 部分子命令必需)
  --rules PATH         rules.json 路径 (默认: 同目录 ../references/rules.json)
  --format json|text   输出格式 (默认: json)
  --quiet              仅输出 exit code

parse-result 专有选项:
  --stdout TEXT        测试 stdout 内容
  --stderr TEXT        测试 stderr 内容
  --exitcode N         测试退出码
```

### 6.2 输出协议

所有子命令统一输出 JSON，格式与 `pypto-skill-reviewer` 的 Finding 协议兼容：

```json
{
  "command": "lint-impl",
  "op_dir": "custom/softmax",
  "stage": 5,
  "passed": false,
  "findings": [
    {
      "rule_id": "OL01",
      "severity": "S0",
      "dimension": "D1",
      "type": "static",
      "status": "FAIL",
      "message": "未找到 @pypto.frontend.jit 装饰器",
      "evidence": {
        "file": "softmax_impl.py",
        "line": 0,
        "snippet": ""
      }
    },
    {
      "rule_id": "OL04",
      "severity": "S1",
      "dimension": "D1",
      "type": "static",
      "status": "PASS",
      "message": "找到 set_vec_tile_shapes 调用",
      "evidence": {
        "file": "softmax_impl.py",
        "line": 23,
        "snippet": "pypto.set_vec_tile_shapes(1, 64, 128)"
      }
    }
  ],
  "summary": {
    "pass": 7,
    "fail": 1,
    "warn": 0,
    "skip": 0,
    "has_s0_fail": true
  }
}
```

`parse-result` 子命令输出：

```json
{
  "command": "parse-result",
  "verdict": "precision_pass",
  "detail": "stdout 包含 [PRECISION_PASS] 标记",
  "sub_type": null
}
```

verdict 取值枚举：

| verdict | 含义 | 下一步 |
|---------|------|--------|
| `precision_pass` | 精度通过 | → Stage 7 |
| `precision_fail` | 精度失败 | → Stage 6 |
| `runtime_error` | 运行失败 (通用) | → Stage 5 内重试 |
| `import_error` | Import 失败 | → 环境检查 |
| `compile_error` | 编译失败 | → Stage 5 内重试 |
| `aicore_error` | AiCore 错误 | → aicore-error-locator |
| `shape_error` | Shape 不匹配 | → Stage 5 内重试 |
| `unknown` | 无法判定 | → 需人工检查 |

### 6.3 退出码协议

| 退出码 | 含义 | Hook 行为 |
|--------|------|----------|
| `0` | 全部通过 | 放行 |
| `1` | 有 FAIL (非 S0) | 放行 + 注入 additionalContext 提示修正 |
| `2` | 有 S0 FAIL | 拦截 (Claude deny / OpenCode 改写命令) |

### 6.4 检查函数注册机制

```python
"""pypto_op_lint.py 核心结构"""

import ast
import json
import os
import re
import sys
import glob as glob_mod
import argparse
from dataclasses import dataclass, field, asdict
from typing import Callable, Optional

# ─── Finding 数据结构 ───

@dataclass
class Evidence:
    file: str = ""
    line: int = 0
    snippet: str = ""

@dataclass
class Finding:
    rule_id: str = ""
    severity: str = ""
    dimension: str = ""
    type: str = "static"
    status: str = "SKIP"      # PASS | FAIL | WARN | SKIP
    message: str = ""
    evidence: Evidence = field(default_factory=Evidence)

# ─── 检查上下文 ───

@dataclass
class CheckContext:
    op_dir: str               # 算子工作目录
    op_name: str              # 算子名（从目录名推断）
    stage: int                # 当前阶段
    rules_data: dict          # rules.json 内容
    _ast_cache: dict = field(default_factory=dict, repr=False)

    def file_path(self, filename: str) -> str:
        return os.path.join(self.op_dir, filename)

    def file_exists(self, filename: str) -> bool:
        return os.path.isfile(self.file_path(filename))

    def read_file(self, filename: str) -> str:
        path = self.file_path(filename)
        if not os.path.isfile(path):
            return ""
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    def parse_file(self, filename: str) -> Optional[ast.Module]:
        if filename in self._ast_cache:
            return self._ast_cache[filename]
        source = self.read_file(filename)
        if not source:
            return None
        try:
            tree = ast.parse(source, filename=filename)
        except SyntaxError:
            return None
        self._ast_cache[filename] = tree
        return tree

    def make_finding(self, rule_id: str, status: str, message: str,
                     file: str = "", line: int = 0, snippet: str = "") -> Finding:
        rule = self._get_rule(rule_id)
        return Finding(
            rule_id=rule_id,
            severity=rule.get("severity", "S2"),
            dimension=rule.get("dimension", ""),
            type=rule.get("type", "static"),
            status=status,
            message=message,
            evidence=Evidence(file=file, line=line, snippet=snippet),
        )

    def pass_finding(self, rule_id: str, message: str, file: str = "",
                     line: int = 0, snippet: str = "") -> Finding:
        return self.make_finding(rule_id, "PASS", message, file, line, snippet)

    def fail_finding(self, rule_id: str, message: str, file: str = "",
                     line: int = 0, snippet: str = "") -> Finding:
        return self.make_finding(rule_id, "FAIL", message, file, line, snippet)

    def skip_finding(self, rule_id: str, message: str = "不适用") -> Finding:
        return self.make_finding(rule_id, "SKIP", message)

    def _get_rule(self, rule_id: str) -> dict:
        for r in self.rules_data.get("rules", []):
            if r["id"] == rule_id:
                return r
        return {}

# ─── 检查函数注册表 ───

CHECKERS: dict[str, Callable[[CheckContext], Finding]] = {}

def register(rule_id: str):
    """装饰器：将检查函数注册到规则 ID"""
    def decorator(fn: Callable[[CheckContext], Finding]):
        CHECKERS[rule_id] = fn
        return fn
    return decorator
```

新增一条规则的完整步骤：
1. 在 `rules.json` 的 `rules` 数组末尾追加规则定义
2. 在 `pypto_op_lint.py` 中添加 `@register("OL26") def check_ol26(ctx): ...`
3. 完毕——无需修改任何 hook 配置

---

## 七、平台 Hook 适配

### 7.1 Claude Code — `settings.json` hooks 配置

```json
{
  "hooks": {
    "PostToolUse": [
      {
        "matcher": "Write|Edit|MultiEdit",
        "hooks": [
          {
            "type": "command",
            "command": "bash \"$CLAUDE_PROJECT_DIR\"/.agents/skills/pypto-op-lint/scripts/hook_claude_post_edit.sh",
            "timeout": 20,
            "statusMessage": "pypto-op-lint: impl/golden/test 检查中..."
          }
        ]
      },
      {
        "matcher": "Bash",
        "hooks": [
          {
            "type": "command",
            "command": "bash \"$CLAUDE_PROJECT_DIR\"/.agents/skills/pypto-op-lint/scripts/hook_claude_post_bash.sh",
            "timeout": 10,
            "statusMessage": "pypto-op-lint: 测试结果解析..."
          }
        ]
      }
    ],
    "PreToolUse": [
      {
        "matcher": "Write|Edit|MultiEdit",
        "hooks": [
          {
            "type": "command",
            "command": "bash \"$CLAUDE_PROJECT_DIR\"/.agents/skills/pypto-op-lint/scripts/hook_claude_pre_edit.sh",
            "timeout": 10
          }
        ]
      }
    ],
    "Stop": [
      {
        "matcher": "",
        "hooks": [
          {
            "type": "command",
            "command": "bash \"$CLAUDE_PROJECT_DIR\"/.agents/skills/pypto-op-lint/scripts/hook_claude_stop.sh",
            "timeout": 30,
            "statusMessage": "pypto-op-lint: 交付门禁检查..."
          }
        ]
      }
    ]
  }
}
```

### 7.2 Claude Hook 适配脚本

#### `hook_claude_post_edit.sh`

```bash
#!/usr/bin/env bash
# Claude PostToolUse[Write|Edit] 适配器
# 文件写入后自动 lint，违规时注入 additionalContext 要求修正
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LINT="python3 ${SCRIPT_DIR}/pypto_op_lint.py"

INPUT=$(cat)
FILE_PATH=$(echo "$INPUT" | python3 -c "
import sys, json
data = json.load(sys.stdin)
print(data.get('tool_input', {}).get('file_path', ''))
" 2>/dev/null || echo "")

# 按文件类型分派子命令
case "$FILE_PATH" in
  *_impl.py)   CMD="lint-impl" ;;
  *_golden.py) CMD="lint-golden" ;;
  *test_*.py)  CMD="lint-test" ;;
  *)           exit 0 ;;
esac

OP_DIR=$(dirname "$FILE_PATH")

RESULT=$($LINT "$CMD" --op-dir "$OP_DIR" --format json 2>/dev/null) || true
PASSED=$(echo "$RESULT" | python3 -c "
import sys, json; print(json.load(sys.stdin).get('passed', True))
" 2>/dev/null || echo "True")

if [ "$PASSED" = "False" ] || [ "$PASSED" = "false" ]; then
  VIOLATIONS=$(echo "$RESULT" | python3 -c "
import sys, json
data = json.load(sys.stdin)
fails = [f for f in data.get('findings', []) if f.get('status') == 'FAIL']
for f in fails:
    print(f'  [{f[\"rule_id\"]}][{f[\"severity\"]}] {f[\"message\"]}')
" 2>/dev/null || echo "  lint check failed")

  python3 -c "
import json
print(json.dumps({
    'hookSpecificOutput': {
        'hookEventName': 'PostToolUse',
        'additionalContext': '''[pypto-op-lint] 以下规则违规，请立即修正后重新写入文件：
${VIOLATIONS}

参考 .agents/skills/pypto-op-develop/references/execution-constraints.md'''
    }
}))
"
fi
exit 0
```

#### `hook_claude_post_bash.sh`

```bash
#!/usr/bin/env bash
# Claude PostToolUse[Bash] 适配器
# 检测到测试执行后，确定性解析三态结果
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LINT="python3 ${SCRIPT_DIR}/pypto_op_lint.py"

INPUT=$(cat)
COMMAND=$(echo "$INPUT" | python3 -c "
import sys, json; print(json.load(sys.stdin).get('tool_input', {}).get('command', ''))
" 2>/dev/null || echo "")

# 只处理 test_*.py 执行
if ! echo "$COMMAND" | grep -qE 'python3?\s+.*test_\w+\.py'; then
  exit 0
fi

STDOUT=$(echo "$INPUT" | python3 -c "
import sys, json; print(json.load(sys.stdin).get('tool_result', {}).get('stdout', ''))
" 2>/dev/null || echo "")
STDERR=$(echo "$INPUT" | python3 -c "
import sys, json; print(json.load(sys.stdin).get('tool_result', {}).get('stderr', ''))
" 2>/dev/null || echo "")
EXIT_CODE=$(echo "$INPUT" | python3 -c "
import sys, json; print(json.load(sys.stdin).get('tool_result', {}).get('exit_code', 0))
" 2>/dev/null || echo "0")

RESULT=$($LINT parse-result \
  --stdout "$STDOUT" --stderr "$STDERR" --exitcode "$EXIT_CODE" \
  --format json 2>/dev/null) || exit 0

VERDICT=$(echo "$RESULT" | python3 -c "
import sys, json; print(json.load(sys.stdin).get('verdict', ''))
" 2>/dev/null || echo "")
DETAIL=$(echo "$RESULT" | python3 -c "
import sys, json; print(json.load(sys.stdin).get('detail', ''))
" 2>/dev/null || echo "")

if [ -n "$VERDICT" ]; then
  python3 -c "
import json
print(json.dumps({
    'hookSpecificOutput': {
        'hookEventName': 'PostToolUse',
        'additionalContext': '[pypto-op-lint parse-result] 确定性判定: ${VERDICT}。${DETAIL}。请以此结果为准，不要自行解读测试输出。'
    }
}))
"
fi
exit 0
```

#### `hook_claude_pre_edit.sh`

```bash
#!/usr/bin/env bash
# Claude PreToolUse[Write|Edit] 适配器
# Stage 6 编辑 impl 前检查是否已备份
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LINT="python3 ${SCRIPT_DIR}/pypto_op_lint.py"

INPUT=$(cat)
FILE_PATH=$(echo "$INPUT" | python3 -c "
import sys, json
data = json.load(sys.stdin)
print(data.get('tool_input', {}).get('file_path', ''))
" 2>/dev/null || echo "")

# 只检查 impl 文件
case "$FILE_PATH" in
  *_impl.py) ;;
  *) exit 0 ;;
esac

OP_DIR=$(dirname "$FILE_PATH")
STATE_FILE="$OP_DIR/.orchestrator_state.json"

# 仅 Stage 6 需要备份检查
if [ ! -f "$STATE_FILE" ]; then
  exit 0
fi

CURRENT_STAGE=$(python3 -c "
import sys, json
with open('$STATE_FILE') as f:
    print(json.load(f).get('current_stage', 0))
" 2>/dev/null || echo "0")

if [ "$CURRENT_STAGE" != "6" ]; then
  exit 0
fi

RESULT=$($LINT check-backup --op-dir "$OP_DIR" --format json 2>/dev/null) || exit 0
PASSED=$(echo "$RESULT" | python3 -c "
import sys, json; print(json.load(sys.stdin).get('passed', True))
" 2>/dev/null || echo "True")

if [ "$PASSED" = "False" ] || [ "$PASSED" = "false" ]; then
  python3 -c "
import json
print(json.dumps({
    'hookSpecificOutput': {
        'hookEventName': 'PreToolUse',
        'permissionDecision': 'deny',
        'permissionDecisionReason': '[pypto-op-lint] Stage 6 修改 impl 前必须先备份到 history_version/。请先执行备份再修改。'
    }
}))
"
fi
exit 0
```

#### `hook_claude_stop.sh`

```bash
#!/usr/bin/env bash
# Claude Stop hook 适配器
# agent 结束前检查交付物完整性
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LINT="python3 ${SCRIPT_DIR}/pypto_op_lint.py"

# 尝试从环境或最近的算子目录推断 op-dir
# Stop hook 没有 tool_input，需要从工作目录搜索
CWD=$(cat | python3 -c "
import sys, json; print(json.load(sys.stdin).get('cwd', '.'))
" 2>/dev/null || echo ".")

# 查找最近修改的算子目录
OP_DIR=""
for state_file in $(find "$CWD" -maxdepth 3 -name ".orchestrator_state.json" -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -1 | cut -d' ' -f2-); do
  OP_DIR=$(dirname "$state_file")
  break
done

if [ -z "$OP_DIR" ] || [ ! -d "$OP_DIR" ]; then
  exit 0  # 无算子目录，放行
fi

RESULT=$($LINT report --op-dir "$OP_DIR" --format json 2>/dev/null) || exit 0
PASSED=$(echo "$RESULT" | python3 -c "
import sys, json; print(json.load(sys.stdin).get('passed', True))
" 2>/dev/null || echo "True")
HAS_S0=$(echo "$RESULT" | python3 -c "
import sys, json; print(json.load(sys.stdin).get('summary', {}).get('has_s0_fail', False))
" 2>/dev/null || echo "False")

if [ "$HAS_S0" = "True" ] || [ "$HAS_S0" = "true" ]; then
  VIOLATIONS=$(echo "$RESULT" | python3 -c "
import sys, json
data = json.load(sys.stdin)
fails = [f for f in data.get('findings', []) if f.get('status') == 'FAIL']
for f in fails:
    print(f'  [{f[\"rule_id\"]}][{f[\"severity\"]}] {f[\"message\"]}')
" 2>/dev/null || echo "  check failed")

  python3 -c "
import json
print(json.dumps({
    'hookSpecificOutput': {
        'hookEventName': 'Stop',
        'decision': 'block',
        'reason': '''[pypto-op-lint] 交付门禁未通过，存在 S0 级违规，不能结束：
${VIOLATIONS}'''
    }
}))
"
fi
exit 0
```

### 7.3 OpenCode Plugin 适配

```typescript
// .opencode/plugin.ts
import type { Plugin } from "@opencode-ai/plugin";

export const id = "pypto-op-lint";

export const server: Plugin = async (input) => {
  const $ = input.$;
  const lint = `python3 ${input.directory}/.agents/skills/pypto-op-lint/scripts/pypto_op_lint.py`;

  return {
    // ── 对标 Claude PostToolUse[Write|Edit] ──
    "tool.execute.after": async ({ tool, args }, output) => {
      const filePath: string = args?.file_path ?? args?.path ?? "";
      let cmd = "";
      if (filePath.endsWith("_impl.py"))        cmd = "lint-impl";
      else if (filePath.endsWith("_golden.py"))  cmd = "lint-golden";
      else if (/test_\w+\.py$/.test(filePath))   cmd = "lint-test";
      else return;

      const opDir = filePath.substring(0, filePath.lastIndexOf("/"));
      try {
        const raw = await $`${lint} ${cmd} --op-dir ${opDir} --format json`.text();
        const result = JSON.parse(raw);
        if (!result.passed) {
          const fails = result.findings
            .filter((f: any) => f.status === "FAIL")
            .map((f: any) => `[${f.rule_id}][${f.severity}] ${f.message}`)
            .join("\n");
          output.output += `\n\n[pypto-op-lint] 框架约束违规，请修正：\n${fails}`;
        }
      } catch { /* lint 异常不阻塞主流程 */ }

      // ── 对标 Claude PostToolUse[Bash]: 测试结果解析 ──
      if (tool === "bash" || tool === "shell") {
        const command: string = args?.command ?? "";
        if (/python3?\s+.*test_\w+\.py/.test(command)) {
          try {
            const stdout = output.output ?? "";
            const stderr = output.metadata?.stderr ?? "";
            const exitCode = output.metadata?.exit_code ?? 0;
            const raw = await $`${lint} parse-result --stdout ${JSON.stringify(stdout)} --stderr ${JSON.stringify(stderr)} --exitcode ${exitCode} --format json`.text();
            const result = JSON.parse(raw);
            if (result.verdict) {
              output.output += `\n\n[pypto-op-lint parse-result] 确定性判定: ${result.verdict}。${result.detail}。请以此结果为准。`;
            }
          } catch { /* 解析异常不阻塞 */ }
        }
      }
    },

    // ── 对标 Claude PreToolUse[Bash]: 测试前门禁 ──
    "tool.execute.before": async ({ tool }, output) => {
      const cmd: string = output.args?.command ?? "";
      const testMatch = cmd.match(/python3?\s+(.*?)test_(\w+)\.py/);
      if (!testMatch) return;

      const pathPrefix = testMatch[1].trim();
      const op = testMatch[2];
      const opDir = pathPrefix || `custom/${op}`;
      try {
        const raw = await $`${lint} check-gate --op-dir ${opDir} --stage 5 --format json`.text();
        const result = JSON.parse(raw);
        if (!result.passed && result.summary?.has_s0_fail) {
          const reason = result.findings
            .filter((f: any) => f.status === "FAIL")
            .map((f: any) => f.message)
            .join("; ");
          output.args = {
            command: `echo '[pypto-op-lint] 门禁未通过: ${reason}' >&2 && exit 2`
          };
        }
      } catch { /* 检查异常不阻塞 */ }
    },

    // ── 对标 Claude PreToolUse[Edit]: Stage 6 备份检查 ──
    "permission.ask": async (input, output) => {
      const filePath = (input as any).input?.file_path ?? "";
      if (!filePath.endsWith("_impl.py")) return;

      const opDir = filePath.substring(0, filePath.lastIndexOf("/"));
      const stateFile = `${opDir}/.orchestrator_state.json`;
      try {
        const stateRaw = await $`cat ${stateFile}`.text();
        const state = JSON.parse(stateRaw);
        if (state.current_stage !== 6) return;

        const raw = await $`${lint} check-backup --op-dir ${opDir} --format json`.text();
        const result = JSON.parse(raw);
        if (!result.passed) {
          output.status = "deny";
        }
      } catch { /* 检查异常不阻塞 */ }
    },
  };
};
```

---

## 八、目录结构总览

```
.agents/skills/pypto-op-lint/
├── SKILL.md                              # Skill 定义
├── PROPOSAL.md                           # 本文档
├── references/
│   ├── rules.json                        # 25 条规则注册表 (Single Source of Truth)
│   └── scoring-spec.md                   # 评分规范 (同 pypto-skill-reviewer 格式)
├── scripts/
│   ├── pypto_op_lint.py                  # 统一 CLI 入口 (全部检查逻辑)
│   ├── score_findings.py                 # 评分引擎 (复用 pypto-skill-reviewer)
│   ├── hook_claude_post_edit.sh          # Claude PostToolUse[Write|Edit] 适配
│   ├── hook_claude_post_bash.sh          # Claude PostToolUse[Bash] 适配
│   ├── hook_claude_pre_edit.sh           # Claude PreToolUse[Write|Edit] 适配
│   └── hook_claude_stop.sh              # Claude Stop 适配
└── templates/
    └── report-template.md                # 全量报告模板

.opencode/
├── plugin.ts                             # OpenCode plugin 适配 (新增)
├── package.json                          # 已有，无需修改

.agents/settings.json                     # 已有，补充 hooks 配置
    或
.claude/settings.json                     # Claude Code hooks 配置
```

---

## 九、与纯 prompt 约束的对比

| 维度 | 当前 (纯 prompt) | Hook + Lint 方案 |
|------|-----------------|----------------|
| **强制力** | LLM 自觉遵循，可遗忘 | hook 硬拦截，无法跳过 |
| **确定性** | LLM 判断，有误判风险 | AST/regex 确定性检查 |
| **反馈速度** | 等到测试跑完才发现问题 | 写完 impl 立即反馈违规 |
| **跨平台** | 每个平台各写一份 prompt | 共享 Python 脚本，hook 层做薄适配 |
| **可维护性** | 规则散落在多个 markdown 中 | 集中在 rules.json + 检查函数 |
| **可扩展性** | 加规则需改多处 prompt | 加规则只需 rules.json + @register |
| **可观测性** | 无日志 | hook 执行有日志可追溯 |
| **评分能力** | 无 | 维度加权 + S0 否决 + 等级报告 |

---

## 十、实施路线

| 阶段 | 交付物 | 包含规则 | 预期收益 |
|------|--------|---------|---------|
| **Phase 1** | `pypto_op_lint.py` 骨架 + `lint-impl` + `parse-result` + rules.json (OL01-OL08) + Claude hook 配置 | D1 全部 + parse-result | 消除最高频的框架约束违规和测试结果误判 |
| **Phase 2** | `check-gate` + `lint-test` + OL09-OL14 + OL19-OL22 | D2 + D4 | 保障阶段门禁和测试规范 |
| **Phase 3** | `lint-golden` + OL15-OL18 + OpenCode plugin.ts | D3 | 保障三文件分离 + OpenCode 平台支持 |
| **Phase 4** | `check-backup` + OL23-OL25 + `report` + `Stop` hook + 评分引擎 | D5 + 全量报告 | 完整流程合规 + 质量报告 |

---

## 附录 A: 与 pypto-skill-reviewer 模式对照

| 组件 | pypto-skill-reviewer | pypto-op-lint |
|------|---------------------|--------------|
| 规则注册表 | `rules.json` (48 规则, 9 维度) | `rules.json` (25 规则, 5 维度) |
| 静态检查器 | `validate_skill.py` (785 行) | `pypto_op_lint.py` (统一 CLI) |
| 评分引擎 | `score_findings.py` (202 行) | 复用同一个 `score_findings.py` |
| Finding 协议 | `{rule_id, severity, dimension, status, message, evidence}` | 完全相同 |
| 严重级体系 | S0/S1/S2/S3 + S0 否决 | 完全相同 |
| 触发方式 | 手动调用 skill | hook 自动触发 + 手动 report |

## 附录 B: 扩展示例 — 新增一条规则

假设需要新增 "impl 中 pypto.tensor() 创建后必须初始化" 的检查：

**Step 1**: `rules.json` 追加:
```json
{
  "id": "OL26",
  "severity": "S2",
  "type": "static",
  "dimension": "D1",
  "stages": [5, 6, 7],
  "target": "impl",
  "rule": "pypto.tensor() 创建的 Tensor 必须在使用前初始化",
  "source": ["C1"],
  "check_function": "check_ol26",
  "check_strategy": "AST: pypto.tensor() 返回值在被读取前必须有赋值操作"
}
```

**Step 2**: `pypto_op_lint.py` 追加:
```python
@register("OL26")
def check_ol26(ctx: CheckContext) -> Finding:
    """pypto.tensor() 创建后必须初始化"""
    tree = ctx.parse_file(f"{ctx.op_name}_impl.py")
    if tree is None:
        return ctx.skip_finding("OL26")
    # ... AST 检查逻辑
```

**Step 3**: 完毕。hook 配置无需修改，新规则在下次编辑 impl 文件时自动生效。

---

## 附录 C: 实现状态与偏差记录

> 更新日期: 2026-04-03

### C.1 MVP 实现范围

已实现 Phase 1-3 的核心功能，采用了比原始设计更简洁的架构：

| 原始设计 | 实际实现 | 偏差原因 |
|---------|---------|---------|
| 7 个 CLI 子命令 (`lint-impl`/`lint-golden`/`lint-test`/`check-gate`/`check-backup`/`parse-result`/`report`) | 4 个手动子命令 (`--lint-impl`/`--lint-golden`/`--lint-test`/`--check-gate`) + 4 个 hook 模式 (`--hook post-edit/post-bash/pre-edit/stop`) | Hook 模式直接从 stdin 读取 JSON 并内联处理，无需独立子命令 |
| Shell 适配脚本层 (`hook_claude_*.sh`) + Python 核心 | 纯 Python 实现，hook 适配逻辑内联在 `pypto_op_lint.py` 中 | 减少一层间接调用，降低维护成本 |
| D5 维度（流程合规）含 OL23/OL24/OL25 | OL23 重定义为 D1 loop 检测（S3 提醒级）；OL24 保留为状态文件结构校验；OL25 未实现 | Stage 6 备份改为 `pre-edit` hook 自动 git commit，比文件命名检查更可靠 |
| 维度权重评分 + 评分引擎 (`score_findings.py`) | 仅 PASS/FAIL/WARN/SKIP 统计 + S0 一票否决 | MVP 阶段评分对 hook 拦截无意义，S0 否决已满足门禁需求 |
| `--format json|text` / `--quiet` 选项 | 仅 JSON 输出 | hook 消费方只��� JSON |
| `report` 全量检查子命令 | `--check-gate` 实现阶段性全量检查；`--hook stop` 实现 agent 结束前全量检查 | 功能已覆盖，无需独立 report 入口 |

### C.2 实际目录结构

```
.agents/hooks/pypto-op-lint/
├── PROPOSAL.md                              # 本文档（设计提案 + 实现记录）
├── pypto_op_lint.py                         # 统一入口（1090 行���24 条规则 + hook 适配）
├── rules.json                               # 规则注册表（4 维度，4 严重级）
└── tests/
    ├── run_tests.py                         # 单元测试（70 cases）
    ├── test_comprehensive.py                # 综合验证（50 cases）
    ├── test_hook_integration.py             # Hook 集成测试（13 cases）
    └── fixtures/                            # 测试固件
        ├── good_op/                         # 全通过 fixture
        ├── bad_op/                          # 全违规 fixture
        ├── flash_attention/                 # 含 loop 的复杂算子 fixture
        ├── partial_bad_op/                  # 有 jit 但内部违规 fixture
        ├── malformed_state_op/              # 损坏的状态文件 fixture
        └── missing_files_op/               # 缺少文件的 fixture

.opencode/plugins/pypto-op-lint.ts           # OpenCode 平台适配插件
.agents/settings.json                        # Claude Code hooks 配置
```

### C.3 实际规则表（24 条）

| 维度 | 规则数 | 规则 ID | 覆盖范围 |
|------|--------|---------|---------|
| D1 框架约束合规 | 9 | OL01-OL08, OL23 | jit 装饰器、写回方式、return、tile shapes、类型注解、min/max、import、wrapper、loop |
| D2 工件完整性 | 7 | OL09-OL14, OL24 | spec/api_report/golden/design 门禁、三件套、精度状态、状态文件 |
| D3 三文件分离 | 4 | OL15-OL18 | golden 纯净性、impl-golden 隔离、test 不含 kernel、test 双导入 |
| D4 测试规范 | 4 | OL19-OL22 | assert_allclose、DEVICE_ID、level0/level1、manual_seed |

### C.4 v2.1 修复记录（2026-04-03）

基于代码 review 修复了以下问题：

| 编号 | 严重级 | 问题 | 修复 |
|------|--------|------|------|
| FIX-1 | P0 | `_parse_verdict` PASS/FAIL 优先级反转：同时出现两个标记时 PASS 优先于 FAIL | FAIL 检查移至 PASS 之前 |
| FIX-2 | P1 | `_find_nearest_op_dir` 从 cwd 向下搜索 3 层，在 /tmp 或仓库根目录可误匹配不相关目录 | 限制搜索范围为 `custom/` 子树，最多 2 层 |
| FIX-3 | P1 | OL02 仅检查 `ast.Assign`，遗漏 `ast.AugAssign`（`y += x`） | 增加 AugAssign 检查 |
| FIX-4 | P2 | OL05 要求所有 kernel 参数为 `pypto.Tensor`，非张量标量参数（`int`/`float`）被误报 | 区分张量参数和非张量参数：张量参数仍要求 `pypto.Tensor` 注解，`int`/`float`/`bool` 标量参数跳过 Tensor 注解检查（符合 `execution-constraints.md` 第 8-9 行：张量参数在前用 Tensor 注解，非张量参数在后） |
| FIX-5 | P2 | OL04 FAIL 信息未说明 "必须在 jit 函数体内" | 补充 FAIL message 提示 |

### C.5 未实现的远景功能

以���功能保留在设计中，视需要在后续迭代实现：

1. **D5 流程合规维度** — OL25 备份文件命名规范检查
2. **维度加权评分引擎** — `score_findings.py` + 等级映射 (A/B/C/D/F)
3. **`report` 全量报��** — Markdown 格式的完整检查报告
4. **`--format text` 模式** — 人类可读的文本输出
5. **`parse-result` 细分 verdict** — `runtime_error`/`import_error`/`compile_error`/`aicore_error`/`shape_error` 细粒度判定
