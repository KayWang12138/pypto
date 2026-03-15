---
name: pypto-op-orchestrator
description: "PyPTO 算子端到端开发编排。管理从需求理解到性能分析的完整状态机，协调 9 个 child skills。当用户请求开发算子时，这是默认入口。"
---

# PyPTO 算子端到端开发编排

## Role

`pypto-op-orchestrator` 是 PyPTO 算子开发的唯一流程 owner。它不写代码，只管理状态机：

- 接收用户开发意图
- 判断新算子 / 续跑已有算子
- 根据工件状态决定当前阶段
- 调用正确的 child skill
- 检查输出工件和质量门禁
- 失败时走明确恢复分支
- 持久化状态到 `.orchestrator_state.json`
- 汇总结构化结果报告

**核心原则**：

1. 只做流程编排，不做专业实现
2. 只以工件驱动状态，不以对话历史驱动状态
3. 只允许明确门禁通过后进入下一阶段
4. 只在可重试的临时失败上重试一次
5. 只输出统一结束态，不输出模糊成功

## Entry Conditions

启动时执行：

1. 检查 `custom/{op}/.orchestrator_state.json` 是否存在
2. 若存在且非 terminal state → 从 `current_stage` 继续
3. 若不存在 → 从 Stage 0 开始
4. 读取 `retry_history` 了解历史失败上下文，避免重复无效修复

## Child Skills Inventory

### 端到端工作流

| 技能 | 用途 | 触发时机 |
|------|------|---------|
| `pypto-op-orchestrator` | 端到端算子开发编排 | **算子开发默认入口** |

### 阶段能力

| 技能 | 用途 | 触发时机 |
|------|------|---------|
| `pypto-intent-understanding` | 需求理解，生成 spec.md | 需求分析阶段 |
| `pypto-golden-generator` | 生成 golden 参考实现 | spec 完成后 |
| `pypto-op-design` | 生成 design 文档 | golden 完成后 |
| `pypto-op-develop` | 实现 kernel + test + README | design 完成后 |
| `pypto-op-perf-autotuner` | 性能采集与调优 | 精度通过后 |
| `pypto-op-perf-analyzer` | 性能分析报告 | 性能数据采集后 |

### 调试辅助

| 技能 | 用途 | 触发时机 |
|------|------|---------|
| `pypto-op-accuracy-verify` | 精度问题初步排查 | 精度不通过时（第一步） |
| `pypto-binary-search-verify` | verify-based 精度定位 | 精度不通过时（默认定位手段） |
| `pypto-binary-search-without-verify` | checkpoint-based 精度定位 | verify 方式不适用时 |

## Unified Artifact Contract

### 标准目录

```text
custom/{op}/
├── spec.md
├── design.md
├── {op}_golden.py
├── test_{op}.py
├── {op}_impl.py
├── README.md
├── .orchestrator_state.json
└── output/
    └── output_*/
```

### 工件 Owner / Consumer

| 工件 | Owner | 主要消费者 | 导出函数 |
|------|-------|------------|----------|
| `spec.md` | `pypto-intent-understanding` | orchestrator / golden / design | — |
| `design.md` | `pypto-op-design` | orchestrator / op-develop | — |
| `{op}_golden.py` | `pypto-golden-generator` | test / accuracy / debug | `{op}_golden()` |
| `test_{op}.py` | `pypto-op-develop` | orchestrator / perf-autotuner | — |
| `{op}_impl.py` | `pypto-op-develop` | test / accuracy / debug / perf | `{op}_wrapper()` |
| `README.md` | `pypto-op-develop` | 用户 / 审查者 | — |
| `.orchestrator_state.json` | orchestrator | orchestrator | — |

### 三文件分离

| 文件 | 职责 | 导出 |
|------|------|------|
| `{op}_golden.py` | 纯 torch 参考实现 | `{op}_golden()` |
| `{op}_impl.py` | PyPTO kernel 实现 | `{op}_wrapper()` |
| `test_{op}.py` | 测试入口：import + 调用 + `assert_allclose` | 不导出 |

```python
# test_{op}.py 的导入方式
from {op}_golden import {op}_golden
from {op}_impl import {op}_wrapper
```

### Overwrite Policy

| 工件 | 覆盖前需确认 |
|------|-------------|
| `spec.md` / `design.md` / `{op}_golden.py` / `test_{op}.py` / `{op}_impl.py` / `README.md` | 是 |
| `output/output_*` / `.orchestrator_state.json` | 否 |

## Stage State Machine

### Stage 总览

| Stage | 名称 | 执行者 | 关键输出 | 成功标准 |
|-------|------|--------|----------|----------|
| 0 | 上下文解析 | orchestrator | 唯一工作目录 | `custom/{op}/` 确定 |
| 1 | 需求理解 | `pypto-intent-understanding` | `spec.md` | 文件存在 |
| 2A | 生成 Golden | `pypto-golden-generator` | `{op}_golden.py` | 文件存在 |
| 2B | 生成 Design | `pypto-op-design` | `design.md` | 文件存在 |
| 3 | 功能实现 | `pypto-op-develop` | 3 文件 | 见 Stage 3 判定 |
| 4 | 精度定位 | 见精度定位策略 | 修复建议 | 产出定位报告 |
| 5 | 性能采集与调优 | `pypto-op-perf-autotuner` | output 数据 | 新 output 存在 |
| 6 | 性能分析 | `pypto-op-perf-analyzer` | 性能摘要 | 报告产出 |

### 流转规则

```
Stage 1: 使用 pypto-intent-understanding 技能，生成 spec.md。

Stage 2A: 使用 pypto-golden-generator 技能，生成 {op}_golden.py。

Stage 2B: 使用 pypto-op-design 技能，生成 design.md。
         （2A → 2B 串行，design 参考 golden 的函数签名和结构）

Stage 3: 使用 pypto-op-develop 技能，生成 test_{op}.py + {op}_impl.py + README.md 并首跑。
         Orchestrator 根据脚本输出判定精度 PASS / FAIL。

Stage 4: 使用 pypto-op-accuracy-verify 技能进行初步排查。
         默认使用 pypto-binary-search-verify 技能定位；
         若不适用，降级使用 pypto-binary-search-without-verify 技能。

Stage 5: 使用 pypto-op-perf-autotuner 技能。

Stage 6: 使用 pypto-op-perf-analyzer 技能，生成性能摘要。
```

### Stage 3 判定逻辑

Orchestrator 在 Stage 3 首跑后根据脚本输出直接判定：

- 脚本 exit 0 → **PASS**（精度通过 → 进入 Stage 5）
- 脚本输出包含 `Not equal to tolerance` 关键字 → **部分通过**（精度不通过 → 进入 Stage 4）
- 脚本因其他错误退出（无 `Not equal to tolerance`） → **FAIL**（Stage 3 内重试修复）

**前提**：`test_{op}.py` 必须使用 `numpy.testing.assert_allclose`，禁止手写 `assert max_diff < tolerance`。

### Stage 4 精度定位策略

三步逐级深入：

| 步骤 | 技能 | 说明 |
|------|------|------|
| 1 | `pypto-op-accuracy-verify` | 快速分析精度失败特征 |
| 2 | `pypto-binary-search-verify` | verify-based，通过 `pass_verify_save()` 捕获中间结果 |
| 3 | `pypto-binary-search-without-verify` | checkpoint-based，修改函数签名 + assemble 原地写入 |

优先级：用户显式指定 > verify-based（默认）> checkpoint-based（降级）

## State Persistence

每个 stage 完成后持久化到 `custom/{op}/.orchestrator_state.json`：

```json
{
  "operator_name": "{op}",
  "current_stage": 3,
  "stage_status": {
    "0": "completed",
    "1": "completed",
    "2a": "completed",
    "2b": "completed",
    "3": "in_progress"
  },
  "artifacts": {
    "spec": "custom/{op}/spec.md",
    "design": "custom/{op}/design.md",
    "golden": "custom/{op}/{op}_golden.py",
    "test_entry": "custom/{op}/test_{op}.py",
    "impl": "custom/{op}/{op}_impl.py"
  },
  "last_updated": "2026-03-15T10:30:00Z",
  "accuracy_fix_loop_count": 0,
  "retry_history": {}
}
```

- `retry_history`：结构化失败记录（attempt/timestamp/failure_reason/action_taken）
- `accuracy_fix_loop_count`：Stage 3→4→3 精度修复循环累计次数

## Gate and Retry Policy

### Gate 原则

- 没有 `spec.md` → 不能进入 Stage 2
- 没有 `design.md` 和 `{op}_golden.py` → 不能进入 Stage 3
- Stage 3 未产出 3 文件 → 不能判定精度
- 精度失败 → 不能进入性能阶段

### 恢复策略

1. 优先读取 `.orchestrator_state.json` 恢复状态
2. 读取 `retry_history` 了解历史失败上下文
3. 只回退到最近失败 stage
4. 尽量复用上游已通过工件
5. 每 stage 最多一次自动重试
6. 语义错误不得盲重试

详细恢复规则见 `references/failure-recovery.md`。

## Accuracy Fix Loop Policy

Stage 3 精度失败时触发 Stage 4（定位）→ Stage 3（修复重跑）的循环。

**上限：10 次。**

```
accuracy_fix_loop_count 初始化为 0

每次 Stage 3 精度失败：
  accuracy_fix_loop_count += 1
  if accuracy_fix_loop_count >= 10:
    → BLOCKED_ACCURACY
    → 输出完整 retry_history 供人工分析
  else:
    → Stage 4 精度定位
    → 修复后回到 Stage 3
```

`accuracy_fix_loop_count` 持久化到 `.orchestrator_state.json`，跨会话保持累计。

## Recovery Routing

根据失败类型路由到不同恢复路径：

| 失败类型 | 检测方式 | 恢复动作 |
|----------|----------|----------|
| 工件缺失 | 文件检查 | 回退到产出该工件的 stage |
| 运行时错误 | exit code ≠ 0，无 `Not equal to tolerance` | Stage 3 内重试修复 |
| 精度失败 | 输出含 `Not equal to tolerance` | Stage 4 定位 → Stage 3 修复 |
| 环境问题 | 特定错误信息 | BLOCKED_ENVIRONMENT |
| 循环超限 | `accuracy_fix_loop_count >= 10` | BLOCKED_ACCURACY |

## Final Output Format

```markdown
## 开发结果
- 算子: {op}
- spec: custom/{op}/spec.md
- design: custom/{op}/design.md
- golden: custom/{op}/{op}_golden.py
- test_entry: custom/{op}/test_{op}.py
- kernel: custom/{op}/{op}_impl.py

## 精度结果
- 状态: PASS / FAIL
- 容差: rtol / atol
- 若失败: 当前定位路径
- 精度修复循环次数: N

## 性能结果
- output_dir: custom/{op}/output/output_xxx/
- core_utilization: ...
- bubble_rate: ...
- load_balance: ...
- max_work_time: ...

## 已知问题
- 环境限制 / 仅 sim 跑通 / NPU 未验证 / 数据缺失
```

详细模板见 `references/final-report-template.md`。

## Unified Terminal States

| 状态 | 含义 |
|------|------|
| `SUCCESS` | 全流程完成 |
| `BLOCKED_CONTRACT` | 缺关键工件或工件契约损坏 |
| `BLOCKED_ACCURACY` | 实现可运行但精度未通过（含达到循环上限） |
| `BLOCKED_PERFORMANCE` | 精度通过但性能分析未完成 |
| `BLOCKED_ENVIRONMENT` | 环境问题阻塞执行 |

## When to Read References

| 文件 | 何时读取 |
|------|---------|
| `references/pipeline-overview.md` | 需要查看完整流程图或 stage 总览时 |
| `references/artifact-contract.md` | 需要确认工件 owner/consumer 或 overwrite 规则时 |
| `references/stage-gates.md` | 需要确认 gate 条件或 Stage 3 判定细节时 |
| `references/overwrite-policy.md` | 需要确认是否需要覆盖确认时 |
| `references/failure-recovery.md` | 处理失败恢复或精度修复循环时 |
| `references/final-report-template.md` | 生成最终输出报告时 |
