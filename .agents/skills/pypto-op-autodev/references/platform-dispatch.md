# 平台调度方式

autodev 的 Step 4（调度算子开发）和 Step 2c（discover）需要调用 agent/skill。
不同平台的调用语法不同，本文件提供模板。

## Step 4：调度算子开发

### orchestrator 策略

**scheduler.py 模式**：
```
opencode run --agent {orchestrator_agent} "{prompt}"
```

**Claude Code 模式**：
```
Agent(
  subagent_type="pypto-op-orchestrator",
  prompt="{prompt}"
)
```

**OpenCode 模式**：
```
@pypto-op-orchestrator {prompt}
```

**prompt 模板**（三种模式通用）：
```
开发算子 {op_name}。
工作目录: {work_dir}/{op_name}/
从 Stage {resume_from_stage} 开始。
复杂度: {complexity}，类别: {category}。

参考已知框架限制（开发时注意规避）:
{autodev/known-limitations.md 的内容}

完成后在工作目录写 .dev_result.json，格式:
{"status": "SUCCESS|FAILED|BLOCKED", "blocked_reason": "...|null",
 "precision_result": "PASS|FAIL|null", "completed_stages": [...],
 "artifacts": [...], "notes": "..."}

开发过程持续追加写入 dev-log.md（报错、方案变更、API 限制、workaround）。
```

### workflow 策略

**Claude Code 模式**：

按依赖顺序逐个调用 Skill（在 autodev 的 context 中直接执行）:
```
Skill(pypto-intent-understanding)  → 检查 spec.md 存在
Skill(pypto-api-explorer)          → 检查 api_report.md 存在
Skill(pypto-golden-generator)      → 检查 {op}_golden.py 可运行
Skill(pypto-op-design)             → 检查 design.md 存在
Skill(pypto-op-develop)            → 三态判定
  [PRECISION_PASS] → Skill(pypto-operator-auto-tuner)
  [PRECISION_FAIL] → Skill(pypto-precision-debugger) → 再测
  运行失败 → 重试（≤3 次）
```

每个 Skill 调用时传入 known-limitations 内容作为参考。
所有 Skill 完成后，autodev 自行组装 .dev_result.json。

**scheduler.py 模式**：

通过 opencode 调用一个 workflow-runner agent（薄壳，内部按上述顺序调 skill）:
```
opencode run --agent pypto-op-workflow-runner "{prompt}"
```

或：scheduler.py 自身顺序调用各 skill 对应的 opencode session（需要封装）。

当前推荐：若 scheduler.py 使用 workflow 策略，创建 `.opencode/agents/pypto-op-workflow-runner.md`
作为薄壳 agent，其 prompt 引用 `.agents/skills/pypto-op-workflow/SKILL.md`。

## Step 2c：discover 调度

**scheduler.py 模式**：
```
opencode run --agent {discover_agent} "扫描 PyPTO 项目，发现 1 个新的待实现算子"
```

**Claude Code 模式**：
```
Agent(
  subagent_type="pypto-op-discover",
  prompt="扫描 PyPTO 项目，发现 1 个新的待实现算子，更新 scan_results.csv"
)
```

## Step 6a：断裂点检测

**Claude Code 模式**：
```
Skill(pypto-fracture-point-detector)
```

**scheduler.py 模式**：
断裂点检测是一个 skill，scheduler.py 通过 opencode 调用：
```
opencode run "调用 pypto-fracture-point-detector skill，分析 {op_dir}/dev-log.md"
```
