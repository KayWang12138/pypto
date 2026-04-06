# 平台调度语法参考

SKILL.md 已内联了 `native`/`cli` 两种 dispatch_mode 的调度语法。
本文件提供 `cli` 模式的完整命令模板，供调试和 scheduler.py 开发参考。

## Step 4：调度算子开发

### orchestrator 策略

```bash
opencode run --agent {orchestrator_agent} "{dev_prompt}"
```

### workflow 策略

```bash
opencode run --agent {workflow_agent} "{dev_prompt}"
```

`{workflow_agent}` 由 `scheduler.conf.json` 配置；若未配置，scheduler.py 回退到 `{orchestrator_agent}`。

### dev_prompt 模板

与 SKILL.md Step 4 中的模板一致，不在此重复。

## Step 2c：discover 调度

```bash
opencode run --agent {discover_agent} "扫描 PyPTO 项目，发现 1 个新的待实现算子"
```

`native` 模式等价写法：

```
Agent(
  subagent_type="pypto-op-discover",
  prompt="扫描 PyPTO 项目，发现 1 个新的待实现算子，更新 scan_results.csv"
)
```

## Step 6a：断裂点检测

```bash
opencode run "调用 pypto-fracture-point-detector skill，分析 {op_dir}/dev-log.md"
```

`native` 模式等价写法：

```
Skill("pypto-fracture-point-detector")
```
