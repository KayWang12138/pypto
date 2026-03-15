# Failure Recovery

## 恢复策略

1. 优先读取 `.orchestrator_state.json` 恢复状态
2. 读取 `retry_history` 了解历史失败上下文，避免重复无效修复
3. 只回退到最近失败 stage
4. 尽量复用上游已通过工件
5. 每 stage 最多一次自动重试
6. 语义错误不得盲重试

## 统一结束态

| 状态 | 含义 |
|------|------|
| `SUCCESS` | 全流程完成 |
| `BLOCKED_CONTRACT` | 缺关键工件或工件契约损坏 |
| `BLOCKED_ACCURACY` | 实现可运行但精度未通过（含达到循环上限） |
| `BLOCKED_PERFORMANCE` | 精度通过但性能分析未完成 |
| `BLOCKED_ENVIRONMENT` | 环境问题阻塞执行 |

## 精度修复循环

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

## retry_history 格式

```json
{
  "3": [
    {
      "attempt": 1,
      "timestamp": "2026-03-15T10:25:00Z",
      "failure_reason": "compilation error: undefined symbol xxx in line 42",
      "action_taken": "retry_with_fix"
    }
  ]
}
```

每次失败都记录具体原因和采取的动作，帮助后续修复避免重复路径。

## 失败类型路由

| 失败类型 | 检测方式 | 恢复动作 |
|----------|----------|----------|
| 工件缺失 | 文件检查 | 回退到产出该工件的 stage |
| 运行时错误 | exit code ≠ 0，无 `Not equal to tolerance` | Stage 3 内重试修复 |
| 精度失败 | 输出含 `Not equal to tolerance` | Stage 4 定位 → Stage 3 修复 |
| 环境问题 | 特定错误信息（TILE_FWK_DEVICE_ID 未设置等） | BLOCKED_ENVIRONMENT |
| 循环超限 | `accuracy_fix_loop_count >= 10` | BLOCKED_ACCURACY |
