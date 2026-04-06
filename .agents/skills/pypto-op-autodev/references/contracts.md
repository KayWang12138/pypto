# autodev 契约定义

## 状态机

```
pending ──→ in_progress ──→ completed
                 │
                 ├──→ failed
                 └──→ pending (--reset-stale / --reset)
```

非法转换（脚本拒绝）：`pending→completed`、`completed→in_progress`、`failed→in_progress`。

## .dev_result.json

开发 agent 完成后必须在算子工作目录写入此文件。

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

autodev 判定逻辑：
- 文件存在且 `status == "SUCCESS"` → 成功，进入验证
- 文件存在且 `status != "SUCCESS"` → 失败
- 文件不存在（超时/崩溃）→ TIMEOUT

## autodev-task.md

Step 3 创建的任务描述文件，放在 `{work_dir}/{op_name}/` 下。

```markdown
# 开发任务
- op_name: {op_name}
- working_dir: {work_dir}/{op_name}/
- resume_from_stage: {1-7}
- complexity: {easy|medium|hard}
- category: {类别}
- reference: {参考实现位置，可为空}
- description: {一句话需求描述，可为空}
- requirement: {需求文件路径，可为空}
- start_time: {ISO 8601}
```

## 退出码约定

所有脚本统一：

| exit code | 含义 |
|-----------|------|
| 0 | 成功 |
| 1 | 业务级拒绝（非错误） |
| 2+ | 脚本级错误 |

## 错误处理

| 级别 | 行为 | 示例 |
|------|------|------|
| 致命 | 终止执行 | 环境检查 exit 2；脚本异常 |
| 阻塞 | 记录状态后退出 | 活跃任务未超时；无候选算子 |
| 可恢复 | 记录警告，继续 | NPU 降级；单个需求添加失败 |
| 非致命 | 记录日志，不影响结果 | 断裂点检测失败；知识更新失败 |
