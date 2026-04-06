# autodev 调用规范

详细设计见 `docs/agents/autodev.md`。本文件供 SKILL.md 执行时快速参考。

## autodev 状态机

```
pending ──→ in_progress ──→ completed
                 │
                 ├──→ failed
                 └──→ pending (--reset-stale / --reset)
```

非法转换（脚本拒绝）：pending→completed、completed→in_progress、failed→in_progress。

## .dev_result.json

autodev 在调用执行层时，通过 prompt 要求完成后写入此文件。

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

判定：
- 文件存在且 `status == "SUCCESS"` → 成功，进入验证
- 文件存在且 `status != "SUCCESS"` → 失败
- 文件不存在（超时/崩溃）→ TIMEOUT

## autodev-task.md 格式

```markdown
# 开发任务
- op_name: {op_name}
- working_dir: {work_dir}/{op_name}/
- resume_from_stage: {1-7}
- complexity: {easy|medium|hard}
- category: {类别}
- start_time: {ISO 8601}
```

## 退出码约定

所有脚本统一：0=成功，1=业务级拒绝（非错误），2+=脚本级错误。

## 错误处理

| 级别 | 行为 | 示例 |
|------|------|------|
| 致命 | 终止执行 | 环境检查 exit 2；脚本异常 exit 2+ |
| 阻塞 | 记录状态后退出 | 活跃任务未超时；无候选算子 |
| 可恢复 | 记录警告，继续 | NPU 降级；单个需求添加失败 |
| 非致命 | 记录日志，不影响结果 | 断裂点检测失败；知识更新失败 |
