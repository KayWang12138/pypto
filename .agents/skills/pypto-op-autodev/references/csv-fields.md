# scan_results.csv 字段定义

**位置**：`{csv_path}`（典型值 `{project_root}/autodev/scan_results.csv`）

| 字段 | 类型 | 可选值 | 说明 |
|------|------|--------|------|
| op_name | string | — | 算子名称，等于目录名 |
| source | enum | `manual` / `auto_discovered` | 需求来源 |
| status | enum | `pending` / `in_progress` / `completed` / `failed` | 当前状态 |
| complexity | enum | `easy` / `medium` / `hard` | 复杂度（LLM 推断） |
| category | enum | elementwise / reduction / normalization / attention / activation / embedding / matmul / pooling / convolution / other | 算子类别 |
| dev_result | enum | 见下表 | 开发结果 |
| fail_count | int | ≥0 | 累计失败次数（reset 不清零） |
| fps_total | int | ≥0 | 检测到的断裂点总数 |
| fps_confirmed | int | ≥0 | confidence=high 的已确认断裂点数 |
| create_time | ISO 8601 | — | 首次写入时间 |
| start_time | ISO 8601 | — | 当前/上次开发开始时间 |
| end_time | ISO 8601 | — | 最近完成/失败时间 |
| blocked_stage | int | 1-7 | 阻塞发生的阶段号（仅失败时） |
| duration_min | float | — | 开发耗时（分钟） |
| depends_on | string | — | 依赖的算子（多个用 `\|` 分隔） |
| last_strategy | enum | `orchestrator` / `workflow` / 空 | 上次使用的开发策略 |
| note | string | — | 自由文本备注 |

## dev_result 值域

| dev_result | 含义 | 后续处理 | 调度惩罚 |
|---|---|---|---|
| SUCCESS | 开发+验证通过 | 完成 | — |
| TIMEOUT | 超时 | 可手动重试 | -5 |
| PRECISION | 精度验证失败 | 记录断裂点 | -15 |
| BLOCKED_API | API 不可行 | 记录断裂点 | -40 |
| BLOCKED_IMPL | 实现阶段阻塞 | 记录断裂点 | -35 |
| BLOCKED_DESIGN | 设计阶段阻塞 | 记录断裂点 | -35 |
| BLOCKED_GOLDEN | Golden 生成阻塞 | 记录断裂点 | -35 |
| BLOCKED_ACCURACY | 精度修复阻塞 | 记录断裂点 | -35 |
| BLOCKED_ENVIRONMENT | 环境问题 | 排查环境后重试 | -35 |
| NOT_IMPLEMENTABLE | 无法实现 | 自动排除，不再选中 | — |

> 所有 CSV 读写必须通过 `{scripts}/` 脚本完成。LLM 不得直接操作 CSV。脚本操作前自动备份为 `.bak`。
