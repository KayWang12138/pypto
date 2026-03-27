# scan_results.csv 字段定义

**位置**: `autodev/scan_results.csv`（项目根目录相对路径）

## 字段表

| 字段 | 类型 | 可选值 | 说明 |
|------|------|--------|------|
| op_name | string | - | 算子名称，等于目录名 `autodev/custom/{op_name}/` |
| source | enum | `manual` / `auto_discovered` | 需求来源 |
| status | enum | `pending` / `in_progress` / `completed` / `failed` | 算子当前状态 |
| complexity | enum | `easy` / `medium` / `hard` | 复杂度（LLM 推断）|
| category | enum | elementwise/reduction/normalization/attention/activation/embedding/matmul/pooling/convolution/other | 算子类别 |
| dev_result | enum | `SUCCESS` / `TIMEOUT` / `BLOCKED_API` / `BLOCKED_ENV` / `PRECISION` / `NOT_IMPLEMENTABLE` / 空 | 开发结果 |
| fail_count | int | ≥0 | 累计失败次数（reset 不清零）|
| fps_total | int | ≥0 | 本次开发检测到的致命断裂点总数 |
| fps_confirmed | int | ≥0 | confidence=high 的已确认断裂点数量 |
| create_time | ISO 8601 | - | 首次写入时间 |
| start_time | ISO 8601 或空 | - | 当前/上次开发开始时间 |
| end_time | ISO 8601 或空 | - | 最近一次完成/失败时间 |
| note | string | - | 自由文本备注（中英文均可）|

## dev_result 后续处理

| dev_result | 后续 |
|------------|------|
| `SUCCESS` | 完成 |
| `TIMEOUT` | 可手动重试 |
| `BLOCKED_API` | 记录断裂点 |
| `BLOCKED_ENV` | 排查环境后可重试 |
| `PRECISION` | 记录断裂点 |
| `NOT_IMPLEMENTABLE` | 自动排除，不再选中 |

## 注意

- 所有 CSV 读写必须通过 `.agents/skills/pypto-op-autodev/scripts/` 下的脚本完成
- LLM 不得直接读取或修改 CSV 文件
- 脚本操作前自动备份为 `.bak`
