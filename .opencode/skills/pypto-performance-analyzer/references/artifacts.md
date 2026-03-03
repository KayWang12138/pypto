# PyPTO 性能产物说明

## 官方文档依据

- `docs/api/config/pypto-set_debug_options.md`：`runtime_debug_mode=1` 会生成运行期调试产物（含泳道图）。
- `docs/tutorials/debug/performance.md`：在 `output/output_*/` 下生成 `merged_swimlane.json`，用于性能分析。
- `docs/tools/swimlane_graph/查看性能报告.md`：AICore Utilization 指标定义和泳道图分析口径。

## 输出目录结构（常见）

运行后产物位于 `{work_dir}/output/output_{timestamp}/`。

```
output_*/
├── merged_swimlane.json               # 泳道图主数据（核心）
├── bubble_analysis.log                # 气泡分析日志（tools/profiling/draw_swim_lane.py 生成）
├── machine_runtime_operator_trace.json# AICPU 控制链路 trace
├── aicpu_dev_pref.json                # AICPU 原始阶段数据
├── tilefwk_L1_prof_data.json          # TileFwk L1 性能数据
├── execute.json                       # 任务执行清单（可选）
├── pipe_usage.csv                     # 流水线使用率统计（可选）
├── topo.json                          # 任务拓扑/依赖信息
└── program.json                       # 程序与函数信息
```

## analyzer 当前可解析文件

| 文件 | 用途 | 关键字段 |
|------|------|----------|
| `merged_swimlane.json` | 基础 timeline/热点/空闲分析 | `traceEvents[].ph/ts/dur/tid/name` |
| `bubble_analysis.log` | 等待时间分解与线程利用率 | `Core Total Work Time`/`Total Wait Time`/`Wait Schedule Time` |
| `machine_runtime_operator_trace.json` | 控制开销分解 | `traceEvents[].cat/name/dur` |
| `aicpu_dev_pref.json` | 控制开销备用数据源 | `coreType`=`AICPU-CTRL`, `tasks[].name/end` |
| `execute.json` | 任务执行统计补充 | `execTime`, `coreType`, `funcName` |
| `pipe_usage.csv` | PIPE 使用率 | `Pipe, AverageTime, TotalExecuteTime, AverageUsage` |
| `topo.json` | 依赖结构补充 | `taskId`, `successors` |
| `program.json` | 函数/张量规模补充 | `functions`, `tensors` |
| `tilefwk_L1_prof_data.json` | 周期级任务统计 | `coreType`, `tasks[].execStart/execEnd` |

## 说明

- 真实数据下，不同文件的时间单位可能不同。报告中会尽量标注“同源单位”。
- 当某个文件缺失时，analyzer 会跳过对应章节并继续输出报告。
