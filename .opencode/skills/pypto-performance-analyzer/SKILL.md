---
name: pypto-performance-analyzer
description: PyPTO 性能分析技能 - 解析泳道图(merged_swimlane.json)及多种 output 产物（bubble/trace/execute/pipe_usage/topo/program/tilefwk），识别性能瓶颈并生成评级与优化建议。触发词：性能分析、swimlane、泳道图、计算图、性能瓶颈、pypto set_pass_options、AICore利用率、AIVector利用率、气泡分析、控制开销、内存分析、执行时间统计、性能评级、pipe usage、execute.json、topo.json
license: Apache-2.0
---

# PyPTO Performance Analyzer

分析 PyPTO 算子的泳道图和计算图，识别性能瓶颈，生成可操作的调优建议。

## 触发场景

- "分析泳道图" / "查看性能数据" / "swimlane 分析"
- "AICore 利用率低" / "性能瓶颈在哪里"
- "生成性能报告" / "pypto 性能分析"
- "计算图分析" / "子图合并问题"
- "气泡分析" / "bubble_analysis.log 怎么看"
- "控制开销分析" / "AICPU-CTRL 占比"
- "内存分析" / "UB 峰值内存"
- "执行时间统计" / "execution-hint"
- "性能评级" / "5 星评分"
- "pipe_usage.csv 分析" / "流水线利用率"
- "execute.json 分析" / "任务执行清单"
- "topo.json 分析" / "依赖关系分析"

## 快速开始

### Dry-run 模式（无需 NPU）

```bash
python3 .opencode/skills/pypto-performance-analyzer/scripts/analyze.py --dry-run
```

### 分析真实产物

```bash
# 自动检测最新 output 目录
python3 .opencode/skills/pypto-performance-analyzer/scripts/analyze.py --pypto-repo /workspace/code/pypto

# 指定输出目录
python3 .opencode/skills/pypto-performance-analyzer/scripts/analyze.py --output-dir /path/to/output --report-out /path/to/report.md
```

## 前置条件

1. 生成泳道图数据（在 PyPTO 代码中启用 `runtime_debug_mode=1`）：

```python
@pypto.frontend.jit(
    runtime_options={"run_mode": pypto.RunMode.NPU},
    debug_options={"runtime_debug_mode": 1}  # 启用泳道图采集
)
def my_kernel(...):
    ...
```

2. 运行算子后，在 `{work_dir}/output/output_*/` 下会生成 `merged_swimlane.json`

## 输出内容

| 指标 | 说明 |
|------|------|
| Timeline 长度 | 总执行时间 |
| AICore 利用率 | Cube 核心平均利用率 |
| AIVector 利用率 | Vector 核心平均利用率 |
| 核心级 Bubble | 每个核心的空闲比例 |
| 气泡分析 | 解析 `bubble_analysis.log` 的线程工作/等待时间与利用率 |
| 控制开销分析 | 解析 `machine_runtime_operator_trace.json`，并支持回退到 `aicpu_dev_pref.json` |
| 内存分析 | 提取 UB 峰值与 operand hint 的内存使用信息 |
| 执行时间统计 | 从 `execution-hint` 汇总 Avg/Max/Min 执行时延 |
| 性能评级 | 基于利用率/气泡率/内存效率/控制开销的 5 星评级 |
| 额外产物分析 | 解析 `execute.json`/`pipe_usage.csv`/`topo.json`/`program.json`/`tilefwk_L1_prof_data.json` |
| Top 热点任务 | 最耗时的任务列表 |
| 空闲间隔 | 核心间的调度间隙 |
| 调优建议 | 可操作的 knob 配置建议 |

## 调优 Knob 映射

| 现象 | 推荐 Knob | 文档 |
|------|-----------|------|
| AICore 利用率低 | `cube_l1_reuse_mode`, `cube_nbuffer_mode` | `docs/api/config/pypto-set_pass_options.md` |
| AIVector 利用率低 | `vec_nbuffer_mode`, `mg_vec_parallel_lb` | `docs/api/config/pypto-set_pass_options.md` |
| 调度间隙大 | `device_sched_mode` (1=L2亲和, 2=公平) | `docs/api/config/pypto-set_runtime_options.md` |
| 子图过碎 | `pg_lower_bound`, `pg_parallel_lower_bound` | `docs/api/config/pypto-set_pass_options.md` |

## Verification

```bash
# 验证脚本语法
python3 -m py_compile .opencode/skills/pypto-performance-analyzer/scripts/analyze.py

# Dry-run 测试
python3 .opencode/skills/pypto-performance-analyzer/scripts/analyze.py --dry-run
```

## References

- `references/sample_swimlane.json` - Chrome Trace Format 样本数据
- `references/knobs.md` - 完整的 Knob 参数字典
- `references/artifacts.md` - 产物文件路径说明
- `docs/api/config/pypto-set_debug_options.md` - 官方 runtime_debug_mode 说明
- `docs/tutorials/debug/performance.md` - 官方泳道图性能分析教程
