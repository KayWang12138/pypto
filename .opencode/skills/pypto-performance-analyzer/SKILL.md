---
name: pypto-performance-analyzer
description: PyPTO 性能分析技能 - 解析泳道图(merged_swimlane.json)和计算图JSON，识别性能瓶颈，提供优化建议。触发词：性能分析、swimlane、泳道图、计算图、性能瓶颈、pypto set_pass_options、AICore利用率、AIVector利用率
license: Apache-2.0
---

# PyPTO Performance Analyzer

分析 PyPTO 算子的泳道图和计算图，识别性能瓶颈，生成可操作的调优建议。

## 触发场景

- "分析泳道图" / "查看性能数据" / "swimlane 分析"
- "AICore 利用率低" / "性能瓶颈在哪里"
- "生成性能报告" / "pypto 性能分析"
- "计算图分析" / "子图合并问题"

## 快速开始

### Dry-run 模式（无需 NPU）

```bash
python3 /workspace/code/skills/library/shared/pypto-performance-analyzer/scripts/analyze.py --dry-run
```

### 分析真实产物

```bash
# 自动检测最新 output 目录
python3 /workspace/code/skills/library/shared/pypto-performance-analyzer/scripts/analyze.py --pypto-repo /workspace/code/pypto

# 指定输出目录
python3 /workspace/code/skills/library/shared/pypto-performance-analyzer/scripts/analyze.py --output-dir /path/to/output --report-out /path/to/report.md
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
python3 -m py_compile /workspace/code/skills/library/shared/pypto-performance-analyzer/scripts/analyze.py

# Dry-run 测试
python3 /workspace/code/skills/library/shared/pypto-performance-analyzer/scripts/analyze.py --dry-run
```

## References

- `references/sample_swimlane.json` - Chrome Trace Format 样本数据
- `references/knobs.md` - 完整的 Knob 参数字典
- `references/artifacts.md` - 产物文件路径说明
