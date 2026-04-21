---
name: pypto-operator-auto-tuner
description: "PyPTO 算子自动化性能调优脚本。包含泳道图数据提取、AIV 依赖链分析和 leafhash 到源码的映射。"
triggers: ["swimlane analysis", "AIV dependency", "leafhash", "auto tune", "performance scripts"]
---

# PyPTO 算子自动调优器

用于 PyPTO 算子性能分析与调优的自动化脚本。

## 脚本

| 脚本 | 用途 | 用法 |
|--------|---------|-------|
| `scripts/analyze_swimlane.py` | 从 profiling 输出中提取并分析泳道图数据 | `python3 scripts/analyze_swimlane.py <profiling_data>` |
| `scripts/analyze_aiv_dep_chains.py` | 分析 AIV 依赖链以识别流水线停顿 | `python3 scripts/analyze_aiv_dep_chains.py <profiling_data>` |
| `scripts/leafhash_to_code.py` | 将 leafhash 标识符映射回源码位置 | `python3 scripts/leafhash_to_code.py <leafhash>` |

## 工作流

1. 在开启 profiling 的情况下运行目标算子。
2. 对 profiling 输出执行 `analyze_swimlane.py`，提取各阶段耗时。
3. 执行 `analyze_aiv_dep_chains.py`，识别依赖瓶颈。
4. 使用 `leafhash_to_code.py` 将性能关键节点映射回源码。
5. 根据分析结果应用优化，重新 profiling 并对比。

## 参考资料

- `references/leafhash-to-code-mapping.md` — leafhash 标识符到前端源码的映射指南。
