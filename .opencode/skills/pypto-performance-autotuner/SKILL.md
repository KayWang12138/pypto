---
name: pypto-performance-autotuner
description: PyPTO 性能自动调优技能 - NPU-only 配置搜索，分阶段优化 tile/pass/runtime knobs，支持 warmup、NPU 同步、中位数聚合。触发词：性能调优、自动调优、autotune、set_pass_options、set_runtime_options、tile shapes
license: Apache-2.0
---

# PyPTO Performance Autotuner

NPU-only 自动调优工具，分阶段搜索最优配置（tile → pass_options → runtime_options）。

## 触发场景

- "自动调优" / "性能调优" / "autotune"
- "搜索最优配置" / "find best config"
- "set_pass_options 调优" / "set_runtime_options 调优"
- "tile shapes 优化"

## 重要约束

⚠️ **所有调优必须在 NPU 模式执行**
- 禁止使用 SIM 结果作为最终 best_config 依据
- SIM 仅用于 dry-run/结构验证

## 快速开始

### Dry-run 模式（验证配置空间）

```bash
python3 /workspace/code/skills/library/shared/pypto-performance-autotuner/scripts/autotune.py --dry-run
```

### 执行自动调优（NPU 模式）

```bash
python3 /workspace/code/skills/library/shared/pypto-performance-autotuner/scripts/autotune.py \
    --bench preset:softmax_npu \
    --space references/space_softmax_npu.json \
    --trials 30 \
    --warmup 10 \
    --measure 30
```

## 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--bench` | `preset:softmax_npu` | 基准测试 |
| `--space` | `references/space_softmax_npu.json` | 搜索空间文件 |
| `--trials` | 30 | 每阶段最大试验次数 |
| `--warmup` | 10 | 预热迭代次数（丢弃） |
| `--measure` | 30 | 测量迭代次数 |
| `--dry-run` | False | 仅验证 schema，不执行 |
| `--out` | `autotune-results.json` | 输出结果文件 |

## 评估协议

1. **Warmup**: 每次试验前执行 N 次预热（丢弃结果）
2. **NPU Synchronize**: 每次测量前后调用 `torch_npu.npu.synchronize()`
3. **Outlier Removal**: IQR 1.5× 规则剔除离群值
4. **Aggregation**: 取中位数 `median_latency_ms`
5. **Best Update**: 新 best 必须优于当前 best 至少 2% (`new < best * 0.98`)
6. **Early Stop**: 连续 5 次试验无改进则提前结束该阶段

## 已废弃参数

以下参数已废弃，不要在搜索空间中使用：
- `stitch_function_inner_memory` → 使用 `stitch_function_max_num`
- `stitch_function_outcast_memory` → 使用 `stitch_function_max_num`
- `stitch_function_num_initial` → 使用 `stitch_function_max_num`

## Verification

```bash
# 验证脚本语法
python3 -m py_compile /workspace/code/skills/library/shared/pypto-performance-autotuner/scripts/autotune.py

# Dry-run 测试
python3 /workspace/code/skills/library/shared/pypto-performance-autotuner/scripts/autotune.py --dry-run
```

## References

- `references/space_softmax_npu.json` - Softmax 搜索空间
- `references/schema.md` - 搜索空间 Schema 说明
