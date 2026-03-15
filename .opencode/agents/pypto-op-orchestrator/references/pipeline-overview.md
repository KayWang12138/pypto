# Pipeline Overview

## 流程图

```mermaid
flowchart TD
    A[用户需求] --> B[pypto-op-orchestrator]
    B --> C[Stage 1: pypto-intent-understanding]
    C --> D[spec.md]
    D --> E[Stage 2A: pypto-golden-generator]
    E --> F["{op}_golden.py"]
    F --> G[Stage 2B: pypto-op-design]
    D --> G
    G --> H[design.md]
    H --> I["Stage 3: pypto-op-develop（功能实现）"]
    F --> I
    I --> J{orchestrator 判定}
    J -->|exit 0, 精度 PASS| L[Stage 5: pypto-op-perf-autotuner]
    J -->|Not equal to tolerance, loop < 10| M[Stage 4: 精度定位]
    J -->|Not equal to tolerance, loop >= 10| BLOCKED[BLOCKED_ACCURACY]
    J -->|其他错误| I
    M --> M0[初步排查: pypto-op-accuracy-verify]
    M0 --> M1[默认: pypto-binary-search-verify]
    M1 -->|定位成功| I
    M1 -->|不适用| M2[降级: pypto-binary-search-without-verify]
    M2 --> I
    L --> P[Stage 6: pypto-op-perf-analyzer]
    P --> Q[结构化结果摘要]
```

## Stage 总览

| Stage | 名称 | 执行者 | 关键输入 | 关键输出 | 成功标准 |
|-------|------|--------|----------|----------|----------|
| 0 | 上下文解析 | orchestrator | 用户请求 / 现有目录 | 唯一工作目录 | `custom/{op}/` 目录确定 |
| 1 | 需求理解 | `pypto-intent-understanding` | 用户需求 | `spec.md` | 文件存在 |
| 2A | 生成 Golden | `pypto-golden-generator` | `spec.md` | `{op}_golden.py` | 文件存在 |
| 2B | 生成 Design | `pypto-op-design` | `spec.md` + `{op}_golden.py` | `design.md` | 文件存在 |
| 3 | 功能实现 | `pypto-op-develop` | `spec.md` + `design.md` + `{op}_golden.py` | `test_{op}.py` + `{op}_impl.py` + `README.md` | 见 Stage 3 判定 |
| 4 | 精度定位 | 见精度定位策略 | 实现 + 中间结果 | 修复建议 | 产出定位报告 |
| 5 | 性能采集与调优 | `pypto-op-perf-autotuner` | 精度通过实现 | 新 output / 调优建议 | 新 `output/output_*` 存在 |
| 6 | 性能分析 | `pypto-op-perf-analyzer` | 性能工件 | 性能摘要 | 报告产出 |

## Stage 2 串行

```
Stage 2A (golden-generator) → Stage 2B (op-design)
```

design 生成时参考 golden 代码结构和函数签名，避免两者不一致。
