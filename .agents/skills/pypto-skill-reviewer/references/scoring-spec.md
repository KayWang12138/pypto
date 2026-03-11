# Scoring Specification

This document defines the scoring algorithm for the PyPTO Skill Reviewer.

## Dimensions and Weights

| Dimension | Name | Weight | Max Score |
|-----------|------|--------|-----------|
| D1 | Frontmatter 元数据 | 25% | 25.0 |
| D2 | 简洁性与效率 | 15% | 15.0 |
| D3 | 文件结构与导航 | 10% | 10.0 |
| D4 | 语言与表达 | 10% | 10.0 |
| D5 | 精确性与可执行性 | 10% | 10.0 |
| D6 | 工作流完整性 | 10% | 10.0 |
| D7 | 模式与最佳实践 | 5% | 5.0 |
| D8 | 反模式检测 | 10% | 10.0 |
| D9 | 脚本与代码质量 | 5% | 5.0 |

**D0 (附加检查)** has weight 0 — its findings contribute to context but do not affect scoring.

**D9 special rule**: When no `scripts/` directory exists, D9 receives full marks automatically.

## Severity Levels and Deductions

| Severity | Deduction | Meaning |
|----------|-----------|---------|
| S0 | 20 points | Fatal defect — triggers veto mechanism |
| S1 | 10 points | Major issue |
| S2 | 5 points | Moderate issue |
| S3 | 2 points | Minor suggestion |

## Scoring Formula

### Per-Dimension Score

Each dimension is scored on a 0–100 internal scale, then multiplied by its weight:

```
dimension_raw  = max(0, 100 - sum_of_deductions_in_dimension)
dimension_score = dimension_raw × weight
```

Where `sum_of_deductions_in_dimension` is the sum of all FAIL deductions for rules in that dimension.

Example: D1 (weight 25%). If R04 (S1, -10) and R05 (S2, -5) both fail:
```
D1_raw   = max(0, 100 - 10 - 5) = 85
D1_score = 85 × 0.25 = 21.25
```

This avoids low-weight dimensions being zeroed out by a single S1 deduction (e.g., D7 at 5% weight: one S1 gives `max(0,100-10)×0.05 = 4.5` instead of `max(0,5-10) = 0`).

### Total Score

```
total_score = Σ dimension_scores (for D1 through D9)
```

### S0 Veto Mechanism

If **any** S0-severity rule has status FAIL:
- Total score is capped at **59.9**
- Maximum grade is **D**
- The veto is noted in the report summary

S0 rules: R01, R02, R03, R34.

## Grade Mapping

| Grade | Score Range |
|-------|------------|
| A | 90.0 – 100.0 |
| B | 75.0 – 89.9 |
| C | 60.0 – 74.9 |
| D | 40.0 – 59.9 |
| F | 0.0 – 39.9 |

## Issue Aggregation

Findings from static and semantic checks are aggregated into **issues** by location:

- **Aggregation key**: `file + line_range` (findings within ±5 lines merge into one issue)
- Each issue records all matching `rule_id` values
- Each issue gets a single unified fix suggestion covering all matched rules
- Fix suggestions include **before** and **after** content comparison

## Counting Rules

- **PASS**: Rule check succeeded — no deduction
- **FAIL**: Rule check failed — deduction applied per severity
- **WARN**: Advisory only (used for S3 rules where the check is inconclusive) — counted separately, no deduction
- **SKIP**: Rule not applicable (e.g., R43 when `context` field is absent) — not counted

Report statistics:
- `pass_count`: Number of PASS results
- `fail_count`: Number of FAIL results
- `warn_count`: Number of WARN results
