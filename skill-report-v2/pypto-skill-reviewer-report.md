# 技能评审报告

## 1. 评审摘要

| 项目 | 结果 |
|------|------|
| 技能名称 | pypto-skill-reviewer |
| 评审时间 | 2026-03-11 13:49:37 |
| 总分 | 99.55 / 100 |
| 等级 | A |
| S0 否决 | 否 |
| 规则统计 | 通过 48 / 失败 2 / 警告 0 / 跳过 1 |

## 2. 维度评分表

| 维度 | 名称 | 原始分(0-100) | 权重 | 加权分 | 扣分明细 |
|------|------|---------------|------|--------|----------|
| D1 | Frontmatter 元数据 | 100 | 25% | 25.00 | 无 |
| D2 | 简洁性与效率 | 100 | 15% | 15.00 | 无 |
| D3 | 文件结构与导航 | 100 | 10% | 10.00 | 无 |
| D4 | 语言与表达 | 98 | 10% | 9.80 | R21(-2) |
| D5 | 精确性与可执行性 | 100 | 10% | 10.00 | 无 |
| D6 | 工作流完整性 | 100 | 10% | 10.00 | 无 |
| D7 | 模式与最佳实践 | 100 | 5% | 5.00 | 无 |
| D8 | 反模式检测 | 100 | 10% | 10.00 | 无 |
| D9 | 脚本与代码质量 | 95 | 5% | 4.75 | R50(-5) |
| 合计 | - | - | 100% | 99.55 | - |

## 3. 规则覆盖率

- 静态规则：PASS 29 + FAIL 0 + SKIP 0 = 29
- 语义规则：PASS 19 + FAIL 2 + SKIP 1 = 22
- 总评估规则数：51 / 51
- 覆盖率：100.00%

### 规则状态明细

| 规则 | 状态 | 维度 | 严重度 | 类型 |
|------|------|------|--------|------|
| R01 | PASS | D1 | S0 | static |
| R02 | PASS | D1 | S0 | static |
| R03 | PASS | D1 | S0 | static |
| R04 | PASS | D1 | S1 | static |
| R05 | PASS | D1 | S2 | static |
| R06 | PASS | D1 | S3 | static |
| R07 | PASS | D1 | S1 | semantic |
| R08 | PASS | D1 | S2 | semantic |
| R09 | PASS | D1 | S2 | semantic |
| R10 | PASS | D1 | S2 | static |
| R11 | PASS | D2 | S1 | static |
| R12 | PASS | D2 | S2 | static |
| R13 | PASS | D2 | S1 | static |
| R14 | PASS | D2 | S2 | semantic |
| R15 | PASS | D3 | S2 | static |
| R16 | PASS | D3 | S2 | static |
| R17 | PASS | D3 | S2 | static |
| R18 | PASS | D3 | S2 | static |
| R19 | PASS | D3 | S2 | semantic |
| R20 | PASS | D4 | S2 | semantic |
| R21 | FAIL | D4 | S3 | semantic |
| R22 | PASS | D4 | S2 | static |
| R23 | PASS | D4 | S2 | semantic |
| R24 | PASS | D5 | S2 | semantic |
| R25 | PASS | D5 | S2 | semantic |
| R26 | PASS | D5 | S1 | semantic |
| R27 | PASS | D5 | S2 | semantic |
| R28 | PASS | D6 | S1 | semantic |
| R29 | PASS | D6 | S2 | semantic |
| R30 | PASS | D6 | S2 | semantic |
| R31 | PASS | D6 | S2 | semantic |
| R32 | PASS | D7 | S3 | semantic |
| R33 | PASS | D7 | S3 | semantic |
| R34 | PASS | D8 | S0 | static |
| R35 | PASS | D8 | S1 | static |
| R36 | PASS | D8 | S1 | static |
| R37 | PASS | D8 | S1 | static |
| R38 | PASS | D8 | S2 | static |
| R39 | PASS | D9 | S2 | static |
| R40 | PASS | D9 | S2 | static |
| R41 | PASS | D9 | S2 | static |
| R42 | PASS | D9 | S2 | semantic |
| R43 | PASS | D0 | S2 | static |
| R44 | SKIP | D0 | S2 | semantic |
| R43 | PASS | D0 | S2 | static |
| R44 | PASS | D1 | S2 | static |
| R45 | PASS | D2 | S2 | static |
| R46 | PASS | D4 | S3 | static |
| R47 | PASS | D8 | S2 | static |
| R50 | FAIL | D9 | S2 | semantic |
| R51 | PASS | D7 | S2 | semantic |

## 4. 质量门禁

| 类型 | 数量 | 说明 |
|------|------|------|
| internal_misbound_rule_or_evidence | 0 | 未发现引用 reviewer 自身而非目标 skill 的错绑证据 |
| low_information_snippet | 0 | 未发现 snippet 无法在源文件逐字匹配的条目 |

本次无被质量门禁过滤的 findings。

## 5. 问题列表

### S0 致命缺陷

无。

### S1 重大问题

无。

### S2 中等问题

#### 问题 1：脚本未优雅处理缺失依赖
- 命中规则：`R50(S2)`
- 位置：`scripts/validate_skill.py:20`
- 证据：`import yaml`
- 问题说明：脚本依赖 `pyyaml`（`scripts/validate_skill.py:11` 中也写明 `Requirements: Python 3.8+, pyyaml.`），但 `import yaml` 未加 `try/except`，在缺失依赖时会直接抛 traceback，不满足“优雅处理缺失依赖”。
- 修改建议（before/after）：

```python
# before
import yaml

# after
try:
    import yaml
except ImportError as e:
    print("缺少依赖: pyyaml，请先执行: pip install pyyaml", file=sys.stderr)
    sys.exit(2)
```

### S3 轻微建议

#### 问题 2：错误处理中出现模糊措辞
- 命中规则：`R21(S3)`
- 位置：`SKILL.md:125`
- 证据：`- **frontmatter 无效**：R01 FAIL 触发 S0 否决。尽可能继续检查其他规则（即不依赖 frontmatter 数据的规则）。`
- 问题说明：`尽可能` 属于弱化词，执行边界不够确定，可能导致不同执行者行为不一致。
- 修改建议（before/after）：

```markdown
before: - **frontmatter 无效**：R01 FAIL 触发 S0 否决。尽可能继续检查其他规则（即不依赖 frontmatter 数据的规则）。
after:  - **frontmatter 无效**：R01 FAIL 触发 S0 否决。继续检查所有不依赖 frontmatter 数据的规则，并将依赖 frontmatter 的规则标记为 SKIP。
```

## 6. 通过规则汇总

| 维度 | 通过规则 |
|------|----------|
| D1 | R01, R02, R03, R04, R05, R06, R07, R08, R09, R10, R44 |
| D2 | R11, R12, R13, R14, R45 |
| D3 | R15, R16, R17, R18, R19 |
| D4 | R20, R22, R23, R46 |
| D5 | R24, R25, R26, R27 |
| D6 | R28, R29, R30, R31 |
| D7 | R32, R33, R51 |
| D8 | R34, R35, R36, R37, R38, R47 |
| D9 | R39, R40, R41, R42 |
| D0 | R43, R43 |