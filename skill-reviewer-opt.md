# pypto-skill-reviewer 优化记录

## 问题 1: R01 规则实现与官方规范不符

**发现日期**: 2026-03-11

### 问题描述

`validate_skill.py` 脚本要求 SKILL.md 的 frontmatter 必须是 **JSON 格式**，但官方文档 (Anthropic Claude Code Skills) 使用的是 **YAML 格式**。

### 影响范围

- **15/16 skills** 被错误判定为 R01 FAIL (S0 级)
- 导致所有使用标准 YAML frontmatter 的 skills 得分被扣除 20 分

### 证据对比

| 来源 | 格式 | 示例 |
|------|------|------|
| **官方文档** | YAML | `name: explain-code` |
| **validate_skill.py** | JSON | `{"name": "explain-code"}` |

#### 官方文档示例 (YAML)

```yaml
---
name: explain-code
description: Explains code with visual diagrams and analogies. Use when explaining how code works, teaching about a codebase, or when the user asks "how does this work?"
---
```

#### validate_skill.py 当前实现

```python
# scripts/validate_skill.py 第 47-51 行
try:
    fm = json.loads(fm_text)  # ❌ 要求 JSON 格式
except json.JSONDecodeError:
    return None, 1, end_idx + 1, end_idx + 1
```

### 修复方案

#### 方案 A: 支持 YAML 格式（推荐）

```python
import yaml

def parse_frontmatter(content: str):
    # ... 提取 frontmatter 文本 ...
    try:
        fm = yaml.safe_load(fm_text)  # 支持 YAML
    except yaml.YAMLError:
        return None, 1, ...
```

需要添加依赖：
```txt
pyyaml>=6.0
```

#### 方案 B: 同时支持 YAML 和 JSON

```python
def parse_frontmatter(content: str):
    # ... 提取 frontmatter 文本 ...
    
    # 先尝试 YAML（官方格式）
    try:
        import yaml
        fm = yaml.safe_load(fm_text)
        if isinstance(fm, dict):
            return fm, ...
    except (yaml.YAMLError, ImportError):
        pass
    
    # 再尝试 JSON（向后兼容）
    try:
        fm = json.loads(fm_text)
        if isinstance(fm, dict):
            return fm, ...
    except json.JSONDecodeError:
        pass
    
    return None, 1, ...
```

### 修改文件清单

| 文件 | 修改内容 |
|------|----------|
| `scripts/validate_skill.py` | 修改 `parse_frontmatter()` 函数，支持 YAML |
| `references/rules.json` | 更新 R01 描述，明确支持 YAML 格式 |
| `requirements.txt` | 添加 `pyyaml` 依赖（如不存在） |

### 验证步骤

1. 修复后运行：`python3 scripts/validate_skill.py .agents/skills/pypto-pr-creator`
2. 预期结果：R01 PASS（而非当前的 R01 FAIL）

### 状态

- [ ] 待修复
- [ ] 待验证
- [ ] 待重新生成评审报告

---

## 问题 2: （待补充）

*后续发现的问题记录在此*


---

## 问题 2: 评审报告评分计算错误

**发现日期**: 2026-03-11

### 问题描述

生成的评审报告中存在多处评分计算错误，导致总分不准确。

### 示例：pypto-pr-creator-report.md

报告中显示总分 **28.5**，但正确计算应为 **59.9**。

#### 静态检查结果

| Rule | Severity | Dimension | 扣分 |
|------|----------|-----------|------|
| R01 | S0 | D1 | 20 |
| R11 | S1 | D2 | 10 |
| R46 × 4 | S3 | D4 | 2 × 4 = **8** |
| R47 | S2 | **D8** | 5 |

#### 报告中的错误

| 问题 | 报告中 | 正确值 |
|------|--------|--------|
| D4 扣分 | R46×4(-4) | R46×4(**-8**)，每个 S3 扣 2 分 |
| R47 归属 | D5 | **D8** |
| D9 得分 | 0 | **5.0**（无 scripts/ 得满分）|
| 总分 | 28.5 | **59.9**（S0 否决后）|

#### 正确计算

根据 `scoring-spec.md` 公式：
```
dimension_raw = max(0, 100 - sum_of_deductions)
dimension_score = dimension_raw × weight
```

| 维度 | 权重 | 扣分 | raw | score |
|------|------|------|-----|-------|
| D1 | 25% | 20 | 80 | 20.0 |
| D2 | 15% | 10 | 90 | 13.5 |
| D3 | 10% | 0 | 100 | 10.0 |
| D4 | 10% | 8 | 92 | 9.2 |
| D5 | 10% | 0 | 100 | 10.0 |
| D6 | 10% | 0 | 100 | 10.0 |
| D7 | 5% | 0 | 100 | 5.0 |
| D8 | 10% | 5 | 95 | 9.5 |
| D9 | 5% | 满分 | - | 5.0 |

**总分**: 92.2 → S0 否决后上限 59.9

### 根因分析

1. **S3 扣分错误**: 误将每个 S3 扣 2 分当作扣 1 分
2. **Rule 归属错误**: R47 属于 D8 而非 D5
3. **D9 特殊规则**: 无 scripts/ 目录时应得满分，而非 0 分
4. **总分计算**: 手动计算时遗漏或重复计算

### 修复方案

1. 将评分计算逻辑自动化，避免手动计算错误
2. 在 `validate_skill.py` 中增加 `--score` 选项，自动计算总分
3. 或创建独立的 `calculate_score.py` 脚本

### 状态

- [ ] 待修复评分计算逻辑
- [ ] 待重新生成所有评审报告

---

## 问题 3: 评分计算缺少自动化

**发现日期**: 2026-03-11

### 问题描述

`validate_skill.py` 只输出 findings JSON，不计算最终评分。评分完全依赖手动计算，导致：

1. **计算错误**: 28.5 分应为 59.9 分
2. **效率低下**: 16 个 skills 需要逐个手动计算
3. **不可复现**: 不同人计算可能得到不同结果

### 根因

- `validate_skill.py` 职责仅限于静态检查，不包含评分逻辑
- 评分公式在 `scoring-spec.md` 中定义，但没有代码实现
- 生成报告时需要人工对照公式计算

### 改进方案

#### 方案 A: 扩展 validate_skill.py

添加 `--score` 选项：

```python
# validate_skill.py
def calculate_score(findings: list, has_scripts: bool) -> dict:
    dimensions = {
        'D1': {'weight': 0.25, 'deductions': 0},
        'D2': {'weight': 0.15, 'deductions': 0},
        # ... D3-D8
        'D9': {'weight': 0.05, 'deductions': 0},
    }
    
    # 累加扣分
    for f in findings:
        if f['status'] == 'FAIL':
            dim = f['dimension']
            sev = f['severity']
            deduction = {'S0': 20, 'S1': 10, 'S2': 5, 'S3': 2}[sev]
            dimensions[dim]['deductions'] += deduction
    
    # 计算各维度分数
    scores = {}
    total = 0
    for dim, data in dimensions.items():
        raw = max(0, 100 - data['deductions'])
        score = raw * data['weight']
        scores[dim] = {'raw': raw, 'score': score, 'deductions': data['deductions']}
        total += score
    
    # D9 特殊规则
    if not has_scripts:
        scores['D9'] = {'raw': 100, 'score': 5.0, 'deductions': 0}
    
    # S0 否决
    s0_failed = any(f['severity'] == 'S0' and f['status'] == 'FAIL' for f in findings)
    if s0_failed:
        total = min(total, 59.9)
    
    # 等级映射
    grade = 'A' if total >= 90 else 'B' if total >= 75 else 'C' if total >= 60 else 'D' if total >= 40 else 'F'
    
    return {
        'total': round(total, 1),
        'grade': grade,
        's0_veto': s0_failed,
        'dimensions': scores
    }
```

#### 方案 B: 创建独立的评分脚本

```bash
python3 scripts/calculate_score.py findings.json --has-scripts false
```

#### 方案 C: 创建完整报告生成脚本

```bash
python3 scripts/generate_report.py <skill-path> --output report.md
```

### 状态

- [ ] 实现评分计算函数
- [ ] 添加到 validate_skill.py 或创建独立脚本
- [ ] 重新生成所有评审报告

---

## 问题 4: 加权公式应用错误

**发现日期**: 2026-03-11

### 问题描述

报告中的评分计算**没有正确应用加权公式**，导致分数严重偏低。

### 错误对比

#### scoring-spec.md 定义的正确公式

```
dimension_raw  = max(0, 100 - sum_of_deductions)  # 先算 100 分制的 raw
dimension_score = dimension_raw × weight            # 再乘权重
```

#### 报告中的错误计算

```
dimension_score = max_score - deductions  # 直接从满分扣分，跳过 raw 计算
```

### 具体对比 (以 D1 为例)

| 计算方式 | 公式 | 结果 |
|----------|------|------|
| **正确** | (100 - 20) × 0.25 | **20.0** |
| **报告中** | 25.0 - 20 | **5.0** |

### 完整计算对比

| 维度 | 权重 | 扣分 | 正确 raw | 正确 score | 报告 score |
|------|------|------|----------|------------|------------|
| D1 | 25% | 20 | 80 | **20.0** | 5.0 |
| D2 | 15% | 10 | 90 | **13.5** | 5.0 |
| D3 | 10% | 0 | 100 | **10.0** | 10.0 |
| D4 | 10% | 8 | 92 | **9.2** | 6.0 |
| D5 | 10% | 0 | 100 | **10.0** | 8.0 |
| D6 | 10% | 0 | 100 | **10.0** | 10.0 |
| D7 | 5% | 0 | 100 | **5.0** | 5.0 |
| D8 | 10% | 5 | 95 | **9.5** | 10.0 |
| D9 | 5% | - | 满分 | **5.0** | 0.0 |
| **总计** | | | | **92.2** | 59.0 |

### 额外问题

1. **S0 否决后**: 92.2 → **59.9** (上限)
2. **报告总分 28.5**: 与维度得分之和 59.0 也不一致

### 修复建议

在评分脚本中严格按照 scoring-spec.md 公式实现：

```python
def calculate_dimension_score(dimension: str, deductions: int, weight: float) -> float:
    raw = max(0, 100 - deductions)  # 先算 raw
    return raw * weight              # 再乘权重
```

### 状态

- [ ] 修复加权计算逻辑
- [ ] 重新生成所有评审报告

---

## 问题 5: 规则覆盖率虚假

**发现日期**: 2026-03-11

### 问题描述

报告声称"已评估 51 条规则 (100%)"，但实际只列出了 15 条规则的检查结果。

### 证据 (pypto-pass-ut-generate-report.md)

#### 报告声称

| 指标 | 值 |
|------|-----|
| 期望规则数 | 51 |
| 已评估 | 51 (100%) |
| 通过 | 36 |
| 失败 | 15 |

#### 实际列出

**问题清单 (FAIL)**:
- R01: 1 条
- R46: 1 条
- 小计: **2 条**

**通过项汇总 (PASS)**:
- D2: R11, R12, R13 = 3 条
- D3: R15 = 1 条
- D5: R07, R08, R09 = 3 条
- D6: R28, R29 = 2 条
- D7: R14, R20 = 2 条
- D8: R34, R35 = 2 条
- 小计: **13 条**

#### 覆盖率对比

| 项目 | 报告声称 | 实际列出 | 差距 |
|------|----------|----------|------|
| 已评估 | 51 | 15 | **36 条未列出** |
| 覆盖率 | 100% | 29.4% | **虚假** |

### 根因

1. **静态检查只输出 FAIL**: `validate_skill.py` 只返回失败项，不返回通过项
2. **语义检查未完整执行**: 手动生成报告时没有逐条检查 51 条规则
3. **通过项随意列举**: 列了几个示例，不是完整列表
4. **统计数据捏造**: 通过/失败数量是估算而非实际统计

### 改进方案

#### 方案 A: 修改 validate_skill.py 输出所有规则状态

```python
def validate_all_rules(skill_path: str) -> list:
    findings = []
    for rule in ALL_RULES:
        result = check_rule(rule, skill_path)
        findings.append({
            "rule_id": rule.id,
            "status": result.status,  # PASS, FAIL, SKIP
            "severity": rule.severity,
            "dimension": rule.dimension,
            "message": result.message
        })
    return findings
```

#### 方案 B: 创建规则覆盖统计脚本

```bash
python3 scripts/check_coverage.py findings.json
# 输出:
# PASS: 36, FAIL: 6, SKIP: 9
# Coverage: 42/51 (82.4%)
```

### 状态

- [ ] 修改 validate_skill.py 输出所有规则状态
- [ ] 创建规则覆盖统计功能
- [ ] 重新生成所有评审报告，确保 100% 覆盖