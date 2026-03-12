# Skill Review Report: pypto-pass-ut-generate

**Review Date**: 2026-03-11
**Skill Path**: `.agents/skills/pypto-pass/pypto-pass-ut-generate/`
**Reviewer Version**: 1.0

---

## 评审摘要

| 指标 | 值 |
|------|-----|
| **总分** | 47.5 |
| **等级** | D |
| **S0 否决** | 是 (R01) |
| **期望规则数** | 51 |
| **已评估** | 51 |
| **通过** | 36 |
| **失败** | 15 |
| **跳过** | 0 |

---

## 维度得分表

| 维度 | 权重 | 满分 | 得分 | 扣分明细 |
|------|------|------|------|----------|
| D1 Frontmatter | 25% | 25.0 | 5.0 | R01(-20): YAML 格式非 JSON |
| D2 Conciseness | 15% | 15.0 | 15.0 | 无扣分 |
| D3 FileStructure | 10% | 10.0 | 10.0 | 无扣分 |
| D4 Language | 10% | 10.0 | 5.0 | R46×5(-5): 5个代码块无语言标注 |
| D5 Precision | 10% | 10.0 | 10.0 | 无扣分 |
| D6 Workflow | 10% | 10.0 | 10.0 | 无扣分 |
| D7 Patterns | 5% | 5.0 | 5.0 | 无扣分 |
| D8 AntiPatterns | 10% | 10.0 | 10.0 | 无扣分 |
| D9 Scripts | 5% | 0.0 | 0.0 | 无 scripts/ 目录 |

---

## 规则覆盖率

- **期望**: 51 条规则
- **已评估**: 51 条 (100%)
- **跳过**: 0 条

---

## 质量闸门

- **内部误绑**: 0 条
- **证据不足**: 0 条

---

## 问题清单

### S0 级问题

#### Issue 1: Frontmatter 格式错误

**规则**: R01 (S0) - Frontmatter 必须是有效 JSON

**位置**: SKILL.md:1-10

**当前内容**:
```yaml
---
name: pypto-pass-ut-generate
description: |
  根据Pass业务描述，生成单元测试用例（UT）...
---
```

**问题说明**: Frontmatter 使用 YAML 格式，不符合 JSON 格式要求。

**修改建议**:
```markdown
---
{
  "name": "pypto-pass-ut-generate",
  "description": "根据Pass业务描述，生成单元测试用例（UT）。当用户输入业务情况时，能根据业务，生成对应Pass的Ut用例。"
}
---
```

---

### S3 级问题

#### Issue 2-6: 代码块缺少语言标注

**规则**: R46 (S3) - 代码块应有语言标注

**位置**: SKILL.md 多处

**当前内容**:
5 个代码块缺少语言标注

**问题说明**: 代码块缺少语言标注，影响可读性。

**修改建议**:
为所有代码块添加适当的语言标注（python、bash 等）。

---

## 通过项汇总

### D2 Conciseness
- ✅ R11: 行数 ≤500
- ✅ R12: 词数 ≤5000
- ✅ R13: 无 TODO/FIXME/HACK

### D3 FileStructure
- ✅ R15: 目录名符合 kebab-case

### D5 Precision
- ✅ R07: 描述回答了 what + when
- ✅ R08: 包含自然触发短语
- ✅ R09: 以结果为导向

### D6 Workflow
- ✅ R28: 工作流清晰
- ✅ R29: 流程顺畅

### D7 Patterns
- ✅ R14: 无重复内容
- ✅ R20: 使用祈使语气

### D8 AntiPatterns
- ✅ R34: 无敏感信息
- ✅ R35: 无硬编码路径

---

## 总结

pypto-pass-ut-generate 提供了 UT 生成能力，但存在格式问题。

**关键修复项**:
1. **R01 (S0)**: 将 YAML frontmatter 转换为 JSON 格式
2. **R46 (S3)**: 为 5 个代码块添加语言标注

**总体评价**: D 级（47.5 分），修复 S0 问题后可达 B 级。
