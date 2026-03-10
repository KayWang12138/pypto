# 评分规范

本规范定义 PyPTO Skill Reviewer 的评分方法：维度权重、扣分计算、S0 否决规则、最终等级映射与覆盖率统计。

## 术语

- **规则（rule）**：在 [rules.json](rules.json) 中定义的检查项（R01..R51）。
- **finding**：规则执行后的单条结果（PASS/FAIL/SKIP），包含严重级别、维度与证据。
- **维度（dimension）**：D1..D9（以及 D0 作为附加检查）。

## 权重

维度权重由 [rules.json](rules.json) 的 `dimensions` 定义：

- D1：25
- D2：15
- D3：10
- D4：10
- D5：10
- D6：10
- D7：5
- D8：10
- D9：5
- D0：0（不计入总分，仅展示）

## 严重级别扣分

每条 FAIL 的 finding 会产生扣分，扣分强度由严重级别决定（同 [rules.json](rules.json) 的 `severity_deductions`）：

- S0：20
- S1：10
- S2：5
- S3：2

## S0 否决（Hard Gate）

若存在任何 **S0 且 FAIL** 的 finding，则触发否决：

- 总分强制为 0
- 等级强制为 F
- 报告仍需包含全部 findings（用于修复）

## 维度得分计算

对每个维度 Dx（不含 D0），计算：

1. 维度原始分 `raw = 100 - sum(severity_deduction for FAIL findings in Dx)`
2. 截断到区间 `[0, 100]`
3. 维度加权分 `weighted = raw * (weight(Dx) / 100)`

说明：

- PASS 与 SKIP 不扣分。
- 同一规则若出现多条 finding，按所有 FAIL 累加扣分（除非质量闸门判定为重复/误绑并过滤）。

## 总分

总分为所有维度加权分之和（不含 D0）：

`total = sum(weighted(D1..D9))`

## 等级映射

在未触发 S0 否决时：

- A：90-100
- B：80-89
- C：70-79
- D：60-69
- F：0-59

## 覆盖率

覆盖率用于表示评审范围的完整性：

- **静态覆盖**：静态规则中有结果的规则数 / 静态规则总数
- **语义覆盖**：语义规则中有结果的规则数 / 语义规则总数
- **总体覆盖**：所有规则中有结果的规则数 / 规则总数

SKIP 也计入“有结果”，因为它体现了明确的判断。

## 质量闸门（Quality Gate）

在计分前应用质量闸门：

- 过滤证据不足（snippet 无法逐字匹配、缺文件/行号、纯泛化描述）
- 合并明显重复的 findings
- 过滤明显误绑（指向非目标 skill 文件）

被过滤/合并的条目必须在报告中记录原因（透明性要求）。
