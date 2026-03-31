# References 检查报告

**检查时间**：{timestamp}  
**检查范围**：`.agents/skills/*/references/`  
**文档基准**：`docs/`

---

## 一、检查概况

| 统计项 | 数量 |
|--------|------|
| 检查 skill 数 | {skill_count} |
| 检查文件数 | {file_count} |
| P0 问题数 | {p0_count} |
| P1 问题数 | {p1_count} |
| P2 问题数 | {p2_count} |

---

## 二、P0 必须修复

> 事实性错误，必须修复

{p0_issues}

---

## 三、P1 建议修复

> 歧义或可能导致误解的问题，建议修复

{p1_issues}

---

## 四、P2 可选修复

> 模糊或不够清晰的问题，可选修复

{p2_issues}

---

## 五、无需修复项说明

{no_fix_items}

---

## 六、修复建议汇总

| 问题 | 文件 | 修复方案 |
|------|------|----------|
{fix_summary}

---

## 七、联动修改检查

{linkage_check}

---

*报告生成完成。建议按 P0 → P1 → P2 顺序修复。*