# API 探索报告

> **生成时间**: {timestamp}

---

<!-- REQUIRED -->
## 1. 概述

### 1.1 输入摘要

{输入内容摘要}

### 1.2 算子分类

- **类型**: {Vector / Cube / 混合}
- **判断依据**: {type_reason}

---

## 2. 公式分解

| 步骤 | 操作类型 | 数学表达 | 说明 |
|------|----------|----------|------|
| 1 | {op_type} | {math_expr} | {desc} |

---

<!-- REQUIRED -->
## 3. API 映射

### 3.1 映射结果

| 步骤 | 数学表达 | PyPTO API | 映射级别 | 约束满足 |
|------|----------|-----------|----------|----------|
| 1 | {expr} | `{api}` | {direct/substitute/unsupported} | {✓/⚠/✗} |

### 3.2 Substitute 配方

<!-- 仅 substitute 时填写 -->

```
{operation}: {recipe}
```

---

## 4. 约束检查

### 4.1 入口约束

| 约束项 | 要求 | 输入值 | 结果 |
|--------|------|--------|------|
| dtype | {supported} | {input_dtype} | {✓/✗} |
| contiguous | 必须 | — | {✓/需确保} |

### 4.2 API 约束

| API | 约束项 | 要求 | 结果 |
|-----|--------|------|------|
| {api} | dtype | {list} | {✓/✗} |

---

## 5. Tiling 需求

| 算子类型 | 需调用 API |
|----------|-----------|
| {type} | `pypto.set_{vec/cube}_tile_shapes()` |

---

## 6. 风险评估

### 6.1 阻断问题

| 问题 | 原因 | 建议 |
|------|------|------|
| {issue} | {reason} | {suggestion} |

### 6.2 注意事项

| 注意点 | 说明 |
|--------|------|
| {warning} | {desc} |

---

<!-- REQUIRED -->
## 7. 证据索引

| 信息 | 文档路径 |
|------|----------|
| API 存在性 | `docs/api/operation/index.md` |
| {api} 文档 | `docs/api/operation/pypto-{api}.md` |
| 入口约束 | `docs/api/others/pypto-from_torch.md` |

---

<!-- REQUIRED -->
## 8. 结论

- **可行性**: {可行 / 需调整 / 不可行}
- **主要问题**: {main_issue}
