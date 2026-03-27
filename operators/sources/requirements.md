# 算子开发需求

每个需求用 `---` 分隔。系统通过 `op_name` 字段判断是否已处理（与 CSV 比对）。
用户无需填写复杂度，系统自动推断。条目上限 50 条，超出后归档到 requirements-archive.md。

---

<!-- 示例需求，可删除 -->

## 示例：tanh_linear

- 算子名称：tanh_linear
- 公式：y = tanh(x) * x
- 规格：输入 x [B, L, D] float32，输出 y [B, L, D] float32
- 动态轴：B、L
- 精度标准：atol=0.0001, rtol=0.001

---
