---
name: pypto-op-discover
description: |
  自动发现新的 PyPTO 算子候选，每次补充 1 个新算子到 scan_results.csv。
  由 pypto-op-autodev 在候选队列不足时触发（通过 OpenCode Agent pypto-op-discover）。
  不由用户直接触发。
---

# PyPTO 算子自动发现

每次执行仅补充 **1 个**新算子候选到 `operators/scan_results.csv`，通过 `add_op.py` 写入。

## 输入

- `operators/scan_results.csv`（通过脚本读取已有算子列表）
- 互联网资源（WebSearch/WebFetch，可选）

## 执行步骤

### Step 1：读取已有算子

```bash
python .agents/skills/pypto-op-autodev/scripts/get_progress.py \
  --csv operators/scan_results.csv
```

从 `by_category` 统计各类别已有算子数量，用于类别差异性筛选。
从输出提取 `op_name` 列表（`status` 任意），避免重复推荐。

### Step 2：生成候选算子

按以下优先级搜索候选（找到满足条件的立即停止）：

**P0 经典算子（内置知识）**
首先检查以下列表中是否有未在 CSV 中出现的算子：
- softmax, gelu, silu, relu, tanh, sigmoid, layernorm, rms_norm, batchnorm
- matmul, batch_matmul, linear
- scaled_dot_product_attention, flash_attention
- max_pool2d, avg_pool2d, adaptive_avg_pool2d
- embedding, rotary_embedding
- sum_reduction, mean_reduction, max_reduction

**P1 官方仓库（需要 WebSearch/WebFetch）**
搜索 `site:github.com FlagGems OR FlagAttention OR AscendC 算子 OR operator` 获取候选。

**P2 开源 LLM 模型**
从 Llama/Qwen/DeepSeek/Mistral 模型结构中提取算子（如 rotary_embedding、grouped_query_attention 等）。

**P3 学术论文**
高引用算子论文（FlashAttention、RoPE、SwiGLU 等）中的算子。

**P4 LLM 知识**
基于 PyTorch/JAX/TensorFlow 内置算子知识直接推断。

### Step 3：筛选候选

从候选中选择 **1 个**满足以下条件的算子：
1. `op_name` 不在 CSV 已有算子中（精确匹配或相似匹配）
2. **类别差异性优先**：优先选择 CSV 中数量最少的类别
3. 不做前置可行性验证（实现时若失败，由 orchestrator 判定 dev_result）

### Step 4：推断 complexity 和 category

根据 `references/category-rules.md` 和 `references/complexity-rules.md` 推断。
不确定时取保守估计。

### Step 5：写入 CSV

```bash
python .agents/skills/pypto-op-autodev/scripts/add_op.py \
  --csv operators/scan_results.csv \
  --op {op_name} --source auto_discovered \
  --complexity {complexity} --category {category}
```

输出 `is_new: true` 则成功；`is_new: false` 说明重复（重新选择另一个，最多尝试 3 次）。

## 参考文档

- `references/category-rules.md`：类别推断规则
- `references/complexity-rules.md`：复杂度判定规则
