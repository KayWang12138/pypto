# PyPTO 算子规格文档：embedding

## 基本信息

| 项目 | 内容 |
|------|------|
| 算子名称 | embedding |
| 类别 | embedding |
| 复杂度 | medium |
| 版本 | 1.0 |
| 生成时间 | 2026-03-30 |

---

## 1. 算子描述

### 1.1 功能描述

Embedding 算子是一个查表操作，将索引张量映射到对应的嵌入向量。这是自然语言处理和推荐系统中常用的基础算子。

### 1.2 数学公式

```
output = embedding_lookup(weight, indices)

其中:
- weight: [vocab_size, embed_dim] 的 embedding 表
- indices: [batch, seq] 的索引张量
- output[i, j, :] = weight[indices[i, j], :]

若 padding_idx 不为 None 且 indices[i, j] == padding_idx:
  output[i, j, :] = 0
```

### 1.3 数据流图

```
    输入 indices              权重 weight (embedding table)
+------------------+      +----------------------------+
|  [batch, seq]    |      |  [vocab_size, embed_dim]   |
|     int64        |      |       float32              |
+--------+---------+      +-------------+--------------+
         |                              |
         |    +-------------------------+
         |    |
         v    v
    +-------------------------------------+
    |         embedding 查表操作          |
    |   output[i,j,:] = weight[indices[i,j], :]
    |   若 padding_idx 不为 None 且      |
    |   indices[i,j] == padding_idx，    |
    |   则 output[i,j,:] = 0             |
    +-----------------+-------------------+
                      |
                      v
              +------------------+
              |  输出 output      |
              |[batch, seq, embed_dim] |
              |     float32      |
              +------------------+
```

---

## 2. 关键特性

| 特性名称 | 是否需要 | 优先级 | 置信度 | 实现说明 |
|----------|----------|--------|--------|----------|
| 动态轴 batch/seq | 需要 | P0 | 高 | 必须支持动态 shape |
| padding_idx | 需要 | P1 | 高 | 填充索引置零处理 |
| max_norm | 不需要 | P3 | 高 | 暂不支持 |
| scale_grad_by_freq | 不需要 | P3 | 高 | 训练相关，推理不需要 |
| sparse | 不需要 | P3 | 高 | 训练相关，推理不需要 |

---

## 3. 输入输出规格

### 3.1 输入

| 名称 | Shape | DType | 动态轴 | 描述 |
|------|-------|-------|--------|------|
| indices | [batch, seq] | int64 | batch, seq | 索引张量，值范围 [0, vocab_size) |
| weight | [vocab_size, embed_dim] | float32 | 无 | Embedding 查找表 |

### 3.2 输出

| 名称 | Shape | DType | 动态轴 | 描述 |
|------|-------|-------|--------|------|
| output | [batch, seq, embed_dim] | float32 | batch, seq | 嵌入向量输出 |

### 3.3 可选参数

| 参数名 | 类型 | 默认值 | 优先级 | 描述 |
|--------|------|--------|--------|------|
| padding_idx | int 或 None | None | P1 | 指定填充索引，该位置的输出置零 |

---

## 4. 动态轴配置

### 4.1 动态轴定义

| 轴名称 | 含义 | 范围 | 说明 |
|--------|------|------|------|
| batch | 批次维度 | [1, 1024] | 支持动态 shape |
| seq | 序列长度 | [1, 4096] | 支持动态 shape |

### 4.2 静态维度

| 维度名称 | 含义 | 说明 |
|----------|------|------|
| vocab_size | 词表大小 | 静态，由模型决定 |
| embed_dim | 嵌入维度 | 静态，由模型决定 |

### 4.3 PyPTO 动态轴配置示例

```python
# 动态轴配置 (用于 PyPTO kernel)
# batch 和 seq 为动态轴，vocab_size 和 embed_dim 为静态
dynamic_axes = {
    "indices": {0: "batch", 1: "seq"},
    "output": {0: "batch", 1: "seq"}
}
```

---

## 5. 精度要求

| 指标 | 值 | 说明 |
|------|-----|------|
| atol | 0.001 | 绝对误差容限 |
| rtol | 0.001 | 相对误差容限 |
| dtype | float32 | 主计算精度 |

---

## 6. 边界条件处理

| 条件 | 处理方式 | 说明 |
|------|----------|------|
| indices 超出范围 | 未定义行为 | 用户需保证 indices ∈ [0, vocab_size) |
| padding_idx | 置零 | 若 indices == padding_idx，输出置零 |
| 空输入 | 正常处理 | batch=0 或 seq=0 时返回空张量 |

---

## 7. 性能目标

| 指标 | 目标值 | 说明 |
|------|--------|------|
| 性能基准 | 首跑精度成功性能的 2 倍 | 基础优化目标 |

---

## 8. 参考信息

### 8.1 PyTorch 参考

```python
torch.nn.functional.embedding(
    input,
    weight,
    padding_idx=None,
    max_norm=None,
    norm_type=2.0,
    scale_grad_by_freq=False,
    sparse=False
)
```

### 8.2 等价 PyTorch 实现

```python
def embedding_golden(indices, weight, padding_idx=None):
    """
    Args:
        indices: [batch, seq] int64
        weight: [vocab_size, embed_dim] float32
        padding_idx: int or None
    Returns:
        output: [batch, seq, embed_dim] float32
    """
    output = weight[indices]
    if padding_idx is not None:
        mask = (indices == padding_idx).unsqueeze(-1)
        output = output.masked_fill(mask, 0.0)
    return output
```

---

## 9. 典型配置

| 配置名称 | 类型 | 优先级 | 参数 | 输入 Shape | 输出 Shape | 说明 |
|----------|------|--------|------|------------|------------|------|
| 小模型 | 性能 | P0 | padding_idx=None | indices: [1, 128], weight: [32000, 4096] | [1, 128, 4096] | 小 batch 测试 |
| 中模型 | 性能 | P0 | padding_idx=None | indices: [8, 512], weight: [32000, 4096] | [8, 512, 4096] | 典型推理场景 |
| 大模型 | 性能 | P0 | padding_idx=None | indices: [32, 2048], weight: [128000, 4096] | [32, 2048, 4096] | 大模型推理 |
| padding 功能 | 功能 | P1 | padding_idx=0 | indices: [4, 256], weight: [10000, 256] | [4, 256, 256] | 验证 padding 功能 |

---

## 10. 验收标准

- [ ] 精度验证通过（atol=0.001, rtol=0.001）
- [ ] 动态轴 batch 和 seq 正确支持
- [ ] padding_idx 功能正确实现
- [ ] 典型配置全部通过
