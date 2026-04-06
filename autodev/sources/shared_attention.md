# shared_attention

## 算法描述

Shared Attention 是 Zamba2 模型提出的架构设计，核心思想是多个 Mamba（状态空间模型）块共享同一个注意力层的参数，从而在保持模型表达能力的同时大幅减少参数量。

架构设计：

1. **Hybrid Mamba-Attention 架构**:
   - Zamba2 采用 Mamba 块和 Attention 块交替堆叠的混合架构
   - 例如：[Mamba, Mamba, SharedAttn, Mamba, Mamba, SharedAttn, ...]
   - 多个 Mamba 块之间的注意力层共享同一套权重参数

2. **共享机制**:
   - 设有 N 个注意力层位置，但实际只有 M 套独立权重（M << N）
   - 多个位置的注意力层指向同一个权重实例
   - 例如：位置 2, 5, 8 的注意力层共享同一套 W_q, W_k, W_v, W_o

3. **注意力计算本身**:
   - 共享的注意力层执行标准的 Multi-Head Attention 或 GQA：
     - Q = X @ W_q, K = X @ W_k, V = X @ W_v
     - Attn = softmax(Q @ K^T / sqrt(d_k)) @ V
     - O = Attn @ W_o
   - 数学计算与标准注意力完全相同，区别在于参数共享

4. **残差连接与 LoRA**:
   - 共享注意力层通常配合独立的 LoRA (Low-Rank Adaptation) 或 adapter 层使用
   - 每个使用共享注意力的位置可有独立的 LoRA 参数，用于引入位置特异性
   - 这样在共享主体权重的同时保留了每层的差异化能力

## 来源模型

| 模型 | 架构类型 | 使用位置 |
|------|----------|----------|
| Zamba2 | SSM-Attention Hybrid | 共享注意力层（多个 Mamba 层共享一个 Attention 层） |

## 参考实现

- `transformers/models/zamba2/modeling_zamba2.py`
  - `Zamba2Model`: 整体模型架构，管理共享层的映射关系
  - `Zamba2AttentionDecoderLayer`: 注意力 decoder 层
  - `Zamba2Attention`: 注意力计算实现
  - 关键逻辑：`self.layers` 中多个层指向相同的 attention 权重
- Zamba2 官方实现:
  - `Zyvra/Zamba2-2.7B` 仓库
  - 关注 `config.attention_pattern` 和 `config.num_shared_attention_layers` 配置

## 输入输出规格

- 输入:
  - hidden_states: [batch_size, seq_len, d_model], dtype: float16/bfloat16, 含义: 来自前序 Mamba 块的隐藏状态
  - attention_weights (共享):
    - W_q: [d_model, num_heads * head_dim], 含义: Query 投影权重（共享）
    - W_k: [d_model, num_kv_heads * head_dim], 含义: Key 投影权重（共享）
    - W_v: [d_model, num_kv_heads * head_dim], 含义: Value 投影权重（共享）
    - W_o: [num_heads * head_dim, d_model], 含义: 输出投影权重（共享）
  - lora_params: (可选) 每层独立的 LoRA 参数，用于引入层间差异
  - position_ids: [batch_size, seq_len], dtype: int64, 含义: 位置编码索引
  - attention_mask: (可选) [batch_size, 1, seq_len, seq_len], dtype: bool/float, 含义: causal mask
- 输出:
  - O: [batch_size, seq_len, d_model], dtype: float16/bfloat16, 含义: 注意力层输出
- 典型 shape:
  - Zamba2-2.7B: batch=1, num_heads=16, num_kv_heads=4 (GQA), seq_len=4096, head_dim=128, d_model=2048
  - Zamba2-7B: batch=1, num_heads=24, num_kv_heads=8, seq_len=4096, head_dim=128, d_model=3072
  - 共享配置: 通常 6-8 个位置共享 1-2 套注意力权重

## 精度要求

与 PyTorch 参考实现的相对误差 ≤ 1e-3（float16/bfloat16）
