# sliding_window_attention

## 算法描述

Sliding Window Attention (SWA) 将每个 token 的注意力范围限制在固定窗口 W 内，仅关注前 W 个 token，降低长序列注意力的计算和显存开销。

数学公式与计算流程：

1. **窗口约束**:
   - 对位置 i 的 query，仅与位置 max(0, i - W + 1) 到 i 的 key 计算注意力
   - 窗口外的位置 attention score 设为 -inf

2. **计算过程**:
   - S_ij = Q_i @ K_j^T / sqrt(d_k),  仅在 |i - j| < W 时计算
   - mask(i, j) = 0 if i - W < j <= i, else -inf
   - A = softmax(S + mask) @ V

3. **信息传播**:
   - 通过多层堆叠，信息可跨越 L * W 个 token（L 为层数）
   - 通常与全局 attention 层交替使用

## 来源模型

| 模型 | 架构类型 | 使用位置 |
|------|----------|----------|
| Qwen3 | Decoder-only Transformer | 滑动窗口注意力 |
| Gemma 3 | Decoder-only Transformer | 全局/局部注意力交替 |
| Cohere2 | Decoder-only Transformer | 滑动窗口注意力 |
| ModernBERT | Encoder-only Transformer | 全局/局部注意力交替 |
| Mistral | Decoder-only Transformer | 滑动窗口注意力 |
| Phi-3 | Decoder-only Transformer | 滑动窗口注意力 |

## 参考实现

- `transformers/models/mistral/modeling_mistral.py::MistralAttention`
  - 通过 `sliding_window` 配置参数控制窗口大小
  - attention mask 中对窗口外位置填充 -inf
- `transformers/models/qwen3/modeling_qwen3.py::Qwen3Attention`
  - `layer_type` 区分 sliding window 层和全局 attention 层
- `transformers/models/gemma3/modeling_gemma3.py::Gemma3Attention`

## 输入输出规格

- 输入:
  - Q: [batch_size, num_heads, seq_len, head_dim], dtype: float16/bfloat16, 含义: Query 张量
  - K: [batch_size, num_kv_heads, seq_len, head_dim], dtype: float16/bfloat16, 含义: Key 张量
  - V: [batch_size, num_kv_heads, seq_len, head_dim], dtype: float16/bfloat16, 含义: Value 张量
  - window_size: int, 含义: 滑动窗口大小 W
- 输出:
  - O: [batch_size, num_heads, seq_len, head_dim], dtype: float16/bfloat16, 含义: 注意力输出
- 典型 shape:
  - Mistral 7B: batch=1, num_heads=32, seq_len=8192, head_dim=128, W=4096
  - Gemma 3: batch=1, num_heads=8, seq_len=8192, head_dim=256, W=1024

## 输入约束

- `window_size` 须为正整数，常见值：256、512、1024、4096
- `head_dim` 必须为 2 的幂次
- `seq_len` ≥ `window_size`（否则退化为标准 attention）

## 精度要求

与 PyTorch 参考实现的相对误差 ≤ 1e-3（float16/bfloat16）
