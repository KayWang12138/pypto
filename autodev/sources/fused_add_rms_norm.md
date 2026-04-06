# fused_add_rms_norm

## 算法描述

残差加法与 RMSNorm 的融合算子：

$$y = \text{RMSNorm}(x + \text{residual})$$

展开为：

$$h = x + \text{residual}$$
$$y = \frac{h}{\sqrt{\frac{1}{H} \sum_{i=1}^{H} h_i^2 + \epsilon}} \cdot \gamma$$

将残差加法和 RMSNorm 融合为一个 kernel，避免中间结果 $h$ 的全局内存写回和再读取，减少一次完整的 Tensor 读写。

## 来源模型

| 模型 | 架构类型 | 使用位置 |
|------|----------|----------|
| Qwen3 | Decoder-only Transformer | 残差连接 + RMSNorm 融合 |
| LLaMA 3 | Decoder-only Transformer | 残差连接 + RMSNorm 融合 |
| Gemma 3 | Decoder-only Transformer | 残差连接 + RMSNorm 融合 |
| DeepSeek-V3 | Decoder-only MoE | 残差连接 + RMSNorm 融合 |

## 参考实现

- 等效 PyTorch 实现：
  ```python
  def fused_add_rms_norm(x, residual, weight, eps=1e-6):
      hidden = x + residual
      variance = hidden.pow(2).mean(-1, keepdim=True)
      hidden_norm = hidden * torch.rsqrt(variance + eps)
      return weight * hidden_norm
  ```
- 融合前的非融合调用路径：
  ```python
  hidden_states = hidden_states + residual
  hidden_states = rms_norm(hidden_states)
  ```

## 输入输出规格

- 输入：
  - `input` ($x$): shape `[batch, seq_len, hidden_size]`，dtype float16 / bfloat16 / float32
  - `residual`: shape `[batch, seq_len, hidden_size]`，与 input 同 shape 和 dtype
  - `weight` ($\gamma$): shape `[hidden_size]`
  - `eps`: float，默认 1e-6
- 输出：
  - `output`: shape `[batch, seq_len, hidden_size]`，与 input 同 dtype
- 典型 shape：
  - `input/residual`: `[1, 2048, 4096]`、`[1, 4096, 8192]`
  - `weight`: `[4096]`、`[8192]`

## 精度要求

与 PyTorch 参考实现的相对误差 ≤ 1e-3（float16）/ 1e-5（float32）
