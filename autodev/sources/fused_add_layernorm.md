# fused_add_layernorm

## 算法描述

残差加法与 LayerNorm 的融合算子：

$$y = \text{LayerNorm}(x + \text{residual})$$

展开为：

$$h = x + \text{residual}$$
$$\mu = \frac{1}{H} \sum_{i=1}^{H} h_i, \quad \sigma^2 = \frac{1}{H} \sum_{i=1}^{H} (h_i - \mu)^2$$
$$y = \frac{h - \mu}{\sqrt{\sigma^2 + \epsilon}} \cdot \gamma + \beta$$

将残差加法和 LayerNorm 融合为一个 kernel，避免中间结果的全局内存写回和再读取。

## 来源模型

| 模型 | 架构类型 | 使用位置 |
|------|----------|----------|
| Cohere2 | Decoder-only Transformer | 残差连接 + LayerNorm 融合 |
| ModernBERT | Encoder-only Transformer | 残差连接 + LayerNorm 融合 |
| Whisper | Encoder-Decoder Transformer | 残差连接 + LayerNorm 融合 |

## 参考实现

- 等效 PyTorch 实现：
  ```python
  def fused_add_layernorm(x, residual, weight, bias, eps=1e-5):
      hidden = x + residual
      return F.layer_norm(hidden, (hidden.size(-1),), weight, bias, eps)
  ```
- 融合前的非融合调用路径：
  ```python
  hidden_states = hidden_states + residual
  hidden_states = self.layer_norm(hidden_states)
  ```

## 输入输出规格

- 输入：
  - `input` ($x$): shape `[batch, seq_len, hidden_size]`，dtype float16 / bfloat16 / float32
  - `residual`: shape `[batch, seq_len, hidden_size]`，与 input 同 shape 和 dtype
  - `weight` ($\gamma$): shape `[hidden_size]`
  - `bias` ($\beta$): shape `[hidden_size]`
  - `eps`: float，默认 1e-5
- 输出：
  - `output`: shape `[batch, seq_len, hidden_size]`，与 input 同 dtype
- 典型 shape：
  - `input/residual`: `[1, 512, 768]`（ModernBERT）、`[1, 1500, 1280]`（Whisper）
  - `weight/bias`: `[768]`、`[1280]`

## 精度要求

与 PyTorch 参考实现的相对误差 ≤ 1e-3（float16）/ 1e-5（float32）
