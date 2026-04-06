# layernorm

## 算法描述

Layer Normalization，沿最后一维进行归一化：

$$y = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} \cdot \gamma + \beta$$

其中：
- $\mu = \frac{1}{H} \sum_{i=1}^{H} x_i$（均值）
- $\sigma^2 = \frac{1}{H} \sum_{i=1}^{H} (x_i - \mu)^2$（方差）
- $\gamma$：可学习的缩放参数，shape `[hidden_size]`
- $\beta$：可学习的偏移参数，shape `[hidden_size]`
- $\epsilon$：防止除零的小常数，默认 1e-5

## 来源模型

| 模型 | 架构类型 | 使用位置 |
|------|----------|----------|
| Cohere2 | Decoder-only Transformer | Attention 和 FFN 后的归一化 |
| ModernBERT | Encoder-only Transformer | 每层 Transformer Block |
| Whisper | Encoder-Decoder Transformer | Encoder/Decoder 各层归一化 |

## 参考实现

- PyTorch：`torch.nn.LayerNorm(normalized_shape, eps=1e-5, elementwise_affine=True)`
- 函数式接口：`torch.nn.functional.layer_norm(input, normalized_shape, weight, bias, eps)`
- Transformers 中使用：
  ```python
  self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
  hidden_states = self.layer_norm(hidden_states)
  ```

## 输入输出规格

- 输入：
  - `input`: shape `[batch, seq_len, hidden_size]`，dtype float16 / bfloat16 / float32
  - `weight` ($\gamma$): shape `[hidden_size]`
  - `bias` ($\beta$): shape `[hidden_size]`
  - `eps`: float，默认 1e-5
- 输出：
  - `output`: shape `[batch, seq_len, hidden_size]`，与 input 同 dtype
- 典型 shape：
  - `input`: `[1, 2048, 768]`、`[8, 512, 1280]`
  - `weight/bias`: `[768]`、`[1280]`

## 精度要求

与 PyTorch 参考实现的相对误差 ≤ 1e-3（float16）/ 1e-5（float32）
