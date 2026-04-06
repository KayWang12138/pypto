# exponential_gating

## 算法描述
指数门控（Exponential Gating）函数：

`gate(x) = exp(x)`

用于 xLSTM 的 mLSTM/sLSTM 指数门控机制，以指数函数替代传统 Sigmoid 门控，使门控值可大于 1，增强记忆的写入和保持能力。数值稳定性方面，实际使用中通常配合 stabilizer 进行 `exp(x - m)` 计算，基础算子实现纯 `exp(x)`。

## 来源模型

| 模型 | 架构类型 | 使用位置 |
|------|----------|----------|
| xLSTM | SSM Hybrid | mLSTM/sLSTM 的指数门控 |
| Falcon-H1 | SSM-Attention Hybrid | Mamba2 层门控 |

## 参考实现
`torch.exp(x)`

xLSTM 门控上下文参考：`xlstm.blocks.mlstm.cell.mLSTMCell::forward()` 中的 `torch.exp(i_tilde)` 和 `torch.exp(f_tilde)`

## 输入输出规格
- 输入: x，任意形状的浮点 tensor，dtype 为 float16/bfloat16/float32，表示门控的原始值（通常为线性投影输出）
- 输出: y，与输入同形状同 dtype 的 tensor，值域 (0, +inf)，表示指数门控值
- 典型 shape: `[batch_size, seq_len, n_heads]`，如 `[1, 2048, 8]`；`[batch_size, seq_len, hidden_size]`，如 `[1, 2048, 4096]`

## 精度要求

与 PyTorch 参考实现的相对误差 ≤ 1e-3（float16）/ 1e-5（float32）
