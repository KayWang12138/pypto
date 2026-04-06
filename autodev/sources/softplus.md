# softplus

## 算法描述
Softplus 激活函数，是 ReLU 的平滑近似：

`softplus(x) = log(1 + exp(x))`

输出恒为正值，保持处处可导。当 x 较大时趋近于 x，当 x 较小时趋近于 0。

## 来源模型

| 模型 | 架构类型 | 使用位置 |
|------|----------|----------|
| Mamba2 | SSM | dt 参数的正值约束 |
| Falcon-H1 | SSM-Attention Hybrid | Mamba2 层中的 dt 参数 |
| Nemotron-H | SSM-Attention Hybrid | Mamba2 层中的 dt 参数 |

## 参考实现
`torch.nn.functional.softplus(x)`

## 输入输出规格
- 输入: x，任意形状的浮点 tensor，dtype 为 float16/bfloat16/float32，表示待激活的特征值
- 输出: y，与输入同形状同 dtype 的 tensor，值域 (0, +inf)，表示 Softplus 激活后的结果
- 典型 shape: `[batch_size, seq_len, inner_size]`，如 `[1, 2048, 256]`、`[8, 1024, 128]`、`[1, 4096, 192]`

## 精度要求

与 PyTorch 参考实现的相对误差 ≤ 1e-3（float16）/ 1e-5（float32）
