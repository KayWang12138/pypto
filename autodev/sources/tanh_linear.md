# tanh_linear

## 算法描述
Tanh-Linear 融合激活函数：

`y = tanh(x) * x`

将 Tanh 激活与线性项相乘，形成自门控结构。Tanh 提供 (-1, 1) 的归一化缩放，线性项保留梯度通路。该算子为自定义融合算子，无直接 PyTorch 单函数对应。

## 来源模型

| 模型 | 架构类型 | 使用位置 |
|------|----------|----------|
| xLSTM | SSM Hybrid | mLSTM/sLSTM 中的 tanh 线性变换 |

## 参考实现
无直接 PyTorch 单函数对应，等价组合实现：
```python
y = torch.tanh(x) * x
```

## 输入输出规格
- 输入: x，任意形状的浮点 tensor，dtype 为 float16/bfloat16/float32，表示待激活的特征值
- 输出: y，与输入同形状同 dtype 的 tensor，表示 tanh(x) * x 的计算结果
- 典型 shape: `[batch_size, seq_len, hidden_size]`，如 `[1, 2048, 4096]`、`[8, 512, 4096]`、`[1, 4096, 11008]`

## 精度要求

与 PyTorch 参考实现的相对误差 ≤ 1e-3（float16/bfloat16）
