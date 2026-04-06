# one_hot_encoding

## 算法描述

将整数索引转换为 one-hot 向量表示。对于每个整数值 $c \in [0, C)$，生成一个长度为 $C$ 的向量，其中第 $c$ 个位置为 1，其余位置为 0：

$$\text{one\_hot}(c, C)_i = \begin{cases} 1 & \text{if } i = c \\ 0 & \text{otherwise} \end{cases}$$

等价于构造单位矩阵的第 $c$ 行：$\text{one\_hot}(c, C) = I_C[c, :]$

常见用途：
- 分类任务中将整数标签转为概率分布格式，用于计算交叉熵损失
- 作为某些网络层的输入编码方式
- Sparse-to-dense 转换的基础操作

## 来源模型

| 模型 | 架构类型 | 使用位置 |
|------|----------|----------|
| MoE 路由器 | 各 MoE 模型通用 | Top-K 专家选择的 one-hot 编码 |

## 参考实现

- PyTorch：`torch.nn.functional.one_hot(tensor, num_classes=-1)`
- 源码路径：`torch/nn/functional.py` → `one_hot`
- 使用示例：
  ```python
  labels = torch.tensor([0, 2, 1, 4])  # [batch]
  one_hot = torch.nn.functional.one_hot(labels, num_classes=5)
  # tensor([[1, 0, 0, 0, 0],
  #         [0, 0, 1, 0, 0],
  #         [0, 1, 0, 0, 0],
  #         [0, 0, 0, 0, 1]])
  ```

## 输入输出规格

- 输入：
  - `indices`: 任意形状 `[*]` 的整数 Tensor，dtype int64，值域 $[0, C)$
  - `num_classes` ($C$): 类别数，int。若为 -1 则自动推断为 `max(indices) + 1`
- 输出：
  - `output`: shape `[*, num_classes]`，dtype int64（与 PyTorch 一致）或 float32
- 典型 shape：
  - Label 编码: indices `[32]` → output `[32, 1000]`
  - Token 编码: indices `[batch, seq_len]` → output `[batch, seq_len, vocab_size]`
  - 注意：当 vocab_size 很大时（如 128K+），one-hot 输出非常稀疏且巨大

## 精度要求

离散操作，无精度损失
