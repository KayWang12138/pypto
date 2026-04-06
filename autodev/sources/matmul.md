# matmul

## 算法描述

矩阵乘法（Matrix Multiplication）是深度学习中最基础、最核心的算子。计算两个矩阵的乘积，支持高维张量的广播机制。

基本公式（2D 情况）：
$$C_{ij} = \sum_{k=0}^{K-1} A_{ik} \cdot B_{kj}$$

其中 $A \in \mathbb{R}^{M \times K}$，$B \in \mathbb{R}^{K \times N}$，$C \in \mathbb{R}^{M \times N}$。

高维张量情况（`torch.matmul` 语义）：
- 1D × 1D：向量内积，返回标量
- 2D × 2D：标准矩阵乘法
- 高维 × 高维：将前面的维度视为 batch 维度，最后两维做矩阵乘，支持广播
- 例如：`[B, M, K] @ [K, N]` → `[B, M, N]`（B 维度广播）

计算复杂度：$O(M \times N \times K)$，是 Transformer 中的主要计算瓶颈。

## 来源模型

| 模型 | 架构类型 | 使用位置 |
|------|----------|----------|
| Qwen3 | Decoder-only Transformer | Attention QK^T、Attention×V、FFN 线性层 |
| LLaMA 3 | Decoder-only Transformer | Attention QK^T、Attention×V、FFN 线性层 |
| DeepSeek-V3 | Decoder-only MoE | MLA 投影、FFN |

## 参考实现

- **PyTorch 函数**：`torch.matmul(input, other)`
- **PyTorch 运算符**：`@` 运算符（等价于 `torch.matmul`）
- **PyTorch 低级**：`torch.mm(input, mat2)`（仅 2D）、`torch.bmm(input, mat2)`（仅 3D batch）
- **cuBLAS**：NVIDIA 的 `cublasSgemm` / `cublasHgemm` 等

## 输入输出规格

- **输入**:
  - `input`: `torch.Tensor`，shape `[..., M, K]`，dtype `float32/float16/bfloat16/int8`，含义：左矩阵
  - `other`: `torch.Tensor`，shape `[..., K, N]`，dtype 同输入（或量化场景下可不同），含义：右矩阵
  - 前导维度 `...` 支持广播

- **输出**:
  - `output`: `torch.Tensor`，shape `[..., M, N]`，dtype 由输入决定（int8 输入通常输出 int32），含义：矩阵乘积

- **典型 shape**:
  - QKV 投影：`[1, 2048, 4096] @ [4096, 4096]` → `[1, 2048, 4096]`
  - Attention Score（Prefill）：`[1, 32, 2048, 128] @ [1, 32, 128, 2048]` → `[1, 32, 2048, 2048]`
  - FFN Up：`[1, 2048, 4096] @ [4096, 11008]` → `[1, 2048, 11008]`
  - LM Head：`[1, 1, 4096] @ [4096, 32000]` → `[1, 1, 32000]`

## 输入约束

- 输入矩阵的内维度必须匹配：A `[..., M, K]` × B `[..., K, N]`
- K 维度须满足硬件对齐要求（通常 16 的倍数）
- 支持 batch 维度广播

## 精度要求

与 PyTorch 参考实现的相对误差 ≤ 1e-3（float16/bfloat16）
