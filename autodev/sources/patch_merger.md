# patch_merger

## 算法描述

Patch Merger（也称 Patch Merging / Token Merging）用于合并或下采样视觉 token，减少后续 Transformer 层需要处理的 token 数量，从而降低计算量和显存占用。

常见实现方式：

1. **空间下采样合并**（Swin Transformer 风格）：
   - 将相邻的 2×2 patch 合并为 1 个 patch
   - 4 个相邻 patch 的 embedding 拼接：$z_{merged} = \text{Concat}(z_{00}, z_{01}, z_{10}, z_{11})$，$z_{merged} \in \mathbb{R}^{4D}$
   - 线性投影降维：$z_{out} = z_{merged} \cdot W$，$W \in \mathbb{R}^{4D \times 2D}$
   - 空间分辨率减半，通道数翻倍

2. **可学习池化合并**（Pixtral / Mistral3 风格）：
   - 使用 2D 池化或 stride=2 的卷积进行下采样
   - 或通过注意力池化（attention pooling）选择性合并 token

3. **SAM3 风格**：
   - 使用多层下采样卷积逐步减少空间分辨率
   - 可能包含跳跃连接

合并效果：token 数量减少 2× 或 4×，显著降低后续 attention 的 $O(N^2)$ 计算开销。

## 来源模型

| 模型 | 架构类型 | 使用位置 |
|------|----------|----------|
| Mistral 3 (Pixtral) | 多模态 VLM | PatchMerger 视觉 token 压缩 |
| SmolVLM | 多模态 VLM | Patch 合并降低视觉 token 数 |

## 参考实现

- **Mistral3 (Pixtral)**：`transformers/models/mistral3/modeling_mistral3.py` 中的视觉 token 合并模块
- **Swin Transformer**：`transformers/models/swin/modeling_swin.py::SwinPatchMerging`
- **InternVL**：`transformers/models/internvl/modeling_internvl.py` 中的 pixel shuffle downsample

## 输入输出规格

- **输入**:
  - `patch_embeds`: `torch.Tensor`，shape `[batch_size, num_patches, embed_dim]` 或 `[batch_size, H, W, embed_dim]`，dtype `float16/bfloat16/float32`，含义：视觉 patch embedding 序列
  - `weight`: `torch.Tensor`（可选），shape `[4*embed_dim, 2*embed_dim]`（Swin 风格）或卷积权重，含义：投影权重

- **输出**:
  - `merged_embeds`: `torch.Tensor`，shape `[batch_size, num_patches//4, embed_dim*2]`（Swin 风格）或 `[batch_size, num_patches//merge_ratio, embed_dim]`，dtype 同输入，含义：合并后的 patch embedding

- **典型 shape**:
  - Pixtral (1024 patches → 256)：`input=[1, 1024, 1024]` → `output=[1, 256, 1024]`
  - Swin-B (56×56 → 28×28)：`input=[1, 3136, 128]` → `output=[1, 784, 256]`
  - SAM3：通过多级卷积逐步下采样

## 精度要求

与 PyTorch 参考实现的相对误差 ≤ 1e-3（float16/bfloat16）
