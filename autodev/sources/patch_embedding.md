# patch_embedding

## 算法描述

Patch Embedding 将输入图像切分为固定大小的非重叠 patch，并将每个 patch 线性投影到 embedding 空间。这是 Vision Transformer（ViT）及所有视觉-语言模型的视觉编码器入口。

计算过程：
1. 将输入图像 $X \in \mathbb{R}^{C \times H \times W}$ 划分为 $N = \frac{H}{P} \times \frac{W}{P}$ 个 patch，每个 patch 尺寸为 $P \times P$
2. 将每个 patch 展平为向量 $x_p \in \mathbb{R}^{C \cdot P^2}$
3. 通过线性投影得到 embedding：$z_p = x_p \cdot W + b$，其中 $W \in \mathbb{R}^{(C \cdot P^2) \times D}$，$D$ 为 embedding 维度

等价实现：使用 stride 等于 kernel_size 的 2D 卷积：
$$Z = \text{Conv2D}(X, \text{kernel\_size}=P, \text{stride}=P)$$

输出 shape：$Z \in \mathbb{R}^{N \times D}$，其中 $N = (H/P) \times (W/P)$

## 来源模型

| 模型 | 架构类型 | 使用位置 |
|------|----------|----------|
| SigLIP 2 | Vision Encoder | 图像 Patch Embedding |
| Qwen3-VL | 多模态 VLM | 视觉编码器 Patch Embedding |
| Gemma 3 | Decoder-only Transformer | 视觉编码器 Patch Embedding |
| InternVL | 多模态 VLM | 视觉编码器 Patch Embedding |

## 参考实现

- **ViT**：`transformers/models/vit/modeling_vit.py::ViTEmbeddings`
- **SigLIP**：`transformers/models/siglip/modeling_siglip.py::SiglipVisionEmbeddings`
- **CLIP**：`transformers/models/clip/modeling_clip.py::CLIPVisionEmbeddings`
- **PyTorch**：通过 `torch.nn.Conv2d(in_channels=3, out_channels=D, kernel_size=P, stride=P)` 实现

## 输入输出规格

- **输入**:
  - `pixel_values`: `torch.Tensor`，shape `[batch_size, channels, height, width]`，dtype `float32/float16/bfloat16`，含义：归一化后的图像像素值
  - channels 通常为 3（RGB）
  - height, width 通常为 224, 336, 448, 896 等（经过 resize 和 padding）

- **输出**:
  - `patch_embeds`: `torch.Tensor`，shape `[batch_size, num_patches, embed_dim]`，dtype 同输入，含义：patch embedding 序列
  - `num_patches = (H // patch_size) * (W // patch_size)`

- **典型 shape**:
  - ViT-L/14 (224×224)：`batch=1, num_patches=256, embed_dim=1024`
  - Qwen-VL (448×448)：`batch=1, num_patches=1024, embed_dim=1280`
  - 动态分辨率（Qwen2-VL）：num_patches 随输入图像尺寸变化

## 精度要求

与 PyTorch Conv2d 参考实现的相对误差 ≤ 1e-3（float16/bfloat16）
