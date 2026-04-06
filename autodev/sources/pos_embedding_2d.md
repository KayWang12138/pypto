# pos_embedding_2d

## 算法描述

二维位置嵌入（2D Positional Embedding）为图像 patch 序列提供空间位置信息。与 1D 位置编码不同，2D 位置嵌入分别编码 patch 的行（height）和列（width）位置，使模型能感知二维空间结构。

常见实现方式：

1. **可学习绝对位置嵌入**：
   - 为每个 (h, w) 位置学习一个独立的 embedding 向量
   - $PE_{2D}(h, w) = E_{pos}[h \times W + w]$，$E_{pos} \in \mathbb{R}^{(H \times W) \times D}$
   - 通过加法融合：$z = z_{patch} + PE_{2D}$

2. **分离式正弦 2D 编码**：
   - 分别对行和列计算正弦编码，然后拼接：
   $$PE_{2D}(h, w) = \text{Concat}(\text{SinPE}(h, D/2), \text{SinPE}(w, D/2))$$
   - 或分别对行和列编码后相加：
   $$PE_{2D}(h, w) = \text{SinPE}_{row}(h, D) + \text{SinPE}_{col}(w, D)$$

3. **插值式位置嵌入**：
   - 在预训练分辨率下学习位置嵌入，推理时通过双线性插值适配不同分辨率
   - $PE_{interp} = \text{interpolate}(PE_{pretrain}, (H_{new}, W_{new}))$

## 来源模型

| 模型 | 架构类型 | 使用位置 |
|------|----------|----------|
| SigLIP 2 | Vision Encoder | 2D 位置编码 |
| SAM 3 | 图像分割 | Window Attention 的 2D 位置偏置 |
| Qwen3-VL | 多模态 VLM | 视觉编码器 2D 位置编码 |

## 参考实现

- **ViT**：`transformers/models/vit/modeling_vit.py::ViTEmbeddings`，`self.position_embeddings = nn.Parameter(torch.randn(1, num_patches+1, hidden_size))`
- **SigLIP**：`transformers/models/siglip/modeling_siglip.py::SiglipVisionEmbeddings`
- **MAE 正弦 2D**：`transformers/models/vit_mae/modeling_vit_mae.py::get_2d_sincos_pos_embed()`
- **PyTorch**：无原生 2D PE API，通常通过 `nn.Parameter` 或手动构建

## 输入输出规格

- **输入**:
  - `patch_embeds`: `torch.Tensor`，shape `[batch_size, num_patches, embed_dim]`，dtype `float16/bfloat16/float32`，含义：patch embedding 序列（来自 patch_embedding 算子）
  - `height`: `int`，含义：patch 网格的高度 (H // patch_size)
  - `width`: `int`，含义：patch 网格的宽度 (W // patch_size)
  - 可学习方式：`position_embedding`: `torch.Tensor`，shape `[1, num_patches, embed_dim]`，含义：预训练的位置嵌入权重

- **输出**:
  - `embeds_with_pos`: `torch.Tensor`，shape `[batch_size, num_patches, embed_dim]`，dtype 同输入，含义：加上位置信息后的 embedding

- **典型 shape**:
  - ViT-L/14 (224×224)：`num_patches=256 (16×16), embed_dim=1024`
  - SigLIP (384×384)：`num_patches=729 (27×27), embed_dim=1152`
  - 动态分辨率：`num_patches` 随输入图像尺寸变化

## 精度要求

与 PyTorch 参考实现的相对误差 ≤ 1e-3（float16）/ 1e-5（float32）
