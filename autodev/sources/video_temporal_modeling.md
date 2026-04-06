# video_temporal_modeling

## 算法描述

视频时序建模（Video Temporal Modeling）用于捕捉视频帧之间的时间关系，使模型能理解动作、状态变化和因果关系。在视觉-语言模型中，这一模块将独立的帧级视觉特征转化为具有时序感知的表示。

常见实现方式：

1. **3D 位置编码 + 标准 Attention**（Qwen-VL 风格）：
   - 使用 3D RoPE 同时编码时间和空间位置
   - 视频帧按 (t, h, w) 展平为 token 序列
   - 标准 Self-Attention 在所有 token 间计算，隐式建模时序关系
   - 时序信息通过 3D 位置编码注入

2. **时序注意力**（TimeSformer 风格）：
   - 分离空间注意力和时间注意力
   - 时间注意力：同一空间位置跨帧的 token 间计算 attention
   - 空间注意力：同一帧内的 token 间计算 attention
   - 计算公式：$Y = \text{TemporalAttn}(\text{SpatialAttn}(X))$

3. **时序卷积/池化**：
   - 在时间维度上使用 1D 卷积或池化
   - 压缩时间冗余，减少 token 数量

## 来源模型

| 模型 | 架构类型 | 使用位置 |
|------|----------|----------|
| Qwen3-VL | 多模态 VLM | 视频帧间时序建模 |

## 参考实现

- **Qwen2-VL**：`transformers/models/qwen2_vl/modeling_qwen2_vl.py`，通过 3D position_ids 实现时序建模
- **Qwen3-VL**：`transformers/models/qwen3_vl/modeling_qwen3_vl.py`
- **TimeSformer**：`transformers/models/timesformer/modeling_timesformer.py::TimesformerLayer`
- **ViViT**：`transformers/models/vivit/modeling_vivit.py`

## 输入输出规格

- **输入**:
  - `video_features`: `torch.Tensor`，shape `[batch_size, num_frames * patches_per_frame, embed_dim]` 或 `[batch_size, num_frames, patches_per_frame, embed_dim]`，dtype `float16/bfloat16`，含义：视频帧的视觉特征序列
  - `temporal_position_ids`: `torch.LongTensor`（可选），shape `[batch_size, total_tokens]`，含义：时间维度位置索引
  - `spatial_position_ids`: `torch.LongTensor`（可选），shape `[batch_size, total_tokens, 2]`，含义：空间维度 (h, w) 位置索引
  - `num_frames`: `int`，含义：视频帧数

- **输出**:
  - `temporal_features`: `torch.Tensor`，shape 同输入或压缩后的 shape，dtype 同输入，含义：具有时序感知的视觉特征

- **典型 shape**:
  - Qwen-VL 视频（8帧, 448×448, patch=14）：`num_frames=8, patches_per_frame=1024, total_tokens=8192, embed_dim=1280`
  - 短视频（4帧）：`total_tokens=4096, embed_dim=1024`
  - 长视频（64帧，经过时序压缩）：`total_tokens=2048-8192`

## 精度要求

与 PyTorch 参考实现的相对误差 ≤ 1e-3（float16/bfloat16）
