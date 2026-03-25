# Attention算子 PyPTO 开发计划

> 基于 `/mnt/workspace/gitCode/cann/ops-transformer/attention` 目录整理
> 创建时间：2026-03-24
> 最后更新：2026-03-24

---

## 开发状态说明

- ⬜ **待开发**：尚未开始
- 🔄 **开发中**：正在开发
- ✅ **已完成**：开发完成并验证通过
- ⏭️ **已跳过**：跳过开发
- ❌ **开发失败**：无法实现或遇到阻碍

---

## 高优先级算子（建议优先开发）

| 序号 | 算子名称 | 功能描述 | 数学公式(简要) | 复杂度 | 状态 | 备注 |
|:---:|:---|:---|:---|:---:|:---:|:---|
| 1 | FlashAttentionScore | 训练场景FlashAttention | Attention = Dropout(Softmax(Mask(scale*(QK^T+pse))))V | 复杂 | ✅ | 核心算子 |
| 2 | FlashAttentionScoreGrad | FlashAttention反向传播 | dV=P^TdY, dQ=((dS)*K)/√d, dK=((dS)^T*Q)/√d | 复杂 | ⏭️ | 训练必需 |
| 3 | FusedInferAttentionScore | 增量&全量推理FlashAttention | Attention(Q,K,V) = Softmax(QK^T/√d)V | 复杂 | ✅ | 推理核心 |
| 4 | IncreFlashAttention | 增量推理场景注意力 | Attention(Q,K,V) = Softmax(QK^T/√d)V, S=1 | 复杂 | ⬜ | 推理核心 |
| 5 | PromptFlashAttention | 全量推理FlashAttention | Attention(Q,K,V) = Softmax(QK^T/√d)V | 复杂 | ⬜ | 推理核心 |
| 6 | SparseFlashAttention | 大序列稀疏注意力 | Softmax(Q@K̃^T/√d)@Ṽ | 复杂 | ⬜ | 大模型必需 |
| 7 | LightningIndexer | Top-k位置选择 | Indices = Top-k([W⊙ReLU(Q_index@K_index^T)]) | 中等 | ⬜ | 稀疏注意力前处理 |
| 8 | QuantLightningIndexer | 量化版Top-k选择 | Top-k(W⊙ReLU(Scale_Q·Scale_K^T⊙(Q^Quant@(K^Quant)^T))) | 复杂 | ⬜ | 量化场景 |
| 9 | MlaPreprocess | MLA前处理 | q^N = RmsNormQuant(x)·W^DQKV·W^UK | 复杂 | ⬜ | DeepSeek-V3核心 |
| 10 | MlaPreprocessV2 | MLA前处理V2 | 同MlaPreprocess，支持更多量化 | 复杂 | ⬜ | MLA增强版 |
| 11 | MlaProlog | MLA前处理(含ROPE) | c^Q = RmsNorm(x·W^DQ), q^N = c^Q·W^UQ·W^UK | 复杂 | ⬜ | DeepSeek-V3核心 |
| 12 | MlaPrologV2 | MLA前处理V2 | 同MlaProlog，支持更多量化 | 复杂 | ⬜ | MLA增强版 |
| 13 | MlaPrologV3 | MLA前处理V3 | 含α_q、α_kv尺度矫正参数 | 复杂 | ⬜ | MLA最新版 |
| 14 | BlockSparseAttention | 块稀疏注意力 | Attention = Softmax(QK^T/√d)V (稀疏块) | 复杂 | ⬜ | 高性能稀疏 |

---

## 中优先级算子

| 序号 | 算子名称 | 功能描述 | 数学公式(简要) | 复杂度 | 状态 | 备注 |
|:---:|:---|:---|:---|:---:|:---:|:---|
| 15 | AttentionUpdate | 更新局部lse/localOut为全局 | lse_max = max(lse_i), O = ΣO_i·exp(lse_i-lse_m) | 中等 | ⬜ | FlashAttention辅助 |
| 16 | GatherPaKvCache | 拼接不连续token为连续序列 | keyRef[dim0] = sum(seqLens) | 简单 | ⬜ | Page Attention辅助 |
| 17 | KvQuantSparseFlashAttention | 量化稀疏注意力 | Softmax(Q@Dequant(K)^T/√d)@Dequant(V) | 复杂 | ⬜ | 量化场景 |
| 18 | KvQuantSparseFlashAttentionPioneer | 新版量化稀疏注意力 | 同上，Per-Token-Head-Tile-128量化 | 复杂 | ⬜ | 新版量化 |
| 19 | NsaCompress | KV序列压缩 | K̃_t^cmp = {φ(k_id+1:id+l)} | 中等 | ⬜ | NSA前处理 |
| 20 | NsaCompressWithCache | NSA推理KV压缩 | outputCache = input*weight | 中等 | ⬜ | NSA推理 |
| 21 | NsaCompressAttention | NSA compress attention | P_cmp=Softmax(query*key^T), topkIndices | 复杂 | ⬜ | NSA核心 |
| 22 | NsaCompressAttentionInfer | NSA推理compress attention | P_cmp=Softmax(scale*query·key^T)·value | 复杂 | ⬜ | NSA推理 |
| 23 | NsaSelectedAttention | NSA selected attention | selected_key = Gather(key, topk_indices) | 复杂 | ⬜ | NSA核心 |
| 24 | NsaSelectedAttentionInfer | NSA推理selected attention | Attention = Softmax(query·key_topk^T/√d)value_topk | 复杂 | ⬜ | NSA推理 |
| 25 | RainFusionAttention | 高性能稀疏注意力 | 稀疏注意力计算 | 复杂 | ⬜ | 块级稀疏 |
| 26 | RecurrentGatedDeltaRule | 变步长Recurrent计算 | S_t = α_t·Diag(α_kt)·S_{t-1} + β_t(...) | 复杂 | ⬜ | RNN变体 |
| 27 | RingAttentionUpdate | 两次FlashAttention输出更新 | softmax_max = max(prev_max, cur_max) | 中等 | ⬜ | 分布式Attention |
| 28 | ScatterPaCache | 更新KCache指定位置 | keyCache = slotMapping(key) | 简单 | ⬜ | Page Attention辅助 |
| 29 | ScatterPaKvCache | 更新KvCache指定位置 | keyCache/slotMapping更新 | 简单 | ⬜ | Page Attention辅助 |
| 30 | DenseLightningIndexerSoftmaxLse | LightningIndexer分支算子 | res = ReduceSum(W⊙ReLU(Q_index@K_index^T)) | 中等 | ⬜ | 辅助算子 |
| 31 | FusedFloydAttention | 多维自注意力 | weights = Softmax(scale*(Q*K_1^T + Q*K_2^T)) | 复杂 | ⬜ | 特殊Attention |

---

## 低优先级算子

| 序号 | 算子名称 | 功能描述 | 数学公式(简要) | 复杂度 | 状态 | 备注 |
|:---:|:---|:---|:---|:---:|:---:|:---|
| 32 | AttentionWorkerCombine | 多计算单元注意力融合 | 多头注意力融合+专家权重加权 | 中等 | ⬜ | 分布式场景 |
| 33 | AttentionWorkerScheduler | Attention数据扫描算子 | 数据扫描与状态检查 | 中等 | ⬜ | 分布式场景 |
| 34 | DenseLightningIndexerGradKLLoss | LightningIndexer反向+KL | Top-k + Softmax + KL散度反向传播 | 复杂 | ⬜ | 训练辅助 |
| 35 | FusedFloydAttentionGrad | FloydAttention反向 | dV_1=P^TdY, dQ=(dS*K_1+dS*K_2)/√d | 复杂 | ⬜ | 训练辅助 |
| 36 | NsaCompressGrad | NsaCompress反向 | dw = dk_cmp·K^T, dk = W^T·dk_cmp | 中等 | ⬜ | 训练辅助 |
| 37 | NsaSelectedAttentionGrad | NsaSelectedAttention反向 | selected_key = Gather(key, topk_indices) | 复杂 | ⬜ | 训练辅助 |
| 38 | LightningIndexerGrad | LightningIndexer反向 | - | 中等 | ⬜ | 训练辅助 |
| 39 | SparseFlashAttentionGrad | SparseFlashAttention反向 | - | 复杂 | ⬜ | 训练辅助 |
| 40 | SparseLightningIndexerGradKLLoss | Sparse版KL Loss反向 | - | 复杂 | ⬜ | 训练辅助 |
| 41 | SwinAttentionScoreQuant | Swin Transformer量化注意力 | - | 复杂 | ⬜ | 视觉模型 |

---

## 开发进度统计

- **总算子数**：41个
- **高优先级**：14个
- **中优先级**：17个
- **低优先级**：10个
- **已完成**：2个
- **已跳过**：1个
- **待开发**：38个

---

## 开发顺序建议

### 第一批：核心FlashAttention系列（高优先级）
1. FlashAttentionScore → 2. FlashAttentionScoreGrad → 3. FusedInferAttentionScore

### 第二批：推理场景注意力（高优先级）
4. IncreFlashAttention → 5. PromptFlashAttention → 6. SparseFlashAttention

### 第三批：稀疏注意力前处理（高优先级）
7. LightningIndexer → 8. QuantLightningIndexer

### 第四批：MLA系列（高优先级）
9. MlaPreprocess → 10. MlaPreprocessV2 → 11. MlaProlog → 12. MlaPrologV2 → 13. MlaPrologV3

### 第五批：其他高优先级
14. BlockSparseAttention

### 后续批次
按中优先级、低优先级顺序依次开发

---

## 更新日志

| 日期 | 更新内容 |
|:---|:---|
| 2026-03-24 | ✅ 完成 FusedInferAttentionScore 开发（全量+增量推理场景） |
| 2026-03-24 | ⏭️ 跳过 FlashAttentionScoreGrad |
| 2026-03-24 | ✅ 完成 FlashAttentionScore 开发（基础版本：BF16，输出 softmax_max/sum） |
| 2026-03-24 | 创建开发计划，整理41个算子信息 |