# mla polog quant 

## 功能说明

MLA Prolog 模块将hidden状态 $\bold{X}$ 转换为查询投影 $\bold{q}$、键投影 $\bold{k}$ 和值投影 $\bold{v}$，其结构与 DeepSeek V3 的架构一致。在解码阶段，采用了权重吸收技术。
### 计算公式

#### RmsNorm公式
$$
\text{RmsNorm}(x) = \gamma \cdot \frac{x_i}{\text{RMS}(x)}
$$
$$
\text{RMS}(x) = \sqrt{\frac{1}{N} \sum_{i=1}^{N} x_i^2 + \epsilon}
$$
#### 路径1：标准Query计算

包括下采样、RmsNorm和两次上采样：
$$
c^Q = RmsNorm(x \cdot W^{DQ})
$$
$$
q^C = c^Q \cdot W^{UQ}
$$
$$
q^N = q^C \cdot W^{UK}
$$
#### 路径2：位置编码Query计算
对Query进行ROPE旋转位置编码：
$$
q^R = ROPE(c^Q \cdot W^{QR})
$$
#### 路径3：标准Key计算
包括下采样、RmsNorm，将计算结果存入Cache：
$$
c^{KV} = RmsNorm(x \cdot W^{DKV})
$$
$$
k^C = Cache(c^{KV})
$$
#### 路径4：位置编码Key计算
对Key进行ROPE旋转位置编码，并将结果存入Cache：
$$
k^R = Cache(ROPE(x \cdot W^{KR}))
$$

## 函数原型
```
mla_prolog_quant_compute(input_tensors, output_tensors, epsilon_cq, epsilon_ckv, cache_mode, tile_config):
```

## 参数说明

>**说明：**<br> 
>
>- B（Batch Size）表示输入样本批量大小、S（Sequence Length）表示输入样本序列长度、He（Head Size）表示隐藏层大小、N（Head Num）表示多头数、Hcq表示q低秩矩阵维度、Hckv表示kv低秩矩阵维度、D表示qk不含位置编码维度、Dr表示qk位置编码维度、Nkv表示kv的head数、BlockNum表示PagedAttention场景下的块数、BlockSize表示PagedAttention场景下的块大小、T表示BS合轴后的大小。

-   **token_x**（`Tensor`）：公式中用于计算Query和Key的输入tensor。不支持非连续，数据格式支持ND，数据类型支持`bfloat16`，shape为[T, He]或[t, h]。
-   **w_dq**（`Tensor`）：公式中用于计算Query的下采样权重矩阵$W^{DQ}$。数据格式支持NZ，数据类型支持`bfloat16`，shape为[h, q_lora_rank]。
-   **w_uq_qr**（`Tensor`）：公式中用于计算Query的上采样权重矩阵$W^{UQ}$和位置编码权重矩阵$W^{QR}$。不支持非连续，数据格式支持NZ，数据类型支持`int8`，shape为[q_lora_rank, n_q*q_head_dim]。
-   **w_uk**（`Tensor`）：公式中用于计算Key的上采样权重$W^{UK}$。不支持非连续，数据格式支持ND，数据类型支持`bfloat16`，shape为[n_q, qk_nope_head_dim, kv_lora_rank]。
-   **w_dkv_kr**（`Tensor`）：公式中用于计算Key的下采样权重矩阵$W^{DKV}$和位置编码权重矩阵$W^{KR}$。不支持非连续，数据格式支持NZ，数据类型支持`bfloat16`，shape为[h, kv_lora_rank+rope_dim]。
-   **rmsnorm_gamma_cq**（`Tensor`）：计算$c^Q$的RmsNorm公式中的$\gamma$参数。不支持非连续，数据格式支持ND，数据类型支持`bfloat16`，shape为[q_lora_rank]。
-   **rmsnorm_gamma_ckv**（`Tensor`）：计算$c^{KV}$的RmsNorm公式中的$\gamma$参数。不支持非连续，数据格式支持ND，数据类型支持`bfloat16`，shape为[kv_lora_rank]。
-   **rope_sin**（`Tensor`）：用于计算旋转位置编码的正弦参数矩阵。不支持非连续，数据格式支持ND，数据类型支持`bfloat16`，shape为[t, rope_dim]。
-   **rope_cos**（`Tensor`）：用于计算旋转位置编码的余弦参数矩阵。不支持非连续，数据格式支持ND，数据类型支持`bfloat16`，shape为[t, rope_dim]。
-   **kv_cache**（`Tensor`）：用于cache索引的aclTensor，计算结果原地更新（对应公式中的$k^C$）。数据格式支持ND，数据类型支持`int8`，cache_mode为"PA_BSND"、shape为[block_num, block_size, n_kv, kv_lora_rank]。
-   **kr_cache**（`Tensor`）：用于key位置编码的cache，计算结果原地更新（对应公式中的$k^R$）。数据格式支持ND，数据类型支持`bfloat16`，cache_mode为"PA_BSND"、shape为[block_num, block_size, n_kv, rope_dim]。
-   **cache_index**（`Tensor`）：用于存储kv_cache和kr_cache的索引。不支持非连续，数据格式支持ND，数据类型支持`int64`，shape为[T]。
-   **dequant_scale_w_uq_qr**（`Tensor`）：用于MatmulQcQr矩阵乘后反量化操作的per-channel参数。数据格式支持ND，数据类型支持`float`，shape为[n_q*q_head_dim, 1]。
-   **epsilon_cq**（`float`）：计算$c^Q$的RmsNorm公式中的$\epsilon$参数。用户未特意指定时，建议传入1e-05，仅支持double类型，默认值为1e-05。
-   **epsilon_cq**（`float`）：计算$c^{KV}$的RmsNorm公式中的$\epsilon$参数。用户未特意指定时，建议传入1e-05，仅支持double类型，默认值为1e-05。
-   **cache_mode**（`str`）：表示kv_cache的模式，支持"PA_BSND"。
-   **tile_config**（`int`）：表示tile切分配置。

## 返回值说明
-   **q_nope**（`Tensor`）：公式中Query的输出tensor（对应$q^N$）。数据格式支持ND，数据类型支持`bfloat16`，shape为[t, n_q, kv_lora_rank]。
-   **q_rope**（`Tensor`）：公式中Query位置编码的输出tensor（对应$q^R$）。数据格式支持ND，数据类型支持`bfloat16`，shape为[t, n_q, rope_dim]。
-   **q_norm_scale**（`Tensor`）：Query输出的反量化参数。数据格式支持ND，数据类型支持`float`，shape为[T1]或[B, S]。
-   **query_norm**（`Tensor`）：Query做RmsNorm_cq后的输出tensor（对应$q^C$）。数据格式支持ND，数据类型支持`int8`，shape为[t, q_lora_rank]。
-   **q_norm_scale**（`Tensor`）：Query做RmsNorm_cq后的反量化参数。数据格式支持ND，数据类型支持`float`，shape为[t, 1]。
-   **kv_cache**（`Tensor`）：Key输出到`kv_cache`中的tensor（对应$k^C$）。数据格式支持ND，数据类型支持`int8`，shape为[block_num, block_size, n_kv, kv_lora_rank]。
-   **kr_cache**（`Tensor`）：Key的位置编码输出到`kr_cache`中的tensor（对应$k^R$）。数据格式支持ND，数据类型支持`bfloat16`，shape为[block_num, block_size, n_kv, qk_rope_dim]。
-   **k_scale_cache**（`Tensor`）：Key做反量化后输出的反量化参数。数据格式支持ND，数据类型支持`float`，shape为[block_num, block_size, n_kv, 4]。

## 调用示例

- 详见 [test_mla_prolog_prefill](test_mla_prolog_prefill.py)

# lightning indexer prolog<a name="ZH-CN_TOPIC_0000001979260729"></a>
## 功能说明<a name="zh-cn_topic_0000001832267082_section14441124184110"></a>

-   算子功能：用于 Deepseek IndexerAttention 中，计算 Lightning Indexer 所需要的 query，key 和 weights。
Indexer Prolog 的量化策略如下：Q_b_proj 使用 W8A8 量化，其他 Linear 均不量化；query 使用 A8 量化，key(cache) 使用 C8 量化；反量化因子以 FP16 存储；weights 以 FP16 存储；

Query 的计算公式如下：

$$
\bold{q}, \bold{q}_{scale} = \text{DynamicQuant}(\text{Hadamard}(\text{RoPE}(\text{DeQuant}(\bold{q} \cdot \bold{w}_{qb}))))
$$

Q 的计算采用了动态的 Per-Token-Head 量化，其中 Hadamard 变换通过矩阵右乘 hadamard_q 实现。而 $\bold{q}, \bold{w}_{qb}$ 均是 Int8 类型。

Key(cache) 的计算公式如下：

$$
\bold{k}, \bold{k}_{scale} = \text{DynamicQuant}(\text{Hadamard}(\text{RoPE}(\text{LayerNorm}(\bold{x} \cdot \bold{w}_k))))
$$

Cache 的计算同样采用了动态的 Per-Token-Head 量化，其中 Hadamard 变换通过矩阵右乘 hadamard_k 实现。


Weights 的计算公式如下：

$$
\bold{weight} = (\bold{x} \cdot \bold{w}_{proj}) * \text{scale}
$$

Weights 的计算没有采用量化，同时需要最后转化为 FP16 数据类型，供后续的 Lightning Indexer 计算使用。

## 函数原型<a name="zh-cn_topic_0000001832267082_section45077510411"></a>

```
def lightning_indexer_prolog_quant(input_tensors, output_tensors, attrs, configs):
```

## 参数说明<a name="zh-cn_topic_0000001832267082_section112637109429"></a>

>**说明：**<br>
>
-   **token\_x**（`Tensor`）：表示 hidden 状态，必选参数，不支持非连续的Tensor，数据格式支持ND，数据类型支持`bfloat16`,shape为[t, h]。
-   **q\_norm**（`Tensor`）：表示经过 rmsnorm 后量化的 query，必选参数，不支持非连续的Tensor，数据格式支持ND，数据类型支持`int8`,shape为[t, q_lora_rank]。
-   **q\_norm\_scale**（`Tensor`）：表示 query 的反量化因子，必选参数，不支持非连续的Tensor，数据格式支持ND，数据类型支持`float32`,shape为[t, 1]。
-   **wq\_b**（`Tensor`）：表示 query 的权重，必选参数，不支持非连续的Tensor，数据格式支持NZ，数据类型支持`int8`,shape为[q_lora_rank, idx_n_heads*idx_head_dim]。
-   **wq\_b\_scale**（`Tensor`）：表示 query 的权重反量化因子，必选参数，不支持非连续的Tensor，数据格式支持ND，数据类型支持`float32`,shape为[idx_n_heads*idx_head_dim, 1]。
-   **wk**（`Tensor`）：表示 key 的权重，必选参数，不支持非连续的Tensor，数据格式支持NZ，数据类型支持`bfloat16`,shape为[h, idx_head_dim]。
-   **weights_proj**（`Tensor`）：表示 weights 的权重，必选参数，不支持非连续的Tensor，数据格式支持NZ，数据类型支持`bfloat16`，shape为[h, idx_n_heads]。
-   **ln_gamma_k**（`Tensor`）：表示 key 的 layernorm 缩放，必选参数，不支持非连续的Tensor，数据格式支持ND，数据类型支持`bfloat16`，shape为[idx_head_dim]。
-   **ln_beta_k**（`Tensor`）：表示 key 的 layernorm 偏移，必选参数，不支持非连续的Tensor，数据格式支持ND，数据类型支持`bfloat16`,shape为[idx_head_dim]。
-   **cos_idx_rope**（`Tensor`）：表示用于 RoPE 的 cos，不支持非连续的 Tensor，数据格式支持 ND，数据类型支持`bfloat16`，shape为[t, rope_dim]。
-   **sin_idx_rope**（`Tensor`）：表示用于 RoPE 的 sin，不支持非连续的 Tensor，数据格式支持 ND，数据类型支持`bfloat16`，shape为[t, rope_dim]。
-   **hadamard_q**（`Tensor`）：表示用于 query Hadamard 变换的权重矩阵，不支持非连续的 Tensor，数据格式支持 ND，数据类型支持`bfloat16`，shape为[idx_head_dim, idx_head_dim]。
-   **hadamard_k**（`Tensor`）：表示用于 key Hadamard 变换的权重矩阵，不支持非连续的 Tensor，数据格式支持 ND，数据类型支持`bfloat16`，shape为[idx_head_dim, idx_head_dim]。
-   **idx_k_cache**（`Tensor`）：表示 key 的缓存，必选参数，不支持非连续的 Tensor，数据格式支持 ND，数据类型支持`int8`，shape为[block_num, block_size, n_kv, idx_head_dim]。
-   **idx_k_scale_cache**（`Tensor`）：表示 key 反量化因子的缓存，必选参数，不支持非连续的 Tensor，数据格式支持 ND，数据类型支持`float16`，shape为[idx_block_num, block_size, n_kv]。
-   **idx_k_cache_index**（`Tensor`）：表示更新 key 缓存的位置，必选参数，不支持非连续的 Tensor，数据格式支持 ND，数据类型支持`int64`，shape为[t]。
-   **layernorm_epsilon_k**（`float`）：表示 key layernorm 防除 0 系数，必选参数，数据类型支持`float32`。
-   **layout\_query**（`str`）：可选参数，用于标识输入`query`的数据排布格式，默认值"TND"。当前仅支持 "TND"。
-   **layout\_key**（`str`）：可选参数，用于标识输入`key`的数据排布格式，默认值"PA_BSND"。当前仅支持 "PA_BSND"。

## 返回值说明

-   **query**（`Tensor`）：公式中 query 的输出 tensor，数据格式支持 ND，数据类型支持`int8`，shape为[t, idx_n_heads, idx_head_dim]。
-   **query_scale**（`Tensor`）：公式中 query 反量化因子的输出 tensor，数据格式支持 ND，数据类型支持`float16`，shape为[t, idx_n_heads]。
-   **weights**（`Tensor`）：公式中 weights 的输出 tensor，数据格式支持 ND，数据类型支持`float16`，shape为[t, idx_n_heads]。

## 调用示例
-   算子源码执行参考[testdsv32_lightning_indexer_prolog_quant.py](testdsv32_lightning_indexer_prolog_quant.py)


# sparse flash attention quant<a name="ZH-CN_TOPIC_0000001979260729"></a>

## 功能说明<a name="zh-cn_topic_0000001832267082_section14441124184110"></a>

对于每个查询 token $\bold{x}_i$，索引模块会为每个键值缓存项（表示键值对或 MLA 潜在表示）计算一个相关性得分 $I_{i,j}$。然后，通过将注意力机制应用于查询 token $\bold{x}_i$ 以及得分最高的前 $k$ 个缓存项，来计算输出 $\bold{o}_i$：

$$
\bold{o}_i = \text{Attn}(\bold{x}_i, \{\bold{c}_j | j \in \text{Top-k}(\bold{I}_{i, :})\})
$$

## 函数原型<a name="zh-cn_topic_0000001832267082_section45077510411"></a>

```
def sparse_flash_attention_quant_d_compute(in_tensors, out_tensors, nq, n_kv, softmax_scale, topk, tile_config):
```

## 参数说明<a name="zh-cn_topic_0000001832267082_section112637109429"></a>

>**说明：**<br> 
>
>- query、key、value参数维度含义：B（Batch Size）表示输入样本批量大小、S（Sequence Length）表示输入样本序列长度、H（Head Size）表示hidden层的大小、N（Head Num）表示多头数、D（Head Dim）表示hidden层最小的单元尺寸，且满足D=H/N、T表示所有Batch输入样本序列长度的累加和。
>- Q\_S和S1表示query shape中的S，KV\_S和S2表示key shape中的S，Q\_N表示num\_query\_heads，KV\_N表示num\_key\_value\_heads。

-   **q_nope**（`Tensor`）：必选参数，表示MLA结构中的query的rope信息，不支持非连续，数据格式支持ND，数据类型支持`bfloat16`，shape为[t * n_q, kv_lora_rank]。 
-   **q_rope**（`Tensor`）：必选参数，表示MLA结构中的query的nope信息，不支持非连续，数据格式支持ND，数据类型支持`bfloat16`，shape为[t * n_q, rope_dim]。 
-   **k_nope**（`Tensor`）：必选参数，表示MLA结构中的key的rope信息，不支持非连续，数据格式支持ND，数据类型支持`bfloat16`，shape为[block_num * block_size, kv_lora_rank]。 
-   **k_rope**（`Tensor`）：必选参数，表示MLA结构中的key的nope信息，不支持非连续，数据格式支持ND，数据类型支持`bfloat16`，shape为[block_num * block_size, rope_dim]。 
-   **k_nope_scale**（`Tensor`）：必选参数，表示k_nope的反量化缩放因子，必选参数，不支持非连续，数据格式支持ND，数据类型支持`float`，shape为[block_num * block_size, 4]。
-   **topk_indcies**（`Tensor`）：必选参数，表示每个token选出的topk索引，必选参数，不支持非连续，数据格式支持ND，数据类型支持`int32`，shape为[t, n_kv * topk]。
-   **block_table**（`double`）：必选参数，表示PageAttention中KV存储使用的block映射表，数据格式支持ND，数据类型支持`int32`，shape为[b, s2_max/block_size]，其中第二维表示长度不小于所有batch中最大的s2对应的block数量，即s2_max / block_size向上取整。
-   **actual_seq_lengths_key**（`Tensor`）：必选参数，数据格式支持ND,表示不同Batch中`key`和`value`的有效token数，数据类型支持`int32`,shape为[b]。
-   **nq**（`int`）：必选参数，代表缩放系数，作为query和key矩阵乘后Muls的scalar值，数据类型支持float。
-   **n_kv**（`int`）：必选参数，代表缩放系数，作为query和key矩阵乘后Muls的scalar值，数据类型支持float。
-   **softmax_scale**（`float`）：必选参数，代表缩放系数，作为query和key矩阵乘后Muls的scalar值，数据类型支持float。
-   **topk**（`int`）：必选参数，代表选取的token个数，数据类型支持int。
-   **tile_config**（`TileShapeConfig`）：TileShapeConfig配置结构体，表示tile切分配置，配置项数据类型支持int。


## 返回值说明<a name="zh-cn_topic_0000001832267082_section22231435517"></a>

-   **attn_res**（`Tensor`）：公式中的输出。数据格式支持ND，数据类型支持`bfloat16`，输出shape[b, s, n_q, kv_lora_rank]。

## 调用示例<a name="zh-cn_topic_0000001832267082_section14459801435"></a>

-   详见[testdsv32_sparse_flash_attention_quant_decode.py](testdsv32_sparse_flash_attention_quant_decode.py)
# mla indexer polog quant 

## 功能说明

MLA Indexer Prolog 模块将MLA Prolog和Lightning Indexer Prolog两个算子进行了更大范围的融合，实现了算子间的流水并行，提升了算子的性能。

## 函数原型
```
mla_indexer_prolog_quant_debug(inputs, outputs, mla_epsilon_cq, mla_epsilon_ckv, mla_cache_mode, mla_tile_config, ip_attrs, ip_configs):
```

## 参数说明

>**说明：**<br> 
>
>- B（Batch Size）表示输入样本批量大小、S（Sequence Length）表示输入样本序列长度、He（Head Size）表示隐藏层大小、N（Head Num）表示多头数、Hcq表示q低秩矩阵维度、Hckv表示kv低秩矩阵维度、D表示qk不含位置编码维度、Dr表示qk位置编码维度、Nkv表示kv的head数、BlockNum表示PagedAttention场景下的块数、BlockSize表示PagedAttention场景下的块大小、T表示BS合轴后的大小。

-   **token_x**（`Tensor`）：公式中用于计算Query和Key的输入tensor。不支持非连续，数据格式支持ND，数据类型支持`bfloat16`，shape为[T, He]或[t, h]。
-   **w_dq**（`Tensor`）：公式中用于计算Query的下采样权重矩阵$W^{DQ}$。数据格式支持NZ，数据类型支持`bfloat16`，shape为[h, q_lora_rank]。
-   **w_uq_qr**（`Tensor`）：公式中用于计算Query的上采样权重矩阵$W^{UQ}$和位置编码权重矩阵$W^{QR}$。不支持非连续，数据格式支持NZ，数据类型支持`int8`，shape为[q_lora_rank, n_q*q_head_dim]。
-   **w_uk**（`Tensor`）：公式中用于计算Key的上采样权重$W^{UK}$。不支持非连续，数据格式支持ND，数据类型支持`bfloat16`，shape为[n_q, qk_nope_head_dim, kv_lora_rank]。
-   **w_dkv_kr**（`Tensor`）：公式中用于计算Key的下采样权重矩阵$W^{DKV}$和位置编码权重矩阵$W^{KR}$。不支持非连续，数据格式支持NZ，数据类型支持`bfloat16`，shape为[h, kv_lora_rank+rope_dim]。
-   **rmsnorm_gamma_cq**（`Tensor`）：计算$c^Q$的RmsNorm公式中的$\gamma$参数。不支持非连续，数据格式支持ND，数据类型支持`bfloat16`，shape为[q_lora_rank]。
-   **rmsnorm_gamma_ckv**（`Tensor`）：计算$c^{KV}$的RmsNorm公式中的$\gamma$参数。不支持非连续，数据格式支持ND，数据类型支持`bfloat16`，shape为[kv_lora_rank]。
-   **rope_sin**（`Tensor`）：用于计算旋转位置编码的正弦参数矩阵。不支持非连续，数据格式支持ND，数据类型支持`bfloat16`，shape为[t, rope_dim]。
-   **rope_cos**（`Tensor`）：用于计算旋转位置编码的余弦参数矩阵。不支持非连续，数据格式支持ND，数据类型支持`bfloat16`，shape为[t, rope_dim]。
-   **kv_cache**（`Tensor`）：用于cache索引的aclTensor，计算结果原地更新（对应公式中的$k^C$）。数据格式支持ND，数据类型支持`int8`，cache_mode为"PA_BSND"、shape为[block_num, block_size, n_kv, kv_lora_rank]。
-   **kr_cache**（`Tensor`）：用于key位置编码的cache，计算结果原地更新（对应公式中的$k^R$）。数据格式支持ND，数据类型支持`bfloat16`，cache_mode为"PA_BSND"、shape为[block_num, block_size, n_kv, rope_dim]。
-   **cache_index**（`Tensor`）：用于存储kv_cache和kr_cache的索引。不支持非连续，数据格式支持ND，数据类型支持`int64`，shape为[T]。
-   **dequant_scale_w_uq_qr**（`Tensor`）：用于MatmulQcQr矩阵乘后反量化操作的per-channel参数。数据格式支持ND，数据类型支持`float`，shape为[n_q*q_head_dim, 1]。
-   **wq\_b**（`Tensor`）：表示 query 的权重，必选参数，不支持非连续的Tensor，数据格式支持NZ，数据类型支持`int8`,shape为[q_lora_rank, idx_n_heads*idx_head_dim]。
-   **wq\_b\_scale**（`Tensor`）：表示 query 的权重反量化因子，必选参数，不支持非连续的Tensor，数据格式支持ND，数据类型支持`float32`,shape为[idx_n_heads*idx_head_dim, 1]。
-   **wk**（`Tensor`）：表示 key 的权重，必选参数，不支持非连续的Tensor，数据格式支持NZ，数据类型支持`bfloat16`,shape为[h, idx_head_dim]。
-   **weights_proj**（`Tensor`）：表示 weights 的权重，必选参数，不支持非连续的Tensor，数据格式支持NZ，数据类型支持`bfloat16`，shape为[h, idx_n_heads]。
-   **ln_gamma_k**（`Tensor`）：表示 key 的 layernorm 缩放，必选参数，不支持非连续的Tensor，数据格式支持ND，数据类型支持`bfloat16`，shape为[idx_head_dim]。
-   **ln_beta_k**（`Tensor`）：表示 key 的 layernorm 偏移，必选参数，不支持非连续的Tensor，数据格式支持ND，数据类型支持`bfloat16`,shape为[idx_head_dim]。
-   **cos_idx_rope**（`Tensor`）：表示用于 RoPE 的 cos，不支持非连续的 Tensor，数据格式支持 ND，数据类型支持`bfloat16`，shape为[t, rope_dim]。
-   **sin_idx_rope**（`Tensor`）：表示用于 RoPE 的 sin，不支持非连续的 Tensor，数据格式支持 ND，数据类型支持`bfloat16`，shape为[t, rope_dim]。
-   **hadamard_q**（`Tensor`）：表示用于 query Hadamard 变换的权重矩阵，不支持非连续的 Tensor，数据格式支持 ND，数据类型支持`bfloat16`，shape为[idx_head_dim, idx_head_dim]。
-   **hadamard_k**（`Tensor`）：表示用于 key Hadamard 变换的权重矩阵，不支持非连续的 Tensor，数据格式支持 ND，数据类型支持`bfloat16`，shape为[idx_head_dim, idx_head_dim]。
-   **idx_k_cache**（`Tensor`）：表示 key 的缓存，必选参数，不支持非连续的 Tensor，数据格式支持 ND，数据类型支持`int8`，shape为[block_num, block_size, n_kv, idx_head_dim]。
-   **idx_k_scale_cache**（`Tensor`）：表示 key 反量化因子的缓存，必选参数，不支持非连续的 Tensor，数据格式支持 ND，数据类型支持`float16`，shape为[idx_block_num, block_size, n_kv]。
-   **idx_k_cache_index**（`Tensor`）：表示更新 key 缓存的位置，必选参数，不支持非连续的 Tensor，数据格式支持 ND，数据类型支持`int64`，shape为[t]。
mla_epsilon_cq, mla_epsilon_ckv, mla_cache_mode
-   **ip_attrs**（`float`）：lightning indexer prolog子图计算所需的属性值，包括layernorm_epsilon_k，layout\_query，layout\_key
-   **layernorm_epsilon_k**（`float`）：表示 key layernorm 防除 0 系数，必选参数，数据类型支持`float32`。
-   **layout\_query**（`str`）：可选参数，用于标识输入`query`的数据排布格式，默认值"TND"。当前仅支持 "TND"。
-   **layout\_key**（`str`）：可选参数，用于标识输入`key`的数据排布格式，默认值"PA_BSND"。当前仅支持 "PA_BSND"。
-   **mla_epsilon_cq**（`float`）：计算$c^Q$的RmsNorm公式中的$\epsilon$参数。用户未特意指定时，建议传入1e-05，仅支持double类型，默认值为1e-05。
-   **mla_epsilon_cq**（`float`）：计算$c^{KV}$的RmsNorm公式中的$\epsilon$参数。用户未特意指定时，建议传入1e-05，仅支持double类型，默认值为1e-05。
-   **mla_cache_mode**（`str`）：表示kv_cache的模式，支持"PA_BSND"。
-   **mla_tile_config**（`int`）：表示mla子图的tile切分配置。
-   **ip_config**（`int`）：表示mla子图的tile切分配置及动态分档配置。

## 返回值说明
-   **q_nope**（`Tensor`）：公式中Query的输出tensor（对应$q^N$）。数据格式支持ND，数据类型支持`bfloat16`，shape为[t, n_q, kv_lora_rank]。
-   **q_rope**（`Tensor`）：公式中Query位置编码的输出tensor（对应$q^R$）。数据格式支持ND，数据类型支持`bfloat16`，shape为[t, n_q, rope_dim]。
-   **q_norm_scale**（`Tensor`）：Query输出的反量化参数。数据格式支持ND，数据类型支持`float`，shape为[T1]或[B, S]。
-   **kv_cache**（`Tensor`）：Key输出到`kv_cache`中的tensor（对应$k^C$）。数据格式支持ND，数据类型支持`int8`，shape为[block_num, block_size, n_kv, kv_lora_rank]。
-   **kr_cache**（`Tensor`）：Key的位置编码输出到`kr_cache`中的tensor（对应$k^R$）。数据格式支持ND，数据类型支持`bfloat16`，shape为[block_num, block_size, n_kv, qk_rope_dim]。
-   **k_scale_cache**（`Tensor`）：Key做反量化后输出的反量化参数。数据格式支持ND，数据类型支持`float`，shape为[block_num, block_size, n_kv, 4]。
-   **query**（`Tensor`）：公式中 query 的输出 tensor，数据格式支持 ND，数据类型支持`int8`，shape为[t, idx_n_heads, idx_head_dim]。
-   **query_scale**（`Tensor`）：公式中 query 反量化因子的输出 tensor，数据格式支持 ND，数据类型支持`float16`，shape为[t, idx_n_heads]。
-   **weights**（`Tensor`）：公式中 weights 的输出 tensor，数据格式支持 ND，数据类型支持`float16`，shape为[t, idx_n_heads]。

## 调用示例

- 详见 [testdsv32_mla_indexer_prolog_prefill](testdsv32_mla_indexer_prolog_prefill.py)