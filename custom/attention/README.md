# incre_flash_attention

## 功能说明
为不断优化提升增量推理性能, 提出支持增量推理的incre_flash_attention融合算子需求。

增量推理相对于全量推理, 主要有如下差异：

1. 输入数据特点是query 的S轴固定为1;
2. Key和Value是经过kv cache后, 将之前推理过的state信息叠加在一起, 每个Batch对应的S轴的实际长度可能不一样, 输入的数据是经过padding后的固定长度数据。

**说明：**
  
  <blockquote>KV Cache是大模型推理性能优化的一个常用技术。采样时，Transformer模型会以给定的prompt/context作为初始输入进行推理（可以并行处理），随后逐一生成额外的token来继续完善生成的序列（体现了模型的自回归性质）。在采样过程中，Transformer会执行自注意力操作，为此需要给当前序列中的每个项目（无论是prompt/context还是生成的token）提取键值（KV）向量。这些向量存储在一个矩阵中，通常被称为KV缓存（KV Cache）。</blockquote>


## 计算公式：

  self-attention（自注意力）利用输入样本自身的关系构建了一种注意力模型。其原理是假设有一个长度为$n$的输入样本序列$x$，$x$的每个元素都是一个$d$维向量，可以将每个$d$维向量看作一个token embedding，将这样一条序列经过3个权重矩阵变换得到3个维度为$n$d的矩阵。

  

    self-attention的计算公式一般定义如下，其中$Q$、$K$、$V$为输入样本的重要属性元素，是输入样本经过空间变换得到，且可以统一到一个特征空间中。

  

  $$
     Attention(Q,K,V)=Score(Q,K)V
  $$

  

    本算子中Score函数采用Softmax函数，self-attention计算公式为：

  

  $$
  Attention(Q,K,V)=Softmax(\frac{QK^T}{\sqrt{d}})V
  $$

  

    其中$Q$和$K^T$的乘积代表输入$x$的注意力，为避免该值变得过大，通常除以$d$的开根号进行缩放，并对每行进行softmax归一化，与$V$相乘后得到一个$n*d$的矩阵。

**说明**：
    <blockquote>query、key、value数据排布格式支持从多种维度解读，其中B（Batch）表示输入样本批量大小、S（Seq-Length）表示输入样本序列长度、H（Hidden-Size）表示隐藏层的大小、N（Head-Num）表示多头数、D（Head-Dim）表示隐藏层最小的单元尺寸，且满足D=H/N、T表示所有Batch输入样本序列长度的累加和。
    <br>Q_S表示query shape中的S，KV_S表示key和value shape中的S，Q_N表示num_query_heads，KV_N表示num_key_value_heads。P表示Softmax(<span>(QK<sup class="superscript">T</sup>) / <span class="sqrt">d</span></span>)的计算结果。</blockquote>

##  函数原型
```Python
def incre_flash_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    actual_seq_lengths: torch.Tensor,
    block_table: torch.Tensor,
    num_heads: int,
    input_layout: str,
    num_key_value_heads: int,
    block_size: int
) -> torch.Tensor:
```

##  参数说明

-  **query**（`Tensor`）：数据格式支持ND，数据类型支持`bfloat16` ，shape为 [num_tokens, num_head, head_size]。

-   **key_cache**（`Tensor`）：数据格式支持ND，数据类型支持`bfloat16` ，shape为 [num_blocks, block_size, kv_head_num, head_size]。

-   **value_cache**（`Tensor`）：数据格式支持ND，数据类型支持`bfloat16` ，shape为 [num_blocks, block_size, kv_head_num, head_size]。

-   **block_tables**（`Tensor`）：数据格式支持ND，数据类型支持`int32` ，shape为 [batch_size, max_num_blocks_per_query]。

-   **actual_seqs**（`Tensor`）：数据格式支持ND，数据类型支持`int32` ，shape为 [batch_size]。

-   **attn_res**（`Tensor`）：数据格式支持ND，数据类型支持`bfloat16` ，shape为 [num_tokens, num_head, head_size]。

**说明：**

  <blockquote>batch_size表示输入样本批量大小（当前支持范围1至32）、seq_len表示输入样本序列长度（当前支持为1）、num_tokens表示batch_size和seq_len合轴的大小、num_head表示查询端的多头数量（当前支持为12）、head_size表示每个注意力头的维度（当前支持为128）、num_blocks表示总共可用的缓存块数量、block_size表示每个缓存块能容纳的词元数量（当前支持为128）、kv_head_num表示键/值端的多头数量（当前支持为1）、max_num_blocks_per_query表示单个请求最多可以占用的缓存块数。</blockquote>