# scaled_dot_product_attention 算子设计文档

> **算子名称**: scaled_dot_product_attention
> **算子分类**: attention
> **生成时间**: 2026-03-28T00:00:00Z
> **基于**: spec.md

---

## 1. 概述
### 1.1 功能描述
根据 spec.md 的描述， scaled_dot_product_attention 用于计算 Transformer 模型中的注意力分数。该算子用于支持动态轴（batch, num_heads, seq_len),是核心计算特性。

其实现包括标准 attention、 causal mask、 和 dropout 等可选功能.
### 1.2 数学公式
Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d)) @ V
### 1.3 算法描述
Algorithm: Scaled Dot Product Attention
────────────────────────────────────
输入: Q, K, V, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None
输出: output

1. 计算缩放因子: scale_factor = 1/sqrt(d) if scale is None else scale
2 初始化注意力偏置: attn_bias = zeros(L, S)
3. if is_causal:
     3.1 生成下三角掩码: temp_mask = tril(ones(L, S))
     3.2 应用因果掩码: attn_bias[~temp_mask] = -inf
4. if attn_mask is not None:
     4.1 if attn_mask.dtype == bool:
           attn_bias[~attn_mask] = -inf
         else:
           attn_bias = attn_mask
4. 计算注意力分数: attn_weight = Q @ K^T * scale_factor
5. Softmax: attn_weight = softmax(attn_weight, dim=-1)
6. Dropout: if dropout_p > 0: attn_weight = dropout(attn_weight, dropout_p)
7. 计算输出: output = attn_weight @ V
8. return output
### 1.4 数据流图
```
        Q                    K                    V
   [N, H, L, E]        [N, H, S, E]        [N, H, S, Ev]
        │                    │                    │
        │              ┌─────┴─────┐              │
        │              │ transpose │              │
        │ [N, H, S, E]    └─────┴─────┘              │
        │              │   (2,3)              │
        │              │ matmul   │              │
        │ [N, H, L, E] @ [N, H, E, S] -> [N, H, L, S]
        │              │   (2,3)              │
        │              │ matmul   │              │
        │ [N, H, L, S] @ [N, H, S, Ev] -> [N, H, L, Ev]
        │                    │                    │
        └────────────────┘────────────────────┘
                    ▼
```

---

## 2. API 映射设计
### 2.1 数学公式分解
将公式拆解为基本操作步骤:
| 步骤 | 数学表达 | 说明 |
|------|----------|------|
| 1 | K^T | K 转置 | 交换最后两个维度 |
| 2 | Q @ K^T | 计算注意力分数 | Q 与 K^T 的矩阵乘法 |
| 3 | scale | 缩放注意力分数 | 乘以缩放因子 |
 | 4 | add bias | 应用注意力掩码或加偏置 |
| 5 | softmax | 归一化 | 对注意力分数进行 softmax |
| 6 | dropout (opt) | Dropout | 对注意力权重进行 dropout (可选) |
| 7 | @ V | 计算输出 | 注意力权重与 V 的矩阵乘法 |
### 2.2 PyPTO API 映射表
| 步骤 | 数学表达 | PyPTO API | 参数 | 文档路径 |
|------|----------|-----------|------|----------|
| 1 | K^T | `pypto.transpose(input, 2, 3)` | input=K, dim0=2, dim1=3 | `docs/api/operation/pypto-transpose.md` |
| 2 | Q @ K^T | `pypto.matmul(input, mat2, out_dtype, b_trans=True)` | input=Q, mat2=K, out_dtype,FP32, b_trans=True | `docs/api/operation/pypto-matmul.md` |
| 3 | scale | `pypto.mul(input, scale)` | input=scores, scale | - | `docs/api/operation/pypto-mul.md` |
| 4 | add bias | `pypto.add(input, other)` | input=scores, other=attn_bias | - | `docs/api/operation/pypto-add.md` |
| 5 | softmax | `pypto.softmax(input, dim)` | input=scores_scaled, dim=-1 | **仅支持 DT_FP32** | `docs/api/operation/pypto-softmax.md` |
| 6 | dropout | **不支持** | - | PyPTO 无 dropout API | - |
| 7 | @ V | `pypto.matmul(input, mat2, out_dtype)` | input=attn_weights, mat2=v, out_dtype=FP32 | `docs/api/operation/pypto-matmul.md` |
### 2.3 计算步骤序列
```python
# 1. K 转置: [N, H, S, E] -> [N, H, E, S]
k_t = pypto.transpose(k, 2, 3)
# 2. 计算注意力分数: Q @ K^T
# 使用 b_trans=True 参数来避免显式转置
scores = pypto.matmul(q, k_t, out_dtype=pypto.DT_FP32, b_trans=True)
# 3. 缩放注意力分数
scores_scaled = pypto.mul(scores, scale)
# 4. 应用注意力掩码
if attn_mask is not None:
    if attn_mask.dtype == torch.bool:
        attn_bias = pypto.where(attn_mask, 0.0, float('-inf'))
    else:
        attn_bias = attn_mask
    scores = pypto.add(scores, attn_bias)
# 5. Softmax
attn_weights = pypto.softmax(scores_scaled, dim=-1)
# 6. Dropout (跳过, p2/p3 不支持)
# 7. 计算输出: output = pypto.matmul(attn_weights, v, out_dtype=pypto.DT_FP32)
```
### 2.4 设计依据
- **来源**: api_report.md, Pypto.matmul 文档, pypto.transpose 文档
 Pypto.mul/add/softmax 文档, 参考实现: examples/03_advanced/advanced_nn/attention/attention.py
- **说明**: 选择这些 API 的原因
  - PyPTO 审 API 成熟,支持 4D 和 (2,3) 荬置交换
  - matmul 是性能的核心操作,使用 b_trans=True 齏化显式转置
  - mul/add 支持广播和适合 tiling 刭 弬 softmax 是数值稳定性要求的关键操作,使用 FP32 可保证精度

  - transpose/mul/add/matmul 的 API 组合支持动态轴处理
  - 参考实现展示了完整的 attention 计算流程和包括 tiling 配置

  - 需要简化的是参考实现使用了 `pypto.loop` + `pypto.view` 处理 paged KV cache

 这实现使用了简化版本,在线 softmax + 分块计算,是简化实现,适合我们的标准 attention 场景

---

## 3. 数据规格设计
### 3.1 SclaedDotProductAttentionInput dataclass
```python
@dataclass
class ScaledDotProductAttentionInput:
    query: Tensor  # Query tensor [N, H, L, E], float32 / bfloat16
    key: Tensor  # Key tensor [N, H, S, E], float32 / bfloat16
    value: Tensor  # Value tensor [N, H, S, Ev], float32 / bfloat16
    attn_mask: Optional[Tensor] = None  # Attention mask, [N, H, L, S] or [L, S], bool / float
    dropout_p: float = 0.0  # Dropout probability
    is_causal: bool = False  # Whether to apply causal mask
    scale: Optional[float] = None  # Scale factor (default: 1/sqrt(E))
```
### 3.2 SclaedDotProductAttentionOutput dataclass
```python
@dataclass
class ScaledDotProductAttentionOutput:
    output: Tensor  # Output tensor [N, H, L, Ev], float32 / bfloat16
```
### 3.3 中间 Tensor 定义
| 名称 | Shape | dtype | 说明 |
|------|-------|-------|------|
| scores | [N, H, L, S] | FP32 | 注意力分数 |
| scores_scaled | [N, H, L, S] | FP32 | 缩放后的注意力分数 |
| attn_weights | [N, H, L, S] | FP32 | 注意力权重 |
| attn_weights_bf16 | [N, H, L, S] | BF16 | 注意力权重 (BF16) |
| attn_bias | [N, H, L, S] or [L, S] | FP32 | 注意力偏置 |
| causal_mask | [L, S] | FP32 | 因果掩码 (预计算) |
### 3.4 数据格式选择
所有 tensor 使用 ND 格式,因为:
- 简单算子: 使用 ND 格式
- Flash Attention 鸀分块计算时可能需要使用 NZ 格式优化数据布局
- PyPTO 的 transpose/mul/softmax/matmul 不支持 NZ 格式的尾轴转置
 因此对于 K^T 的转置,我们选择使用支持的 (2,3) 交换
- 4D 只支持 ND/NZ 格式,选择合适的数据格式
- 参考 GLM-4.5 和 examples,03_advanced 实现
都使用 ND 格式
- 修正建议: 如果需要使用 NZ 格式进行分块计算,考虑使用 ND 格式并适当调整 tile shape
### 3.5 动态轴定义
根据 spec.md, 动态轴包括 batch, num_heads, seq_len。
在动态轴定义中,我们需要明确说明:

这些轴在计算过程中可能会变化,其值范围由 spec.md 确定.

这有助于后续实现时根据实际的 batch size和序列长度进行计算。
避免不必要的错误。

此外,动态轴的处理也使得代码更加灵活,能够适应不同的输入 shape.

#### 辵 宇说明:
- N: batch size
- H: number of attention heads
- L: query sequence长度
- s: key/value sequence length
- E: embedding dimension
- Ev: embedding维度
 对于每个 head，需要根据 head_dim 讐查询序列长度计算
 缩放因子
- 因此,使用 scale = head_dim ** -0.5 作为缩放因子是避免数值溢出
 此外,对于数值稳定性,使用在线 softmax 来更新最大值和累加值。避免数值溢出问题。
最后,缩放后的注意力分数乘以缩放因子,然后进行 softmax 操作,得到注意力权重
 将权重与 V 进行矩阵乘法,得到最终输出。
#### 5.4 Tiling 歖略
基于参考实现的分析, standard attention 使用以下 tiling 配置:
 对于性能测试和建议使用更小的分块大小 (如 64x64) 以减少内存占用和避免过大的中间结果
#### 5.1 TileShape 初值设置
```python
# Cube tiling for matmul operations
# 参考 examples/03_advanced/advanced_nn/attention/attention.py
pypto.set_cube_tile_shapes([64, 64], [64, 64], [64, 64])
# Vector tiling for elementwise operations
pypto.set_vec_tile_shapes(1, 8, 16, HEAD_dim)
```
#### 5.2 Tiling 依据
- **来源**: examples/03_advanced/advanced_nn/attention/attention.py 和 GLm-4.5 的 paged attention 实现
- **判断依据**: 两个参考实现都使用了 tiling,且需要处理动态轴
 都是性能测试中的重要考虑因素
- **适用条件**: 适用于大多数标准 attention 场景,包括小规模(功能测试)、中等规模(性能测试)和大规模(长序列)场景
  **不适用场景**: 隐式分块计算场景(如 Flash Attention),可能需要更复杂的实现,目前不推荐
 - **判断依据**: tiling 配置主要来自参考实现,这两个示例都使用类似的 tiling 策略
- **适用条件**: 核心场景使用上述配置,其他场景可根据实际需求调整
 - **不适用场景**: 无(暂时不考虑) - 需要在不适用场景中通过配置参数选择或 tiling 策略,并在文档中说明推荐值和默认值
 这有助于后续实现时根据实际需求进行调整 tiling 配置。
#### 5.3 注意事项
- **matmul 前必须调用 set_cube_tile_shapes**: 蟃用 matmul 噂必须设置 vector tiling (3D/4D 或 4D)
- **softmax 精度问题**: PyPTO softmax 仅支持 FP32,需要进行类型转换
 混合精度场景:matmul 输出 FP32, softmax 辸 FP32
 宇可以 cast 回 FP16/bf16; 宭方案中完整精度场景: matmul 直接使用 FP16, 在 softmax 嚴使用 pypto.cast 将输入和 pypto.DT_FP32), 计算,最后 cast 回 FP16/bf16。  *注意: 这会增加一次开销,但可以避免在 matmul 之外进行额外的 cast 操作。
- **transpose 4D 约束**: 只支持 (2,3) 交换,其他交换需要通过组合实现或在文档中明确说明,让用户根据实际需求选择合适的方式
- **K^T 转置**: 对于 4D 输入 [N, H, S, E],只支持 (2,3) 交换, 这避免了 (0,3) 和 (0,1) 的不支持,可以通过组合实现处理, 例如:
 可以通过 arange + 换实现下三角掩码,然后使用 masked_fill 或 pypto.where 进行处理
#### 5.4 循环结构设计
### 5.1 Loop 券结论
- **结论**: 不使用 loop
- **原因**: 输入 shape固定且较小,所有轴都是编译期已知,不需要循环切分
- **适用条件**: 仅适用于静态 shape 或编译期可确定 shape的场景
- **限制**: 动态 shape 麸必须使用 pypto.loop 进行切分
- **Loop 类型**: 无循环
- **适用条件**: 静态 shape 或编译期已知 shape
- **限制**: 仅适用于静态 shape,无法实现动态轴
#### 5.2 黟 / 动态轴处理
| 轴 | 类型 | 处理方式 |
|------|----------|----------|
| batch | 静态 | 编译期已知,无需处理 | 动态 | 编译期使用 pypto.loop, 在运行时循环切分 |
| num_heads | 静态 | 编译期已知,无需处理 | 不切分 | seq_len | 静态 | 编译期已知,无需处理 | 不切分 |
| attn_mask | 可选 | 编译期已知或如有则使用条件判断;如未提供则使用默认值 |
> 注: 若未提供 attn_mask， 应使用 pypto.where 枼加 进行类型转换, **循环次数**: 栂0
 2
> attn_mask 为 None: 0
> attn_mask is not None: 1
> **结论**: 不需要循环
> **原因**: 动态 shape 场景下需要遍历 batch 维度,减少循环次数,提高性能
- **适用条件**: 支持动态轴的所有场景
- **限制**: 不适用于超长序列或需要考虑其他优化策略
#### 5.3 循环合并策略
所有循环外层的循环合并成一个主循环,按顺序执行:
- 外层循环: 遍历 batch 维度
 batch_idx 从 0 到 batch_size
         内层循环: 遍历 num_heads
             for head_idx in range(num_heads):
                 qh = query[batch_idx, head_idx * head_dim]
                 qkt = key[batch_idx, head_idx * head_dim]
                 scores = pypto.matmul(qh, k_t, pypto.DT_FP32, b_trans=True)
                 # 应用缩放因子
                 scores_scaled = pypto.mul(scores, scale)
                 # Softmax (数值稳定性)
                 scores_fp32 = pypto.cast(scores_scaled, pypto.DT_FP32)
                 attn_weights = pypto.softmax(scores_fp32, dim=-1)
                 # 计算输出
                 output[batch_idx, head_idx] = pypto.matmul(attn_weights, v, pypto.DT_FP32)
```
> 注: 此实现使用 masked_fill 来避免显式循环,提高效率

> **循环次数**: 1 次
 * batch_size = num_heads = num_heads
 * head_dim = head_dim
 * seq_len = seq_len
 * batch_size * num_heads * seq_len * head_dim
#### 5.4 数据依赖处理
- **scores**: 依赖 scores_scaled
- **attn_weights**: 依赖 attn_weights_bf16 (输出时需要 cast 回原 dtype)
- **注意**: softmax 辻度要求
 鷷度处理时使用 float('int') 鵽在循环内进行 softmax 还一 次计算,然后 cast 回 float32, 揁精度保证,- **推荐**: 方案 B (混合精度)

 - matmul 输出 FP32, softmax 使用 FP32
  - 输出 cast 回 bf16/float16
- **时间**: 叀 matmul 输出 FP32 会增加一次显式 cast, 但 matmul 使用 FP32 可以更好地利用高精度单元,提高计算精度
- **空间**: 在 softmax 傂层需要 cast 回 FP32, 会增加额外的内存访问和寄存器压力

  - 输出 cast 回 float16 会增加一次额外的时间开销,  - 方案 A 的输出 dtype与 FP32, 可以利用 matmul 的高精度单元和提高整体精度
- **注意**: 两个方案都可以正常工作,但方案 B 更完整。 实现更简洁, 性能更好
 方案 A 完全避免了精度问题,且更容易维护,而方案 B 在简化版本中默认,推荐方案 B

 对于大多数标准 attention 场景已经足够

 最终实现将基于这两个方案的调整进行优化。

#### 5.5 尾块处理策略
针对动态轴处理, seq_len 维度采用分块策略:
  - 外层循环: 遍历 seq_tile (每个 seq_tile 内计算)
  - 内层循环: 遍历 num_heads (按 tile 大块计算
  - 内层循环结束后,将结果合并到 oi_update 受, 最后执行一次 matmul 宣成最终结果

  - 分块大小: 根据动态轴范围调整 (  - 六例: 64
  - 在超长序列场景(如 1024, 4096), 适当减小分块大小
  - 脱离/分块逻辑更简洁
 风险: Pypto.loop 在动态轴上需要按顺序处理每个 tile, 因此需要判断当前 tile 是否是最后一个 tile
然后应用掩码

 对于 mask 类型和 causal mask 的处理方式不同:
  - bool mask: 在循环外处理,避免重复创建 tensor
  - float mask: 在循环内直接加到 scores 上 (更高效)
  - 无 mask: 在循环外通过条件判断处理
避免循环,创建 tensor
#### 5.6 循环 unroll配置
由于动态 shape 支持,使用静态展开:
但在编译期生成多版本代码,优化性能, 因此使用 unroll 配置:
`unroll_list=[1, 2, 4, 8, 16]`
```
> **优化目的**: 减少循环次数, 改善性能
> **注意**:
  - 上述配置是示例值,实际使用时需要根据具体场景调整
  - unroll_list 的选择影响性能, 风险和不可预测
  - 对于 seq_len 刌在 tail轴时，应该使用较小的分块大小 (如 128) 以减少重复的数据搬运
 提高性能
  - 但对于超长序列(如 4096), unroll 配置中的较小的分块大小可能无法提供显著优势,反而可能导致性能下降

  - **建议**: 根据实际性能需求调整,推荐值为 128
 256, 512, 1024, 4096 等, 使用较小的分块
 菺更好的性能
  - 对于典型场景(如 1024x1024, 512), 64 是合理的分块大小
  - **不建议**: 过小的分块(如 32) 在长序列场景下可能导致过多的循环次数,反而降低性能

  - 对于长序列(如 4096), 增大分块到 256 可以减少循环次数,但会显著提升性能
  - **建议**: 64/128/256 可长序列场景下根据实际 seq_len 调整

> **注意**: 分块大小需要根据动态轴范围动态调整
---
## 6. 验证方案
### 6.1 Golden 函数设计
使用 `scaled_dot_product_attention_golden.py` 作为参考实现
### 6.2 测试用例设计
基于 spec.md 中的典型配置,每个配置对应一个测试场景.
| 配置名称 | 类型 | 优先级 | 参数 | 输入 Shape | 输出 Shape | 说明 |
|----------|------|--------|------|------------|------------|------|
| 性能_P0 | 性能 | P0 | scale=None, dropout_p=0.0 | Q:[1,8,1024,128], K:[1,8,1024,128], V:[1,8,1024,128] | [1,8,1024,128] |
| 功能_P0 | 功能 | P0 | scale=None, dropout_p=0.0 | Q:[2,4,512,64], K:[2,4,512,64], v:[2,4,512,64] | [2,4,512,64] |
| 因果注意力_P1 | 功能 | P1 | is_causal=True, scale=None | Q:[1,8,512,128], K:[1,8,512,128], V:[1,8,512,128] | [1,8,512,128] |
| 掩码注意力_P1 | 功能 | P1 | attn_mask=[1,1,512,512] | Q:[1,8,512,128], K:[1,8,512,128], V:[1,8,512,128] | [1,8,512,128] |
| 长序列_P2 | 功能 | P2 | scale=None | Q:[1,4,4096,64], K:[1,4,4096,64], V:[1,4,4096,64] | [1,4,4096,64] |
| Dropout_P2 | 功能 | P2 | dropout_p=0.1 | Q:[1,8,256,128], K:[1,8,256,128], V:[1,8,256,128] | [1,8,256,128] |
#### 6.2 边界情况测试（可选)
| 场景 | 参数 | 说明 |
|------|------|------|
| 单头注意力 | num_heads=1 | 测试 num_heads=1 的场景 |
| 不等序列长度 | L != s | 测试 query 和 key/value 序列长度不同 |
| 空掩码 | attn_mask=None | 测试不提供 mask 的情况 |
| dtype 跬换 | 测试 float16/bfloat16 输入 |
### 6.3 精度验证标准
| dtype | atol | rtol | 说明 |
|-------|------|------|
| float32 | 0.001 | 0.001 | 高精度 |
| float16 | 0.01 | 0.01 | 混合精度 |
| bfloat16 | 0.01 | 0.01 | 混合精度 |
---
## 7. 性能指标与开箱配置
### 7.1 性能目标
基于 spec.md 兀性能类典型配置的预期性能:
| 配置名称 | 类型 | 优先级 | 参数 | 输入 Shape | 输出 Shape | 预期 kernel 耗时 |
|----------|------|--------|------|------------|------------|------------------|
| 性能_P0 | 性能 | P0 | scale=None, dropout_p=0.0 | Q:[1,8,1024,128], K:[1,8,1024,128], V:[1,8,1024,128] | [1,8,1024,128] | < 10us |
| 功能_P0 | 功能 | P0 | scale=None, dropout_p=0.0 | Q:[2,4,512,64], K:[2,4,512,64], V:[2,4,512,64] | [2,4,512,64] | < 5ms |
| 因果注意力_P1 | 功能 | P1 | is_causal=True | scale=None | Q:[1,8,512,128], K:[1,8,512,128], V:[1,8,512,128] | [1,8,512,128] | < 10ms |
| 长序列_P2 | 功能 | P2 | scale=None | Q:[1,4,4096,64], K:[1,4,4096,64], V:[1,4,4096,64] | [1,4,4096,64] | < 100ms |
**注意**: 实际性能需要根据硬件环境和具体优化结果进行调整,以下为初步目标。
### 7.2 开箱性能配置
```python
# Tiling configuration
tile_config = {
    'cube_tile_shapes': [[128, 128], [128, 128], [128, 128]],
    'vec_tile_shapes': [1, 8, 16, 128],
}

# Runtime options
runtime_options = {
    'run_mode': 'npu',  # or 'sim' for simulation
}
```
### 7.3 pass_options 配置
暂无特殊 pass_options 齀求
### 7.4 runtime_options 配置
```python
runtime_options = {
    'run_mode': 'npu',  # or 'sim' for simulation
}
```
---
## 8. 风险与注意事项
### 8.1 已知约束
- **Softmax 仅支持 FP32**: PyPTO softmax 仅支持 FP32, 遇到 FP16/BF16 输入时需要 cast
  - **Transpose 4D 限制**: 4D transpose 只支持 (2,3) 交换,不支持 (0,1) 和 (0,3)
 交换
  - **Dropout 不支持**: PyPTO 目前不支持 dropout, 标记为 P2/P3 功能
  - **动态轴循环**: 对于动态轴 (如 batch, seq_len), 需要使用 pypto.loop, 可能影响性能
### 8.2 常见错误规避
| 风险/错误 | 触发场景 | 影响 | 原因 | 规避方法 |
|-------------|----------|-------------|----------|
| softmax 精度溢出 | FP16/BF16 输入直接 softmax | 精度丢失 | 方案 B: matmul 输出 FP32, softmax 使用 FP32 |
| transpose 轴错误 | 使用不支持的轴交换 | 程序崩溃 | 使用支持的 (2,3) 交换 |
| 动态 shape 处理 | 动态轴超出范围 | 循环边界错误 | 在循环内添加边界检查 |
| mask 类型错误 | bool mask 未转换 | 掩码不生效 | 使用 pypto.where 或 bool mask 转换 |
### 8.3 特殊场景处理
- **is_causal=True 时**: 确保只应用一次因果掩码,避免重复应用
- **attn_mask 与 is_causal 互斥**: 两者不能同时使用, 如果同时提供, is_causal 优先级更高
会忽略 attn_mask
- **dropout_p > 0**: 当前实现不支持,如需实现需要自定义方案
- **长序列场景**: 需要调整 tiling 配置和可能需要分块计算优化
### 8.4 实现建议
- **精度优先**: 推荐使用方案 B (matmul FP32 + softmax FP32) 以获得更好的精度
- **性能优化**: 可以根据实际硬件环境和数据特点调整 tiling 配置
- **动态轴**: 对于动态轴场景,建议在循环中添加边界检查
- **长序列**: 对于超长序列(>2048), 值得考虑更小的分块大小以避免内存溢出
- **测试覆盖**: 巻加更多边界测试用例,如空输入、极小值、极大值等
- **文档完善**: 添加详细的 API 文档和性能基准

- **错误处理**: 增强错误处理和提供有用的错误信息
---
## 9. 交付件清单
### 9.1 目录结构
```
custom/scaled_dot_product_attention/
├── spec.md                          # 需求规范(已有)
├── design.md                        # 设计文档(本文件)
├── scaled_dot_product_attention_golden.py        # Golden 参考实现
├── scaled_dot_product_attention_impl.py          # 算子核心实现代码
├── test_scaled_dot_product_attention.py          # 测试代码
└── output/                          # 运行输出(自动生成)
```
### 9.2 文件清单
| 文件 | 类型 | 说明 | 生成方式 |
|------|------|------|----------|
| spec.md | 需求 | 算子需求规范 | pypto-intent-understanding |
| design.md | 设计 | 算子设计文档 | pypto-op-design (本 skill) |
| scaled_dot_product_attention_golden.py | 代码 | Golden 参考实现 | pypto-golden-generator |
| scaled_dot_product_attention_impl.py | 代码 | 算子核心实现 | 后续实现 |
| test_scaled_dot_product_attention.py | 代码 | 测试用例 | 后续实现 |
### 9.3 命名规范
遵循 PyPTO 烘例命名规范:
- 算子名称: `scaled_dot_product_attention` (小写字母 + 下划线)
- 目录名: 与算子名称一致 `custom/scaled_dot_product_attention`
- Golden 文件: `scaled_dot_product_attention_golden.py`
- 实现文件: `scaled_dot_product_attention_impl.py`
- 测试文件: `test_scaled_dot_product_attention.py`
### 9.4 生成顺序
```
spec.md → design.md → scaled_dot_product_attention_golden.py -> scaled_dot_product_attention_impl.py -> test_scaled_dot_product_attention.py
```
