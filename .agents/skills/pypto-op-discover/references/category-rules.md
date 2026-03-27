# 算子类别推断规则

## 类别定义

| 类别 | 判定条件 | 典型算子 |
|------|----------|---------|
| `elementwise` | 逐元素操作，输入输出 shape 相同 | add, mul, relu, sigmoid, tanh, abs, neg, exp, log |
| `reduction` | 沿一个或多个维度规约，输出 shape 缩小 | sum, mean, max, min, argmax, argmin, prod, any, all |
| `normalization` | 归一化操作，通常含均值/方差计算 | layernorm, batchnorm, groupnorm, rms_norm, softmax |
| `attention` | 注意力机制，含 QKV 和 softmax | scaled_dot_product_attention, flash_attention, mha |
| `activation` | 激活函数（无状态参数，elementwise 但语义为激活）| gelu, silu, swish, mish, hardswish, leaky_relu |
| `embedding` | 嵌入查表或位置编码 | embedding_lookup, positional_embedding, rotary_embedding |
| `matmul` | 矩阵乘法或线性变换 | matmul, bmm, batch_matmul, linear, gemm |
| `pooling` | 池化操作（空间降采样）| max_pool1d/2d, avg_pool1d/2d, adaptive_pool |
| `convolution` | 卷积操作 | conv1d, conv2d, conv3d, depthwise_conv, transposed_conv |
| `other` | 不属于上述类别 | gather, scatter, index_select, sort, topk, unique |

## 判定顺序

1. 若算子名称含 `attention` / `attn` → `attention`
2. 若算子名称含 `matmul` / `mm` / `gemm` / `linear` → `matmul`
3. 若算子名称含 `norm` / `normalize` → `normalization`（softmax 也归此类）
4. 若算子名称含 `pool` → `pooling`
5. 若算子名称含 `conv` → `convolution`
6. 若算子名称含 `embed` / `rope` / `rotary` → `embedding`
7. 若计算含规约（sum/mean/max/min 等）且输出维度减少 → `reduction`
8. 若是常见激活函数（gelu/silu/swish/mish 等）→ `activation`
9. 若逐元素操作 → `elementwise`
10. 其他 → `other`
