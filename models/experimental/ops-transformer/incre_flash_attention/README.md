# IncreFlashAttention

## 概述

IncreFlashAttention 是增量推理场景的 FlashAttention 算子。与全量推理不同，增量推理时 query 的 S 轴固定为 1，key 和 value 来自 KV Cache。

## 数学公式

$$
Attention(Q,K,V) = Softmax(\frac{QK^T}{\sqrt{d}})V
$$

其中：
- $Q$: Query 张量，shape 为 (B, N, 1, D)，S轴固定为1
- $K$: Key 张量，shape 为 (B, N, Skv, D)
- $V$: Value 张量，shape 为 (B, N, Skv, D)
- $d$: head_dim，用于缩放因子 $\frac{1}{\sqrt{d}}$

## 实现说明

### 算子接口

```python
@pypto.frontend.jit
def incre_flash_attention_kernel(
    query: pypto.Tensor((BATCH_SIZE, NUM_HEADS, 1, HEAD_DIM), pypto.DT_BF16),
    key: pypto.Tensor((BATCH_SIZE, NUM_HEADS, SEQ_LEN_KV, HEAD_DIM), pypto.DT_BF16),
    value: pypto.Tensor((BATCH_SIZE, NUM_HEADS, SEQ_LEN_KV, HEAD_DIM), pypto.DT_BF16),
    attention_out: pypto.Tensor((BATCH_SIZE, NUM_HEADS, 1, HEAD_DIM), pypto.DT_BF16),
    scale: float,
):
```

### 计算流程

1. **计算注意力分数**: `scores = Q @ K^T` (BF16 → FP32)
2. **缩放**: `scores_scaled = scores * scale`
3. **Softmax 归一化**: `attn_weights = softmax(scores_scaled, dim=-1)`
4. **加权求和**: `output = attn_weights @ V`

## 编译运行

### 环境要求

- CANN 8.5.0+
- PyPTO 环境
- NPU 设备

### 设置环境变量

```bash
export TILE_FWK_DEVICE_ID=0
```

### 编译

```bash
python3 build_ci.py -f python3 --disable_auto_execute
```

### 运行测试

```bash
python3 custom/incre_flash_attention/incre_flash_attention.py
```

## 测试结果

### 测试配置

| 参数 | 值 |
|-----|-----|
| Batch Size | 2 |
| Num Heads | 8 |
| Seq Len Q | 1 (固定) |
| Seq Len KV | 16 |
| Head Dim | 64 |
| Data Type | BF16 |

### 精度结果

- **max diff**: 0.007812
- **精度要求**: < 0.02
- **状态**: ✓ 通过

## 文件列表

```
custom/incre_flash_attention/
├── incre_flash_attention.py  # 算子实现与测试
└── README.md                 # 本文档
```

## 已知限制

1. 当前实现使用静态 shape，Seq Len KV 在编译时确定
2. 仅支持 BF16 数据类型
3. 未实现 attention mask、位置编码等高级特性

## 参考

- [IncreFlashAttention 官方文档](/mnt/workspace/gitCode/cann/ops-transformer/attention/incre_flash_attention/README.md)
- [PyPTO Attention 示例](../../examples/03_advanced/advanced_nn/attention/attention.py)