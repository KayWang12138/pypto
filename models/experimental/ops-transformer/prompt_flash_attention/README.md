# PromptFlashAttention

## 概述

PromptFlashAttention 是用于全量推理场景的 FlashAttention 算子，实现了高效的注意力计算机制。

## 数学公式

$$
\text{Attention}(Q, K, V) = \text{Softmax}\left(\frac{QK^T}{\sqrt{d}}\right)V
$$

其中：
- $Q$: Query 矩阵，形状为 `[B, N, Sq, D]`
- $K$: Key 矩阵，形状为 `[B, N, Skv, D]`
- $V$: Value 矩阵，形状为 `[B, N, Skv, D]`
- $d$: Head 维度
- $B$: Batch size
- $N$: 注意力头数（Num heads）
- $Sq$: Query 序列长度
- $Skv$: Key/Value 序列长度

## 实现说明

### API 映射

| 公式步骤 | PyPTO API |
|---------|-----------|
| $K^T$ | `pypto.transpose(key, 2, 3)` |
| $QK^T$ | `pypto.matmul(query, k_t, out_dtype=pypto.DT_BF16)` |
| $QK^T / \sqrt{d}$ | `pypto.mul(scores, scale)` |
| Softmax | `pypto.softmax(scores_scaled, dim=-1)` |
| Attention @ V | `pypto.matmul(attn_weights, value, out_dtype=pypto.DT_BF16)` |

### 数据类型

- 输入/输出：BF16 (BFloat16)

### 张量布局

- 输入格式：BNSD（Batch, Num heads, Sequence, Dimension）
- 输出格式：BNSD（与输入一致）

## 编译运行

### 环境要求

```bash
# 设置 NPU 设备 ID
export TILE_FWK_DEVICE_ID=0

# 设置 PTO_TILE_LIB_CODE_PATH
export PTO_TILE_LIB_CODE_PATH=${ASCEND_HOME_PATH:-/usr/local/Ascend/cann}/aarch64-linux
```

### 运行测试

```bash
# 运行所有测试
python3 custom/prompt_flash_attention/prompt_flash_attention.py --test all

# 仅运行基础测试（B=1, N=8, Sq=128, Skv=128, D=64）
python3 custom/prompt_flash_attention/prompt_flash_attention.py --test basic

# 仅运行大规模测试（B=2, N=16, Sq=256, Skv=256, D=128）
python3 custom/prompt_flash_attention/prompt_flash_attention.py --test large
```

## 测试结果

### 测试场景 1：基础场景

- **配置**：B=1, N=8, Sq=128, Skv=128, D=64
- **精度**：max diff = 0.007812
- **结果**：✓ 通过（< 0.02）

### 测试场景 2：大规模场景

- **配置**：B=2, N=16, Sq=256, Skv=256, D=128
- **精度**：max diff = 0.007812
- **结果**：✓ 通过（< 0.02）

## 约束说明

1. **数据类型**：支持 BF16
2. **序列长度约束**：
   - Sq 和 Skv 支持不同长度
   - 序列长度建议 128 对齐以获得最佳性能
3. **维度约束**：
   - Head 维度（D）支持 ≤ 512
   - 支持的 N 值：≤ 256

## 已知限制

1. 当前实现不支持 attention mask
2. 当前实现不支持 sparse attention
3. 当前实现不支持量化功能

## 参考文档

- [PromptFlashAttention 官方文档](/mnt/workspace/gitCode/cann/ops-transformer/attention/prompt_flash_attention/README.md)
- [PyPTO Attention 示例](../../examples/03_advanced/advanced_nn/attention/attention.py)