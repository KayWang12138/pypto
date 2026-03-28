# Flash Attention Score Grad 算子实现

## 概述

本算子实现 Flash Attention 的反向传播梯度计算，用于 Transformer 模型训练场景。

### 数学公式

正向计算：
$$Y = Dropout(Softmax(Mask(\frac{QK^T}{\sqrt{d}} + pse), atten\_mask), keep\_prob) @ V$$

反向计算：
- $dV = P^T @ dY$
- $dQ = \frac{(dS @ K)}{\sqrt{d}}$
- $dK = \frac{(dS^T @ Q)}{\sqrt{d}}$

其中 $S = Mask(\frac{QK^T}{\sqrt{d}} + pse, atten\_mask)$，$P = Dropout(Softmax(S), keep\_prob)$

Softmax 梯度公式：
$$dS = P \times (dP - \sum(P \times dP, dim=-1, keepdim=True))$$

### 关键特性

- **Online Softmax**: 数值稳定的分块 softmax 计算
- **Causal Mask**: 支持 sparse_mode=3 的因果掩码
- **FP32 中间计算**: 所有 softmax 相关操作使用 FP32 保证精度
- **matmul b_trans/a_trans**: 避免单独 transpose 操作

## 目录结构

```
custom/flash_attention_score_grad/
├── spec.md                                    # 需求规范
├── api_report.md                              # API 探索报告
├── design.md                                  # 设计文档
├── flash_attention_score_grad_golden.py       # PyTorch 参考实现
├── flash_attention_score_grad_impl.py         # PyPTO 算子实现
├── test_flash_attention_score_grad.py         # 测试入口
└── README.md                                  # 本文件
```

## 数据规格

### 输入 (BNSD 布局)

| 张量 | Shape | Dtype | 说明 |
|------|-------|-------|------|
| query | [B, N, S, D] | FP16 | Query 输入 |
| key | [B, N, S, D] | FP16 | Key 输入 |
| value | [B, N, S, D] | FP16 | Value 输入 |
| dy | [B, N, S, D] | FP16 | 梯度输入 |

### 输出

| 张量 | Shape | Dtype | 说明 |
|------|-------|-------|------|
| dq_out | [B, N, S, D] | FP16 | Query 梯度 |
| dk_out | [B, N, S, D] | FP16 | Key 梯度 |
| dv_out | [B, N, S, D] | FP16 | Value 梯度 |

### Shape 配置

**动态轴支持：**
- **动态轴**: batch (axis 0), seq_len (axis 2)
- **静态轴**: num_heads (axis 1) = 8, head_dim (axis 3) = 64

```python
NUM_HEADS = 8    # 静态
HEAD_DIM = 64    # 静态
SEQ_LEN = 128    # 用于 view/reshape 的最大容量
# batch_size 和实际 seq_len 在运行时动态获取
```

## 运行方式

### 环境准备

```bash
# 设置 NPU 设备 ID
export TILE_FWK_DEVICE_ID=0

# 设置 pto-isa 路径
export PTO_TILE_LIB_CODE_PATH=/path/to/pto-isa
```

### 运行测试

```bash
# 运行所有测试
python test_flash_attention_score_grad.py

# 运行特定测试
python test_flash_attention_score_grad.py flash_attention_score_grad::test_causal_p0

# 列出所有测试用例
python test_flash_attention_score_grad.py --list
```

## 测试用例

| 用例 | Shape | 参数 | 说明 |
|------|-------|------|------|
| Causal P0 | [B,8,128,64] | scale=1/√64, sparse_mode=3 | 因果掩码验证 (B动态) |

### 精度要求

- FP16: atol=0.005, rtol=0.005
- BF16: atol=0.005, rtol=0.005

## 实现说明

### API 映射

| 操作 | PyPTO API | 说明 |
|------|-----------|------|
| Q @ K^T | `matmul(q, k, DT_FP32, b_trans=True)` | 避免单独 transpose |
| dS^T @ Q | `matmul(ds, q, DT_FP32, a_trans=True)` | 同上 |
| rowmax | `amax(x, dim=-1, keepdim=True)` | Softmax 数值稳定 |
| rowsum | `sum(x, dim=-1, keepdim=True)` | 仅支持 FP32 |
| softmax_grad | 手动实现 | p * (dp - sum(p*dp)) |

### Tiling 配置

```python
# Cube tiling (matmul)
pypto.set_cube_tile_shapes([64, 64], [64, 64], [64, 64])

# Vector tiling (softmax, exp, amax, sum, etc.)
pypto.set_vec_tile_shapes(1, 8, 16, 1024)
```

## 已知限制

1. **动态轴限制**: 
   - batch 和 seq_len 支持动态，但实际 seq_len 不能超过 SEQ_LEN (128)
   - num_heads 和 head_dim 为静态配置，需与编译时一致
2. **sparse_mode**: 仅实现 mode=3 (causal)，其他模式待扩展
3. **GQA**: 当前 N=N2，GQA (N≠N2) 支持待实现
4. **PSE/Dropout**: 未实现，待后续扩展

## 参考信息

- 设计文档: `design.md`
- API 探索: `api_report.md`
- Golden 实现: `flash_attention_score_grad_golden.py`
- 正向参考: `models/experimental/ops-transformer/flash_attention_score/flash_attention_score_impl.py`

---
*生成时间: 2026-03-28*