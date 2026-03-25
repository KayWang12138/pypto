# FlashAttentionScore

## 算子概述

训练场景下使用 FlashAttention 算法实现 self-attention 计算，输出 attention 结果及 softmax 的 max/sum 中间结果（用于反向传播）。

## 数学公式

```
attention_out = Softmax(scale * (query @ key^T)) @ value

softmax_max = max(scores, dim=-1)
softmax_sum = sum(exp(scores - softmax_max), dim=-1)
```

## 输入输出规格

### 输入
| 参数 | Shape | 数据类型 | 说明 |
|------|-------|----------|------|
| query | [B, N, Sq, D] | BF16 | Query 矩阵 |
| key | [B, N, Skv, D] | BF16 | Key 矩阵 |
| value | [B, N, Skv, D] | BF16 | Value 矩阵 |
| scale | scalar | float | 缩放系数 (通常 1/√D) |

### 输出
| 参数 | Shape | 数据类型 | 说明 |
|------|-------|----------|------|
| attention_out | [B, N, Sq, D] | BF16 | Attention 输出 |
| softmax_max | [B, N, Sq, 1] | BF16 | Softmax max 值 |
| softmax_sum | [B, N, Sq, 1] | BF16 | Softmax sum 值 |

## 实现说明

1. 使用 FP32 进行所有中间计算，确保数值精度
2. 手动实现 softmax 以获取 max/sum 中间结果
3. 输出转换回 BF16

## 编译运行

```bash
# 设置环境变量
export TILE_FWK_DEVICE_ID=0
export PTO_TILE_LIB_CODE_PATH=${ASCEND_HOME_PATH:-/usr/local/Ascend/cann}/aarch64-linux

# 运行测试
python3 custom/flash_attention_score/flash_attention_score.py
```

## 测试结果

| 测试用例 | Shape | attention_out diff | softmax_max diff | softmax_sum diff | 状态 |
|---------|-------|-------------------|------------------|------------------|------|
| Level 0 | B=2, N=8, Sq=16, Skv=16, D=64 | 0.007812 | 0.000000 | 0.000000 | ✓ |

## 已知限制

1. 当前版本仅支持 BF16 数据类型
2. 不支持 pse 位置编码、atten_mask、dropout 等扩展功能
3. 不支持 FP8 量化

## 后续优化

- [ ] 支持 pse 位置编码
- [ ] 支持 atten_mask
- [ ] 支持 dropout
- [ ] 支持 FP16/FP32 数据类型
- [ ] 性能优化（分块计算）

## 文件结构

```
custom/flash_attention_score/
├── flash_attention_score.py    # 主实现文件（golden + kernel）
└── README.md                   # 本文档
```