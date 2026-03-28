# FlashAttentionScore 开发计划

> 创建时间：2026-03-24
> 状态：开发中

---

## 一、需求分析

### 1.1 算子功能
训练场景下，使用 FlashAttention 算法实现 self-attention 计算。

### 1.2 数学公式
```
pseType=1: attention_out = Dropout(Softmax(Mask(scale*(pse + query*(key)^T)))) * value
pseType≠1: attention_out = Dropout(Softmax(Mask(scale*(query*(key)^T) + pse))) * value
```

简化版（先实现）：
```
attention_out = Softmax(scale * (query @ key^T)) @ value
```

### 1.3 输入输出规格
| 参数 | 类型 | Shape | 说明 |
|------|------|-------|------|
| query | Tensor | [B, N, Sq, D] | Query 矩阵 |
| key | Tensor | [B, N, Skv, D] | Key 矩阵 |
| value | Tensor | [B, N, Skv, D] | Value 矩阵 |
| attention_out | Tensor | [B, N, Sq, D] | 输出 |
| softmax_max | Tensor | [B, N, Sq, 1] | Softmax max 中间结果 |
| softmax_sum | Tensor | [B, N, Sq, 1] | Softmax sum 中间结果 |

### 1.4 数据类型
- 输入：BF16 / FP16 / FP32
- 中间计算：FP32（确保精度）
- 输出：与输入一致

---

## 二、API 映射方案

### 2.1 PyPTO API 使用
| 操作 | PyPTO API | 说明 |
|------|-----------|------|
| 矩阵乘法 | pypto.matmul | Q @ K^T, weights @ V |
| 转置 | pypto.transpose | K 转置 |
| 缩放 | pypto.mul | scores * scale |
| 最大值 | pypto.amax | softmax max |
| 减法 | pypto.sub | scores - max |
| 指数 | pypto.exp | e^(scores - max) |
| 求和 | pypto.sum | softmax sum |
| 除法 | pypto.div | 归一化 |

### 2.2 Softmax 手动实现
由于 PyPTO softmax 不返回中间结果，需要手动实现：
```python
scores_max = amax(scores, dim=-1, keepdim=True)  # [B, N, Sq, 1]
scores_shifted = scores - scores_max              # [B, N, Sq, Skv]
exp_scores = exp(scores_shifted)                  # [B, N, Sq, Skv]
scores_sum = sum(exp_scores, dim=-1, keepdim=True)  # [B, N, Sq, 1]
attn_weights = exp_scores / scores_sum            # [B, N, Sq, Skv]
```

---

## 三、开发步骤

### 阶段 1：基础版本（当前）
- [x] 创建目录结构
- [ ] 编写 golden 函数（PyTorch 参考）
- [ ] 编写测试用例
- [ ] 编写 jit kernel 实现
- [ ] 验证精度

### 阶段 2：扩展功能（后续）
- [ ] 支持 pse 位置编码
- [ ] 支持 atten_mask
- [ ] 支持 dropout
- [ ] 支持 FP8 量化

---

## 四、关键约束

1. **TileShape 设置**：matmul 前需调用 set_cube_tile_shapes
2. **数据类型**：中间计算使用 FP32 确保精度
3. **对齐要求**：matmul 的 K、N 轴需 32 字节对齐

---

## 五、测试用例

| 级别 | Shape | 说明 |
|------|-------|------|
| Level 0 | B=1, N=2, Sq=16, Skv=16, D=64 | 基础验证 |
| Level 1 | B=2, N=8, Sq=128, Skv=128, D=64 | 典型场景 |
| Level 2 | B=1, N=4, Sq=64, Skv=128, D=128 | 不等长序列 |

---

## 六、开发记录

### 2026-03-24
- 开始开发基础版本
- 创建目录结构