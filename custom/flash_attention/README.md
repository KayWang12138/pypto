# 算子规格描述：Flash Attention

> 版本：v1.0
> 日期：2026-03-13
> 来源：ifa_flash_torch函数

---

## 一、概述

基于分页 KV Cache 的高效 Flash Attention 实现，支持动态序列长度和 GQA (Grouped Query Attention)。

---

## 二、函数签名

```
flash_attention_kernel(
    q: Tensor[(bs1, n1, d)], dtype=BF16,
    k: Tensor[(block_num, block_size, n2, d)], dtype=BF16,
    v: Tensor[(block_num, block_size, n2, d)], dtype=BF16,
    block_table: Tensor[(b, block_num)], dtype=INT32,
    kv_act_seqs: Tensor[(b,)], dtype=INT32,
    scale: float,
    d_tile: int,
    g_tile: int,
    block_size_tile: int,
) -> Tensor[(bs1, n1, d)], dtype=BF16
```

### 形状符号定义

| 符号 | 含义 | 典型值范围 | 约束 |
|------|------|-----------|------|
| bs1 | batch × seq_len (query 总数) | 1-8192 | 正整数 |
| b | batch 维度 | 1-128 | 正整数 |
| s1 | query 序列长度 | 1-64 | 正整数，bs1 = b × s1 |
| n1 | query 头数 | 1-128 | 正整数 |
| n2 | kv 头数 | 1-32 | 正整数，n1 为 n2 的整数倍 |
| d | 特征维度 | 16-512 | 通常为 2 的幂 |
| block_num | kv block 数量 | 1-1024 | 正整数 |
| block_size | 每个 block 的序列长度 | 32-128 | 通常为 2 的幂 |

---

## 三、数学定义

### 3.1 核心计算公式

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d}}\right)V
$$

其中：
- $Q \in \mathbb{R}^{bs1 \times n1 \times d}$：Query 矩阵
- $K \in \mathbb{R}^{block\_num \times block\_size \times n2 \times d}$：Key 矩阵（分页存储）
- $V \in \mathbb{R}^{block\_num \times block\_size \times n2 \times d}$：Value 矩阵（分页存储）

### 3.2 分步计算

1. **分块 matmul**: $S_{ij} = Q_i K_j^T / \sqrt{d}$
2. **Online Softmax Max**: $m_{ij} = \max(m_{i,j-1}, \text{rowmax}(S_{ij}))$
3. **Online Softmax Exp**: $P_{ij} = \exp(S_{ij} - m_{ij})$
4. **Online Softmax Sum**: $l_{ij} = e^{m_{i,j-1} - m_{ij}} l_{i,j-1} + \text{rowsum}(P_{ij})$
5. **分块 matmul**: $O_{ij} = e^{m_{i,j-1} - m_{ij}} O_{i,j-1} + P_{ij} V_j$
6. **归一化输出**: $O_i = O_{i,last} / l_{i,last}$

### 3.3 归约操作说明

- 归约维度：dim=-1 (K 维度)
- 归约类型：max + sum
- keepdim：True

---

## 四、计算特征分析

### 4.1 计算类型

- [ ] 逐元素运算 (elementwise)
- [ ] 归约运算 (reduction)
- [ ] 矩阵运算 (matmul / GEMM)
- [ ] 结构变换 (reshape / transpose / gather / scatter)
- [x] 混合 (组合以上多种)

### 4.2 计算复杂度

- 时间复杂度：O(bs1 × n1 × actual_seq_len × d)
- 空间复杂度：O(bs1 × n1 × d + block_num × block_size × n2 × d)
- FLOPs 估算：2 × bs1 × n1 × actual_seq_len × d (QK^T + PV)

### 4.3 数据访问模式

- 输入访问：跨步 (通过 block_table 访问不连续的 K/V blocks)
- 输出访问：连续
- 算术强度：高 (计算密集型)

---

## 五、数值精度要求

### 5.1 数据类型

| Tensor | 输入 dtype | 计算 dtype | 输出 dtype |
|--------|----------|----------|----------|
| q | BF16 | FP32 | - |
| k | BF16 | FP32 | - |
| v | BF16 | FP32 | - |
| out | - | FP32 | BF16 |

### 5.2 累加类型

- 归约累加：FP32
- Matmul 累加：FP32

### 5.3 数值稳定性

- Softmax 计算时减去 max 值，避免 exp 溢出
- Online softmax 维护 running max 和 running sum

### 5.4 验证容差

| dtype | rtol | atol |
|-------|------|------|
| BF16 | 5e-3 | 5e-3 |

---

## 六、边界行为

### 6.1 padding / masking

- 通过 kv_act_seqs 指定每个 batch 的实际序列长度
- 未使用的 K/V block 通过 block_table 索引为 -1 标记
- actual_s2_tile 通过 min() 确保不超过实际序列长度

### 6.2 越界处理

- block_size 不整除序列长度时，最后一块使用 valid_shape 指定实际大小
- valid_shape 通过 (cur_seq_val - s2_idx * block_size).min(block_size) 动态计算

### 6.3 特殊值处理

- NaN 处理：传播
- Inf 处理：传播
- 零值处理：softmax 的 exp(0) = 1

---

## 七、测试域

### 7.1 标准测试形状

| 用例 | 形状 | 说明 |
|------|------|------|
| 小规模 | b=4, s1=2, n1=8, n2=2, d=16, block_size=32 | 快速验证 |
| 典型规模 | b=16, s1=1, n1=32, n2=4, d=128, block_size=128 | 常见工作负载 |
| 大规模 | b=32, s1=8, n1=64, n2=8, d=256, block_size=128 | 压力测试 |

### 7.2 边界测试形状

| 用例 | 形状 | 说明 |
|------|------|------|
| 最小 | b=1, s1=1, n1=1, n2=1, d=16, block_size=32 | 最小有效输入 |
| 非对齐 | b=3, s1=1, n1=5, n2=1, d=17, block_size=16 | 不整除 tile 大小 |
| GQA | b=4, s1=1, n1=8, n2=2, d=64, block_size=32 | Grouped Query Attention |

### 7.3 对抗测试用例

- 极大值输入（接近 BF16 最大值）
- 极小值输入（接近零）
- 全零输入
- 全相同值输入
- 变长序列（actual_seq_len 差异大）

---

## 八、假设与限制

- 假设输入已在 NPU 设备上
- 假设 n1 为 n2 的整数倍 (GQA 要求)
- 假设 block_table 已正确初始化
- 限制：不支持反向传播
- 限制：不支持 dropout
- 限制：不支持 alibi positional embedding

---

## 九、实现特性

### 9.1 动态轴支持

- 使用 `pypto.frontend.dynamic()` 定义动态维度
- 运行时通过 `tensor.shape` 获取实际大小
- 支持不同形状的输入，无需重新编译

### 9.2 性能优化

- 使用 `unroll_list=[8,4,2,1]` 优化动态循环
- 使用 `valid_shape` 处理尾块，避免无效计算
- Online softmax 减少 HBM 访问次数
- 分页 KV Cache 支持内存高效管理

---

## 十、编译运行

### 10.1 环境设置

```bash
export TILE_FWK_DEVICE_ID=0
export PTO_TILE_LIB_CODE_PATH=/mnt/workspace/pto_isa/pto-isa/
```

### 10.2 运行测试

```bash
python3 custom/flash_attention/flash_attention.py
```

### 10.3 预期输出

```
Q shape: torch.Size([8, 8, 16])
K shape: torch.Size([4, 32, 2, 16])
V shape: torch.Size([4, 32, 2, 16])
Output shape: torch.Size([8, 8, 16])

Max diff: 0.003906
✓ Flash Attention output matches golden output
```
