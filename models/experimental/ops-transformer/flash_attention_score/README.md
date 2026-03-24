# Flash Attention Score 算子

本算子实现了基于 Online Softmax 的 Flash Attention Score 计算，通过分块计算和在线更新避免存储完整的注意力矩阵，提高了内存效率和数值稳定性。

## 总览介绍

Flash Attention Score 是一种优化的注意力机制实现，具有以下特点：
- **Online Softmax**: 分块计算 softmax，避免存储完整的 N×N 注意力矩阵
- **内存高效**: 复杂度从 O(N²) 降低到 O(N)
- **数值稳定**: 通过在线更新最大值保证 softmax 计算的数值稳定性
- **支持掩码**: 提供带掩码和不带掩码两种版本

## 数学公式

标准注意力计算公式：
```
Attention(Q, K, V) = Softmax(Q @ K^T / sqrt(D)) @ V
```

Online Softmax 算法通过分块计算和在线更新实现：
1. 分块计算 Q @ K^T
2. 在线更新最大值 m_i 和指数和 l_i
3. 在线更新输出向量 o_i
4. 最终归一化输出

输入：
- Query: [B, N, Sq, D] (bfloat16)
- Key: [B, N, Skv, D] (bfloat16)  
- Value: [B, N, Skv, D] (bfloat16)
- Attention Mask (可选): [Sq, Skv] (float32)，值为 1 表示不参与计算

输出：
- Attention Output: [B, N, Sq, D] (bfloat16)

## 代码结构

- **`flash_attention_score_impl.py`**: PyPTO kernel 实现，包含两个核心函数：
  - `flash_attention_score_kernel`: 不带掩码版本
  - `flash_attention_score_kernel_with_mask`: 带掩码版本
- **`flash_attention_score.py`**: 测试代码和 golden 参考实现

## 运行方法

### 环境准备

```bash
# 配置 CANN 环境变量
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 设置设备 ID
export TILE_FWK_DEVICE_ID=0

# 设置 PTO_TILE_LIB_CODE_PATH
export PTO_TILE_LIB_CODE_PATH=${ASCEND_HOME_PATH:-/usr/local/Ascend/cann}/aarch64-linux
```

### 执行测试

```bash
# 运行所有测试
python3 flash_attention_score.py

# 仅运行带掩码的测试
python3 flash_attention_score.py with_mask

# 仅运行不带掩码的测试  
python3 flash_attention_score.py no_mask

# 指定运行模式（默认 npu）
python3 flash_attention_score.py --run_mode npu
python3 flash_attention_score.py --run_mode sim
```

## 核心算法实现

### 分块计算策略

```python
# 分块大小
BLOCK_SIZE_KV = 16

# 对 KV 序列进行分块处理
for kv_block_idx in range(num_blocks_kv):
    # 加载 KV 块
    k_block = load_key_block(kv_block_idx)
    v_block = load_value_block(kv_block_idx)
    
    # 计算注意力分数
    scores = Q @ K_block^T * scale
    
    # Online Softmax 更新
    m_ij = max(scores)
    p_ij = exp(scores - m_ij)
    l_ij = sum(p_ij)
    
    # 更新累积输出
    update_online(m_i, l_i, o_i, m_ij, l_ij, p_ij @ V_block)
```

### Online Softmax 更新

```python
# 计算新的最大值
m_new = max(m_old, m_current)

# 计算缩放因子
alpha = exp(m_old - m_new)
beta = exp(m_current - m_new)

# 更新指数和
l_new = alpha * l_old + beta * l_current

# 更新输出
o_new = alpha * o_old + beta * o_current
```

## 关键技术点

- **分块大小**: KV 序列分块大小为 16，平衡计算效率和内存访问
- **数据类型**: 使用 bfloat16 作为计算精度，内部计算使用 float32 保证精度
- **Tiling 配置**: 
  - Cube tile shapes: [64, 64] × [64, 64] → [64, 64]
  - Vec tile shapes: 16 × 128
- **掩码处理**: 通过 valid_mask 机制在 softmax 计算前应用掩码
- **数值稳定性**: 在 softmax 计算前减去最大值，防止指数溢出

## 测试配置

默认测试参数：
- Batch Size: 4
- Num Heads: 8
- Query Seq Length: 64
- KV Seq Length: 128
- Head Dimension: 64

精度验证：
- 相对容差 (rtol): 0.0078125 (1/128)
- 绝对容差 (atol): 0.0001

## 注意事项

1. **内存优化**: 本实现通过 online softmax 避免存储完整的注意力矩阵，大幅降低内存占用
2. **精度权衡**: 使用 bfloat16 精度，相比 float32 会有一定精度损失
3. **掩码语义**: 掩码值为 1 表示该位置不参与注意力计算（被屏蔽）
4. **序列长度**: 当前实现为固定序列长度，如需动态序列长度需修改 kernel 参数
5. **设备要求**: 必须在支持 bfloat16 数据类型的 NPU 设备上运行

## 已知限制

- 序列长度固定，不支持动态形状
- 仅支持 bfloat16 输入数据类型
- 掩码仅支持 2D 形状 [Sq, Skv]

## 性能特点

- **内存复杂度**: O(N) 而非 O(N²)
- **计算效率**: 通过分块计算优化缓存利用率
- **数值稳定**: Online softmax 保证计算稳定性