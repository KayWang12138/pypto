# DenseLightningIndexerSoftmaxLse 算子

## 算子概述

本算子实现了带因果掩码的 Dense Lightning Indexer Softmax LSE 计算，支持变长序列和多头注意力机制。主要用于自回归模型中的注意力计算，能够计算 softmax 的最大值和指数和（用于后续的 attention logit 计算）。

### 核心功能

- 支持因果注意力掩码（Causal Attention Mask）
- 支持变长序列处理
- 支持多头注意力（Multi-head Attention）
- 支持提前查看未来 N 个 token（通过 next_tokens 参数控制）

## 数学公式

对于每个查询位置 `i` 和键位置 `j`：

1. **计算注意力分数**：
   ```
   score[i, j] = ReLU(Q[i] @ K[j]^T)
   weighted_score[i, j] = score[i, j] * weights[i]
   ```

2. **应用因果掩码**：
   - 当 `j > i + next_tokens` 时，该位置被掩盖
   - 掩盖位置设为 `-inf`，有效位置保持原值

3. **计算 Softmax LSE（Log-Sum-Exp）**：
   ```
   max_val[i] = max(weighted_score[i, :])
   sum_val[i] = sum(exp(weighted_score[i, :] - max_val[i]))
   ```

## API 映射

| PyTorch 操作 | PyPTO API | 说明 |
|-------------|-----------|------|
| `torch.matmul` | `pypto.matmul` | 矩阵乘法 |
| `torch.relu` | `pypto.relu` | ReLU 激活函数 |
| `torch.exp` | `pypto.exp` | 指数函数 |
| `torch.max` | `pypto.amax` | 最大值归约 |
| `torch.sum` | `pypto.sum` | 求和归约 |
| `torch.where` | `pypto.add(mask)` | 通过加法实现掩码 |

## 编译与运行

### 环境要求

```bash
# 设置 NPU 设备 ID
export TILE_FWK_DEVICE_ID=0

# 设置 PyPTO 库路径
export PTO_TILE_LIB_CODE_PATH=${ASCEND_HOME_PATH:-/usr/local/Ascend/cann}/aarch64-linux
```

### 运行测试

```bash
python3 dense_lightning_indexer_softmax_lse.py
```

## 输入输出

### 输入参数

| 参数名 | 类型 | 形状 | 说明 |
|--------|------|------|------|
| `query` | FP16 | `[T1, N1, D]` | 查询向量，T1为序列长度，N1为头数，D为维度 |
| `key` | FP16 | `[T2, N2, D]` | 键向量，T2为序列长度，N2为头数，D为维度 |
| `weights` | FP16 | `[T1, N1]` | 权重矩阵 |
| `s1_starts` | INT32 | `[B]` | 每个batch的查询序列起始位置 |
| `s1_ends` | INT32 | `[B]` | 每个batch的查询序列结束位置 |
| `s2_starts` | INT32 | `[B]` | 每个batch的键序列起始位置 |
| `s2_ends` | INT32 | `[B]` | 每个batch的键序列结束位置 |
| `causal_mask` | FP32 | `[T1, T2]` | 因果掩码矩阵，无效位置为 -1e9 |
| `next_tokens` | INT | - | 允许提前查看的未来token数量 |

### 输出参数

| 参数名 | 类型 | 形状 | 说明 |
|--------|------|------|------|
| `softmax_max` | FP32 | `[N2, T1]` | 每行的最大值 |
| `softmax_sum` | FP32 | `[N2, T1]` | 每行的指数和 |

## 测试结果

测试配置：
- 序列长度：T1 = T2 = 16
- 头数：N1 = 8, N2 = 1
- 维度：D = 128
- next_tokens = 2

测试输出：
```
Max diff (softmax_max): 0.000000
Max diff (softmax_sum): 0.000000
✓ Test passed with causal mask!
```

## 实现细节

### 因果掩码实现

本算子使用浮点数掩码而非布尔掩码：
- 有效位置：值为 0
- 无效位置：值为 -1e9（接近负无穷）

通过加法操作 `masked_res = res + mask` 实现掩码效果，避免使用 PyPTO 不支持的动态条件操作。

### 变长序列处理

使用 `pypto.view` 提取当前 batch 的有效数据区域，通过 `valid_shape` 参数指定实际数据长度，实现对变长序列的支持。

### 多头注意力聚合

对于每个 key head，处理 G = N1/N2 个 query heads，通过 `pypto.sum` 沿着 head 维度聚合权重。

## 性能优化

- 使用 Cube 指令进行矩阵乘法加速（tile_shapes: [128, 128]）
- 使用 Vector 指令进行逐元素操作（tile_shapes: 1, S1G, T2）
- 避免动态循环条件，提高编译器优化效果

## 已知限制

1. 当前实现中掩码值使用 -1e9，在极端情况下可能存在精度损失
2. 序列长度 T1 和 T2 在编译时固定，需要根据实际场景调整
3. 仅支持 FP16 输入和 FP32 输出

## 参考实现

- PyTorch 参考实现：`dense_lightning_indexer_softmax_lse_golden()`
- PyPTO 实现：`dense_lightning_indexer_softmax_lse_kernel_with_mask()`