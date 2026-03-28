# DenseLightningIndexerSoftmaxLse 算子

## 算子概述

本算子实现了带因果掩码的 Dense Lightning Indexer Softmax LSE 计算，支持变长序列和多头注意力机制。主要用于自回归模型中的注意力计算，能够计算 softmax 的最大值和指数和（用于后续的 attention logit 计算）。

### 核心功能

- 支持因果注意力掩码（Causal Attention Mask）
- 支持变长序列处理
- 支持多头注意力（Multi-head Attention）
- 支持提前查看未来 N 个 token（通过 next_tokens 参数控制）
- 支持动态 batch 维度（通过 `pypto.DYNAMIC` 实现）

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
| `s1_starts` | INT32 | `[pypto.DYNAMIC]` | 每个batch的查询序列起始位置（支持动态 batch） |
| `s1_ends` | INT32 | `[pypto.DYNAMIC]` | 每个batch的查询序列结束位置（支持动态 batch） |
| `s2_starts` | INT32 | `[pypto.DYNAMIC]` | 每个batch的键序列起始位置（支持动态 batch） |
| `s2_ends` | INT32 | `[pypto.DYNAMIC]` | 每个batch的键序列结束位置（支持动态 batch） |
| `causal_mask` | FP32 | `[T1, T2]` | 因果掩码矩阵，无效位置为 -1e9 |
| `next_tokens` | INT | - | 允许提前查看的未来token数量 |

### 输出参数

| 参数名 | 类型 | 形状 | 说明 |
|--------|------|------|------|
| `softmax_max` | FP32 | `[N2, T1]` | 每行的最大值 |
| `softmax_sum` | FP32 | `[N2, T1]` | 每行的指数和 |

## 测试结果

测试配置：
- 序列长度：T1 = T2 = 32
- 头数：N1 = 8, N2 = 1
- 维度：D = 128
- next_tokens = 2
- batch size：B = 1, 2, 3（测试动态 batch 支持）

测试输出：
```
--- Test Config 1: B=1, T1=32, T2=32, next_tokens=2 ---
Max diff (softmax_max): 0.000000
Max diff (softmax_sum): 0.000002
✓ Test config 1 passed!

--- Test Config 2: B=2, T1=32, T2=32, next_tokens=2 ---
Max diff (softmax_max): 0.000000
Max diff (softmax_sum): 0.000000
✓ Test config 2 passed!

--- Test Config 3: B=3, T1=32, T2=32, next_tokens=2 ---
Max diff (softmax_max): 0.000000
Max diff (softmax_sum): 0.000000
✓ Test config 3 passed!

✓ All tests passed with dynamic batch axis support!
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

## 动态轴支持

本算子支持动态 batch 维度，主要实现方式：

1. **动态参数定义**：
   - `s1_starts`/`s1_ends`/`s2_starts`/`s2_ends` 使用 `[pypto.DYNAMIC]` 形状定义
   - kernel 内部通过 `B = s1_starts.shape[0]` 动态获取 batch 数量

2. **固定序列长度**：
   - T1 和 T2 保持固定值（32）
   - 原因：PyPTO 的 `view` 和 `reshape` API 要求 shapes 参数为固定值
   - 这是框架限制，无法为序列长度添加动态轴

3. **使用方式**：
   ```python
   # 不同 batch 数量的调用示例
   # B=1
   s1_starts = torch.tensor([0], dtype=torch.int32)
   s1_ends = torch.tensor([T1], dtype=torch.int32)
   
   # B=2
   s1_starts = torch.tensor([0, 0], dtype=torch.int32)
   s1_ends = torch.tensor([T1, T1], dtype=torch.int32)
   ```

4. **验证测试**：
   - 测试了 B=1, 2, 3 三种场景
   - 所有测试精度均满足要求（max diff < 1e-3）

## 性能优化

- 使用 Cube 指令进行矩阵乘法加速（tile_shapes: [128, 128]）
- 使用 Vector 指令进行逐元素操作（tile_shapes: 1, S1G, T2）
- 避免动态循环条件，提高编译器优化效果

## 已知限制

1. 当前实现中掩码值使用 -1e9，在极端情况下可能存在精度损失
2. 序列长度 T1 和 T2 在编译时固定为 32，由于 PyPTO 框架限制，`view`/`reshape` 等 API 的 shapes 参数不支持动态值
3. 仅支持 FP16 输入和 FP32 输出
4. batch 维度已支持动态，可通过 `s1_starts`/`s1_ends` 等参数的长度动态调整 batch 数量

## 参考实现

- PyTorch 参考实现：`dense_lightning_indexer_softmax_lse_golden()`
- PyPTO 实现：`dense_lightning_indexer_softmax_lse_kernel_with_mask()`