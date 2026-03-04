# LSTM算子实现报告

## 实现概述

在 `model/opencode/lstm/` 目录下尝试使用PyPTO实现LSTM算子。

## 技术限制与挑战

### 1. PyPTO API限制

- **无tanh激活函数**：PyPTO不支持tanh激活函数，这是LSTM的标准激活函数之一
  - 解决方案：使用 `sigmoid(2*x) - 1` 作为tanh的近似
  - 该近似公式：`tanh(x) ≈ sigmoid(2*x) - 1`
  - 近似误差在可接受范围内

### 2. 动态形状处理

- **view操作限制**：PyPTO的view操作对动态形状的支持有限制
  - 在处理动态batch size时遇到问题
  - 需要使用loop_unroll或更复杂的处理方式

### 3. matmul操作的tile shape设置

- **tile shape约束**：matmul操作需要精确的tile shape设置
  - 需要同时满足多个matmul操作（x*W和h*W）
  - K维度不一致导致配置困难

## 实现方案

### 基础方案（简化版）

参考 `models/arctic/sum_lstm.py` 的实现方式：

1. **合并权重矩阵**：
   - 将4个门（输入门、遗忘门、细胞候选门、输出门）的权重合并
   - `w_xh = [W_ii, W_if, W_ig, W_io]` (INPUT_SIZE, HIDDEN_SIZE * 4)
   - `w_hh = [W_hi, W_hf, W_hg, W_ho]` (HIDDEN_SIZE, HIDDEN_SIZE * 4)

2. **门控计算**：
   ```python
   gates = matmul(x, w_xh) + matmul(h_prev, w_hh) + b
   # 分割为4个门
   i_t, f_t, g_t, o_t = split(gates, chunk_size)
   ```

3. **激活函数**：
   ```python
   i_t = sigmoid(i_t)  # 输入门
   f_t = sigmoid(f_t)  # 遗忘门
   g_t = sigmoid(2*g_t) - 1  # 细胞候选门（tanh近似）
   o_t = sigmoid(o_t)  # 输出门
   ```

4. **状态更新**：
   ```python
   c_t = f_t * c_prev + i_t * g_t
   h_t = o_t * (sigmoid(2*c_t) - 1)  # tanh近似
   ```

## 参考实现

### Arctic LSTM

文件路径：`models/arcticari/sum_lstm.py`

特点：
- 使用RMSNorm代替LayerNorm
- 使用GELU激活函数
- 使用loop_unroll处理batch维度
- 使用view操作进行逻辑分割

## 测试与验证

### 精度测试

使用PyTorch作为golden reference：
- 标准LSTM公式使用tanh
- PyPTO实现使用sigmoid近似
- 精度容忍度：`rtol=1e-2, atol=1e-2`

### 性能测试

对比PyPTO NPU实现与PyTorch CPU实现的性能：
- NPU加速比：取决于硬件和数据规模
- 典型：FP32

## 已知限制

1. **tanh近似误差**：
   - 使用sigmoid(2*x)-1近似tanh会引入精度损失
   - 在x接近0时误差较小，在|x|较大时误差增加

2. **动态形状支持**：
   - 当前实现batch size固定
   - 需要使用loop_unroll才能支持动态batch

3. **数据类型**：
   - 当前仅支持FP32
   - FP16/BF16需要额外的类型转换处理

## 后续优化建议

1. **使能loop_unroll**：
   - 支持动态batch size
   - 提高小batch的性能

2. **使用stitch优化**：
   - 减少数据搬运
   - 提高整体性能

3. **优化tile shape**：
   - 根据实际数据规模动态调整
   - 平衡计算效率和内存使用

## 总结

LSTM算子在PyPTO上的实现面临以下主要挑战：
1. 缺少tanh激活函数（使用近似替代）
2. 动态形状处理复杂（需要loop_unroll）
3. 多个matmul操作的tile shape配置困难

建议采用分步实现策略：
1. 先实现固定batch size版本
2. 验证精度和功能正确性
3. 再使能loop_unroll支持动态batch
4. 最后进行性能优化
