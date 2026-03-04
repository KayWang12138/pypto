# OpenCode AI LSTM

## 概述

OpenCode AI LSTM 是基于 PyPTO 框架实现的长短期记忆网络（Long Short-Term Memory）算子，专为 NPU 优化，用于序列建模和自然语言处理任务。

## 算子功能

LSTM 是一种特殊的循环神经网络（RNN），能够学习长期依赖关系。该算子实现了以下核心功能：

1. **输入融合**：将状态输入与 z4 输入按 alpha 系数融合
2. **门控机制**：包含遗忘门（Forget Gate）、输入门（Input Gate）和输出门（Output Gate）
3. **候选细胞状态**：使用 RMSNorm 和 GELU 激活函数
4. **细胞状态更新**：c_new = prev_cell * f_gate + c_act * i_gate
5. **隐藏状态生成**：h_new = h_act * o_gate

## 数学公式

### RMSNorm
```
RMSNorm(x) = x / sqrt(mean(x^2) + eps)
```

### GELU 激活函数（使用 Sigmoid 近似）
```
GELU(x) = x * sigmoid(1.702 * x)
```

### LSTM 前向传播
```
# 1. 输入融合
fused = states_4d + alpha * z4_4d

# 2. 分割为四个门
[pre_f, pre_i, pre_o, pre_c] = split(fused, 4)

# 3. 门控计算
f_gate = sigmoid(pre_f)  # 遗忘门
i_gate = sigmoid(pre_i)  # 输入门
o_gate = sigmoid(pre_o)  # 输出门

# 4. 候选细胞状态
c_cand_norm = RMSNorm(pre_c, eps_cell)
if w_cell is not None:
    c_cand_norm = c_cand_norm * w_cell + b_cell
c_act = GELU(c_cand_norm)

# 5. 细胞状态更新
c_new = prev_cell * f_gate + c_act * i_gate

# 6. 隐藏状态计算
h_temp = RMSNorm(c_new, eps_state)
if w_state is not None:
    h_temp = h_temp * w_state + b_state
h_act = GELU(h_temp)

# 7. 最终输出
h_new = h_act * o_gate
```

## 文件结构

```
models/opencode_ai/
├── opencode_ai_lstm.py          # 测试和 golden 参考实现
├── opencode_ai_lstm_impl.py      # PyPTO 核心实现
└── README.md                     # 本文档
```

## 输入输出规格

### 输入

| 参数 | 形状 | 数据类型 | 说明 |
|------|------|----------|------|
| states_4d | [batch_size, hidden_dim * 4] | FP16 | 状态输入 |
| z4_4d | [batch_size, hidden_dim * 4] | FP16 | z4 输入 |
| prev_cell | [batch_size, hidden_dim] | FP16 | 前一时刻的细胞状态 |
| w_cell | [hidden_dim] | FP16 | 细胞权重（可选） |
| b_cell | [hidden_dim] | FP16 | 细胞偏置（可选） |
| w_state | [hidden_dim] | FP16 | 状态权重（可选） |
| b_state | [hidden_dim] | FP16 | 状态偏置（可选） |

### 输出

| 参数 | 形状 | 数据类型 | 说明 |
|------|------|----------|------|
| h_out | [batch_size, hidden_dim] | FP16 | 隐藏状态输出 |
| c_out | [batch_size, hidden_dim] | FP16 | 细胞状态输出 |

### 配置参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| alpha | float | 0.1 | 输入融合系数 |
| eps_cell | float | 1e-6 | 细胞 RMSNorm epsilon |
| eps_state | float | 1e-6 | 状态 RMSNorm epsilon |

## 编译和运行

### 环境变量设置

```bash
# 设置 NPU 设备 ID
export TILE_FWK_DEVICE_ID=0

# 设置 PyPTO ISA 库路径
export PTO_TILE_LIB_CODE_PATH=/mnt/workspace/gitCode/jie-zhang/pto-isa/
```

### 运行测试

```bash
python3 models/opencode_ai/opencode_ai_lstm.py
```

## 测试结果

### 测试用例

- **测试名称**：opencode_ai_lstm_32_bs_4096_d
- **Batch Size**：32
- **Hidden Dimension**：4096

### 精度测试结果

```
Max Diff Hidden: 0.005900
Max Diff Cell:   0.003955
>> Precision Test PASSED!
```

### 精度标准

- **相对误差容限**：0.001
- **绝对误差容限**：5e-3

## 性能优化

### Tiling 配置

- **Batch Tile Size**：1
- **Hidden Tile Size**：4096（128 字节对齐）
- **Loop Unrolling**：[1, 2, 4]

### 运行时选项

```python
runtime_options={
    "device_sched_mode": 1,
    "stitch_cfgcache_size": 2700000
}
```

## 实现特点

1. **高精度计算**：关键计算路径使用 FP32 精度
2. **循环展开**：使用 loop_unroll 优化批量处理
3. **语义标签**：添加语义标签便于调试和性能分析
4. **内存优化**：使用 inplace 操作减少内存分配

## 已知限制

1. 当前实现仅支持固定的 batch_size 和 hidden_dim
2. 权重和偏置为可选参数，但必须同时提供或不提供
3. 数据类型固定为 FP16

## 参考资料

- PyPTO 官方文档：`./docs/`
- PyPTO 示例：`./examples/`
- Arctic LSTM 参考实现：`models/arctic/sum_lstm.py`

## 许可证

Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
