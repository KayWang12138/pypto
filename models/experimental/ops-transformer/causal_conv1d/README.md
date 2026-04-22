# causal_conv1d 因果卷积算子

## 概述

因果卷积算子（Causal Conv1D）用于 Mamba/SSM 模型的序列处理。支持两种运行模式：

- **Prefill 模式 (FN VARLEN)**: 处理完整序列，支持变长序列（packed layout）
- **Decode 模式 (UPDATE)**: 处理单个或多个 token（投机解码），状态缓存滚动更新

## 数学公式

$$
y[t] = \text{activation}\left(\text{bias} + \sum_{i=0}^{\text{width}-1} w[i] \cdot x[t-\text{width}+1+i]\right)
$$

其中：
- $x[t]$ 为当前 token，$x[t-k]$ 为历史 token（从 conv_state 缓存读取）
- $w[i]$ 为卷积权重
- silu 激活函数: $\text{silu}(x) = x / (1 + e^{-x})$

## 目录结构

```
custom/causal_conv1d/
├── SPEC.md                    # 需求规格文档
├── API_REPORT.md              # API 探索报告
├── DESIGN.md                  # 设计方案文档
├── causal_conv1d_golden.py    # Golden 参考实现
├── causal_conv1d_impl.py      # PyPTO 实现
├── test_causal_conv1d.py      # 测试入口
└── README.md                  # 本文件
```

## 运行方式

### 环境准备

```bash
# 设置 NPU 设备
export TILE_FWK_DEVICE_ID=11

# 或使用空闲卡查询
export TILE_FWK_DEVICE_ID=$(bash .agents/skills/pypto-op-develop/scripts/list_idle_chip_ids.sh | awk '{print $1}')
```

### 运行测试

```bash
cd /data/x00952168/pypto/custom/causal_conv1d

# 运行所有测试
python test_causal_conv1d.py

# 运行单个测试
python test_causal_conv1d.py causal_conv1d::test_prefill_p0
python test_causal_conv1d.py causal_conv1d::test_decode_p0

# 查看可用测试
python test_causal_conv1d.py --list
```

### 测试用例

| 用例 | 参数 | 说明 |
|------|------|------|
| Prefill_P0 | seqlen=2048, dim=2048, batch=1, width=4 | 核心 Prefill 场景 |
| Decode_P0 | seqlen=1, dim=2048, batch=1, width=4 | 核心 Decode 场景 |
| Decode_multi_token | seqlen=4, dim=2048, batch=1, width=4 | 投机解码支持 |

## 精度要求

- **rtol**: 0.01
- **atol**: 0.01
- **dtype**: float16（输入输出），float32（内部计算）

## 实现要点

1. **silu 手动实现**: 使用 exp/div 组合，不使用 `pypto.sigmoid`（仅支持 FP32）
2. **FP32 计算**: 累加和 silu 使用 FP32，避免 FP16 精度问题
3. **动态轴处理**: 使用 `pypto.view` + `pypto.assemble` 处理动态轴
4. **状态管理**: 从 conv_state 或历史 token 加载，处理完后更新 conv_state
5. **Tiling**: Prefill [1, 512], Decode [1, 512]

## 已知限制

1. **当前版本不支持**:
   - Bias 参数
   - cache_indices 间接索引
   - initial_state_mode

2. **width 范围**: 3-6（默认 4）

3. **seqlen 范围**: Prefill [1, 8192], Decode [1, 4]

4. **dim 范围**: [256, 4096]

## 参考

- **参考实现**: vLLM mamba ops causal_conv1d, tilelang 实现
- **论文**: Mamba: Linear-Time Sequence Modeling with Selective State Spaces
- **应用场景**: Mamba, Jamba, SSM 模型的序列处理层