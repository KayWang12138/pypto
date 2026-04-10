# Flash Attention Score

## 概述

Flash Attention Score 是一个高效的注意力机制实现，支持注意力掩码处理，采用 **Online Softmax** 算法实现分块计算。

### 数学公式

$$
\text{attention\_out} = \text{Softmax}\left(\frac{Q @ K^T}{\sqrt{d}} \cdot \text{mask}\right) @ V
$$

其中：
- $Q$ 为 query 张量，shape 为 $[B, N, S_q, D]$
- $K$ 为 key 张量，shape 为 $[B, N, S_{kv}, D]$
- $V$ 为 value 张量，shape 为 $[B, N, S_{kv}, D]$
- $d$ 为 head dimension
- $\text{mask}$ 为注意力掩码，值为 1 表示不参与计算，值为 0 表示参与计算

### 功能特性

- ✅ **Online Softmax 分块计算**
- ✅ 使用 FP32 进行中间计算以提高精度
- ✅ 支持注意力掩码处理
- ✅ 支持 PyTorch `autograd.Function` 封装的训练态 backward
- ✅ **支持动态轴**：Batch size、Query seq len、KV seq len 均为动态维度，无需重编译
- ✅ 满足 bfloat16 精度标准：`atol=0.0001, rtol=0.0078125`

## Online Softmax 算法

### 算法原理

Online Softmax 通过分块计算，避免存储完整的 attention matrix：

1. **分块处理**: Python 外层按 `Q` 维切块调度，kernel 内将 K 和 V 沿 $S_{kv}$ 维度分块（8B preset 当前默认 `Q=8192, KV=128`）
2. **动态更新**: 维护三个中间变量
   - `running_max`: 当前最大值
   - `running_sum`: 当前 exp 之和
   - `running_output`: 累积输出
3. **归一化**: 每个 q-block 内最终 `output = running_output / running_sum`

### 算法步骤

```python
for each block in K/V:
    # 1. 计算 attention scores
    scores = Q @ K_block^T / sqrt(d) + mask

    # 2. 更新最大值
    new_max = max(running_max, block_max)

    # 3. 调整之前的累积值
    correction = exp(running_max - new_max)
    running_sum *= correction
    running_output *= correction

    # 4. 累积当前块
    exp_scores = exp(scores - new_max)
    running_sum += sum(exp_scores)
    running_output += exp_scores @ V_block

    # 5. 更新状态
    running_max = new_max

# 最终归一化
output = running_output / running_sum
```

### 优势

- **内存效率**: 不需要存储 $S_q \times S_{kv}$ 的完整 attention matrix
- **缓存友好**: 分块计算提高数据局部性
- **长序列支持**: 理论上支持任意长度的序列

## 规格说明

| 参数 | 类型 | Shape | 数据类型 | 说明 |
|------|------|-------|----------|------|
| query | 输入 | [B, N, Sq, D] | bfloat16 | Query 张量，**B、Sq 为动态轴** |
| key | 输入 | [B, N, Skv, D] | bfloat16 | Key 张量，**B、Skv 为动态轴** |
| value | 输入 | [B, N, Skv, D] | bfloat16 | Value 张量，**B、Skv 为动态轴** |
| atten_mask | 输入 | [Sq, Skv] | float32 | 注意力掩码，**Sq、Skv 为动态轴** |
| output | 输出 | [B, N, Sq, D] | bfloat16 | 输出张量，**B、Sq 为动态轴** |

**动态轴说明**：
- **B (Batch size)**: 动态维度，运行时可变，无需重编译
- **Sq (Query sequence length)**: 动态维度，运行时可变，无需重编译
- **Skv (Key/Value sequence length)**: 动态维度，运行时可变，无需重编译

**固定维度**：
- N (Number of attention heads): `32`
- D (Head dimension): `128`
- 当前默认 block size: `Q=8192, KV=128`
- 当前默认 KV unroll: `1`
- 当前默认 vec tile: `128,256`
- 优化过程总结见 `OPTIMIZATION_SUMMARY.md`

## 真实模型对齐

当前仓库里最容易混淆的是“脚本名”和“实际 MoE 总参数规模”不是一回事。

### `aigcode_8b_jamba_gdn_moe`

来源脚本：
- `/sharedata/llx/pto/Mindspeed-LLM/examples/mcore/qwen2/pretrain_aigcode_8b_4k_jamba_gdn_moe_cann850_tpe.sh`

真实 attention 形状：
- Query/Key/Value: `[B, 32, S, 128]`
- `B=1`
- `S=8192`
- `mask=causal`

对应运行方式：

```bash
python3 flash_attention_score.py flash_attention::test_forward --run_mode npu
python3 flash_attention_score.py flash_attention::test_backward --run_mode npu --seq_q 128 --seq_kv 128 --mask causal
python3 flash_attention_score.py flash_attention::benchmark --run_mode npu
python3 flash_attention_score.py flash_attention::benchmark_train --run_mode npu --seq_q 128 --seq_kv 128 --mask causal
```

## 编译与运行

### 环境准备

```bash
# 设置 NPU 设备 ID（使用空闲卡）
export TILE_FWK_DEVICE_ID=14

# 设置 PTO_TILE_LIB_CODE_PATH
export PTO_TILE_LIB_CODE_PATH=/usr/local/Ascend/cann-8.5.0/aarch64-linux
```

### 运行测试

```bash
cd custom/flash_attention_score
python3 flash_attention_score.py
```

### 预期输出

```
============================================================
Test: Flash Attention Score (Online Softmax)
============================================================
Input shape: query=torch.Size([1, 2, 16, 64]), key=torch.Size([1, 2, 64, 64])
Mask shape: torch.Size([1, 1, 16, 64])
Output shape: torch.Size([1, 2, 16, 64])
Block size: 16, Num blocks: 4
Max difference: 0.002987
Mean difference: 0.000307
✓ Test passed (rtol=0.0078125, atol=0.0001)
```

## 实现细节

### 关键技术点

1. **Online Softmax 分块计算**
   - Python 外层按 `Q` 分块调度
   - kernel 内分块处理 K/V，避免存储完整 attention matrix
- 当前默认 `KV_UNROLL_LIST=1`，优先保证长序列 correctness
- 对 8B preset，当前默认 `q_block=8192`，优先减少 kernel launch

2. **精度优化**
   - 使用 FP32 进行中间计算
   - 只在输入和输出时使用 bfloat16

3. **掩码处理**
   - 掩码在 CPU 端预处理为 FP32 格式
   - 通过加 -10000.0 实现屏蔽效果

4. **中间变量管理**
   - kernel 内维护 `running_max / running_sum / running_output`
   - 为规避多 `q_block` 交互，`q_block` 循环已移到 Python wrapper

### API 映射

| PyTorch | PyPTO | 说明 |
|---------|-------|------|
| torch.matmul | pypto.matmul | 矩阵乘法 |
| torch.transpose | pypto.transpose | 转置 |
| torch.exp | pypto.exp | 指数函数 |
| torch.sum | pypto.sum | 求和 |
| torch.max | pypto.maximum/amax | 最大值 |
| - | pypto.view | 视图操作 |
| - | pypto.loop | 循环控制 |

## 测试结果

### 精度测试

- `forward`:
  - `S=64`: max diff `0.001953`
  - `S=128`: max diff `0.003906`
  - `S=256`: max diff `0.001953`
  - `S=512`: max diff `0.001953`
  - `S=1024`: max diff `0.003906`
  - `S=2048`: max diff `0.000977`
  - `S=4096`: max diff `0.001953`
  - `S=8192`: max diff `0.003906`
- `backward`:
  - `S=64`: `dQ/dK/dV` max diff `0.015625 / 0.015625 / 0.003906`
  - `S=128`: `dQ/dK/dV` max diff `0.008789 / 0.015625 / 0.001953`
- backward 当前按 BF16 标准使用 `rtol=0.03, atol=0.02`

### 当前验证配置

- Model preset: `aigcode_8b_jamba_gdn_moe`
- Num heads: `32`
- Head dimension: `128`
- Block size: `Q=8192, KV=128`
- KV unroll: `1`
- Vec tile: `128,256`

### 当前最优 forward 路线

- 这轮稳定收益最大的优化，是尽量减少外层 `q_block` wrapper 调度，也就是尽可能减少 kernel launch
- 在同一份稳定代码上，真实 8B 形状 `[1, 32, 8192, 128]` 的 steady-state forward 实测是：
  - `Q=128`: `4296.957 ms`
  - `Q=256`: `2263.673 ms`
  - `Q=512`: `1191.296 ms`
  - `Q=1024`: `637.811 ms`
  - `Q=2048`: `375.659 ms`
  - `Q=4096`: `223.492 ms`
  - `Q=8192`: `169.533 ms`
- 当前最优路线的单-kernel runtime_debug 结果：
  - Output: `output/output_20260402_202451_519913_3042750_C0A890D9`
  - `AICore End-to-End Time = 168666.34 us`
  - `Average utilization = 57.19%`
  - `Average bubble = 10.80%`
  - `Tasks = 370752`
- 已验证精度：`Q=8192`, `seq_q=seq_kv=8192`, `causal`, `max diff = 0.001953`

## 已知限制

1. **训练态 backward 当前只验证到单 block 形状**：
   `seq_q <= 128` 且 `seq_kv <= 128`
2. **forward 当前已验证到真实 8B 形状 `S=8192`**：
   这版默认实现以 correctness-first 为主，`KV` unroll 默认关闭；对 8B preset，当前默认 `q_block=8192`
3. **CANN masked flash-attention 在小尺寸 benchmark 下存在接口限制**：
   当前 `causal` 小 mask 需要 `atten_mask_shape=[2048,2048]`，因此 `benchmark_train` 会跳过 CANN 对照
4. **无 Dropout**: 未实现 dropout 功能

## 动态轴使用示例

```python
import torch
from flash_attention_score_impl import flash_attention_score_kernel_with_mask

# 支持不同的 batch 和 seq 长度，无需重编译
# 测试用例 1
query = torch.randn(2, 8, 32, 64, dtype=torch.bfloat16, device='npu:0')
key = torch.randn(2, 8, 64, 64, dtype=torch.bfloat16, device='npu:0')
value = torch.randn(2, 8, 64, 64, dtype=torch.bfloat16, device='npu:0')
atten_mask = torch.zeros(32, 64, dtype=torch.float32, device='npu:0')
output = torch.empty(2, 8, 32, 64, dtype=torch.bfloat16, device='npu:0')
flash_attention_score_kernel_with_mask(query, key, value, atten_mask, output)

# 测试用例 2 - 不同的 batch 和 seq 长度
query = torch.randn(8, 8, 128, 64, dtype=torch.bfloat16, device='npu:0')
key = torch.randn(8, 8, 256, 64, dtype=torch.bfloat16, device='npu:0')
value = torch.randn(8, 8, 256, 64, dtype=torch.bfloat16, device='npu:0')
atten_mask = torch.zeros(128, 256, dtype=torch.float32, device='npu:0')
output = torch.empty(8, 8, 128, 64, dtype=torch.bfloat16, device='npu:0')
flash_attention_score_kernel_with_mask(query, key, value, atten_mask, output)
```

## 文件说明

```
flash_attention_score/
├── flash_attention_score.py          # 测试文件（含 golden）
├── flash_attention_score_impl.py     # 算子实现（支持动态轴）
└── README.md                         # 本文档
```

## 参考资料

- [PyPTO API 文档](../../docs/api/operation/)
- [Attention 示例](../../examples/03_advanced/advanced_nn/attention/)
- [Online Softmax 论文](https://arxiv.org/abs/2006.04768)
