# attention_worker_combine 算子

## 概述

`attention_worker_combine` 是 MoE（Mixture of Experts）模型中的注意力 token 融合算子，将多个专家的计算结果按权重加权求和。

## 数学公式

$$y = \sum_{i=0}^{K-1}(expert\_scales[i] \times token\_data[i]) + token\_data[K]$$

其中：
- $K$ 为 TopK 选出的专家数量
- $expert\_scales[i]$ 为第 $i$ 个专家的权重
- $token\_data[i]$ 为第 $i$ 个专家处理的 token 数据
- $token\_data[K]$ 为共享专家（shared expert）的输出数据

## 输入输出规格

### Kernel 接口（batch_size=1）

| 类型 | name | shape | dtype | 说明 |
|------|------|-------|-------|------|
| 输入 | expert_scales | [1, K] | float32 | 专家权重 |
| 输入 | routed_expert_data | [K, HiddenSize] | float16/bfloat16 | K 个路由专家的输出数据 |
| 输入 | shared_expert_data | [1, HiddenSize] | float16/bfloat16 | 共享专家的输出数据 |
| 输出 | y | [1, HiddenSize] | float16/bfloat16 | 融合输出 |

### Wrapper 接口（支持多批次）

| 类型 | name | shape | dtype | 说明 |
|------|------|-------|-------|------|
| 输入 | expert_scales | [BatchSize, K] | float32 | 专家权重 |
| 输入 | routed_expert_data | [BatchSize * K, HiddenSize] | float16/bfloat16 | K 个路由专家的输出数据（展平） |
| 输入 | shared_expert_data | [BatchSize, HiddenSize] | float16/bfloat16 | 共享专家的输出数据 |
| 输出 | y | [BatchSize, HiddenSize] | float16/bfloat16 | 融合输出 |

## 使用方法

### 方法一：使用 Kernel（batch_size=1）

```python
import torch
import pypto
from attention_worker_combine import attention_worker_combine_kernel

# 创建 kernel
K = 8
hidden_size = 2048
token_dtype = 0  # 0 for float16, 1 for bfloat16
kernel = attention_worker_combine_kernel(K, hidden_size, token_dtype)

# 准备输入数据
expert_scales = torch.randn(1, K, dtype=torch.float32)  # [1, K]
routed_expert_data = torch.randn(K, hidden_size, dtype=torch.float16)  # [K, HiddenSize]
shared_expert_data = torch.randn(1, hidden_size, dtype=torch.float16)  # [1, HiddenSize]
y = torch.zeros(1, hidden_size, dtype=torch.float16)  # 输出

# 调用 kernel
kernel(expert_scales, routed_expert_data, shared_expert_data, y)
```

### 方法二：使用 Wrapper（支持多批次）

```python
import torch
from attention_worker_combine import attention_worker_combine_wrapper

# 创建 wrapper
batch_size = 4
K = 8
hidden_size = 2048
token_dtype = 0  # 0 for float16, 1 for bfloat16
wrapper = attention_worker_combine_wrapper(batch_size, K, hidden_size, token_dtype)

# 准备输入数据
expert_scales = torch.randn(batch_size, K, dtype=torch.float32)  # [BatchSize, K]
routed_expert_data = torch.randn(batch_size * K, hidden_size, dtype=torch.float16)  # [BatchSize*K, HiddenSize]
shared_expert_data = torch.randn(batch_size, hidden_size, dtype=torch.float16)  # [BatchSize, HiddenSize]
y = torch.zeros(batch_size, hidden_size, dtype=torch.float16)  # [BatchSize, HiddenSize]

# 调用 wrapper（内部循环调用 kernel）
wrapper(expert_scales, routed_expert_data, shared_expert_data, y)
```

## 数据布局说明

### 路由专家数据布局

**Kernel 接口**：`routed_expert_data` 形状为 `[K, HiddenSize]`，包含 K 个路由专家对当前 token 的输出数据。

**Wrapper 接口**：`routed_expert_data` 形状为 `[BatchSize * K, HiddenSize]`，展平的多批次数据，每个 token 的 K 个专家数据连续排列。

### 共享专家数据布局

**Kernel 接口**：`shared_expert_data` 形状为 `[1, HiddenSize]`，包含共享专家对当前 token 的输出数据。

**Wrapper 接口**：`shared_expert_data` 形状为 `[BatchSize, HiddenSize]`，包含共享专家对多个 token 的输出数据。

## 已知限制

1. **Kernel 接口限制**：`attention_worker_combine_kernel` 仅支持 `batch_size=1`
2. **Wrapper 实现方式**：`attention_worker_combine_wrapper` 通过循环调用 kernel 支持多批次，非并行实现
3. **调度机制未实现**：`schedule_context`、`layer_id`、`need_schedule` 等参数暂未使用
4. **三轴切分策略未实现**：暂未实现 SplitBS、SplitH、SplitK 策略

## 编译运行

```bash
# 设置环境变量
export TILE_FWK_DEVICE_ID=0
export PTO_TILE_LIB_CODE_PATH=${ASCEND_HOME_PATH:-/usr/local/Ascend/cann}/aarch64-linux

# 运行测试
python3 attention_worker_combine.py
```

## 测试结果

```
使用设备: npu:0
配置: batch_size=1, K=8, hidden_size=2048, dtype=float16
✓ 测试通过
```

## 精度标准

- `atol = 0.001`
- `rtol = 0.01`

## 参考

- MoE distributed combine 实现：`models/glm_v4_5/glm_moe_distributed_dispatch_combine.py`
- PyPTO API 文档：`docs/api/`