# AttentionWorkerCombine PyPTO 算子

## 算子概述

将多个计算单元处理的注意力 token 数据进行融合，结合专家权重对结果进行加权，输出最终的注意力融合结果。

## 数学公式

$$
y = \sum_{i=0}^{K-1} (token\_data[:, i, :] \times expert\_scales[:, i]) + token\_data[:, K, :]
$$

其中：
- $K$ 为选中的路由专家数量（TopK）
- $token\_data[:, i, :]$ 为第 $i$ 个专家的输出 token
- $expert\_scales[:, i]$ 为第 $i$ 个专家的权重
- $token\_data[:, K, :]$ 为共享专家的输出 token（无权重）

## 规格

| 类型 | shape | dtype |
|------|-------|-------|
| 输入 token_data | [BS, K+1, H] | bfloat16 |
| 输入 expert_scales | [BS, K] | float32 |
| 输出 y | [BS, H] | bfloat16 |

## 切分策略

### 1. SplitBS（默认）
- **适用场景**：BS 较大，H 和 K 较小
- **实现方式**：向量化实现，一次性计算所有 batch
- **特点**：最高效，推荐优先使用

### 2. SplitH
- **适用场景**：H 较大
- **实现方式**：按 hidden 维度切分，每次处理 H_tile 个 hidden 元素
- **特点**：适合大 hidden 维度场景

### 3. SplitK
- **适用场景**：K 较大
- **实现方式**：向量化实现
- **特点**：向量化求和，避免循环累加精度问题

## 关键发现

在开发过程中发现 PyPTO 框架的关键限制：

**循环累加器问题**：
- 问题：PyPTO 循环中的累加器更新存在精度问题
- 解决方案：使用向量化实现替代循环累加

## 使用方法

### 基本用法

```python
import torch
from attention_worker_combine_impl import attention_worker_combine

# 准备数据
BS, K, H = 8, 2, 32
token_data = torch.randn(BS, K + 1, H, dtype=torch.bfloat16, device='npu:14')
expert_scales = torch.rand(BS, K, dtype=torch.float32, device='npu:14')

# 执行计算
y = attention_worker_combine(token_data, expert_scales, strategy="split_bs")
```

### 环境要求

```bash
# 设置设备 ID
export TILE_FWK_DEVICE_ID=14

# 设置库路径
export PTO_TILE_LIB_CODE_PATH=${ASCEND_HOME_PATH:-/usr/local/Ascend/cann}/aarch64-linux
```

## 测试结果

```
Strategy     BS     K      H      Result     Max Diff    
----------------------------------------------------------------------
split_bs     8      2      32     PASS       0.00000000  
split_h      8      2      32     PASS       0.00000000  
split_k      8      2      32     PASS       0.00000000  
----------------------------------------------------------------------
Total: 3/3 passed
```

## 文件结构

```
custom/attention_worker_combine/
├── needs_analysis.md                      # 需求分析
├── attention_worker_combine_golden.py     # Golden 实现
├── attention_worker_combine_kernel.py     # PyPTO kernel 实现
├── attention_worker_combine_impl.py       # Wrapper 函数
├── test_attention_worker_combine.py       # 测试脚本
└── README.md                              # 本文档
```

## 注意事项

1. **H 必须是 16 的倍数**：BF16 类型需要 32 bytes 对齐，即 16 元素
2. **只支持 BF16**：当前实现仅支持 bfloat16 数据类型
3. **向量化实现**：为避免精度问题，所有策略均使用向量化实现

## 开发历程

1. **Golden 开发**：完成 PyTorch 参考实现并验证通过
2. **SplitBS/SplitH 开发**：向量化实现，一次性通过
3. **SplitK 调试**：发现循环累加器问题，使用二分法定位后改用向量化实现

## 相关文档

- [需求分析](./needs_analysis.md)
- [PyPTO API 文档](../../docs/api/)
- [二分查找调试技能](../../.agents/skills/pypto-binary-search-verify/)