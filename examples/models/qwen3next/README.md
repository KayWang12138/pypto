## Overview

This directory contains PyPTO implementations of key components from the Qwen3-Next model architecture, including:

- **Gated Attention**: 引入门控的注意力实现
- **Gated Delta Net**: 引入门控，实现选择性遗忘和灵活记忆管理
- **Set KV Buffer**: 据映射关系存储k，v
- **Attention Pre**: Pre-attention computation
- **RMSNorm**: Residual connection with RMS normalization

## Files

### `test_gated_attention_prefill.py`
实现推理预填充阶段的TND内存格式输入下的，门控注意力及出口线性层映射的融合算子操作

**Features**:
- Grouped-Query Attention
- TND Memory Format Input
- Gated Attention and Output Linear Layer

### `test_gated_attention_decode.py`
实现推理解码阶段的TND内存格式输入下的，门控注意力及出口线性层映射的融合算子操作

**Features**:
- Grouped-Query Attention
- TND Memory Format Input
- Paged Attention
- Gated Attention and Output Linear Layer


### `test_gated_delta_net_prefill.py`
实现Qwen3-Next网络中Gated_Delta_Net的prefill部分

**Features**:
- 通过hidden state生成query, key, value, z, b, a
- 实现针对query, key和value的conv1d操作
- 实现query和key的l2norm操作
- 实现chunk_gated_delta_rule函数功能
- 实现RMS Norm和门控
- 实现结果由attention_output生成hidden_state
- 支持动态Sequence长度输入\

### `test_gated_delta_net_decode.py`
实现Qwen3-Next网络中Gated_Delta_Net的decode部分

**Features**:
- 通过hidden state生成query, key, value, z, b, a
- 实现针对query, key和value的conv1d操作
- 实现query和key的l2norm操作
- 实现recurrent_gated_delta_rule函数功能
- 实现RMS Norm和门控
- 实现结果由attention_output生成hidden_state
- 针对s=1场景设计了算法优化

### `test_attention_pre.py`
实现了pypto进行注意力预处理的过程

**Features**:
- 根据输入的x和模型中的linear层权重, 计算得到Q,K,V,Gate
- 对Q和K进行RMSNorm归一化
- 对Q和K进行旋转位置编码
- 对Gate进行sigmoid激活函数处理


### `test_zero_centered_rmsnorm.py`
实现动态维度的参数约束在零中心的RMS归一化

**Features**:
- 残差连接
- RMS 归一化
- 参数约束在零中心
- 动态批次大小支持

### `test_set_kv_buffer.py`
实现根据映射关系将attention_pre之后的k，v存储到k_buffer,v_buffer




