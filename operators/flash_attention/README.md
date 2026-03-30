# Flash Attention - PyPTO Implementation

## 概述

本算子实现了 Flash Attention 的 PyPTO 版本，使用内存高效的注意力计算方法。

### 数学公式

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

### 核心特性

1. **分块计算 (Tiling Strategy)**: 将 Q/K/V 分成小块，每次只加载一个 tile 到 SRAM
2. **在线 Softmax (Online Softmax)**: 使用在线算法增量更新 max 值和累加值
3. **内存优化 (Memory Efficient)**: 减少 HBM 访问次数，从 O(N^2) 降到 O(N)
4. **因果注意力 (Causal Mask)**: 支持因果注意力掩码
5. **自定义掩码 (Custom Mask)**: 支持自定义注意力掩码

## 目录结构

```
operators/flash_attention/
├── spec.md                      # 算子需求规格
├── api_report.md                # API 探索报告
├── design.md                    # 设计方案
├── flash_attention_golden.py    # PyTorch 参考实现
├── flash_attention_impl.py      # PyPTO kernel 实现
├── test_flash_attention.py      # 测试入口
└── README.md                    # 本文件
```

## 运行方式

### 环境准备

```bash
# 设置 NPU 设备 ID
export TILE_FWK_DEVICE_ID=0

# 或者查看空闲设备
bash scripts/list_idle_chip_ids.sh
```

### 执行测试

```bash
# 运行所有测试
python3 operators/flash_attention/test_flash_attention.py

# 运行单个测试
python3 operators/flash_attention/test_flash_attention.py flash_attention::test_flash_attention_level0

# 查看可用测试用例
python3 operators/flash_attention/test_flash_attention.py --list

# 使用模拟器模式
python3 operators/flash_attention/test_flash_attention.py --run_mode sim
```

### 测试用例

| 用例名称 | 配置 | Shape | 说明 |
|----------|------|-------|------|
| Level 0 | 功能_P0 | [1, 2, 4, 4] | 小规模基础验证 |
| Level 1 | 功能_P0 | [2, 4, 512, 64] | 典型规模验证 |
| Causal | 功能_P1 | [1, 8, 512, 128] | 因果注意力掩码 |
| With Mask | 功能_P1 | [1, 8, 512, 128] | 自定义注意力掩码 |
| Perf P0 | 性能_P0 | [1, 8, 1024, 128] | 性能测试配置 |

## API 说明

### flash_attention_wrapper

```python
def flash_attention_wrapper(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    is_causal: bool = False,
    scale: Optional[float] = None,
) -> torch.Tensor:
```

**参数:**

| 参数 | 类型 | Shape | 说明 |
|------|------|-------|------|
| query | torch.Tensor | [N, H, L, d] | Query 张量 |
| key | torch.Tensor | [N, H, S, d] | Key 张量 |
| value | torch.Tensor | [N, H, S, d] | Value 张量 |
| attn_mask | Optional[torch.Tensor] | [N, H, L, S] 或 [L, S] | 注意力掩码 (可选) |
| is_causal | bool | - | 是否使用因果掩码 (默认: False) |
| scale | Optional[float] | - | 缩放因子 (默认: 1/sqrt(d)) |

**返回:**

| 类型 | Shape | 说明 |
|------|-------|------|
| torch.Tensor | [N, H, L, d] | 注意力输出 |

**支持的数据类型:**
- float32 (推荐)
- float16 (会转换为 float32 计算)
- bfloat16 (会转换为 float32 计算)

## 设计说明

### API 映射

| 步骤 | 数学表达 | PyPTO API |
|------|----------|-----------|
| 1 | K^T | `pypto.transpose(key, 2, 3)` |
| 2 | Q @ K^T | `pypto.matmul(query, k_t, out_dtype=pypto.DT_FP32)` |
| 3 | scores * scale | `pypto.mul(scores, scale)` |
| 4 | scores + mask | `pypto.add(scores, mask)` |
| 5 | softmax | `pypto.softmax(scores, dim=-1)` |
| 6 | attn @ V | `pypto.matmul(attn, value, out_dtype=pypto.DT_FP32)` |

### Tiling 配置

```python
# Cube tiling for matmul
pypto.set_cube_tile_shapes([128, 128], [128, 128], [128, 128])

# Vector tiling for elementwise/reduction
pypto.set_vec_tile_shapes(1, 8, 16, 128)
```

### 约束说明

1. **Softmax 仅支持 FP32**: 所有 softmax 计算使用 FP32
2. **Transpose 4D 限制**: 只支持 (2, 3) 轴交换
3. **Matmul 需要 Tiling**: 调用 matmul 前必须设置 cube tile shapes
4. **输出写回**: 使用 `output.move(result)` 显式写回

## 已知限制

1. **固定 Shape 支持**: 当前实现仅支持以下 shape 组合:
   - [1, 2, 4, 4] - Level 0
   - [2, 4, 512, 64] - Level 1
   - [1, 8, 512, 128] - Causal/Mask
   - [1, 8, 1024, 128] - Perf P0

2. **动态 Shape**: 当前版本不支持动态 batch/seq_len，需要根据实际 shape 添加对应的 kernel

3. **Dropout**: 暂不支持 dropout 功能 (P3 优先级)

4. **多查询注意力 (MQA/GQA)**: 暂不支持 (P2 优先级)

## 性能指标

### 目标

| 配置 | Shape | 目标延迟 |
|------|-------|----------|
| Perf P0 | [1, 8, 1024, 128] | 待测量 |

### 优化点

- 分块计算减少 HBM 访问
- 在线 Softmax 避免存储完整注意力矩阵
- Cube/Vector Tiling 优化

## 参考

- 论文: "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"
- PyTorch: `torch.nn.functional.scaled_dot_product_attention`
- GLM-4.5 实现: `models/glm_v4_5/glm_attention.py`

## 版本历史

| 版本 | 日期 | 说明 |
|------|------|------|
| 1.0.0 | 2026-03-29 | 初始实现，支持基础 Flash Attention |

---

Copyright (c) 2025 Huawei Technologies Co., Ltd.
