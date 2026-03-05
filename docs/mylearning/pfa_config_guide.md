# PFA 算子参数配置指南

**版本：** 1.0  
**日期：** 2026-02-28  
**适用范围：** GLM-4.5 PFA (Prompt Flash Attention) 算子优化

---

## 1. 概述

本指南提供 PFA 算子的参数配置建议，帮助开发者根据不同场景选择最优配置。

### 1.1 配置参数分类

| 参数类别 | 影响范围 | 调优难度 | 效果 |
|----------|----------|----------|------|
| Tile配置 | 计算效率、内存访问 | 中 | 高 |
| JIT配置 | 编译优化、调度 | 低 | 中高 |
| 循环配置 | 并行度、循环开销 | 低 | 中 |

---

## 2. Tile 配置指南

### 2.1 Tile 配置参数说明

```python
tile_cfg = AttentionTileConfig(
    g_tile=nq,              # Query Head 分组大小
    s2_tile=256,            # KV 序列分块大小
    c1_tile_shape=[[mL0, mL1], [kL0, kL1], [nL0, nL1]],  # QK^T 的 Tile 配置
    v1_tile_shape=[m, n],   # Vector 计算 Tile 配置
    c2_tile_shape=[[mL0, mL1], [kL0, kL1], [nL0, nL1]],  # PV 的 Tile 配置
    v2_tile_shape=[m, n]    # Vector 计算 Tile 配置
)
```

### 2.2 s2_tile 选择指南

**决策树：**

```
输入: s1 (Query序列长度)
│
├─ s1 <= 64
│   └─ 推荐 s2_tile = 128
│
├─ 64 < s1 <= 256
│   └─ 推荐 s2_tile = 256 (默认)
│
├─ 256 < s1 <= 1024
│   └─ 推荐 s2_tile = 512
│
└─ s1 > 1024
    └─ 推荐 s2_tile = 512 或 1024
```

**约束条件：**
1. s2_tile 应为 block_size 的整数倍（通常 block_size=128）
2. s2_tile * d * sizeof(dtype) ≤ L0 Buffer 大小
3. 对于 BF16，推荐范围：128-512

### 2.3 c1_tile_shape 配置指南

**推荐配置（BF16）：**

| 场景 | M维度 [L0, L1] | K维度 [L0, L1] | N维度 [L0, L1] | 说明 |
|------|----------------|----------------|----------------|------|
| 平衡配置 | [128, 128] | [64, 256] | [128, 128] | 通用场景 |
| 大K轴配置 | [128, 128] | [128, 512] | [128, 128] | K维度较大时 |
| 紧凑配置 | [128, 128] | [128, 128] | [256, 256] | 内存受限时 |

**约束条件：**
1. mL0 * nL0 * sizeof(float) ≤ L0C_SIZE (131072)
2. K轴需要32字节对齐
3. L1配置影响L2命中率

### 2.4 v1_tile_shape 配置指南

**计算公式：**
```
v1_tile = [m, s2_tile]
其中 m = g_tile (通常 = nq)
```

**约束条件：**
1. m * n * sizeof(dtype) 应在 16KB - 64KB 范围内
2. 尾轴（n）需要32字节对齐

---

## 3. JIT 配置指南

### 3.1 runtime_options 配置

```python
runtime_options = {
    "stitch_function_num_initial": 128,      # 子图切分粒度
    "stitch_function_outcast_memory": 4096,  # 外部内存 (KB)
    "stitch_function_inner_memory": 4096,    # 内部内存 (KB)
    "device_sched_mode": 1                   # 调度模式
}
```

#### 3.1.1 stitch_function 内存配置

**推荐值选择：**

| 配置场景 | outcast_memory | inner_memory | 适用情况 |
|----------|----------------|--------------|----------|
| 保守配置 | 1024 | 1024 | 内存受限环境 |
| 平衡配置 | 4096 | 4096 | 通用场景（推荐） |
| 激进配置 | 8192 | 8192 | 追求极致性能 |

#### 3.1.2 device_sched_mode 选择

| 模式 | 名称 | 适用场景 | 效果 |
|------|------|----------|------|
| 0 | 默认调度 | 通用场景 | 基线 |
| 1 | L2亲和调度 | 数据复用多 | 推荐 |
| 2 | 公平调度 | 多核均衡 | 可选 |

### 3.2 pass_options 配置

```python
pass_options = {
    "pg_upper_bound": 2048,                    # 子图大小上界
    "cube_l1_reuse_setting": {0: N},          # L1复用次数
    "cube_l1_reuse_mode": 1                    # 复用模式
}
```

#### 3.2.1 pg_upper_bound 选择

| 场景 | 推荐值 | 说明 |
|------|--------|------|
| 保守 | 1536 | 稳定编译 |
| 平衡 | 2048 | 通用场景（推荐） |
| 激进 | 3072 | 更多融合机会 |

#### 3.2.2 cube_l1_reuse_setting 计算

**动态计算公式：**
```python
def calculate_l1_reuse(s1, s2_tile):
    """
    计算最优的 cube_l1_reuse_setting
    
    对于PFA因果注意力：
    - s2_loop ≈ s1 / (2 * s2_tile)
    - 推荐设置为 avg_s2_loop 或略小
    """
    avg_s2_loop = (s1 + s2_tile - 1) // (2 * s2_tile)
    return max(2, avg_s2_loop)  # 至少复用2次
```

**常见配置：**

| s1 | s2_tile | avg_s2_loop | 推荐设置 |
|----|---------|-------------|----------|
| 128 | 128 | 0-1 | {0: 2} |
| 128 | 256 | 0 | {0: 2} |
| 256 | 256 | 0-1 | {0: 2} |
| 512 | 256 | 1 | {0: 2} |
| 1024 | 256 | 2 | {0: 2} |
| 2048 | 512 | 2 | {0: 2} |

### 3.3 完整配置示例

#### 3.3.1 稳健优化配置 (V1)

```python
def get_pfa_config_v1():
    """稳健优化配置 - 预期提升 40-50%"""
    return {
        "tile": {
            "s2_tile": 256,
            "c1_tile_shape": [[128, 128], [64, 256], [128, 128]],
            "v1_tile_shape": [128, 256],
            "c2_tile_shape": [[128, 128], [64, 256], [128, 128]],
            "v2_tile_shape": [128, 128]
        },
        "jit": {
            "runtime_options": {
                "stitch_function_num_initial": 128,
                "stitch_function_outcast_memory": 4096,
                "stitch_function_inner_memory": 4096,
                "device_sched_mode": 1
            },
            "pass_options": {
                "pg_upper_bound": 2048,
                "cube_l1_reuse_setting": {0: 2},
                "cube_l1_reuse_mode": 1
            }
        }
    }
```

#### 3.3.2 激进优化配置 (V2)

```python
def get_pfa_config_v2():
    """激进优化配置 - 预期提升 50-70%"""
    return {
        "tile": {
            "s2_tile": 256,
            "c1_tile_shape": [[128, 128], [64, 256], [128, 128]],
            "v1_tile_shape": [128, 256],
            "c2_tile_shape": [[128, 128], [64, 256], [128, 128]],
            "v2_tile_shape": [128, 128]
        },
        "jit": {
            "runtime_options": {
                "stitch_function_num_initial": 256,
                "stitch_function_outcast_memory": 8192,
                "stitch_function_inner_memory": 8192,
                "device_sched_mode": 1
            },
            "pass_options": {
                "pg_upper_bound": 3072,
                "cube_l1_reuse_setting": {0: 4},
                "cube_l1_reuse_mode": 1
            }
        },
        "loop": {
            "unroll_list": [16, 8, 4, 2, 1]
        }
    }
```

---

## 4. 循环配置指南

### 4.1 循环展开策略 (unroll_list)

**选择指南：**

| s1 范围 | 推荐 unroll_list | 说明 |
|---------|------------------|------|
| ≤ 128 | [4, 2, 1] | s2_loop 较小 |
| 128-512 | [8, 4, 2, 1] | 平衡配置（推荐） |
| ≥ 1024 | [16, 8, 4, 2, 1] | 减少循环开销 |

### 4.2 并行循环配置

**推荐做法：**
```python
# 外层循环使用 pypto.loop 启用并行
for b_idx in pypto.loop(b_scalar, name="LOOP_b", idx_name="b_idx"):
    for s1_idx in pypto.loop(s1_scalar, name="LOOP_s1", idx_name="s1_idx"):
        for n2_idx in pypto.loop(n2_sym, name="LOOP_n2", idx_name="n2_idx"):
            for g_idx in pypto.loop(g_loop, name="LOOP_g", idx_name="g_idx"):
                # 核心计算...

# 内层小循环保持 range()
for i in range(block_num):  # block_num 通常很小 (1-4)
    # 小循环，不需要并行
```

---

## 5. 场景化配置推荐

### 5.1 常见场景配置

#### 场景1：短序列 (s1 ≤ 128)

```python
config = {
    "s2_tile": 128,
    "c1_tile_shape": [[128, 128], [64, 256], [128, 128]],
    "v1_tile_shape": [128, 128],
    "cube_l1_reuse_setting": {0: 2},
    "unroll_list": [4, 2, 1]
}
```

#### 场景2：中等序列 (s1 = 256-512)

```python
config = {
    "s2_tile": 256,
    "c1_tile_shape": [[128, 128], [64, 256], [128, 128]],
    "v1_tile_shape": [128, 256],
    "cube_l1_reuse_setting": {0: 2},
    "unroll_list": [8, 4, 2, 1]
}
```

#### 场景3：长序列 (s1 ≥ 1024)

```python
config = {
    "s2_tile": 512,
    "c1_tile_shape": [[128, 128], [64, 512], [128, 128]],
    "v1_tile_shape": [128, 512],
    "cube_l1_reuse_setting": {0: 2},
    "unroll_list": [16, 8, 4, 2, 1]
}
```

### 5.2 性能目标导向配置

| 目标 | 推荐配置 | 风险 |
|------|----------|------|
| 稳定优先 | V1保守配置 | 低 |
| 性能优先 | V1激进配置 | 中 |
| 极致性能 | V2配置 | 高 |

---

## 6. 配置调优流程

### 6.1 标准调优流程

```
Step 1: 基线测试
    └─ 运行原始版本，记录性能数据
    
Step 2: Tile配置调优
    ├─ 调整 s2_tile
    ├─ 调整 c1_tile K轴
    └─ 验证性能提升

Step 3: JIT配置调优
    ├─ 调整内存配置
    ├─ 调整 L1 复用
    └─ 验证性能提升

Step 4: 循环配置调优
    ├─ 调整展开策略
    └─ 验证性能提升

Step 5: 综合验证
    ├─ 精度验证
    ├─ 性能验证
    └─ 稳定性验证
```

### 6.2 快速配置选择

**如果不确定从何开始：**

1. **使用 V1 稳健配置作为起点**
   - 预期提升：40-50%
   - 风险：低

2. **如果 V1 效果好，尝试 V2 激进配置**
   - 预期提升：50-70%
   - 风险：中高

3. **根据实际效果微调**
   - 性能未达预期 → 增大 Tile 配置
   - 出现 OOM → 减小内存配置
   - 精度问题 → 减小 Tile 配置

---

## 7. 常见问题

### Q1: s2_tile 增大后性能反而下降？

**可能原因：**
1. L0/L1 Buffer 空间不足
2. block_num 增加导致组装开销增加
3. s2_loop 过小，展开效果不佳

**解决方案：**
1. 检查 Buffer 空间使用
2. 尝试更小的 s2_tile
3. 调整循环展开策略

### Q2: cube_l1_reuse_setting 设置多少合适？

**建议：**
- 设置为 avg_s2_loop 或略小
- 最小值为 2
- 过大可能导致 L1 空间不足

### Q3: 如何判断是否需要调整调度策略？

**判断方法：**
- 观察泳道图核间负载
- 负载不均衡 → 尝试公平调度 (mode=2)
- L2 命中率低 → 尝试 L2 亲和调度 (mode=1)

---

## 8. 附录

### 8.1 配置参数速查表

| 参数 | 推荐范围 | 默认值 | 说明 |
|------|----------|--------|------|
| s2_tile | 128-512 | 256 | KV分块 |
| c1_tile K轴 L1 | 256-512 | 256 | K轴切分 |
| v1_tile | [128, 256] | [128, 256] | Vector Tile |
| stitch_function内存 | 1024-8192 | 4096 | KB |
| pg_upper_bound | 1536-3072 | 2048 | 子图大小 |
| cube_l1_reuse | 2-8 | 2 | 复用次数 |

### 8.2 相关文档

- 优化指南：`docs/mylearning/pfa_optimization_guide.md`
- 优化记录：`docs/mylearning/pfa_optimization_record.md`
- Matmul性能指南：`docs/tutorials/debug/matmul-performance-guide.md`
