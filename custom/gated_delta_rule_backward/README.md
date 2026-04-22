# gated_delta_rule_backward

## 概述

Gated Delta Rule 反向传播算子的 PyPTO 实现。该算子是 Gated Delta Rule 前向算子的精确反向，用于 Qwen3-Next 模型的注意力层反向传播。

## 数学公式

反向传播核心逻辑：
1. **局部注意力梯度**: `dv0 = (A_local^T @ do) * scale`
2. **状态递推反向传播**（逆序 chunk 循环）: `d_s = d_s * exp(g_last) + q_eff^T @ do * scale - w^T @ dv_total`
3. **q/k/g 梯度**: 局部注意力贡献 + 状态贡献 + WY 低秩表示分解
4. **L2 norm 反向**: 对 dq/dk 施加 L2 归一化的反向传播

## 目录结构

```
custom/gated_delta_rule_backward/
├── SPEC.md                                    # 需求规格文档
├── DESIGN.md                                  # 设计方案文档
├── API_REPORT.md                              # API 可行性报告
├── gated_delta_rule_backward_golden.py        # PyTorch golden 参考实现
├── gated_delta_rule_backward_impl.py          # PyPTO kernel 实现
├── test_gated_delta_rule_backward.py          # 精度验证测试
├── README.md                                  # 本文档
└── debug_log.md                               # 调试日志（运行后生成）
```

## 输入输出规格

### 输入

| 变量 | Shape | Dtype | 说明 |
|------|-------|-------|------|
| `q` | `[B, T, H, K]` | FP32 | 原始 query |
| `k` | `[B, T, H, K]` | FP32 | 原始 key |
| `v` | `[B, T, H, V]` | FP32 | value |
| `g_raw` | `[B, T, H]` | FP32 | 门控原始值 |
| `beta` | `[B, T, H]` | FP32 | beta 参数 |
| `initial_state` | `[B, H, K, V]` | FP32 | 初始隐状态 |
| `do` | `[B, T, H, V]` | FP32 | 输出梯度 |
| `dht` | `[B, H, K, V]` | FP32 | 终态梯度 |
| `bt` | int | - | 分块大小 |

### 输出

| 变量 | Shape | Dtype | 说明 |
|------|-------|-------|------|
| `dq` | `[B, T, H, K]` | FP32 | dL/d(q) |
| `dk` | `[B, T, H, K]` | FP32 | dL/d(k) |
| `dv` | `[B, T, H, V]` | FP32 | dL/d(v) |
| `db` | `[B, T, H]` | FP32 | dL/d(beta) |
| `dg_raw` | `[B, T, H]` | FP32 | dL/d(g_raw) |
| `dh0` | `[B, H, K, V]` | FP32 | dL/d(initial_state) |

### 约束

- T 必须能被 BT 整除: `T % BT == 0`
- 动态轴: B, T
- 编译期参数: H, K, V, BT
- 精度容忍度: `rtol=1e-3, atol=1e-3`

## 实现要点

1. **函数工厂模式**: K, V, H, BT 作为编译期参数通过闭包传入 kernel
2. **三层嵌套 loop**: `B → H → chunk`（逆序）
3. **状态梯度 d_s**: 跨 chunk 循环迭代携带，使用 `pypto.tensor` + `[:]` 写回
4. **全链路 FP32**: 无 dtype 转换
5. **常量矩阵**: i_mat, m_le, m_lt, c_cum, c_rcum 由 host 端预构造

## 运行方式

### 环境准备

```bash
# 设置 NPU 设备 ID（查看可用芯片）
export TILE_FWK_DEVICE_ID=$(bash .agents/skills/pypto-op-develop/scripts/list_idle_chip_ids.sh | awk '{print $1}')
```

### 执行测试

```bash
cd custom/gated_delta_rule_backward

# 运行默认测试 (level0)
python test_gated_delta_rule_backward.py

# 运行指定测试
python test_gated_delta_rule_backward.py gated_delta_rule_backward::test_gated_delta_rule_backward_level0
python test_gated_delta_rule_backward.py gated_delta_rule_backward::test_gated_delta_rule_backward_level1

# 列出所有用例
python test_gated_delta_rule_backward.py --list
```

## 验证入口

- 精度对比: `numpy.testing.assert_allclose(impl, golden, rtol=1e-3, atol=1e-3)`
- 通过标记: `[PRECISION_PASS]`
- 失败标记: `[PRECISION_FAIL]`

## 已知限制

1. 当前仅支持 FP32 数据类型
2. T 必须能被 BT 整除（非对齐场景需使用 fillpad 处理，暂未实现）
3. kernel 编译时间较长（大量 matmul 操作），首次运行需耐心等待
4. dg_cum 末元素修正使用 view 写回模式，依赖 PyPTO view 的正确行为
