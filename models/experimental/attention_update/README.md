# AttentionUpdate PyPTO 算子

## 算子概述

**功能**: 在序列并行（Sequence Parallelism, SP）场景下，将各 SP 域的 PA 算子输出的中间结果 lse 和 localOut 合并更新为全局结果。

**数学公式**:

$$
\begin{aligned}
lse_{max} &= \max_i(lse_i) \\
lse_{sum} &= \sum_i \exp(lse_i - lse_{max}) \\
lse_m &= lse_{max} + \log(lse_{sum}) \\
O &= \sum_i O_i \cdot \exp(lse_i - lse_m)
\end{aligned}
$$

## 目录结构

```
attention_update/
├── attention_update.py           # PyPTO 实现 + Golden + 测试（合并文件）
├── attention_update_golden.py    # PyTorch Golden 参考实现（独立文件）
├── spec.md                       # 算子规格说明
├── needs_analysis.md             # 需求分析文档
└── README.md                     # 本文档
```

## 编译运行

### 环境准备

```bash
# 设置 NPU 设备 ID
export TILE_FWK_DEVICE_ID=14
export PTO_TILE_LIB_CODE_PATH=${ASCEND_HOME_PATH}/aarch64-linux
```

### 编译 PyPTO

```bash
cd /data/x00952168/pypto_agent/pypto
python3 build_ci.py -f python3 --disable_auto_execute
pip install --force-reinstall build_out/pypto-*.whl
```

### 运行测试

```bash
cd custom/attention_update
python3 attention_update.py                    # 运行所有测试
python3 attention_update.py --list             # 列出测试用例
python3 attention_update.py {case_id}          # 运行指定用例
```

## 测试结果

| 测试级别 | 配置 | 结果 |
|----------|------|------|
| Level 0 | sp=2, BSN=4, D=8 | ✓ 通过 |
| Level 1 | sp=4, BSN=128, D=64 | ✓ 通过 |
| Level 2 | sp=8, BSN=16, D=32 | ✓ 通过 |

## 输入输出规格

### 输入

| 参数 | 类型 | Shape | Dtype | 说明 |
|------|------|-------|-------|------|
| lse_list | List[Tensor] | [[BSN]] * sp | float32 | 各 SP 域的局部 lse |
| local_out_list | List[Tensor] | [[BSN, D]] * sp | float32/fp16/bf16 | 各 SP 域的局部输出 |
| update_type | int | 标量 | int | 0=不输出lseOut，1=输出lseOut |
| sp | int | 标量 | int | 序列并行度（支持 2, 4, 8） |

### 输出

| 参数 | 类型 | Shape | Dtype | 说明 |
|------|------|-------|-------|------|
| out | Tensor | [BSN, D] | 与输入一致 | 全局 attention 输出 |
| lse_out | Tensor | [BSN] | float32 | 全局 lse（可选） |

## 实现要点

1. **针对不同 sp 优化**: 为 sp=2, 4, 8 分别实现专用 kernel，提升性能
2. **数值稳定性**: 使用 max trick 避免数值溢出
3. **广播操作**: 使用 `unsqueeze` 将 lse 广播到 headDim 维度

## 已知限制

1. 当前仅支持 sp = 2, 4, 8
2. 输入 tensor 需要为连续内存

## 参考文档

- AscendC 原始实现: `ops-transformer/attention/attention_update/`
- API 文档: `docs/aclnnAttentionUpdate.md`