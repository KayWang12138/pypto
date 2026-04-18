# GroupedMatmulFinalizeRoutingV3 算子

## 算子简介

GroupedMatmulFinalizeRoutingV3 是一个 MoE（Mixture of Experts）场景下的融合算子，将原本需要多个独立算子的操作合并为一个高效算子：

1. **分组矩阵乘法（GMM）**: 对每个专家组执行 MXFP8/MXFP4 量化矩阵乘法
2. **路由分配（Routing）**: 按 rowIndex 将专家输出 scatter add 到对应位置
3. **共享专家融合**: 将共享专家输出与 MoE 专家结果加权融合

该算子适用于：
- MoE 模型推理（GLM、LLaMA 等）
- 大规模专家模型的路由计算
- MXFP8/MXFP4 量化推理场景

## 数学公式

### 分组矩阵乘法（GMM）
$$
y_i = (x_i \times weight_i) \times scale_i \times perTokenScale_i
$$

### 路由分配
$$
out[rowIndex[j], :] \mathrel{+}= intermediate[j, :]
$$

### 共享专家融合
$$
out[row,:] \mathrel{+}= sharedInputWeight \times sharedInput[j,:]
$$

### 最终输出
$$
y[rowIndex[i], :] = \sum_{i \in \mathcal{E}[j]} y_i [j - start_i] + sharedInputWeight \times sharedInput[j, :]
$$

## 目录结构

```
custom/GroupedMatmulFinalizeRoutingV3/
├── SPEC.md                                    # 需求规范文档
├── API_REPORT.md                              # API 探索报告
├── DESIGN.md                                  # 设计文档
├── grouped_matmul_finalize_routing_v3_golden.py  # Golden 参考实现
├── grouped_matmul_finalize_routing_v3_impl.py    # PyPTO 算子实现
├── test_grouped_matmul_finalize_routing_v3.py    # 测试代码
└── README.md                                  # 本文件
```

## 使用方法

### 环境准备

```bash
# 设置 NPU 设备 ID
export TILE_FWK_DEVICE_ID=7

# 验证环境
npu-smi info
```

### 运行测试

```bash
# 进入算子目录
cd custom/GroupedMatmulFinalizeRoutingV3

# 运行所有测试
python test_grouped_matmul_finalize_routing_v3.py

# 运行单个测试
python test_grouped_matmul_finalize_routing_v3.py GroupedMatmulFinalizeRoutingV3::test_mxfp8_basic

# 列出所有测试用例
python test_grouped_matmul_finalize_routing_v3.py --list
```

### 调用算子

```python
from grouped_matmul_finalize_routing_v3_impl import grouped_matmul_finalize_routing_v3_wrapper

# 准备输入数据
result = grouped_matmul_finalize_routing_v3_wrapper(
    x1=x1,                     # [M, K] MXFP8
    x2=x2,                     # [E, K, N] MXFP8
    scale=scale,               # [E, Ceil(K/64), N, 2] FLOAT8_E8M0
    pertoken_scale=pertoken_scale,  # [M, Ceil(K/64), 2] FLOAT8_E8M0
    group_list=[7, 9],         # 专家分组列表
    row_index=row_index,       # [M] INT64
    logit=logit,               # [M] FLOAT32
    batch=8,                   # 输出 batch 维度
    n=7168,                    # 输出特征维度
)
```

## 测试用例

| 测试名称 | 描述 | 配置 |
|---------|------|------|
| test_mxfp8_basic | MXFP8 基础配置 | m=16, k=512, n=7168, e=2 |
| test_mxfp8_transpose | 转置权重场景 | transpose_x2=True |
| test_mxfp8_with_shared_input | 包含共享专家 | shared_input=True |
| test_mxfp8_with_bias | 包含 bias | bias=True |
| test_mxfp8_large_experts | 大专家数量 | e=8, m=128 |
| test_mxfp8_full_config | 完整配置 | 所有可选参数 |

## 精度标准

- **相对误差（rtol）**: ≤ 1e-3
- **绝对误差（atol）**: ≤ 1e-3

精度验证使用 `numpy.testing.assert_allclose`。

## 输入规格

### 必选参数（MXFP8 场景）

| 参数 | Shape | Dtype | 说明 |
|------|-------|-------|------|
| x1 | (M, K) | FLOAT8_E4M3FN/E5M2 | 输入左矩阵 |
| x2 | (E, K, N) | FLOAT8_E4M3FN/E5M2 | 权重矩阵 |
| scale | (E, Ceil(K/64), N, 2) | FLOAT8_E8M0 | 权重缩放因子 |
| pertoken_scale | (M, Ceil(K/64), 2) | FLOAT8_E8M0 | Token级缩放因子 |
| group_list | (E) | INT64 | 专家分组列表 |
| row_index | (M) | INT64 | 路由索引 |
| logit | (M) | FLOAT32 | MoE logit |

### 可选参数

| 参数 | Shape | Dtype | 说明 |
|------|-------|-------|------|
| bias | (E, N) | BFLOAT16 | 专家偏置 |
| shared_input | (bsdp, N) | BFLOAT16 | 共享专家输出 |

## 输出规格

| 参数 | Shape | Dtype | 说明 |
|------|-------|-------|------|
| out | (batch, N) | FLOAT32 | 最终输出 |

## 约束与限制

1. **K 维度对齐**: MXFP8 场景需满足 K % 64 == 0
2. **scale 形状匹配**: scale 的 shape 必须与 x2 的转置属性一致
3. **专家数量**: E ≤ 1024
4. **连续性**: 所有 tensor 必须连续
5. **产品支持**: 
   - Ascend 950PR/950DT: 支持 MXFP8/MXFP4 量化
   - Atlas A2/A3: 支持伪量化场景

## 参考实现

- 参考: `../gmm_mxfp8.py` (scaled_matmul_kernel)
- 设计文档: `DESIGN.md`
- API 文档: `docs/api/math/pypto-scaled_mm.md`

## 已知问题

- 无

---

**版本**: v1.0
**生成日期**: 2026-04-14
**产品支持**: Ascend 950PR/950DT, Atlas A2/A3