# Tiling 规则

> **版本**: 1.0
> **最后更新**: 2026-03-15
> **说明**: 本文件用于"设计期可行性闸门"。每条规则必须能回链到本仓库内的 docs/ 或代码证据。
> - **HARD**：不满足会直接导致编译失败、运行错误或不被支持。
> - **HEURISTIC**：经验建议/调优起点，允许按实测调整。

---

## 1. 算子分类与 Tiling 决策入口

```
算子公式分析
    │
    ├── 仅含逐元素/向量运算 → Vector Tiling（§2）
    │
    ├── 含 matmul/bmm → Cube Tiling（§3）
    │
    └── matmul + 逐元素混合 → 混合策略（§4）
```

---

## 2. HARD 约束

### Vector Tiling

| rule_id | rule_text | level | trigger | impact | warning_code | evidence |
|---------|-----------|-------|---------|--------|--------------|----------|
| TILE_VEC_01 | set_vec_tile_shapes 参数个数不超过 4 且每维 > 0 | HARD | vec 算子配置 tiling | 编译失败 | TILE_VEC_ARG_INVALID | `docs/api/config/pypto-set_vec_tile_shapes.md`; `docs/tutorials/development/tiling.md` |
| TILE_VEC_02 | vec 尾轴满足 32B 对齐约束 | HARD | vec tiling 配置 | 运行失败/性能异常 | TILE_VEC_ALIGN_INVALID | `docs/tutorials/development/tiling.md` |
| TILE_EXPR_01 | 控制 (TensorShape/TileShape)*(1+算子输入个数) < 18000 | HARD | tile 过小导致切分过多 | 表达式表编译失败 | TILE_EXPR_TABLE_OVERFLOW | `docs/tutorials/development/tiling.md` |
| TILE_RED_01 | reduction tile 满足容量与轴限制 | HARD | sum/amax/amin/prod/var | 编译失败或结果异常 | TILE_REDUCTION_INVALID | `docs/api/operation/pypto-sum.md`; `docs/api/operation/pypto-var.md` |
| TILE_TOPK_01 | topk 的排序轴 tiling 需满足该 API 的轴/维度约束 | HARD | pypto.topk 调用 | 编译失败或结果错误 | TILE_TOPK_AXIS_INVALID | `docs/api/operation/pypto-topk.md` |
| TILE_ARGSORT_01 | argsort 的排序轴 tiling 需满足该 API 的维度约束 | HARD | pypto.argsort 调用 | 编译失败或结果错误 | TILE_ARGSORT_DIM_INVALID | `docs/api/operation/pypto-argsort.md` |

### Cube Tiling（Matmul）

| rule_id | rule_text | level | trigger | impact | warning_code | evidence |
|---------|-----------|-------|---------|--------|--------------|----------|
| TILE_CUBE_01 | matmul 前必须调用 set_cube_tile_shapes | HARD | pypto.matmul 调用 | 编译失败 | TILE_CUBE_NOT_SET | `docs/api/config/pypto-set_cube_tile_shapes.md` |
| TILE_CUBE_02 | cube tile 满足 32-byte 对齐与层级关系 | HARD | matmul 场景 | 编译/运行失败 | TILE_CUBE_ALIGN_INVALID | `docs/api/config/pypto-set_cube_tile_shapes.md` |
| TILE_CUBE_03 | L0/L1 与 buffer 预算满足上界 | HARD | matmul 场景 | 运行失败 | TILE_CUBE_BUFFER_OVERFLOW | `docs/api/config/pypto-set_cube_tile_shapes.md` |
| TILE_CUBE_04 | 满足 0 < mL0<=mL1, kL0<=kL1, nL0<=nL1 且 L1%L0==0 | HARD | cube tile 配置 | 编译失败 | TILE_CUBE_DIVISIBILITY_INVALID | `docs/api/config/pypto-set_cube_tile_shapes.md` |
| TILE_CUBE_05 | Bias/FixPipe 场景满足 nL0*4<=1KB 与 nL0*8<=2KB | HARD | bias/fixpipe 场景 | 运行失败 | TILE_CUBE_SPECIAL_BUFFER_OVERFLOW | `docs/api/config/pypto-set_cube_tile_shapes.md` |
| TILE_CUBE_06 | 输入为 3D/4D 时 enable_split_k 只能为 False | HARD | 高维 matmul 场景 | 编译或运行失败 | TILE_SPLITK_DIM_UNSUPPORTED | `docs/api/config/pypto-set_cube_tile_shapes.md` |

### enable_multi_data_load 一致性

| rule_id | rule_text | level | trigger | impact | warning_code | evidence |
|---------|-----------|-------|---------|--------|--------------|----------|
| TILE_CUBE_07 | enable_multi_data_load 需在 API 中确认可用后才使用 | HARD | 教程提及但 API 签名未显式列出 | 依赖不稳定参数 | TILE_CUBE_MDL_UNVERIFIED | `docs/tutorials/debug/performance.md`; `docs/api/config/pypto-set_cube_tile_shapes.md` |

---

## 3. HEURISTIC 建议

以下为经验建议，不是硬约束，可根据实测结果调整。

| rule_id | rule_text | level | trigger | impact | warning_code | evidence |
|---------|-----------|-------|---------|--------|--------------|----------|
| TILE_HEUR_01 | Vector 初值建议 set_vec_tile_shapes(64, 512)，数据块 16~64KB | HEURISTIC | 初次配置 vec tiling | 性能不稳定 | TILE_HEUR_VEC_INIT | `docs/tutorials/debug/performance.md` |
| TILE_HEUR_02 | Reduce 轴尽量不切分，避免额外 GM 搬运 | HEURISTIC | 包含 reduction 子图 | 性能退化 | TILE_HEUR_REDUCE_NOSPLIT | `docs/tutorials/debug/performance.md` |
| TILE_HEUR_03 | 上下游 Vector TileShape 尽量对齐，提高合图概率 | HEURISTIC | 连续 vec op 链路 | 合图失败/性能下降 | TILE_HEUR_VEC_ALIGN | `docs/tutorials/debug/performance.md` |
| TILE_HEUR_04 | Cube 初值（FP16/BF16）可选 [128,128],[64,256],[256,256] 等组合 | HEURISTIC | 初次配置 cube tiling | 性能不稳定 | TILE_HEUR_CUBE_INIT | `docs/tutorials/debug/performance.md` |
| TILE_HEUR_05 | 避免 tile 过小导致循环开销过高 | HEURISTIC | tile 远小于 tensor 尺寸 | 性能退化 | TILE_HEUR_SMALL_TILE | `docs/tutorials/debug/performance.md` |

---

## 4. 混合算子（Cube + Vector）

当算子同时包含 matmul（Cube）和逐元素/归约（Vector）运算时，按以下流程设计 tiling：

**执行顺序**：

1. **识别主算力阶段**（通常为 matmul）。
2. **对主阶段配置 cube tile** 并验证 §2 中所有 TILE_CUBE_* 硬约束。
3. **对尾部逐元素/归约阶段配置 vec tile** 并验证 TILE_VEC_* 硬约束。
4. **校验跨阶段 shape/dtype/format 连贯性**。
5. 任一步失败触发：`TILE_MIXED_PIPELINE_INVALID`。

**注意事项**：

| rule_id | rule_text | level | trigger | impact | warning_code | evidence |
|---------|-----------|-------|---------|--------|--------------|----------|
| TILE_MIX_01 | 当前暂不支持将 Matmul 与 Vector 自动合图 | HEURISTIC | matmul + elementwise 混合 | 不可依赖跨 Cube/Vector 自动融合 | TILE_MIX_NO_AUTOFUSE | `docs/tutorials/debug/performance.md` |
| TILE_MIX_02 | 降低 Cube 与 Vector 子图间的多对多依赖 | HEURISTIC | 混合算子 | 并行与合图受阻 | TILE_MIX_DEPENDENCY_COMPLEX | `docs/tutorials/debug/performance.md` |

---

## 5. 非可视化校验清单

在设计完成后，按以下清单逐项检查 tiling 配置的合法性：

| 检查项 | 方法 | 判定 |
|--------|------|------|
| 对齐检查 | 尾轴 TileShape * dtype_bytes % 32 == 0 | PASS/FAIL |
| Buffer 预算 | TileShape 对应数据块 < UB 容量（如 192KB） | PASS/FAIL |
| 轴切分合法性 | 切分后各维 > 0 且不违反 reduction 约束 | PASS/FAIL |
| 表达式表容量 | (TensorShape/TileShape)*(1+输入数) < 18000 | PASS/FAIL |
| Cube 层级约束 | L0 <= L1 且 L1 % L0 == 0 | PASS/FAIL |
| 编译日志签名 | 编译后检查是否命中 tiling 相关错误 | 命中/未命中 |

---

## 6. 失败签名与定位

常见 tiling 相关编译/运行失败的识别和定位：

| 失败特征 | 可能原因 | 关联规则 | 排查方向 |
|----------|----------|----------|----------|
| 编译报表达式表溢出 | tile 过小导致切分数超上限 | TILE_EXPR_01 | 增大 tile，减少切分数 |
| 编译报 tile shape 非法 | 对齐/维度/范围不满足 | TILE_VEC_01/02, TILE_CUBE_02/04 | 检查对齐和层级约束 |
| 运行报 buffer 溢出 | tile 过大超 UB/L1 容量 | TILE_CUBE_03/05 | 减小 tile |
| split-k 相关失败 | 高维输入开启 split-k | TILE_CUBE_06 | 3D/4D 输入关闭 split-k |
| 性能远低于预期 | tile 过小/过大、reduce 轴被切 | TILE_HEUR_* | 调整 tile 大小 |

---

## 7. 证据索引

| 证据文件 | 内容 |
|----------|------|
| `docs/tutorials/development/tiling.md` | Tiling 基本原理与通用约束 |
| `docs/tutorials/debug/performance.md` | 开箱性能与 TileShape 初值建议 |
| `docs/api/config/pypto-set_vec_tile_shapes.md` | Vector TileShape API 约束 |
| `docs/api/config/pypto-set_cube_tile_shapes.md` | Cube TileShape API 硬约束清单 |
