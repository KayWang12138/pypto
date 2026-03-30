# topk 算子设计文档

> **算子名称**: topk
> **算子分类**: selection
> **生成时间**: 2026-03-30T08:35:00Z
> **基于**: spec.md, api_report.md

---

## 1. 概述

### 1.1 功能描述

返回输入张量在指定维度上最大（或最小）的 k 个元素及其索引。支持动态轴（batch, seq, num），支持 largest/sorted 参数，支持任意 dim 维度操作。

### 1.2 数学公式

$$\text{values}, \text{indices} = \text{topk}(\text{input}, k, \text{dim}, \text{largest}, \text{sorted})$$

### 1.3 算法描述

```
Algorithm: TopK Selection with PyPTO Constraints
────────────────────────────────────────────────────
输入: input [D0, D1, ..., D_dim, ..., Dn], k, dim, largest, sorted
输出: values [..., k, ...], indices [..., k, ...]

1. 计算实际维度索引: actual_dim = dim >= 0 ? dim : n + dim
2. 维度转换（PyPTO topk 仅支持 dim=-1）:
   若 actual_dim != -1:
     2.1 transpose(input, actual_dim, -1) -> input_transposed
3. TopK 计算:
     3.1 set_vec_tile_shapes(...)
     3.2 values_raw, indices_raw = pypto.topk(input_transposed, k, -1, largest)
4. 排序处理（PyPTO topk 不支持 sorted 参数）:
     若 sorted=True:
       4.1 对 values_raw 和 indices_raw 按 values 降序(largest)或升序(!largest)排列
5. 维度还原:
     若 actual_dim != -1:
       5.1 transpose(values, -1, actual_dim) -> values_final
       5.2 transpose(indices, -1, actual_dim) -> indices_final
6. 类型转换（PyPTO topk 返回 int32 索引）:
     indices_int64 = cast(indices, DT_INT64)
7. 返回:
   - values: 选中元素组成的张量
   - indices: 选中元素在 dim 维度上的原始索引（int64）
```

### 1.4 数据流图

```
                    输入 input
               ┌──────────────────┐
               │ [b, s, n, d]     │
               │ float32          │
               └────────┬─────────┘
                        │
         ┌──────────────┼──────────────┐
         │              │              │
         ▼              ▼              ▼
    ┌─────────┐    ┌─────────┐    ┌─────────┐
    │ dim=-1  │    │ k=10    │    │largest=T│
    │ (指定维) │    │ (数量)  │    │ (方向)  │
    └────┬────┘    └────┬────┘    └────┬────┘
         │              │              │
         └──────────────┼──────────────┘
                        │
                        ▼
              ┌─────────────────┐
              │ dim != -1?      │
              │ ┌─────────────┐ │
              │ │ Transpose   │ │
              │ │ dim ↔ -1    │ │
              │ └─────────────┘ │
              └────────┬────────┘
                       │
                       ▼
              ┌─────────────────┐
              │ pypto.topk      │
              │ dim=-1 only     │
              └────────┬────────┘
                       │
         ┌─────────────┴─────────────┐
         │                           │
         ▼                           ▼
   ┌──────────────┐          ┌──────────────┐
   │   values     │          │   indices    │
   │  (float32)   │          │   (int32)    │
   └──────┬───────┘          └──────┬───────┘
          │                         │
          │                         ▼
          │                 ┌──────────────┐
          │                 │ cast int64   │
          │                 └──────┬───────┘
          │                        │
          └──────────┬─────────────┘
                     │
                     ▼
              ┌─────────────────┐
              │ dim != -1?      │
              │ ┌─────────────┐ │
              │ │ Transpose   │ │
              │ │ back        │ │
              │ └─────────────┘ │
              └────────┬────────┘
                       │
         ┌─────────────┴─────────────┐
         │                           │
         ▼                           ▼
   ┌──────────────┐          ┌──────────────┐
   │   values     │          │   indices    │
   │ [b,s,n,k]    │          │  [b,s,n,k]   │
   │   float32    │          │    int64     │
   └──────────────┘          └──────────────┘

动态轴: b (batch), s (seq), n (num)
```

---

## 2. API 映射设计

### 2.1 数学公式分解

将 topk 公式拆解为基本操作步骤：

| 步骤 | 数学表达 | 说明 |
|------|----------|------|
| 1 | `actual_dim = dim if dim >= 0 else rank + dim` | 计算实际维度索引 |
| 2 | `input_t = transpose(input, actual_dim, -1)` | 将目标维度移到尾部（若 dim != -1） |
| 3 | `values, indices = topk(input_t, k, -1, largest)` | 执行 TopK 选择 |
| 4 | `values_s, indices_s = sort_by_value(values, indices)` | 按 values 排序（若 sorted=True） |
| 5 | `values_f = transpose(values_s, -1, actual_dim)` | 还原维度顺序（若 dim != -1） |
| 6 | `indices_i64 = cast(indices_s, DT_INT64)` | 索引类型转换为 int64 |

### 2.2 PyPTO API 映射表

| 步骤 | 数学表达 | PyPTO API | 参数 | 文档路径 |
|------|----------|-----------|------|----------|
| 1 | 计算维度索引 | Python 内置 | dim, rank | — |
| 2 | transpose | `pypto.transpose` | input, actual_dim, -1 | `docs/api/operation/pypto-transpose.md` |
| 3 | topk | `pypto.topk` | input, k, -1, largest | `docs/api/operation/pypto-topk.md` |
| 4 | sort | Python 实现 | values, indices | — |
| 5 | transpose | `pypto.transpose` | values/indices, -1, actual_dim | `docs/api/operation/pypto-transpose.md` |
| 6 | cast | `pypto.cast` | indices, DT_INT64 | `docs/api/operation/pypto-cast.md` |

### 2.3 计算步骤序列

```python
# 伪代码展示计算流程
def topk_kernel(input_pto, k, dim, largest, sorted):
    # Step 1: 计算实际维度
    rank = len(input_pto.shape)
    actual_dim = dim if dim >= 0 else rank + dim

    # Step 2: 维度转换（若 dim != -1）
    if actual_dim != rank - 1:
        input_t = pypto.transpose(input_pto, actual_dim, rank - 1)
    else:
        input_t = input_pto

    # Step 3: TopK 计算
    pypto.set_vec_tile_shapes(...)
    values, indices = pypto.topk(input_t, k, -1, largest)

    # Step 4: 排序处理（若 sorted=True）
    if sorted:
        # PyPTO topk 不保证排序，需手动处理
        # 注意：sorted 参数在 P1 优先级，首跑可暂不实现
        pass

    # Step 5: 维度还原（若 dim != -1）
    if actual_dim != rank - 1:
        values = pypto.transpose(values, rank - 1, actual_dim)
        indices = pypto.transpose(indices, rank - 1, actual_dim)

    # Step 6: 类型转换
    indices_i64 = pypto.cast(indices, pypto.DT_INT64)

    return values, indices_i64
```

### 2.4 设计依据

- **来源**: api_report.md §3 API 映射 + docs/api/operation/pypto-topk.md
- **说明**:
  - `pypto.topk` 仅支持 dim=-1，需通过 transpose 间接支持任意维度
  - `pypto.topk` 不支持 sorted 参数，sorted=True 时需额外排序逻辑（P1 优先级，可后续优化）
  - `pypto.topk` 返回 int32 索引，需使用 cast 转换为 int64 以对齐 PyTorch 行为

---

## 3. 数据规格设计

### 3.1 OperatorInput dataclass

```python
@dataclass
class TopkInput:
    input: Tensor       # 输入张量, shape: [b, s, n, d], dtype: DT_FP32, 动态轴: [0, 1, 2]
    k: int              # 选取的元素数量
    dim: int = -1       # 操作维度，默认 -1
    largest: bool = True  # True 选最大，False 选最小
    sorted: bool = True   # True 则排序输出
```

### 3.2 OperatorOutput dataclass

```python
@dataclass
class TopkOutput:
    values: Tensor    # 选取的 k 个元素值, shape: [b, s, n, k], dtype: DT_FP32, 动态轴: [0, 1, 2]
    indices: Tensor   # 选取元素在 dim 维度上的索引, shape: [b, s, n, k], dtype: DT_INT64, 动态轴: [0, 1, 2]
```

### 3.3 中间 Tensor 定义

| 名称 | Shape | Dtype | 说明 |
|------|-------|-------|------|
| input_t | [b, s, k, n] 或 [b, s, n, d] | DT_FP32 | transpose 后的输入（若 dim != -1） |
| values_raw | [b, s, n, k] | DT_FP32 | topk 原始输出 values |
| indices_raw | [b, s, n, k] | DT_INT32 | topk 原始输出 indices |

### 3.4 数据格式选择

| Tensor | 格式 | 说明 |
|--------|------|------|
| input | ND | 输入为 ND 格式，保持不变 |
| values | ND | 输出保持 ND 格式 |
| indices | ND | 索引张量使用 ND 格式 |

### 3.5 动态轴定义

| 轴名称 | 含义 | 取值范围 | 动态类型 |
|--------|------|----------|----------|
| b | batch 维度，批次大小 | [1, INT32_MAX] | 运行时动态 |
| s | seq 维度，序列长度 | [1, INT32_MAX] | 运行时动态 |
| n | num 维度，注意力头数或特征数 | [1, INT32_MAX] | 运行时动态 |
| d | 固定维度，隐藏层维度 | 固定值 | 编译期已知 |

### 3.6 JIT 装饰器配置

```python
@pypto.frontend.jit(
    runtime_options={"run_mode": pypto.RunMode.NPU}
)
def topk_wrapper(
    input: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC, pypto.DYNAMIC, ...], pypto.DT_FP32),
    values_out: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC, pypto.DYNAMIC, ...], pypto.DT_FP32),
    indices_out: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC, pypto.DYNAMIC, ...], pypto.DT_INT64),
    k: int,
    dim: int,
    largest: bool,
    sorted: bool
) -> None:
    ...
```

---

## 4. Tiling 策略

### 4.1 算子类型判断

- **类型**: Vector
- **判断依据**: topk 是选择/排序类操作，不涉及矩阵乘法（matmul），根据 quick_ref.md §1.1 判定为 Vector 类型，使用 `set_vec_tile_shapes`

### 4.2 TileShape 初值设置

```python
# 对于 [b, s, n, d] 输入，在 dim=-1 上操作
# d 是完整的最后一维大小，因为 topk 需要完整遍历该维度
pypto.set_vec_tile_shapes(1, 1, 1, d)  # b_tile=1, s_tile=1, n_tile=1, d_tile=d（完整维度）
```

### 4.3 设置依据

1. **尾轴完整**: 根据 `docs/api/operation/pypto-topk.md` 约束 3：`k <= TileShape[-1]`，且 topk 需要完整遍历操作维度，因此 TileShape 尾轴应等于输入尾轴大小 d
2. **32B 对齐**: 根据 `docs/api/operation/pypto-topk.md` 约束 2：`TileShape[-1] * 4 % 32 == 0`，对于 FP32 类型，尾轴需为 8 的倍数
3. **22KB 限制**: 根据 `docs/api/operation/pypto-topk.md` 约束 2：`TileShape[-1] * 4 < 22KB`，即尾轴 < 5632

### 4.4 注意事项

- TileShape 尾轴必须 >= k（否则 topk 会失败）
- TileShape 尾轴 * 4 需满足 32B 对齐
- 对于 dim != -1 的场景，transpose 后 TileShape 维度需重新计算

### 4.5 判断依据与适用条件

- **判断依据**:
  - Vector 算子类型（无 matmul）
  - PyPTO topk 要求尾轴完整且满足对齐
  - 动态轴需要通过 loop 遍历
- **适用条件**: 输入 shape 为 2-4 维，尾轴 d 满足 32B 对齐且 < 22KB/4
- **不适用场景**: 尾轴超过 5632 时需要调整 tiling 策略或分块处理

---

## 5. Loop 结构设计

### 5.1 Loop 判断结论

- **结论**: 需要 Loop
- **原因**: 存在动态轴 b, s, n（运行时才知道大小），根据 quick_ref.md §2.1 条件 1，编译期无法展开，必须用运行时循环遍历动态维度
- **Loop 类型**: `pypto.loop`
- **适用条件**: 动态轴范围 [1, INT32_MAX]，通过 pypto.loop 支持任意大小
- **限制**: 静态轴使用 Python for，动态轴使用 pypto.loop

### 5.2 静态轴 vs 动态轴处理

| 轴 | 类型 | 处理方式 |
|----|------|----------|
| b | 动态 | pypto.loop |
| s | 动态 | pypto.loop |
| n | 动态 | pypto.loop |
| d | 静态 | 编译期已知，无需循环 |

### 5.3 Loop 合并策略

```python
# 合并 b, s, n 三维动态轴为单一 loop 维度
# 展平为 [b*s*n, d] 进行处理，减少 loop 嵌套层级
total_tiles = (batch_size + tile_b - 1) // tile_b
for tile_idx in pypto.loop(total_tiles, name="LOOP_TOPK"):
    # 处理每个 tile
    ...
```

### 5.4 数据依赖处理

topk 算子无数据依赖问题，每个 tile 可独立计算。

### 5.5 尾块处理策略

```python
# 使用 valid_shape 处理尾块
valid_shape = [
    (batch_size - tile_idx * tile_b).min(tile_b),
    seq_len,
    num_heads,
    hidden_dim
]
tile_input = pypto.view(input, [tile_b, seq_len, num_heads, hidden_dim],
                        [tile_idx * tile_b, 0, 0, 0],
                        valid_shape=valid_shape)
```

### 5.6 loop_unroll 配置

不使用 loop_unroll。动态轴范围虽大但使用 pypto.loop 即可满足需求。

---

## 6. 验证方案

### 6.1 Golden 函数设计

```python
def topk_golden(
    input: torch.Tensor,
    k: int,
    dim: int = -1,
    largest: bool = True,
    sorted: bool = True
) -> Tuple[torch.Tensor, torch.Tensor]:
    """topk 参考实现"""
    values, indices = torch.topk(input, k=k, dim=dim, largest=largest, sorted=sorted)
    return values, indices
```

### 6.2 测试用例设计

#### 基于 spec.md 所有典型配置

| 配置名称 | 类型 | 优先级 | 参数 | 输入 Shape | 输出 Shape | 说明 |
|----------|------|--------|------|------------|------------|------|
| 性能_P0 | 性能 | P0 | k=10, dim=-1, largest=True, sorted=True | [1, 1024, 16, 64] | values: [1,1024,16,10], indices: [1,1024,16,10] | 核心性能场景 |
| 功能_P0 | 功能 | P0 | k=5, dim=-1, largest=False, sorted=True | [2, 512, 8, 128] | values: [2,512,8,5], indices: [2,512,8,5] | 最小值选取验证 |
| 动态轴_P1 | 功能 | P1 | k=20, dim=1, largest=True, sorted=False | [4, 2048, 32, 64] | values: [4,20,32,64], indices: [4,20,32,64] | seq 维动态场景，dim!= -1 |

#### 边界情况测试（可选）

| 场景 | 参数 | 说明 |
|------|------|------|
| k == dim_size | k=64, dim=-1 | 边界情况，返回全部元素 |
| 最小 shape | k=2, shape=[1,1,1,4] | 最小输入测试 |
| 包含 Inf/-Inf | input 含 inf | 极值处理验证 |

### 6.3 精度验证标准

| Dtype | atol | rtol |
|-------|------|------|
| DT_FP32 | 0.001 | 0.001 |
| DT_FP16 | 0.01 | 0.01 |

---

## 7. 性能指标与开箱配置

### 7.1 性能目标

基于 spec.md 典型配置（性能类）的预期性能：

| 配置名称 | 类型 | 优先级 | 参数 | 输入 Shape | 输出 Shape | 预期 kernel 耗时 |
|----------|------|--------|------|------------|------------|------------------|
| 性能_P0 | 性能 | P0 | k=10, dim=-1, largest=True, sorted=True | [1, 1024, 16, 64] | values: [1,1024,16,10], indices: [1,1024,16,10] | 首跑成功后 2 倍优化 |

### 7.2 开箱性能配置

```python
# 性能_P0 配置: [1, 1024, 16, 64], k=10, dim=-1
pypto.set_vec_tile_shapes(1, 1, 1, 64)  # 尾轴完整，32B 对齐 (64*4=256)
```

### 7.3 pass_options 配置

```python
# 暂无特殊 pass_options 配置需求
pass_options = {}
```

### 7.4 runtime_options 配置

```python
runtime_options = {"run_mode": pypto.RunMode.NPU}
```

---

## 8. 风险点与注意事项

### 8.1 已知约束

- **dim 限制**: PyPTO `pypto.topk` 仅支持 dim=-1，其他维度需通过 transpose 间接支持，transpose 可能带来额外开销
- **sorted 缺失**: PyPTO `pypto.topk` 不保证排序输出，若需排序需额外实现（P1 优先级，首跑可暂不实现）
- **indices 类型**: PyPTO 返回 int32 索引，需 cast 转换为 int64，带来额外开销
- **TileShape 约束**: 尾轴需 32B 对齐且 < 22KB，k <= TileShape[-1]

### 8.2 常见错误规避

| 风险 / 错误 | 触发场景 | 影响 / 原因 | 规避方法 |
|-------------|----------|-------------|----------|
| TileShape 尾轴不对齐 | 尾轴不是 8 的倍数（FP32） | 编译失败 | 确保 TileShape[-1] * 4 % 32 == 0 |
| k > TileShape[-1] | k 设置过大 | topk 返回错误 | 确保 k <= TileShape[-1] 且 k <= input.shape[-1] |
| transpose 不支持 | 4 维 tensor 的 0 轴和 3 轴转置 | 运行时错误 | 参考 transpose 文档约束，避免不支持的场景 |
| 动态轴未标记 | from_torch 未设置 dynamic_axis | 编译期静态化 | 正确设置 dynamic_axis=[0, 1, 2] |

### 8.3 特殊场景处理

1. **dim != -1 场景**: 需要先 transpose 将目标维度移到尾部，计算完成后再 transpose 回去。注意 4 维 tensor transpose 约束（不支持 0 轴和 3 轴转置）
2. **sorted=True 场景**: PyPTO topk 不保证排序，若 spec 要求 sorted=True，需对输出额外排序（P1 优先级，可后续优化）
3. **k == dim_size 场景**: 返回全部元素，索引为 [0, 1, ..., dim_size-1]

### 8.4 实现建议

| 建议项 | 说明 |
|--------|------|
| 首跑优先 dim=-1 | 先实现 dim=-1 场景，确保基本功能正确后再扩展 dim 支持 |
| sorted 参数延后 | sorted 参数优先级 P1，首跑可暂不实现排序逻辑 |
| 参考 glm_select_experts.py | 参考 models/glm_v4_5/glm_select_experts.py 中的 topk 调用模式和 loop 结构 |

---

## 9. 交付件清单

### 9.1 目录结构

```
operators/topk/
├── spec.md                          # 需求规范（已有）
├── api_report.md                    # API 探索报告（已有）
├── design.md                        # 设计文档（本文件）
├── topk_golden.py                   # Golden 参考实现（已有）
├── topk_impl.py                     # 算子实现代码
├── test_topk.py                     # 测试代码
└── .orchestrator_state.json         # 状态文件
```

### 9.2 文件清单

| 文件 | 类型 | 说明 | 生成方式 |
|------|------|------|----------|
| spec.md | 需求 | 算子需求规范 | pypto-intent-understanding |
| api_report.md | 探索 | API 探索报告 | pypto-api-explorer |
| design.md | 设计 | 算子设计文档 | pypto-op-design（本文件） |
| topk_golden.py | 代码 | Golden 参考实现 | pypto-golden-generator |
| topk_impl.py | 代码 | 算子核心实现 | pypto-op-develop |
| test_topk.py | 代码 | 测试用例 | pypto-op-develop |

### 9.3 命名规范

| 项目 | 规范 | 示例 |
|------|------|------|
| 算子名称 | 小写字母 | `topk` |
| 目录名 | 与算子名称一致 | `operators/topk/` |
| Golden 文件 | `{op}_golden.py` | `topk_golden.py` |
| 实现文件 | `{op}_impl.py` | `topk_impl.py` |
| 测试文件 | `test_{op}.py` | `test_topk.py` |

### 9.4 生成顺序

```
spec.md → api_report.md → topk_golden.py → design.md → topk_impl.py → test_topk.py
```
