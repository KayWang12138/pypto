---
name: tune-frontend
description: PyPTO算子开箱性能调优技能。主要关注代码级的调优、前端写法不同导致的性能差异，包括loop写法优化、TileShape设置优化、数据操作优化等。当用户需要进行算子初始开发性能优化、开箱性能调优时使用此技能。触发词：开箱性能调优、代码级优化、loop优化、TileShape设置、前端优化。
---

# PyPTO 算子开箱性能调优

## 概述

开箱性能调优主要关注代码级的调优、前端写法不同导致的性能差异。在算子初始编写过程中直接得到较好的开箱性能。

## 核心原则

**增加 root function 的大小，减少它们的个数**

由于不同 root function 之间的子图不能合并，而子图合并是 PyPTO 优化性能的关键手段。

## 调优方向

### 1. Loop 写法优化

#### 1.1 静态轴使用 Python for 循环

`pypto.loop` 方法会按当前轴循环展开成不同的 root function。因此静态轴上的循环应使用 Python 的 for 循环。

```python
# ✅ 推荐：静态轴使用 Python for
for i in range(batch_size):
    result[i] = process(data[i])

# ❌ 避免：静态轴使用 PyPTO loop
for i in pypto.loop(batch_size, name="LOOP_1", idx_name="i"):
    result[i] = process(data[i])
```

#### 1.2 动态轴使用 PyPTO loop 并合理配置 view

当算子内有动态 Shape 时，动态轴的 dim 数值范围往往较广，需要使用 loop 循环处理。

**注意事项**：
- view 视图的参数配置选取的 Shape 范围不能过小
- 否则会限制后续 TileShape 的配置范围
- 导致每次循环的计算量过小，循环次数增加

```python
# 推荐：动态轴使用 loop + unroll
bsz, h = x.shape
b = 128
b_loop = (bsz + b - 1) // b
for b_idx in pypto.loop(b_loop, name="LOOP_1", idx_name="b_idx"):
    b_valid = (bsz - b_idx * b).min(b)
    x_view = pypto.view(x, [b, h], [b_idx * b, 0], valid_shape=[b_valid, h])
    # Matmul
    pypto.set_cube_tile_shapes([128, 128], [128, 128], [128, 128])
    y = pypto.matmul(x_view, W)
```

#### 1.3 尽可能合并 loop

检查算子代码是否有可以合并的 loop 块：

```python
# ❌ 不推荐：两个独立的 loop
bsz = x1.shape[0]
for b_idx in pypto.loop(bsz, name="LOOP_1", idx_name="b_idx"):
    out_1 = Operation1(x1[b_idx, :], y)
for b_idx in pypto.loop(bsz, name="LOOP_2", idx_name="b_idx"):
    out_2 = Operation2(x2[b_idx, :], y)

# ✅ 推荐：合并 loop
for b_idx in pypto.loop(bsz, name="LOOP_1", idx_name="b_idx"):
    out_1 = Operation1(x1[b_idx, :], y)
    out_2 = Operation2(x2[b_idx, :], y)
```

#### 1.4 动态轴范围较广时使用 loop_unroll

当算子使用动态 Shape，且 Shape 范围较广的场景，应考虑使用 `loop_unroll` 代替 `loop` 接口。

**注意事项**：
- `unroll_list` 参数支持多个展开方式
- 挡位数量增加会导致编译时间变长
- 建议使用较短的 `unroll_list`，如 `[64, 16, 4]`
- **只有最内层的 loop_unroll 可以成功使用 unroll_list 参数**

```python
for b, k in pypto.loop_unroll(A.shape[0] // 64, unroll_list=[64, 16, 4], name="A", idx_name='b'):
    # 支持在不同的展开档位设置不同调优参数
    if k <= 16:
        pypto.set_vec_tile_shapes(16, 64)
    else:
        pypto.set_vec_tile_shapes(64, 64)

    tile_a = A[b * 64:(b + k) * 64, :]
    tile_a = tile_a + 2
    B[b * 64:, :] = tile_a
```

### 2. TileShape 设置优化

TileShape 切分大小直接决定：
- 算子切分后的任务数量
- 实际执行时的分核数、计算轮次
- 算子的算数强度

**优化关键**：优化 Tiling 配置

#### 2.1 Matmul 初始 Tiling 配置

针对矩阵运算场景（A、B 矩阵均为 DT_BF16 或 DT_FP16 类型）：

```python
# Cube 的相关计算建议采用如下的 TileShape
pypto.set_cube_tile_shapes([128, 128], [64, 256], [256, 256])
pypto.set_cube_tile_shapes([256, 256], [64, 256], [128, 128])
pypto.set_cube_tile_shapes([128, 128], [128, 512], [128, 128])
```

**优点**：
- 在满足 L0 Buffer 约束的条件下达到较大的算数强度
- 后续进一步使用合图相关接口进行深度调优时，有机会开启 Double Buffer

#### 2.2 Vector 初始 Tiling 配置

针对向量运算场景：

**配置原则**：
1. 满足特定 Operation 对 TileShape 的规格约束
2. 保证 Operation 的输入与输出 Tensor 可以在 UB 中分配内存
3. TileShape 不能过大也不能过小（数据块大小在 16 到 64KB 之间）
4. 尾轴 32B 对齐
5. 归约类计算尽可能不要在归约轴上进行切分

```python
# Vector 的相关计算建议采用如下的 TileShape
pypto.set_vec_tile_shapes(64, 512)
```

**归约轴切分问题示例**：

对于输入 Shape 为 (56, 1024) 的 RMSNorm：
- ❌ 对 reduce 轴切分：多个子图的输出需要在同一个子图进行 reduce 操作，产生 GM 搬运和调度开销
- ✅ 不对 reduce 轴切分：上下游子图合并，没有 GM 搬运和调度开销

### 3. 数据操作优化

#### 3.1 输入矩阵格式优化

检查输入矩阵、尤其是 Shape 较大的权重矩阵是否可以提前以 NZ 格式存储。

**NZ 格式的数据搬运到 L1 的带宽更高。**

#### 3.2 Transpose 优化

矩阵乘前后有 transpose 时，可以尝试更换左右矩阵并使用左右矩阵转置的配置。

当 M 轴较大、N 轴较小时，使得左右矩阵有更大的尾轴，提升搬运带宽。

#### 3.3 冗余搬运优化

检查是否有不合理数据操作导致的冗余搬运：

- 更换 concat 为 assemble
- 尝试对 reshape 配置 `inplace = True` 参数

## 性能优化建议库

### 建议 1：Loop 优化

| 问题 | 解决方案 | 代码示例 |
|------|---------|---------|
| 静态轴使用 pypto.loop | 改用 Python for | `for i in range(n):` |
| 多个独立 loop | 合并 loop | 合并到同一个 loop 内 |
| 动态轴范围广 | 使用 loop_unroll | `pypto.loop_unroll(..., unroll_list=[64, 16, 4])` |

### 建议 2：TileShape 优化

| 场景 | 推荐配置 | 说明 |
|------|---------|------|
| Cube 计算 | `[128, 128], [64, 256], [256, 256]` | 高算数强度 |
| Vector 计算 | `64, 512` | UB 利用率高 |
| Reduce 操作 | 不切归约轴 | 避免额外 GM 搬运 |

### 建议 3：数据操作优化

| 问题 | 解决方案 |
|------|---------|
| 大矩阵搬运慢 | 使用 NZ 格式存储 |
| transpose 性能差 | 调整左右矩阵顺序 |
| concat 冗余搬运 | 使用 assemble |
| reshape 冗余搬运 | 配置 `inplace=True` |

## 调优流程

**⚠️ 重要：开箱性能调优不需要查看性能报告！**

### 1. 建立性能基准

**首次运行算子用例**，记录基准性能：
```bash
python3 custom/operator_name/operator.py --run-mode npu
```

**记录基准执行时间**：
```
基准执行时间: XXX us
```

### 2. 迭代优化循环

```
┌──────────────────────────────────┐
│     开箱调优迭代流程              │
├──────────────────────────────────┤
│                                  │
│  1. 选择一个优化点                │
│     ├─ Loop 写法优化              │
│     ├─ TileShape 设置优化         │
│     └─ 数据操作优化               │
│                                  │
│  2. 修改代码                      │
│     └─ 每次只修改一个参数         │
│                                  │
│  3. 验证精度 ⭐                   │
│     ├─ 运行测试用例               │
│     └─ 失败则立即回退             │
│                                  │
│  4. 对比性能 ⭐                   │
│     ├─ 记录新执行时间             │
│     ├─ 对比基准执行时间           │
│     └─ 计算提升百分比             │
│                                  │
│  5. 判断是否保留                  │
│     ├─ 性能提升：保留修改         │
│     └─ 性能下降：回退修改         │
│                                  │
│  6. 检查终止条件                  │
│     ├─ 达到性能目标               │
│     └─ 连续5次优化无提升          │
│                                  │
└──────────────────────────────────┘
```

### 3. 优化检查清单

**Loop 写法检查**：
- [ ] 静态轴是否使用 Python for
- [ ] 是否可以合并 loop
- [ ] 动态轴是否使用 loop_unroll

**TileShape 设置检查**：
- [ ] Cube 计算：是否使用推荐配置
- [ ] Vector 计算：是否使用推荐配置
- [ ] 归约轴是否避免切分

**数据操作检查**：
- [ ] 输入矩阵格式是否优化
- [ ] transpose 配置是否合理
- [ ] 是否存在冗余搬运

### 4. 性能对比示例

```markdown
## 优化记录

| 轮次 | 优化内容 | 执行时间(us) | 提升比例 | 精度结果 |
|------|---------|-------------|---------|---------|
| 基准 | 无优化 | 27469.66 | - | 通过 |
| 1 | 静态轴改用Python for | 25123.45 | 8.5% | 通过 |
| 2 | TileShape优化 | 22456.78 | 10.6% | 通过 |
| 3 | 合并loop | 21345.12 | 4.9% | 通过 |
```

## 参考资料

- [性能调优文档](../../../../docs/tutorials/debug/performance.md)
- [GDR 算子案例](../../../../docs/tutorials/debug/performance_case_GDR.md)
- [Matmul 高性能编程](../../../../docs/tutorials/debug/matmul_performance_guide.md)
