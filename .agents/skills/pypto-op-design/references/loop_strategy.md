# Loop 策略

> **版本**: 1.0
> **最后更新**: 2026-03-15
> **说明**: 本文件用于"设计期可行性闸门"。每条规则必须能回链到本仓库内的 docs/ 或代码证据。
> - **HARD**：不满足会直接导致编译失败、运行错误或结果异常。
> - **HEURISTIC**：经验建议，允许按实测调整。

---

## 1. 是否需要 Loop（判定树）

```
算子特征分析
    │
    ├── 所有轴编译期已知 & 无多步骤分块
    │   └── 不需要 loop，仅靠 TileShape 配置即可
    │
    ├── 存在动态轴（运行期才确定长度）
    │   └── 需要 pypto.loop 或 pypto.loop_unroll
    │
    ├── 多步骤计算（如 attention 按 seq_len 分块）
    │   └── 需要 loop 管理分块迭代
    │
    └── Reduction 跨多维且需分块处理
        └── 视情况决定，可能需要 loop + 中间累加
```

**evidence**: `docs/tutorials/development/loops.md`; `docs/tutorials/debug/performance.md`

---

## 2. 静态轴 vs 动态轴（硬规则）

| rule_id | rule_text | level | trigger | impact | warning_code | evidence |
|---------|-----------|-------|---------|--------|--------------|----------|
| LOOP_AXIS_01 | 静态轴优先 Python for，不优先 pypto.loop | HARD | 轴长度编译期可知 | pypto.loop 将静态轴展开为过多 root function，增加编译复杂度 | LOOP_STATIC_AXIS_MISUSE | `docs/tutorials/debug/performance.md` |
| LOOP_AXIS_02 | 动态轴使用 pypto.loop 并补齐边界控制 | HARD | 轴长度运行期确定 | 运行错误/结果异常 | LOOP_DYNAMIC_AXIS_INVALID | `docs/tutorials/development/loops.md` |
| LOOP_VIEW_01 | 动态轴场景 view 的 shape 范围不可过小 | HARD | 动态 shape + loop | view shape 过小会限制 TileShape 导致性能劣化 | LOOP_VIEW_TILE_TOO_SMALL | `docs/tutorials/debug/performance.md` |

---

## 3. 标准写法模板

### 3.1 静态轴 — Python for

```python
# 轴长度编译期已知，直接用 Python for
for i in range(num_heads):
    head_i = pypto.view(x, [seq_len, head_dim], [0, i * head_dim])
    result_i = process(head_i)
```

### 3.2 动态轴 — pypto.loop

```python
# 轴长度运行期确定，使用 pypto.loop
for i in pypto.loop(batch_size, name="LOOP_BATCH", idx_name="i"):
    x_i = pypto.view(x, [seq_len, hidden], [i * seq_len, 0],
                     valid_shape=[seq_len, hidden])
    y_i = compute(x_i)
```

**API**: `pypto.loop` 返回 `Iterator[SymInt]`。

**evidence**: `docs/api/controlflow/pypto-loop.md`

### 3.3 尾块处理模板

当动态轴按块大小分块时，尾块需要 `valid_shape` 约束：

```python
bsz, h = x.shape
b = 128
b_loop = (bsz + b - 1) // b
for b_idx in pypto.loop(b_loop, name="LOOP_BLOCK", idx_name="b_idx"):
    b_valid = (bsz - b_idx * b).min(b)
    x_view = pypto.view(x, [b, h], [b_idx * b, 0], valid_shape=[b_valid, h])
    y = compute(x_view)
```

**evidence**: `docs/tutorials/debug/performance.md`

### 3.4 依赖链模板

当后续循环依赖前序结果时，使用 `submit_before_loop=True` 约束调度顺序：

```python
for idx in pypto.loop(n, name="LOOP_DEP", idx_name="idx",
                      submit_before_loop=True):
    # 每轮迭代依赖上一轮的写回结果
    prev_result = read_from_gm(idx - 1)
    new_result = compute(prev_result, x[idx])
    write_to_gm(new_result, idx)
```

**evidence**: `docs/tutorials/development/loops.md`; `docs/api/controlflow/pypto-loop.md`

---

## 4. loop_unroll 使用边界

| rule_id | rule_text | level | trigger | impact | warning_code | evidence |
|---------|-----------|-------|---------|--------|--------------|----------|
| LOOP_UNROLL_01 | 多层嵌套时仅最内层 loop_unroll 可使用 unroll_list | HARD | 嵌套循环场景 | 展开策略失效/性能退化 | LOOP_UNROLL_NESTED_INVALID | `docs/tutorials/debug/performance.md` |
| LOOP_UNROLL_02 | unroll_list 会被排序去重且总包含 1，按从大到小排序 | HARD | 配置 unroll_list | 展开档位不可控 | LOOP_UNROLL_FACTOR_INVALID | `docs/api/controlflow/pypto-loop_unroll.md` |

### loop_unroll 模板

```python
# 动态轴范围较广（如 1~64k），使用 loop_unroll
for b, k in pypto.loop_unroll(A.shape[0] // 64,
                               unroll_list=[64, 16, 4],
                               name="LOOP_UNROLL_A", idx_name="b"):
    if k <= 16:
        pypto.set_vec_tile_shapes(16, 64)
    else:
        pypto.set_vec_tile_shapes(64, 64)
    tile_a = A[b * 64:(b + k) * 64, :]
    tile_a = tile_a + 2
    B[b * 64:, :] = tile_a
```

**API**: `pypto.loop_unroll` 返回 `(idx, unroll_factor)`。

**HEURISTIC 建议**：
- 初版算子建议使用较短的 unroll_list（如 `[64, 16, 4]`），避免编译时间随档位数增加而显著变长。
- 仅在热点且分支简单场景启用 unroll；若编译时长显著上升，回退 unroll。

**evidence**: `docs/api/controlflow/pypto-loop_unroll.md`; `docs/tutorials/debug/performance.md`

---

## 5. submit_before_loop 触发条件

| 场景 | 是否需要 submit_before_loop | 说明 |
|------|----------------------------|------|
| 无跨迭代依赖 | 否（默认） | 各迭代可并行执行 |
| 迭代 i+1 读取迭代 i 的写回结果 | 是 | 必须保证上一轮写回完成 |
| 累加/在线更新（如 online softmax） | 是 | 状态更新有严格顺序要求 |

**evidence**: `docs/tutorials/development/loops.md`; `docs/api/controlflow/pypto-loop.md`

---

## 6. 循环合并与尾块处理

### 循环合并

当多个 loop 遍历相同轴/范围且无跨迭代依赖时，优先合并以增大 root function、减少调度开销：

```python
# 合并前：两次 loop，两个 root function
for i in pypto.loop(n, name="L1", idx_name="i"):
    op1(x[i])
for i in pypto.loop(n, name="L2", idx_name="i"):
    op2(x[i])

# 合并后：一次 loop，一个 root function
for i in pypto.loop(n, name="L3", idx_name="i"):
    op1(x[i])
    op2(x[i])
```

**注意**：存在跨 op 数据依赖时不可盲目合并。

**evidence**: `docs/tutorials/debug/performance.md`

---

## 7. 失败签名与规避

| warning_code | trigger | impact | mitigation | evidence |
|--------------|---------|--------|------------|----------|
| LOOP_UNROLL_COMPILE_BLOWUP | 过多 unroll 档位或过深嵌套 | 编译时间爆炸 | 减少 unroll_list 长度或关闭 unroll | `docs/tutorials/debug/performance.md` |
| LOOP_DEPENDENCY_HAZARD | 循环间读写依赖未约束 | 结果错误 | 添加 submit_before_loop=True | `docs/tutorials/development/loops.md` |
| LOOP_TAIL_VALIDITY_MISMATCH | 尾块有效区间处理不当 | 数值错误 | 增加 valid_shape 校验 | `docs/tutorials/development/loops.md` |
| LOOP_MERGE_MISUSE | 存在依赖时不当合并循环 | 正确性或性能问题 | 回退为独立循环 | `docs/tutorials/debug/performance.md` |
| LOOP_STATIC_AXIS_MISUSE | 静态轴使用 pypto.loop | 编译复杂度增加 | 改用 Python for | `docs/tutorials/debug/performance.md` |

---

## 8. 证据索引

| 证据文件 | 内容 |
|----------|------|
| `docs/tutorials/development/loops.md` | Loop 基本概念、pypto.loop 使用方式、依赖处理 |
| `docs/tutorials/debug/performance.md` | Loop 写法选择、合并策略、unroll 建议 |
| `docs/api/controlflow/pypto-loop.md` | pypto.loop API 签名与参数说明 |
| `docs/api/controlflow/pypto-loop_unroll.md` | pypto.loop_unroll API 签名、unroll_list 约束 |
