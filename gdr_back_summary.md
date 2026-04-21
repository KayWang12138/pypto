# gdr_back.py 精度问题定位与修复总结

## 1. 问题描述

`gdr_back.py` 实现了 Gated Delta Rule Backward 的 PyPTO 算子，包含 7 个计算模块。在精度验证中，`db`（对 beta 的梯度）输出与 golden 参考实现对比出现 99.2% 的元素超差，最大偏差 5.5。

## 2. 定位方法：二分对比（Binary Search）

在 kernel 函数的每个模块边界添加检查点 tensor，与 golden 中间结果逐模块对比，定位首个不匹配的位置。

### 检查点设计

kernel 函数为三重循环结构（batch × nv_heads × chunks_reverse），在循环内按模块顺序执行计算。检查点 tensor 采用 `[Nv, ...]` 形状，使用 `cp_tensor[nv_idx] = val` 写入每个 nv 的中间结果。

### 检查点位置与结果

| 检查点 | 模块 | 对比值 | max_diff | 结果 |
|--------|------|--------|----------|------|
| - | Module 1 | 输入切片（view/reshape） | - | 正确（后续模块输入一致） |
| - | Module 2: `g_and_decay` | g_cum, eg, decay | - | 正确 |
| - | Module 3: `local_attn_dv0` | dv0 | - | 正确 |
| - | Module 4: `recurrence_backprop` | dv_total, dS_final | - | 正确 |
| - | Module 5: `qkg_grads` | dg_cum, dw_final | - | 正确 |
| `cp_db_c` | Module 6: `wy_repr` | db_c | **0.0** | **PASS** |
| `db_out` | Assemble 阶段 | 最终 db 输出 | **5.50** | **FAIL** |

**结论：所有 Module 2-6 的计算逻辑完全正确，问题出在 Assemble 阶段。**

## 3. 根因分析

### 问题代码

```python
db_out[bs_ofs:bs_ofs + l, nv_idx:nv_idx+1] = db_c
# db_out shape: [T=128, Nv=2],  db_c shape: [L=128, 1]
```

### 根因：pypto slice 赋值未正确处理非连续（strided）内存写入

`db_out` 的内存布局（行优先）：

```
内存偏移:  0    1    2    3    4    5    6    7   ...
内容:    d0,0  d0,1  d1,0  d1,1  d2,0  d2,1  d3,0  d3,1 ...
```

`db_out[0:128, 0:1] = db_c` 应写入第 0 列，实际需要写入的位置是 **0, 2, 4, 6, ...**（stride=2，非连续）。

pypto 的 slice 赋值将其当作**连续写入**处理，实际写入了位置 **0, 1, 2, 3, ...**，导致数据错乱：

```
期望结果（strided 写入）:      实际结果（contiguous 写入）:
[db_c[0],    0         ]       [db_c[0],   db_c[1]   ]
[db_c[1],    0         ]       [db_c[2],   db_c[3]   ]
[db_c[2],    0         ]       [db_c[4],   db_c[5]   ]
...                             ...
[db_c[127],   0         ]       [0,         0         ]
[0,           db_c_nv1[0]]       [0,         0         ]
...                             ...
```

### 验证证据

| 对比项 | max_diff |
|--------|----------|
| `cp_db_c`（检查点）vs golden `db_c` | 0.0（完全匹配） |
| `db_out`（slice 赋值）vs `cp_db_c` 组装结果 | 5.5（严重不匹配） |
| `cp_db_c` 经 torch 手动组装 vs golden `db` | 0.0（完全匹配） |

### 为什么 `cp_db_c[nv_idx] = db_c` 正确

```
cp_db_c shape: [Nv=2, L=128, 1]
cp_db_c[0] = db_c  → 写入偏移 0~127，stride=1，连续内存
cp_db_c[1] = db_c  → 写入偏移 128~255，stride=1，连续内存
```

整数索引 `tensor[idx]` 取出的是整页连续内存，赋值操作自然是连续写入，与 pypto 实际行为一致。

## 4. 修复方案

### 修改内容

1. **kernel 函数**：新增 `cp_db_c: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_FP32)` 参数，在 Module 6 之后写入 `cp_db_c[nv_idx] = db_c`，移除原始的 `db_out` slice 赋值

2. **pypto_function**：创建 `cp_db_c = torch.zeros([Nv, L, 1])`，传入 kernel，运行后通过 `cp_db_c.squeeze(-1).permute(1, 0).contiguous()` 组装为 `[T, Nv]` 输出

### 关键代码变更

```python
# kernel 内（写入检查点替代 slice 赋值）
cp_db_c[nv_idx] = db_c

# pypto_function 内（Python 端组装）
db_out_from_cp = cp_db_c.squeeze(-1).permute(1, 0).contiguous()
return dq_out, dk_out, dv_out, db_out_from_cp, dg_raw_out, dh0_out
```

## 5. 影响范围

该问题为 pypto 前端对 **2D tensor 的 range slice 赋值（如 `a[offset:offset+L, idx:idx+1] = val`）** 的通用 bug，会影响到所有跨步非连续写入场景。当前文件中其他被注释的 assemble 操作也使用了类似模式：

```python
# 以下均可能存在同样问题，需要同样的 workaround：
# dq_out[bs_ofs:bs_ofs + l, nqk_idx] = dq_raw_c
# dk_out[bs_ofs:bs_ofs + l, nqk_idx] = dk_raw_c
# dv_out[bs_ofs:bs_ofs + l, nv_idx] = dv_c
# dg_raw_out[bs_ofs:bs_ofs + l, nv_idx:nv_idx+1] = dg_raw_c
```

**建议：** 对所有输出 tensor 的 assemble，统一使用 `cp_tensor[loop_idx] = val`（整数索引）+ Python 端 reshape 的模式，避免 2D range slice 赋值。
