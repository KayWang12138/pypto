# gdr_origin.py dq 精度问题调试进展

## 问题描述

运行 `python3 gdr_origin.py`，反向传播的 `dq` 输出与 golden 对比存在精度差异：
- **dq**: FAIL（max_diff=0.253，98.2% 元素超差）
- dk, dv, db, dg_raw, dh0: 均正确

## 测试参数

| 参数 | 值 |
|------|------|
| T (seq_len) | 128 |
| Nqk | 1 |
| Nv | 2 |
| D | 128 |
| L (chunk_size) | 128 |
| B (batch) | 1 |
| group (Nv//Nqk) | 2 |
| use_l2norm | True |

## 计算流程（内层循环）

```
Module 1: view/reshape 输入切片
Module 2: pypto_g_and_decay_kernel      → g_cum, eg, gl, decay
Module 3: pypto_local_attn_dv0_kernel    → qk, dv0
Module 4: pypto_recurrence_backprop      → s_tok, dv_total, dS_final
Module 5: pypto_compute_qkg_grads_dw_du  → dq_c, dk_c, dg_cum, dw
Module 6: pypto_wy_repr_fused_updates    → dv_c, db_c, dk_c, dg_cum（dq_c 不被修改）
Module 7: pypto_finalize_chunk_grads     → dq_raw_c, dk_raw_c（l2norm_bwd）
Assemble: dq_out[bs_ofs:bs_ofs+l, nqk_idx] = dq_raw_c
```

## 已完成的二分搜索

### 方法

在 `gdr_binary_search_1.py` 和 `gdr_binary_search_2.py` 中，通过 `pypto.assemble` 添加 checkpoint tensor，对比 PyPTO 中间结果与 golden 中间结果。

### 添加的检查点

| 检查点 | 位置 | 形状 | 结果 |
|--------|------|------|------|
| dq_c (Module 5 后) | `pypto_compute_qkg_grads_dw_du` 输出 | [B*Nv, L, D] | h=0: PASS (8.9e-8), h=1: PASS (1.3e-7) |
| dq_raw_c (Module 7 后) | `pypto_finalize_chunk_grads` 输出 | [B*Nv, L, D] | h=0: PASS (8.9e-8), h=1: PASS (1.2e-7) |
| dk_c (Module 5 后) | 同上 | [B*Nv, L, D] | PASS |
| dk_raw_c (Module 7 后) | 同上 | [B*Nv, L, D] | PASS |

### 初步结论

所有中间计算 checkpoint 均 PASS，但最终 `dq_out` FAIL → 初步判断问题在 Assemble 阶段（`gdr_origin.py:754`）。

### 结论存疑的原因

1. **添加 checkpoint 改变了编译调度**：加入额外的 `pypto.assemble` 操作后，dq 从 FAIL 变为 PASS（dk 从 PASS 变为 FAIL），说明 checkpoint 影响了执行行为
2. **独立用例无法复现**：在 `test_tensor_slice_assemble.py` 中构造的 `test_slice_overwrite_in_loop` 用例，使用 `pypto.function` 底层 API 和 `@pypto.frontend.jit()` 装饰器两种方式，均无法复现 assemble 覆盖问题（通过 `build_ci.py` 运行 PASS）
3. **覆盖行为非确定性**：多次运行同一 checkpoint 版本，`dq_out` 中保留的 h 值不一致（有时保留 h=0，有时保留 h=1）

## 下一步计划

已完成（见下方"第二轮二分搜索"）。

## 第二轮二分搜索（已完成，确认定位）

### 方法

不修改 Module 内部计算，仅在 Module 7 输出后添加单个 checkpoint tensor `cp_dq_raw_c`，验证 Assemble 前后的精度。

### 测试文件

| 文件 | 说明 |
|------|------|
| `gdr_bs_1.py` | 添加 dq_raw_c checkpoint（Module 7 后），验证中间值 |
| `gdr_bs_2.py` | 在 bs_1 基础上增加 Assemble 覆盖验证 |

### gdr_bs_1.py 结果

| 检查项 | 结果 |
|--------|------|
| dq_raw_c[idx=0] (h=0) vs golden | PASS (max_diff=2.98e-8) |
| dq_raw_c[idx=1] (h=1) vs golden | PASS (max_diff=2.98e-8) |
| dq_out (最终输出) vs golden | FAIL |

**结论**：dq_raw_c 在两个 h 迭代中均正确，但最终 dq_out 错误 → 问题在 Assemble 阶段。

### gdr_bs_2.py 结果（Assemble 覆盖验证）

| 比较 | 结果 | max_diff |
|------|------|----------|
| dq_out vs cp_dq_raw_c[idx=0] (h=0) | **MATCH** | 0.0 |
| dq_out vs cp_dq_raw_c[idx=1] (h=1) | NO MATCH | 0.253 |
| golden_dq vs dq_raw_c_all[idx=0] (h=0) | NO MATCH | 0.253 |
| golden_dq vs dq_raw_c_all[idx=1] (h=1) | **MATCH** | 0.0 |

**关键发现**：
- **PyPTO dq_out = h=0 的值**（第一次写入被保留，第二次写入丢失）
- **Golden dq = h=1 的值**（第二次写入正确覆盖第一次，符合预期）

### 最终结论

**精度问题定位在 `gdr_origin.py:754` 的 Assemble 行**：

```python
dq_out[bs_ofs:bs_ofs + l, nqk_idx] = dq_raw_c
```

**根因分析**：
- 输入 tensor `dq_raw_c` 精度正确（两个 h 迭代均 PASS）
- 输出 tensor `dq_out` 精度错误
- 当 `group = Nv // Nqk > 1` 时，多个 `nv_idx` 迭代（h=0, h=1）写入同一位置 `dq_out[bs_ofs:bs_ofs+l, nqk_idx]`（nqk_idx=0）
- Golden 中后执行的 h=1 正确覆盖 h=0 的结果
- PyPTO 中该 slice 赋值的覆盖行为不确定：h=1 的写入未能正确覆盖 h=0，导致 dq_out 保留了 h=0 的值
- 这与此前观察到的"多次运行保留的 h 值不一致"现象一致

**该行不是计算精度问题，而是 PyPTO slice 赋值在多轮循环写同一地址时的覆盖语义问题。**

## 相关文件

| 文件 | 说明 |
|------|------|
| `gdr_origin.py` | 原始代码（dq FAIL） |
| `gdr_binary_search_1.py` | 二分搜索 v1（4个 checkpoint，全部 PASS） |
| `gdr_binary_search_2.py` | 二分搜索 v2（2个 checkpoint，全部 PASS） |
| `gdr_bs_1.py` | 二分搜索 v3（1个 checkpoint：dq_raw_c，PASS） |
| `gdr_bs_2.py` | 二分搜索 v4（Assemble 覆盖验证：dq_out = h=0 的值） |
| `python/tests/st/test_tensor_slice_assemble.py` | assemble 覆盖测试用例（无法复现问题） |
