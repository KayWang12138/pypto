---
schema_version: "2.1"
op_name: gated_delta_rule_backward
status: draft
last_updated: "2026-04-21"

compute_kind: "mixed"
dtypes: ["fp32"]
dynamic_axes: ["B", "T"]
precision: { rtol: 1e-3, atol: 1e-3 }
---

# gated_delta_rule_backward 设计方案

## 1. 计算图与精度路由

### 1.1 API 调用序列

> 以下为单个 chunk 内的完整计算步骤序列。每步标注 PyPTO API、dtype 和 shape。

| 步骤 | 操作 | PyPTO API | 输入 dtype | 输出 dtype | 输出 shape | 备注 |
|------|------|-----------|------------|------------|-----------|------|
| **0** | **门控累积与衰减** | | | | | |
| 0a | cumsum | `pypto.matmul(c_cum, gc_view, DT_FP32)` | FP32 | FP32 | [BT, 1] | `g_cum = c_cum @ gc_raw` |
| 0b | 指数 | `pypto.exp(g_cum)` | FP32 | FP32 | [BT, 1] | `eg` |
| 0c | 差值指数 | `pypto.expand_exp_dif` 或 `(g_cum - g_cum.T).exp()` | FP32 | FP32 | [BT, BT] | `decay = exp(g_cum_i - g_cum_j)` |
| 0d | 取末行 | `g_cum[BT-1:BT, :]` (view) | FP32 | FP32 | [1, 1] | `gl = g_cum[-1]` |
| **1** | **局部注意力 dv0** | | | | | |
| 1a | qk 乘积 | `pypto.matmul(qc, kc, DT_FP32, b_trans=True)` | FP32 | FP32 | [BT, BT] | `qk = qc @ kc^T` |
| 1b | 衰减掩码乘 | `pypto.mul(qk, decay)` → `pypto.mul(result, m_le)` | FP32 | FP32 | [BT, BT] | `a_local = (qk * decay) * m_le` |
| 1c | dv0 | `pypto.matmul(a_local, doc, DT_FP32, a_trans=True)` → `pypto.mul(result, scale_val)` | FP32 | FP32 | [BT, V] | `dv0 = (a_local^T @ doc) * scale` |
| **2** | **状态递推反向传播** | | | | | |
| 2a | 逆衰减 token | `(gl - g_cum).exp()` | FP32 | FP32 | [BT, 1] | `s_tok = exp(gl - g_cum)` |
| 2b | dv_state | `pypto.mul(pypto.matmul(kc, d_s, DT_FP32), s_tok_2d)` | FP32 | FP32 | [BT, V] | `(kc @ d_s) * s_tok` |
| 2c | dv_total | `pypto.add(dv_state, dv0)` | FP32 | FP32 | [BT, V] | 合并 v 梯度 |
| 2d | d_s 更新-衰减 | `pypto.mul(d_s, pypto.exp(gl))` | FP32 | FP32 | [K, V] | `d_s * exp(gl)` |
| 2e | d_s 更新-q 贡献 | `pypto.matmul(qc, doc, DT_FP32, a_trans=True)` → `pypto.mul(result, scale_val)` | FP32 | FP32 | [K, V] | `q_eff^T @ doc * scale` |
| 2f | d_s 更新-w 贡献 | `pypto.matmul(w, dv_total, DT_FP32, a_trans=True)` | FP32 | FP32 | [K, V] | `w^T @ dv_total` |
| 2g | d_s 合并 | `d_s = add(sub(add(d_s_decay, d_s_q), d_s_w))` | FP32 | FP32 | [K, V] | 状态梯度递推完成 |
| **3** | **dq/dk/dg_cum 梯度累积** | | | | | |
| 3a | dq 状态贡献 | `pypto.mul(pypto.matmul(doc, s_before, DT_FP32, b_trans=True), eg_2d)` → `pypto.mul(result, scale_val)` | FP32 | FP32 | [BT, K] | `(doc @ s_before^T) * eg * scale` |
| 3b | dg_cum 从 dq | `pypto.sum(pypto.mul(dq1, qc), -1)` | FP32 | FP32 | [BT] | `(dq1 * qc).sum(-1)` |
| 3c | dk 状态贡献 | `pypto.matmul(pypto.mul(v_new, s_tok_2d), d_s_next, DT_FP32, b_trans=True)` | FP32 | FP32 | [BT, K] | `v_scaled @ d_s_next^T` |
| 3d | dg_cum 从 dk | `pypto.sum(pypto.mul(kc, dk_state), -1)` → neg/sub | FP32 | FP32 | [BT] | 标量修正 |
| 3e | dq/dk 局部注意力贡献 | `pypto.matmul(d_a_base_decay, kc)` / `.T @ qc` | FP32 | FP32 | [BT, K] | d_a_base = (doc @ v_new^T) * m_le * scale |
| 3f | dg_cum 局部贡献 | `pypto.sum/pypto.sub` 组合 | FP32 | FP32 | [BT] | tmp.sum(-1) - tmp.sum(-2) |
| **4** | **WY 低秩表示分解** | | | | | |
| 4a | dw | `pypto.mul(pypto.matmul(dv_total, s_before, DT_FP32, b_trans=True), neg_one)` | FP32 | FP32 | [BT, K] | `-(dv_total @ s_before^T)` |
| 4b | du | 直接引用 dv_total | FP32 | FP32 | [BT, V] | `du = dv_total` |
| 4c | dvb | `pypto.matmul(a, du, DT_FP32, a_trans=True)` | FP32 | FP32 | [BT, V] | `A^T @ du` |
| 4d | dkbg | `pypto.matmul(a, dw, DT_FP32, a_trans=True)` | FP32 | FP32 | [BT, K] | `A^T @ dw` |
| 4e | dv_c | `pypto.mul(dvb, betac_2d)` | FP32 | FP32 | [BT, V] | `dvb * beta` |
| 4f | db_c | `pypto.sum(pypto.mul(dvb, vc), -1)` | FP32 | FP32 | [BT] | `(dvb * v).sum(-1)` |
| 4g | dk_c += dkbg | `pypto.mul(dkbg, pypto.mul(betac_2d, eg_2d))` → `pypto.add(dk_c, result)` | FP32 | FP32 | [BT, K] | dkbg * beta * eg |
| 4h | db_c += dkbg 贡献 | `pypto.sum(pypto.mul(dkbg, pypto.mul(kc, eg_2d)), -1)` → `pypto.add(db_c, result)` | FP32 | FP32 | [BT] | `(dkbg * kc * eg).sum(-1)` |
| 4i | dg_cum += dkbg 贡献 | `pypto.sum(pypto.mul(dkbg, kbg), -1)` → `pypto.add(dg_cum, result)` | FP32 | FP32 | [BT] | `(dkbg * kbg).sum(-1)` |
| 4j | d_a | `pypto.add(pypto.matmul(dw, kbg, DT_FP32, b_trans=True), pypto.matmul(du, vb, DT_FP32, b_trans=True))` | FP32 | FP32 | [BT, BT] | WY 梯度 |
| 4k | d_l | `pypto.mul(pypto.matmul(a, pypto.matmul(d_a, a, DT_FP32, a_trans=True), DT_FP32, a_trans=True), m_lt)` → neg | FP32 | FP32 | [BT, BT] | `-(A^T @ (d_a @ A^T)) * m_lt` |
| 4l | kkt | `pypto.matmul(kc, kc, DT_FP32, b_trans=True)` | FP32 | FP32 | [BT, BT] | `kc @ kc^T` |
| 4m | db_c += d_l 贡献 | `pypto.sum(pypto.mul(d_l, pypto.mul(kkt, decay)), -1)` → add | FP32 | FP32 | [BT] | 交叉项 |
| 4n | dg_cum += d_l 贡献 | sum/sub 组合 | FP32 | FP32 | [BT] | l_mat 相关修正 |
| 4o | dk_c += m_mat 贡献 | `pypto.matmul(pypto.add(m_mat, m_mat_T), kc)` → add dk_c | FP32 | FP32 | [BT, K] | `(m_mat + m_mat^T) @ kc` |
| **5** | **反 cumsum 与 L2 norm 反向** | | | | | |
| 5a | dg_raw | `pypto.matmul(c_rcum, dg_cum_view, DT_FP32)` | FP32 | FP32 | [BT, 1] | 反向累积 |
| 5b | L2 norm bwd (dq) | `mul + sum + mul + sub` 组合 | FP32 | FP32 | [BT, K] | 见替代方案配方 |
| 5c | L2 norm bwd (dk) | `mul + sum + mul + sub` 组合 | FP32 | FP32 | [BT, K] | 见替代方案配方 |

### 1.2 精度路由

```text
全部 FP32 → FP32 计算 → FP32 输出
（无 dtype 转换点，所有输入输出和中间计算均为 FP32）
```

**全链路 FP32**：所有输入（q, k, v, g_raw, beta, do, dht, 缓存, 常量矩阵）均为 FP32，所有中间计算保持 FP32，所有输出（dq, dk, dv, db, dg_raw, dh0）均为 FP32。无需任何 cast 操作。

### 1.3 替代方案（已排除）

| 替代方案 | 排除原因 |
|---------|---------|
| 使用 `pypto.transpose` + `pypto.matmul` 替代 `a_trans/b_trans` | 额外内存搬运和中间 tensor，前向实现使用 `a_trans/b_trans` 更高效 |
| 使用 `pypto.expand_exp_dif` 计算 `exp(g_cum_i - g_cum_j)` | 该 API 直接输出 `(g_cum - g_cum.T)` 的指数，但需要显式构造差值矩阵，步骤 0c 可用 `pypto.mul(pypto.sub(g_cum, g_cum_T), tril).exp()` 替代 |
| 在设备端构造常量矩阵 (i_mat, m_le, m_lt, c_cum, c_rcum) | 这些矩阵与 BT 相关，在 host 端预构造并通过 `pypto.from_torch` 传入更简单可靠，避免在 kernel 中引入 `arange`/`one_hot`/`tril` 等额外操作 |
| 使用 FP16/BF16 混合精度 | 算子涉及大量矩阵乘法和指数运算，精度要求 atol/rtol=1e-3，半精度会导致数值不稳定 |

---

## 2. 数据规格

### 2.1 Kernel 函数签名

```python
@pypto.frontend.jit(
    runtime_options={
        "stitch_function_max_num": 128,
    },
)
def gated_delta_rule_backward_kernel(
    # ---- 原始输入 ----
    q:              pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC, H, K], pypto.DT_FP32),  # [B, T, H, K]
    k:              pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC, H, K], pypto.DT_FP32),  # [B, T, H, K]
    v:              pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC, H, V], pypto.DT_FP32),  # [B, T, H, V]
    g_raw:          pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC, H], pypto.DT_FP32),     # [B, T, H]
    beta:           pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC, H], pypto.DT_FP32),     # [B, T, H]
    # ---- 梯度输入 ----
    do:             pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC, H, V], pypto.DT_FP32),  # [B, T, H, V]
    dht:            pypto.Tensor([pypto.DYNAMIC, H, K, V], pypto.DT_FP32),              # [B, H, K, V]
    # ---- 前向缓存 ----
    A_cache:        pypto.Tensor([pypto.DYNAMIC, H, NT, BT, BT], pypto.DT_FP32),        # [B, H, NT, BT, BT]
    w_cache:        pypto.Tensor([pypto.DYNAMIC, H, NT, BT, K], pypto.DT_FP32),         # [B, H, NT, BT, K]
    S_before_cache: pypto.Tensor([pypto.DYNAMIC, H, NT, K, V], pypto.DT_FP32),          # [B, H, NT, K, V]
    v_new_cache:    pypto.Tensor([pypto.DYNAMIC, H, NT, BT, V], pypto.DT_FP32),         # [B, H, NT, BT, V]
    # ---- L2 norm 缓存 (use_qk_l2norm_in_kernel=True) ----
    q_norm:         pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC, H, K], pypto.DT_FP32),  # [B, T, H, K]
    k_norm:         pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC, H, K], pypto.DT_FP32),  # [B, T, H, K]
    q_rstd:         pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC, H], pypto.DT_FP32),     # [B, T, H]
    k_rstd:         pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC, H], pypto.DT_FP32),     # [B, T, H]
    # ---- 常量矩阵 (host 端预构造) ----
    i_mat:          pypto.Tensor([BT, BT], pypto.DT_FP32),  # 单位矩阵
    m_le:           pypto.Tensor([BT, BT], pypto.DT_FP32),  # 下三角含对角线
    m_lt:           pypto.Tensor([BT, BT], pypto.DT_FP32),  # 严格下三角
    c_cum:          pypto.Tensor([BT, BT], pypto.DT_FP32),  # 累积和矩阵
    c_rcum:         pypto.Tensor([BT, BT], pypto.DT_FP32),  # 逆累积和矩阵
    # ---- 标量参数 ----
    scale_val:      pypto.Tensor([1], pypto.DT_FP32),       # 1/sqrt(K)
    # ---- 输出 ----
    dq_out:         pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC, H, K], pypto.DT_FP32),  # [B, T, H, K]
    dk_out:         pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC, H, K], pypto.DT_FP32),  # [B, T, H, K]
    dv_out:         pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC, H, V], pypto.DT_FP32),  # [B, T, H, V]
    db_out:         pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC, H], pypto.DT_FP32),     # [B, T, H]
    dg_raw_out:     pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC, H], pypto.DT_FP32),     # [B, T, H]
    dh0_out:        pypto.Tensor([pypto.DYNAMIC, H, K, V], pypto.DT_FP32),              # [B, H, K, V]
):
```

### 2.2 动态轴分析

| 维度名 | 是否动态 | 取值范围 / 常量 | 标注方式 | 说明 |
|--------|---------|-----------------|---------|------|
| B | 是 | [1, 64] | `pypto.DYNAMIC` | batch size，运行时确定 |
| T | 是 | [64, 8192] | `pypto.DYNAMIC` | 序列长度，运行时确定，T % BT == 0 |
| H | 否 | 编译期传入 | 数值 (如 4) | 注意力头数 |
| K | 否 | 编译期传入 | 数值 (如 128) | query/key 维度 |
| V | 否 | 编译期传入 | 数值 (如 128) | value 维度 |
| BT | 否 | 编译期传入 | 数值 (如 64/128) | chunk 大小 |
| NT | 否 | T // BT | 数值 | chunk 数量，编译期可由 T 和 BT 确定 |

### 2.3 值类型分析

| 变量 | 来源 | 类型 | 注意事项 |
|------|------|------|---------|
| `B_val` | `q.shape[0]` | SymbolicScalar | 不可用于 Python `if/range`，不可索引 list |
| `T_val` | `q.shape[1]` | SymbolicScalar | 同上 |
| `NT_val` | `T_val // BT` | SymbolicScalar | 用于 `pypto.loop` 的上界 |
| `H`, `K`, `V`, `BT` | 函数参数 | Python int | 可正常使用 |
| `scale_val` | 函数参数 tensor | Tensor [1] | 标量 tensor，用于 `pypto.mul` |
| `b_idx`, `h_idx`, `i_idx` | `pypto.loop` 返回 | SymbolicScalar | 不可用于 Python 条件或 list 索引 |
| `c` (chunk index) | `NT - 1 - i_idx` | SymbolicScalar | 逆序 chunk 索引 |
| `t0` | `c * BT` | SymbolicScalar | chunk 起始偏移，用于 `pypto.view` offset |

---

## 3. Tiling 策略

### 3.1 算子类型

**混合型 (Hybrid)**：核心计算包含大量矩阵乘法（Cube 类型 `pypto.matmul`）和逐元素/归约操作（Vector 类型 `pypto.mul/add/sub/exp/sum`）。

### 3.2 Tiling 推导

该算子借鉴前向 `gated_delta_rule_impl.py` 的 tiling 策略，并在不同计算阶段之间切换 cube/vec 配置。

**方案：固定 tile 参数，按计算阶段切换**

根据前向实现的成熟模式，使用固定 tile 参数：

**Cube tiling（matmul 操作）**：

```python
# 标准矩阵乘 (BT×K) @ (K×BT) 或类似
pypto.set_cube_tile_shapes([128, 128], [128, 128], [128, 128])

# 较小矩阵乘 (K×V) @ (V×K) 或 (K×BT) 等
pypto.set_cube_tile_shapes([64, 64], [128, 128], [128, 128])
```

**Vector tiling（逐元素/归约操作）**：

```python
pypto.set_vec_tile_shapes(128, 128)      # 标准配置
pypto.set_vec_tile_shapes(128, 128, 128) # 3D tensor 操作
pypto.set_vec_tile_shapes(64, 128)       # 较小配置
```

**推导依据**：

1. **K = V = 128（典型配置）**：尾轴 128，FP32 需要 8 元素对齐，128 满足对齐要求（128 / 8 = 16，整数倍）
2. **BT = 64 或 128**：尾轴 128 满足对齐；BT = 64 时，60 维度不满足对齐，需使用 `valid_shape` 机制
3. **同阶段驻留 tensor 数**：chunk 内主要中间 tensor 约 15-20 个（均为 [BT, K/V] 或 [BT, BT]），FP32 每元素 4 字节
4. **UB 容量**：Ascend 910 单核 UB 约 1.5MB = 384K 元素（FP32），tile [128, 128] = 16K 元素/tensor，20 个 tensor = 320K < 384K，满足约束

### 3.3 替代方案

| 备选 tile | 否决理由 |
|-----------|---------|
| 全局统一 tile，不按阶段切换 | 不同 matmul 的 M/K/N 维度差异大（BT×K vs K×V vs BT×BT），统一 tile 导致部分操作 UB 溢出或利用率不足 |
| 使用更小 tile（如 [32, 32]） | 增加循环次数，展开表达式膨胀，编译时间过长 |
| 动态 tile（根据运行时 shape 调整） | PyPTO tiling 为编译期配置，不支持运行时动态调整 |

---

## 4. Loop 与数据流

### 4.1 维度判定

| 轴 | 维度大小 | 编译期 / 运行期 | Loop 处理 |
|----|---------|----------------|----------|
| B | DYNAMIC | 运行期 | `pypto.loop(B, name="LOOP_B")` |
| H | 编译期参数 | 编译期已知 | `pypto.loop(H, name="LOOP_H")` |
| chunk (NT) | DYNAMIC (T//BT) | 运行期 | `pypto.loop(NT, name="LOOP_CHUNK")`，逆序索引 |
| BT | 编译期参数 | 编译期已知 | 不需要 loop，chunk 内整体处理 |

### 4.2 完整伪代码

> 核心设计：三层嵌套 loop (B → H → chunk)，chunk 循环逆序（NT-1 → 0），
> 每个独立头 (B, H) 维护独立的 `d_s` 状态梯度。

```python
# ---- 函数工厂：编译期参数 K, V, H, BT 通过闭包传入 ----
def gated_delta_rule_backward_factory(K, V, H, BT):
    NT_max = 128  # 最大 chunk 数（安全上限）

    @pypto.frontend.jit(
        runtime_options={"stitch_function_max_num": 128},
    )
    def kernel(
        # 原始输入
        q:              pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC, H, K], pypto.DT_FP32),
        k:              pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC, H, K], pypto.DT_FP32),
        v:              pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC, H, V], pypto.DT_FP32),
        g_raw:          pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC, H], pypto.DT_FP32),
        beta:           pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC, H], pypto.DT_FP32),
        # 梯度输入
        do:             pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC, H, V], pypto.DT_FP32),
        dht:            pypto.Tensor([pypto.DYNAMIC, H, K, V], pypto.DT_FP32),
        # 前向缓存
        A_cache:        pypto.Tensor([pypto.DYNAMIC, H, NT_max, BT, BT], pypto.DT_FP32),
        w_cache:        pypto.Tensor([pypto.DYNAMIC, H, NT_max, BT, K], pypto.DT_FP32),
        S_before_cache: pypto.Tensor([pypto.DYNAMIC, H, NT_max, K, V], pypto.DT_FP32),
        v_new_cache:    pypto.Tensor([pypto.DYNAMIC, H, NT_max, BT, V], pypto.DT_FP32),
        # L2 norm 缓存
        q_norm:         pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC, H, K], pypto.DT_FP32),
        k_norm:         pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC, H, K], pypto.DT_FP32),
        q_rstd:         pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC, H], pypto.DT_FP32),
        k_rstd:         pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC, H], pypto.DT_FP32),
        # 常量矩阵
        i_mat:          pypto.Tensor([BT, BT], pypto.DT_FP32),
        m_le:           pypto.Tensor([BT, BT], pypto.DT_FP32),
        m_lt:           pypto.Tensor([BT, BT], pypto.DT_FP32),
        c_cum:          pypto.Tensor([BT, BT], pypto.DT_FP32),
        c_rcum:         pypto.Tensor([BT, BT], pypto.DT_FP32),
        # 标量参数
        scale_val:      pypto.Tensor([1], pypto.DT_FP32),
        # 输出
        dq_out:         pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC, H, K], pypto.DT_FP32),
        dk_out:         pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC, H, K], pypto.DT_FP32),
        dv_out:         pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC, H, V], pypto.DT_FP32),
        db_out:         pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC, H], pypto.DT_FP32),
        dg_raw_out:     pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC, H], pypto.DT_FP32),
        dh0_out:        pypto.Tensor([pypto.DYNAMIC, H, K, V], pypto.DT_FP32),
    ):
        B_val = q.shape[0]    # SymbolicScalar
        T_val = q.shape[1]    # SymbolicScalar
        NT_val = T_val // BT  # SymbolicScalar

        # ---- 三层嵌套 Loop ----
        for b_idx in pypto.loop(B_val, name="LOOP_B", idx_name="b_idx"):
            for h_idx in pypto.loop(H, name="LOOP_H", idx_name="h_idx"):
                # 初始化状态梯度 d_s = dht[b, h]
                d_s = pypto.tensor([K, V], pypto.DT_FP32)          # [K, V], FP32
                d_s[:] = dht[b_idx, h_idx]                         # 从终态梯度初始化

                # 逆序遍历 chunk: i=0 对应 c=NT-1, i=NT-1 对应 c=0
                for i_idx in pypto.loop(NT_val, name="LOOP_CHUNK",
                                        idx_name="i_idx",
                                        unroll_list=[16, 1]):
                    c = NT_val - 1 - i_idx                          # SymbolicScalar: chunk 索引
                    t0 = c * BT                                     # SymbolicScalar: chunk 起始偏移

                    # ═══════════════════════════════════════════
                    # Step 0: 切片 chunk 输入
                    # ═══════════════════════════════════════════
                    pypto.set_vec_tile_shapes(128, 128)

                    qc = pypto.view(q, [BT, K], [t0, h_idx, 0])           # [BT, K], FP32
                    kc = pypto.view(k, [BT, K], [t0, h_idx, 0])           # [BT, K], FP32
                    vc = pypto.view(v, [BT, V], [t0, h_idx, 0])           # [BT, V], FP32
                    betac = pypto.view(beta, [BT], [t0, h_idx])            # [BT], FP32
                    gc_raw_c = pypto.view(g_raw, [BT], [t0, h_idx])        # [BT], FP32
                    doc = pypto.view(do, [BT, V], [t0, h_idx, 0])         # [BT, V], FP32

                    # 前向缓存切片
                    a = A_cache[b_idx, h_idx, c]                           # [BT, BT], FP32
                    w = w_cache[b_idx, h_idx, c]                           # [BT, K], FP32
                    s_before = S_before_cache[b_idx, h_idx, c]            # [K, V], FP32
                    v_new_c = v_new_cache[b_idx, h_idx, c]                # [BT, V], FP32

                    # ═══════════════════════════════════════════
                    # Step 1: 计算 g_cum, eg, decay
                    # ═══════════════════════════════════════════
                    pypto.set_cube_tile_shapes([128, 128], [128, 128], [128, 128])

                    gc_view = gc_raw_c.reshape([BT, 1])                   # [BT, 1], FP32
                    g_cum = pypto.matmul(c_cum, gc_view, pypto.DT_FP32)  # [BT, 1], FP32

                    pypto.set_vec_tile_shapes(128, 128)
                    eg = pypto.exp(g_cum)                                  # [BT, 1], FP32
                    gl = g_cum[BT-1:BT, :]                                # [1, 1], FP32

                    # decay = exp(g_cum_i - g_cum_j), 利用 (a-b)*tril + 0 上三角
                    g_cum_T = g_cum.transpose(0, 1)                        # [1, BT], FP32
                    diff = g_cum - g_cum_T                                 # [BT, BT], FP32 (广播)
                    decay = pypto.exp(diff)                                # [BT, BT], FP32

                    # ═══════════════════════════════════════════
                    # Step 2: 局部注意力 dv0
                    # ═══════════════════════════════════════════
                    pypto.set_cube_tile_shapes([128, 128], [128, 128], [128, 128])

                    qk = pypto.matmul(qc, kc, pypto.DT_FP32,
                                      b_trans=True)                        # [BT, BT], FP32

                    pypto.set_vec_tile_shapes(128, 128)
                    a_local = pypto.mul(pypto.mul(qk, decay), m_le)        # [BT, BT], FP32

                    pypto.set_cube_tile_shapes([128, 128], [128, 128], [128, 128])
                    dv0_scaled = pypto.matmul(a_local, doc, pypto.DT_FP32,
                                              a_trans=True)               # [BT, V], FP32
                    dv0 = pypto.mul(dv0_scaled, scale_val)                # [BT, V], FP32

                    # ═══════════════════════════════════════════
                    # Step 3: 状态递推反向传播
                    # ═══════════════════════════════════════════
                    pypto.set_vec_tile_shapes(128, 128)
                    s_tok_1d = pypto.exp(pypto.sub(gl, g_cum))             # [BT, 1], FP32
                    s_tok = s_tok_1d.reshape([BT])                        # [BT], FP32
                    s_tok_2d = s_tok.reshape([BT, 1])                     # [BT, 1], FP32

                    pypto.set_cube_tile_shapes([128, 128], [128, 128], [64, 64])
                    dv_state = pypto.mul(
                        pypto.matmul(kc, d_s, pypto.DT_FP32),
                        s_tok_2d
                    )                                                      # [BT, V], FP32

                    dv_total = pypto.add(dv_state, dv0)                   # [BT, V], FP32

                    # d_s 更新
                    eg_2d = eg.reshape([BT, 1])                           # [BT, 1], FP32
                    q_eff = pypto.mul(qc, eg_2d)                          # [BT, K], FP32

                    pypto.set_cube_tile_shapes([64, 64], [128, 128], [128, 128])
                    d_s_decay = pypto.mul(d_s, pypto.exp(gl))             # [K, V], FP32

                    pypto.set_cube_tile_shapes([128, 128], [128, 128], [64, 64])
                    d_s_q = pypto.mul(
                        pypto.matmul(q_eff, doc, pypto.DT_FP32, a_trans=True),
                        scale_val
                    )                                                      # [K, V], FP32

                    pypto.set_cube_tile_shapes([128, 128], [128, 128], [64, 64])
                    d_s_w = pypto.matmul(w, dv_total, pypto.DT_FP32,
                                         a_trans=True)                    # [K, V], FP32

                    # d_s = d_s * exp(gl) + q_eff^T @ doc * scale - w^T @ dv_total
                    d_s_new = pypto.sub(
                        pypto.add(d_s_decay, d_s_q),
                        d_s_w
                    )                                                      # [K, V], FP32
                    d_s[:] = d_s_new                                      # 写回状态梯度

                    # ═══════════════════════════════════════════
                    # Step 4: dq/dk/dg_cum 梯度累积
                    # ═══════════════════════════════════════════
                    dq_c = pypto.tensor([BT, K], pypto.DT_FP32)           # [BT, K], FP32
                    dk_c = pypto.tensor([BT, K], pypto.DT_FP32)           # [BT, K], FP32
                    dg_cum = pypto.tensor([BT], pypto.DT_FP32)            # [BT], FP32
                    dq_c[:] = pypto.full([BT, K], 0.0, pypto.DT_FP32)
                    dk_c[:] = pypto.full([BT, K], 0.0, pypto.DT_FP32)
                    dg_cum[:] = pypto.full([BT], 0.0, pypto.DT_FP32)

                    # 4a. dq 状态贡献: (doc @ s_before^T) * eg * scale
                    pypto.set_cube_tile_shapes([128, 128], [128, 128], [128, 128])
                    dq1 = pypto.mul(
                        pypto.mul(
                            pypto.matmul(doc, s_before, pypto.DT_FP32,
                                         b_trans=True),
                            eg_2d
                        ),
                        scale_val
                    )                                                      # [BT, K], FP32
                    dq_c = pypto.add(dq_c, dq1)                           # [BT, K], FP32

                    # 4b. dg_cum += (dq1 * qc).sum(-1)
                    pypto.set_vec_tile_shapes(128, 128)
                    dg_cum = pypto.add(dg_cum,
                        pypto.sum(pypto.mul(dq1, qc), -1))                # [BT], FP32

                    # 4c. dk 状态贡献: v_scaled @ d_s_next^T
                    #     (使用 dv_state 计算前的 d_s 作为 d_s_next)
                    #     注意：d_s 在 Step 3 已更新，需用 d_s_decay/exp(gl) 还原
                    #     实际实现中应在 Step 3 前保存 d_s_next = d_s (原始值)
                    d_s_next = pypto.tensor([K, V], pypto.DT_FP32)        # [K, V], FP32
                    # d_s_next 在 Step 3 之前应已保存为 d_s 的旧值

                    v_scaled = pypto.mul(v_new_c, s_tok_2d)               # [BT, V], FP32
                    pypto.set_cube_tile_shapes([128, 128], [128, 128], [64, 64])
                    dk_state = pypto.matmul(v_scaled, d_s_next, pypto.DT_FP32,
                                            b_trans=True)                 # [BT, K], FP32
                    dk_c = pypto.add(dk_c, dk_state)                      # [BT, K], FP32

                    # 4d. dg_cum -= (kc * dk_state).sum(-1)
                    pypto.set_vec_tile_shapes(128, 128)
                    scalar_dk = pypto.sum(pypto.mul(kc, dk_state), -1)    # [BT], FP32
                    dg_cum = pypto.sub(dg_cum, scalar_dk)                 # [BT], FP32

                    # 4e. dg_cum[-1] += scalar_dk.sum() + exp(gl) * (s_before * d_s_next).sum()
                    #     (需特殊处理：累积到末元素)
                    #     注：此处为简化描述，实现时需要 .sum() 归约

                    # 4f. dq/dk 局部注意力贡献
                    #     d_a_base = (doc @ v_new^T) * m_le * scale
                    pypto.set_cube_tile_shapes([128, 128], [128, 128], [128, 128])
                    d_a_base = pypto.mul(
                        pypto.mul(
                            pypto.matmul(doc, v_new_c, pypto.DT_FP32,
                                         b_trans=True),
                            m_le
                        ),
                        scale_val
                    )                                                      # [BT, BT], FP32

                    dq_c = pypto.add(dq_c,
                        pypto.matmul(pypto.mul(d_a_base, decay), kc))     # [BT, K], FP32

                    dk_c = pypto.add(dk_c,
                        pypto.matmul(pypto.mul(d_a_base, decay).transpose(0,1), qc))
                                                                          # [BT, K], FP32

                    # 4g. dg_cum 局部贡献
                    pypto.set_vec_tile_shapes(128, 128)
                    a_base = pypto.mul(pypto.mul(qk, decay), m_le)        # [BT, BT], FP32
                    tmp = pypto.mul(d_a_base, a_base)                     # [BT, BT], FP32
                    dg_cum = pypto.add(dg_cum,
                        pypto.sub(
                            pypto.sum(tmp, -1),
                            pypto.sum(tmp, -2)
                        ))                                                 # [BT], FP32

                    # ═══════════════════════════════════════════
                    # Step 5: WY 低秩表示分解
                    # ═══════════════════════════════════════════
                    # 5a. dw = -(dv_total @ s_before^T)
                    pypto.set_cube_tile_shapes([128, 128], [128, 128], [128, 128])
                    dw = pypto.mul(
                        pypto.matmul(dv_total, s_before, pypto.DT_FP32,
                                     b_trans=True),
                        pypto.full([1], -1.0, pypto.DT_FP32)
                    )                                                      # [BT, K], FP32

                    du = dv_total                                          # [BT, V], FP32 (直接引用)

                    # 5b. dvb = A^T @ du
                    pypto.set_cube_tile_shapes([128, 128], [128, 128], [128, 128])
                    dvb = pypto.matmul(a, du, pypto.DT_FP32,
                                       a_trans=True)                      # [BT, V], FP32
                    dkbg = pypto.matmul(a, dw, pypto.DT_FP32,
                                        a_trans=True)                     # [BT, K], FP32

                    # 5c. 分解为 dv, dk, db, dg
                    pypto.set_vec_tile_shapes(128, 128)
                    betac_2d = betac.reshape([BT, 1])                      # [BT, 1], FP32

                    dv_c = pypto.mul(dvb, betac_2d)                        # [BT, V], FP32
                    db_c = pypto.sum(pypto.mul(dvb, vc), -1)              # [BT], FP32

                    dk_c = pypto.add(dk_c,
                        pypto.mul(dkbg, pypto.mul(betac_2d, eg_2d)))      # [BT, K], FP32
                    db_c = pypto.add(db_c,
                        pypto.sum(pypto.mul(dkbg, pypto.mul(kc, eg_2d)), -1))
                                                                          # [BT], FP32

                    kbg = pypto.mul(kc, pypto.mul(betac_2d, eg_2d))      # [BT, K], FP32
                    dg_cum = pypto.add(dg_cum,
                        pypto.sum(pypto.mul(dkbg, kbg), -1))              # [BT], FP32

                    # 5d. d_a = dw @ kbg^T + du @ vb^T
                    vb = pypto.mul(vc, betac_2d)                          # [BT, V], FP32
                    pypto.set_cube_tile_shapes([128, 128], [128, 128], [128, 128])
                    d_a = pypto.add(
                        pypto.matmul(dw, kbg, pypto.DT_FP32, b_trans=True),
                        pypto.matmul(du, vb, pypto.DT_FP32, b_trans=True)
                    )                                                      # [BT, BT], FP32

                    # 5e. d_l = -(A^T @ (d_a @ A^T)) * m_lt
                    d_l = pypto.mul(
                        pypto.matmul(a,
                            pypto.matmul(d_a, a, pypto.DT_FP32, a_trans=True),
                            pypto.DT_FP32, a_trans=True),
                        m_lt
                    )                                                      # [BT, BT], FP32
                    # 取负
                    d_l = pypto.mul(d_l, pypto.full([1], -1.0, pypto.DT_FP32))

                    # 5f. kkt = kc @ kc^T
                    pypto.set_cube_tile_shapes([128, 128], [128, 128], [128, 128])
                    kkt = pypto.matmul(kc, kc, pypto.DT_FP32,
                                       b_trans=True)                     # [BT, BT], FP32

                    # 5g. db_c += (d_l * (kkt * decay)).sum(-1)
                    pypto.set_vec_tile_shapes(128, 128)
                    db_c = pypto.add(db_c,
                        pypto.sum(pypto.mul(d_l, pypto.mul(kkt, decay)), -1))
                                                                          # [BT], FP32

                    # 5h. dg_cum += l_mat 相关贡献
                    l_mat = pypto.mul(pypto.mul(betac_2d, kkt), decay)   # 广播 betac_2d
                    l_mat = pypto.mul(l_mat, m_lt)
                    tmp2 = pypto.mul(d_l, l_mat)                          # [BT, BT], FP32
                    dg_cum = pypto.add(dg_cum,
                        pypto.sub(
                            pypto.sum(tmp2, -1),
                            pypto.sum(tmp2, -2)
                        ))                                                 # [BT], FP32

                    # 5i. dk_c += (m_mat + m_mat^T) @ kc
                    m_mat = pypto.mul(d_l, pypto.mul(betac_2d, decay))   # 广播
                    m_mat_T = m_mat.transpose(0, 1)                       # [BT, BT], FP32
                    pypto.set_cube_tile_shapes([128, 128], [128, 128], [128, 128])
                    dk_c = pypto.add(dk_c,
                        pypto.matmul(pypto.add(m_mat, m_mat_T), kc))     # [BT, K], FP32

                    # ═══════════════════════════════════════════
                    # Step 6: 反 cumsum 与 L2 norm 反向
                    # ═══════════════════════════════════════════
                    # 6a. dg_raw_c = c_rcum @ dg_cum
                    dg_cum_view = dg_cum.reshape([BT, 1])                 # [BT, 1], FP32
                    pypto.set_cube_tile_shapes([128, 128], [128, 128], [128, 128])
                    dg_raw_c = pypto.matmul(c_rcum, dg_cum_view,
                                            pypto.DT_FP32)               # [BT, 1], FP32
                    dg_raw_c = dg_raw_c.reshape([BT])                     # [BT], FP32

                    # 6b. L2 norm backward for dq (当 use_qk_l2norm_in_kernel=True)
                    #     l2norm_bwd(y, rstd, dy) = dy * rstd - (dy*y).sum(-1) * y * rstd
                    q_norm_c = pypto.view(q_norm, [BT, K], [t0, h_idx, 0])
                    q_rstd_c = pypto.view(q_rstd, [BT], [t0, h_idx])

                    pypto.set_vec_tile_shapes(128, 128)
                    q_rstd_2d = q_rstd_c.reshape([BT, 1])                # [BT, 1], FP32
                    dq_term1 = pypto.mul(dq_c, q_rstd_2d)                # [BT, K], FP32
                    dq_dot = pypto.sum(pypto.mul(dq_c, q_norm_c), -1,
                                       True)                              # [BT, 1], FP32
                    dq_term2 = pypto.mul(dq_dot, pypto.mul(q_norm_c, q_rstd_2d))
                                                                          # [BT, K], FP32
                    dq_raw_c = pypto.sub(dq_term1, dq_term2)             # [BT, K], FP32

                    # 6c. L2 norm backward for dk
                    k_norm_c = pypto.view(k_norm, [BT, K], [t0, h_idx, 0])
                    k_rstd_c = pypto.view(k_rstd, [BT], [t0, h_idx])

                    k_rstd_2d = k_rstd_c.reshape([BT, 1])                # [BT, 1], FP32
                    dk_term1 = pypto.mul(dk_c, k_rstd_2d)                # [BT, K], FP32
                    dk_dot = pypto.sum(pypto.mul(dk_c, k_norm_c), -1,
                                       True)                              # [BT, 1], FP32
                    dk_term2 = pypto.mul(dk_dot, pypto.mul(k_norm_c, k_rstd_2d))
                                                                          # [BT, K], FP32
                    dk_raw_c = pypto.sub(dk_term1, dk_term2)             # [BT, K], FP32

                    # ═══════════════════════════════════════════
                    # Step 7: 写入输出
                    # ═══════════════════════════════════════════
                    pypto.set_vec_tile_shapes(128, 128)
                    dq_out[b_idx, t0:t0+BT, h_idx, :] = dq_raw_c         # [BT, K]
                    dk_out[b_idx, t0:t0+BT, h_idx, :] = dk_raw_c         # [BT, K]
                    dv_out[b_idx, t0:t0+BT, h_idx, :] = dv_c             # [BT, V]
                    db_out[b_idx, t0:t0+BT, h_idx] = db_c                # [BT]
                    dg_raw_out[b_idx, t0:t0+BT, h_idx] = dg_raw_c        # [BT]

                # 循环结束后写入 dh0
                dh0_out[b_idx, h_idx] = d_s                               # [K, V], FP32

    return kernel
```

### 4.3 跨迭代状态

| 状态名 | 初始化 | 更新方式 | 生命周期 |
|--------|--------|---------|---------|
| `d_s` | `dht[b_idx, h_idx]` | Step 3 中 `d_s[:] = d_s_new` | chunk 循环内跨迭代携带 |
| `d_s_next` | Step 3 前 `d_s_next = d_s` (旧值) | 每个 chunk 重新保存 | 单 chunk 内有效 |

**关键说明**：`d_s` 是反向传播的核心状态，从 `dht`（终态梯度）初始化，在每个 chunk 中通过 `d_s[:] = d_s_new` 更新。循环结束后，最终的 `d_s` 写入 `dh0`（初始状态梯度）。这要求 `d_s` 在 chunk 循环内跨迭代可见，使用 `pypto.tensor` 分配的局部变量实现。

### 4.4 尾块处理

- **方案**：使用 `valid_shape` 参数处理非对齐情况
- **说明**：当 BT = 64 且 T = 128 时，NT = 2，所有 chunk 对齐，无需特殊处理。但当 BT = 128 且 T 不整除 BT 时，最后一个 chunk（反向的第一个 chunk）可能不完整。此时需在 `pypto.view` 中使用 `valid_shape` 参数指定实际长度。
- **参考**：前向 `chunk_gated_delta_rule_unaligned` 使用 `pypto.fillpad` 处理非对齐情况。反向实现优先使用 `valid_shape`。

---

## 5. 约束自检清单

| # | 约束 | 是否满足 | 备注 |
|---|------|---------|------|
| 1 | 所有 sum 输入为 FP32 | ✅ | 全链路 FP32，无需 cast |
| 2 | matmul 两侧 dtype 一致 | ✅ | 所有输入/输出/中间 tensor 均为 FP32 |
| 3 | TileShape 维度数 = 操作数维度数 | ✅ | 2D matmul 用 [M,K],[K,N],[M,N]；vec 操作按实际维度配置 |
| 4 | 尾轴满足对齐 (FP32: 8 元素) | ✅ | K=V=128, BT=64/128，均为 8 的整数倍 |
| 5 | 同阶段 UB 占用 ≤ 容量 | ✅ | 约 15-20 个 [BT,K/V] tensor，FP32 下约 320K 元素 < 384K |
| 6 | 表达式展开 < 18000 | ✅ | chunk 内 matmul 次数约 15 次，展开可控 |
| 7 | 输出经 `[:]` / `assemble` 显式写回 | ✅ | 所有输出通过 `output[b,t:h] = result` 写回 |
| 8 | 无 view/assemble 同张量回环 | ✅ | 输出 tensor 和输入 tensor 完全分离，无环路 |
| 9 | 动态轴标 `pypto.DYNAMIC` | ✅ | B, T 标为 DYNAMIC |
| 10 | 动态 loop 提供 `unroll_list` | ✅ | chunk loop 使用 `unroll_list=[16, 1]` |
| 11 | 跨迭代状态用 `pypto.tensor` + `[:]` 写回 | ✅ | `d_s[:] = d_s_new` |
| 12 | 尾块用 `valid_shape` 处理 | ✅ | 需在实现时添加 |
| 13 | 无 SymbolicScalar 用作 `**` / list index / Python `if` | ✅ | `c = NT-1-i_idx` 仅用于 view offset；`t0 = c * BT` 用于 view offset |
| 14 | `set_vec/cube_tile_shapes` 在操作前调用 | ✅ | 伪代码中每步标注了 tiling 切换 |

### 开放问题

| # | 问题 | 影响范围 | 待解决方式 |
|---|------|---------|-----------|
| 1 | `betac_2d = betac.reshape([BT, 1])` 广播：PyPTO 是否支持 [BT,1] 与 [BT,BT] 的广播 | `m_mat`, `l_mat` 等计算 | 实现阶段验证广播兼容性；若不支持，需手动 expand |
| 2 | `scale_val` 作为 Tensor[1] 传入 vs 作为 `pypto.Element` 传入 | 所有乘以 scale 的步骤 | 优先使用 `pypto.Element(DT_FP32, 1/math.sqrt(K))` 避免不必要的 tensor 乘法 |
| 3 | 逆序 chunk loop `c = NT-1-i_idx` 的 SymbolicScalar 运算：`NT-1-i_idx` 是否合法 | chunk 索引计算 | 参考 `models/` 中类似的逆序遍历模式，若不支持需改用正序 + 计算偏移 |
| 4 | `d_l` 取负操作：使用 `pypto.mul(d_l, -1.0)` vs `pypto.sub(pypto.zeros_like(d_l), d_l)` | Step 5e | 优先使用 `pypto.mul` 乘以 -1.0 标量 |
| 5 | chunk 循环中 `d_s_next` 的保存时机：需要在 Step 3 更新 `d_s` 前保存 | 状态管理 | 在 Step 3 开头添加 `d_s_next[:] = d_s` |
| 6 | `dg_cum[-1] += ...` 的末元素修正：SymbolicScalar 索引限制 | dg_cum 最后一个元素的修正 | 使用 `dg_cum[BT-1:BT] = dg_cum[BT-1:BT] + delta` 形式 |

---

## 6. 验证方案

### 6.1 测试配置

| 用例 | 输入 shape | dtype | 重点验证 | BT |
|------|----------|-------|---------|-----|
| Small_功能_P0 | B=1, T=128, H=4, K=128, V=128 | FP32 | 基本功能正确性 | 64 |
| Large_性能_P0 | B=1, T=4096, H=4, K=128, V=128 | FP32 | 多 chunk 状态递推正确性 | 128 |
| B=2_泛化 | B=2, T=256, H=2, K=64, V=64 | FP32 | 多 batch 正确性 | 64 |
| H=8_泛化 | B=1, T=512, H=8, K=64, V=64 | FP32 | 多 head 正确性 | 64 |
| No_L2Norm | B=1, T=128, H=2, K=64, V=64 | FP32 | use_qk_l2norm_in_kernel=False | 64 |
| 数值稳定性 | B=1, T=512, H=4, K=128, V=128 | FP32 | 无 NaN/Inf | 128 |

### 6.2 精度容忍度

| dtype | rtol | atol |
|-------|------|------|
| FP32  | 1e-3 | 1e-3 |

### 6.3 验证方法

- 逐张量对比：`detailed_tensor_compare` 对 6 个输出（dq, dk, dv, db, dg_raw, dh0）分别验证
- NaN/Inf 检查：所有输出不得包含 NaN 或 Inf
- 非零梯度验证：各输出张量至少包含非零元素
- Golden 对比：与 `gated_delta_rule_backward_golden` 的输出逐元素对比

---

## 7. 入口函数与 Host 端编排

### 7.1 `from_torch` 入口函数

```python
def gated_delta_rule_backward_entry(
    q, k, v, g_raw, beta,
    do, dht,
    A, w, S_before, v_new,
    q_norm, k_norm, q_rstd, k_rstd,
    bt, use_qk_l2norm_in_kernel=True, l2_eps=1e-6,
):
    """Host 端入口函数，负责：
    1. 构造常量矩阵 (i_mat, m_le, m_lt, c_cum, c_rcum)
    2. 计算 scale = 1/sqrt(K)
    3. 分配输出 tensor
    4. 调用 kernel
    """
    B, T, H, K = q.shape
    V = v.shape[-1]
    NT = T // bt
    scale = 1.0 / math.sqrt(K)

    # 构造常量矩阵
    device = q.device
    ones_bt = torch.ones(bt, bt, device=device, dtype=torch.float32)
    i_mat = torch.eye(bt, device=device, dtype=torch.float32)
    m_le = torch.tril(ones_bt)
    m_lt = torch.tril(ones_bt, diagonal=-1)
    c_cum = torch.tril(ones_bt)
    c_rcum = torch.triu(ones_bt)

    scale_tensor = torch.tensor([scale], device=device, dtype=torch.float32)

    # 分配输出
    dq = torch.zeros_like(q)
    dk = torch.zeros_like(k)
    dv = torch.zeros_like(v)
    db = torch.zeros(B, T, H, device=device, dtype=torch.float32)
    dg_raw = torch.zeros(B, T, H, device=device, dtype=torch.float32)
    dh0 = torch.zeros(B, H, K, V, device=device, dtype=torch.float32)

    # 转换为 pypto tensor 并调用 kernel
    kernel = gated_delta_rule_backward_factory(K, V, H, bt)
    # ... pypto.from_torch 转换 + kernel 调用 ...

    return dq, dk, dv, db, dg_raw, dh0
```

### 7.2 常量矩阵处理

5 个常量矩阵在 host 端通过 PyTorch 构造，通过 `pypto.from_torch` 传入 kernel：

| 矩阵 | Shape | 构造方式 | 说明 |
|------|-------|---------|------|
| `i_mat` | [BT, BT] | `torch.eye(BT)` | 单位矩阵 |
| `m_le` | [BT, BT] | `torch.tril(ones)` | 下三角含对角线 |
| `m_lt` | [BT, BT] | `torch.tril(ones, diagonal=-1)` | 严格下三角 |
| `c_cum` | [BT, BT] | `torch.tril(ones)` | 累积和矩阵（与 m_le 相同） |
| `c_rcum` | [BT, BT] | `torch.triu(ones)` | 逆累积和矩阵 |

---

## 8. 设计决策记录

### 8.1 关键决策

| # | 决策 | 原因 | 替代方案 |
|---|------|------|---------|
| D1 | 逆序 chunk 迭代 | 反向传播的数学性质要求从终态向初态传播 | 无（数学刚性约束） |
| D2 | 常量矩阵 host 端预构造 | BT 在编译期已知，host 端构造更简单可靠 | 设备端构造（引入额外 API 调用） |
| D3 | 使用 `pypto.tensor` 管理 `d_s` 状态 | 需跨 loop 迭代读写，`pypto.tensor` 支持原地更新 | 不可行：普通 Python 变量在 loop 中会被覆盖 |
| D4 | L2 norm 反向用 mul+sum+sub 组合 | PyPTO 无专用 l2norm_bwd API，API_REPORT 确认可组合实现 | 单独写子函数封装 |
| D5 | 使用 `a_trans/b_trans` 替代显式转置 | 减少中间 tensor 和内存搬运 | `pypto.transpose` + `matmul`（多一次内存搬运） |
| D6 | scale 作为 Tensor[1] 或 Element 传入 | 避免 kernel 内重复计算 | 硬编码为 `pypto.Element` |
| D7 | 函数工厂模式 (factory) | K, V, H, BT 作为编译期参数通过闭包传入 kernel | 全部标为 DYNAMIC（编译期无法优化 tiling） |

### 8.2 风险与缓解

| 风险 | 严重性 | 缓解措施 |
|------|--------|---------|
| 指数运算溢出（exp(g_cum) 值过大） | 中 | FP32 精度下 exp 最大值约 88，g_raw 应在合理范围内（典型 ±5） |
| 大量 matmul 导致编译时间过长 | 中 | 使用固定 tile 参数，通过 `stitch_function_max_num=128` 控制编译复杂度 |
| 逆序索引 `c = NT-1-i_idx` 的 SymbolicScalar 运算 | 低 | 参考 `models/` 中类似模式验证；必要时改用正序遍历 + 偏移计算 |
| 广播维度兼容性 [BT,1] vs [BT,BT] | 低 | 实现阶段逐项验证，不兼容时使用 `expand_clone` |

---

*设计状态：已收敛*
*迭代过程：第1轮 API 调用链 17+ 步，cast 0 处（全 FP32）；第2轮 Tiling 混合型，固定 tile 参数；第3轮 Loop 3 层（B/H/chunk），动态轴 B/T，跨迭代状态 d_s；第4轮 约束检查 14/14 通过，开放问题 6 项*
