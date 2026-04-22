---
schema_version: "2.1"
op_name: "causal_conv1d"
status: draft
last_updated: "2026-04-21"

# 关键接口契约
compute_kind: "vector"
dtypes: ["fp16"]
dynamic_axes: ["total_len", "batch", "dim", "num_cache_lines"]
precision: { rtol: 0.01, atol: 0.01 }
---

# causal_conv1d 设计方案

## 1. 计算图与精度路由

### 1.1 API 调用序列

#### Prefill 模式

| 步骤 | 操作 | PyPTO API | 输入 dtype | 输出 dtype | 输出 shape | 备注 |
|------|------|-----------|------------|------------|-----------|------|
| 1 | 加载权重 w | `pypto.from_torch(weight)` | FP16 | FP16 | [width, dim] | 编译期固定 |
| 2 | Cast 权重 | `pypto.cast(w, pypto.DT_FP32)` | FP16 | FP32 | [width, dim] | 精度计算需要 |
| 3 | 初始化历史 | `pypto.view(conv_state, ...)` | FP16 | FP16 | [width-1, block_D] | 从缓存或历史 token 加载 |
| 4 | Cast 历史 | `pypto.cast(hist, pypto.DT_FP32)` | FP16 | FP32 | [width-1, block_D] | 精度计算需要 |
| 5 | 加载当前 token | `pypto.view(x, ...)` | FP16 | FP16 | [1, block_D] | view 当前 tile |
| 6 | Cast token | `pypto.cast(x_cur, pypto.DT_FP32)` | FP16 | FP32 | [1, block_D] | 精度计算需要 |
| 7 | 初始化累加器 | `pypto.zeros([block_M, block_D], pypto.DT_FP32)` | - | FP32 | [block_M, block_D] | 累加 buffer |
| 8a | w[0] * hist[0] | `pypto.mul(w_view[0], hist_view[0])` | FP32 | FP32 | [block_D] | 逐元素乘 |
| 8b | acc += ... | `pypto.add(acc, result)` | FP32 | FP32 | [block_D] | 累加 |
| 8c | ... | (重复 width-1 次) | FP32 | FP32 | [block_D] | 历史权重乘加 |
| 8d | w[n] * x_cur | `pypto.mul(w_view[width-1], x_cur)` | FP32 | FP32 | [block_D] | 当前 token 乘权重 |
| 8e | acc += ... | `pypto.add(acc, result)` | FP32 | FP32 | [block_D] | 最终累加 |
| 9a | -acc | `pypto.mul(acc, -1.0)` | FP32 | FP32 | [block_M, block_D] | 手动实现 silu |
| 9b | exp(-acc) | `pypto.exp(neg_acc)` | FP32 | FP32 | [block_M, block_D] | exp 操作 |
| 9c | 1 + exp | `pypto.add(exp_neg, ones)` | FP32 | FP32 | [block_M, block_D] | 分母计算 |
| 9d | acc / denom | `pypto.div(acc, denom)` | FP32 | FP32 | [block_M, block_D] | silu 输出 |
| 10 | Cast 输出 | `pypto.cast(out, pypto.DT_FP16)` | FP32 | FP16 | [block_M, block_D] | 输出 dtype |
| 11 | 写回输出 | `pypto.assemble(out_fp16, [offsets], y)` | FP16 | FP16 | [total_len, dim] | 写回全局输出 |
| 12 | 更新状态 | `pypto.view + pypto.assemble` | FP16 | FP16 | [width-1, dim] | 滚动更新 conv_state |

#### Decode 模式

| 步骤 | 操作 | PyPTO API | 输入 dtype | 输出 dtype | 输出 shape | 备注 |
|------|------|-----------|------------|------------|-----------|------|
| 1 | 加载权重 w | `pypto.from_torch(weight)` | FP16 | FP16 | [width, dim] | 编译期固定 |
| 2 | Cast 权重 | `pypto.cast(w, pypto.DT_FP32)` | FP16 | FP32 | [width, dim] | 精度计算需要 |
| 3 | 加载历史 | `pypto.view(conv_state, ...)` | FP16 | FP16 | [width-1, block_D] | 从 offset 位置加载 |
| 4 | Cast 历史 | `pypto.cast(hist, pypto.DT_FP32)` | FP16 | FP32 | [width-1, block_D] | 精度计算需要 |
| 5 | 加载当前 token | `pypto.view(x, ...)` | FP16 | FP16 | [seqlen, block_D] | 投机解码支持 seqlen>1 |
| 6 | Cast token | `pypto.cast(x_cur, pypto.DT_FP32)` | FP16 | FP32 | [seqlen, block_D] | 粰度计算需要 |
| 7-11 | 卷积计算 + silu | (同 Prefill 步骤 7-10) | FP32 | FP16 | [seqlen, block_D] | 循环处理每个 token |
| 12 | 写回输出 | `pypto.assemble(out_fp16, [offsets], y)` | FP16 | FP16 | [batch, seqlen, dim] | 写回全局输出 |
| 13 | 滚动更新 conv_state | `pypto.view + pypto.assemble` | FP16 | FP16 | [width-1, dim] | 滚动存储新 token |

### 1.2 精度路由

```text
输入(FP16) → [步骤 2: cast] → 权重(FP32)
             [步骤 4: cast] → 历史(FP32)
             [步骤 6: cast] → token(FP32)
             → FP32 计算链(累加 + silu)
             → [步骤 10: cast] → 输出(FP16)
```

| 转换位置 | 转换方向 | 原因 |
|---------|---------|------|
| 步骤 2, 4, 6 | FP16 → FP32 | 累加和 silu 需要高精度,避免精度损失 |
| 步骤 10 | FP32 → FP16 | 输出需要与输入 dtype 一致 |

### 1.3 替代方案（已排除）

| 替代方案 | 排除原因 |
|---------|---------|
| 使用 `pypto.sigmoid` 实现 silu | sigmoid 仅支持 FP32,需要额外 cast,不如手动 exp/div 实现高效 |
| 直接在 FP16 下计算 | silu 激活涉及 exp 操作,FP16 精度不足可能导致数值不稳定 |
| 使用 FP32 输入输出 | SPEC 要求 FP16,且模型推理通常使用 FP16 |

---

## 2. 数据规格

### 2.1 Kernel 函数签名

#### Prefill 模式 Kernel

```python
@pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.NPU})
def causal_conv1d_prefill_kernel(
    x: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC], pypto.DT_FP16),           # [total_len, dim]
    weight: pypto.Tensor([width, pypto.DYNAMIC], pypto.DT_FP16),              # [width, dim], width 是编译期常量
    conv_state: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC, pypto.DYNAMIC], pypto.DT_FP16),  # [num_cache, state_len, dim]
    cu_seqlens: pypto.Tensor([pypto.DYNAMIC], pypto.DT_INT32),                 # [batch+1]
    y: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC], pypto.DT_FP16),           # [total_len, dim]
    activation: str = "silu",
    width: int = 4,
):
    ...
```

#### Decode 模式 Kernel

```python
@pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.NPU})
def causal_conv1d_decode_kernel(
    x: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC, pypto.DYNAMIC], pypto.DT_FP16),  # [batch, seqlen, dim]
    weight: pypto.Tensor([width, pypto.DYNAMIC], pypto.DT_FP16),                   # [width, dim]
    conv_state: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC, pypto.DYNAMIC], pypto.DT_FP16),  # [num_cache, state_len, dim]
    y: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC, pypto.DYNAMIC], pypto.DT_FP16),  # [batch, seqlen, dim]
    activation: str = "silu",
    width: int = 4,
):
    ...
```

### 2.2 动态轴分析

| 维度名 | 是否动态 | 取值范围 / 常量 | 标注方式 |
|--------|---------|-----------------|---------|
| total_len | 是 | [1, 8192] | `pypto.DYNAMIC` |
| batch | 是 | [1, 256] | `pypto.DYNAMIC` |
| dim | 是 | [256, 4096] | `pypto.DYNAMIC` |
| num_cache_lines | 是 | [1, 1024] | `pypto.DYNAMIC` |
| width | 否 | [3, 6] (默认 4) | 编译期常量 |
| state_len | 否 | >= width-1 | 编译期常量或从 conv_state.shape 推断 |
| seqlen (Decode) | 是 | [1, 4] (投机解码) | `pypto.DYNAMIC` |

### 2.3 值类型分析（避免 SymbolicScalar 误用）

| 变量 | 来源 | 类型 | 注意事项 |
|------|------|------|---------|
| `total_len` | `x.shape[0]`（Prefill） | SymbolicScalar | 不可用于 Python `if/range`,不可索引 list |
| `batch_size` | `cu_seqlens.shape[0] - 1`（Prefill） | SymbolicScalar | 必须用 `pypto.loop` 遍历 |
| `dim` | `x.shape[-1]` | SymbolicScalar | 用于 view shape 参数时需用常量 tile |
| `seqlen` | `x.shape[1]`（Decode） | SymbolicScalar | 投机解码时可能 > 1 |
| `width` | 编译期参数 | int | 可用于 Python range 和 list 索引 |
| `block_M` | Tile 配置参数 | int | 常量,用于 view shape |
| `block_D` | Tile 配置参数 | int | 常量,用于 view shape |
| `b_offset` | `batch_idx * ...` | SymbolicScalar | 用于 view offset 参数 |
| `d_offset` | `dim_idx * block_D` | SymbolicScalar | 用于 view offset 参数 |
| `-1.0` | 字面量 | float | 用于 `pypto.mul` 标量乘法 |

---

## 3. Tiling 策略

### 3.1 算子类型

**Vector**（纯逐元素操作）

判断依据：
- 计算仅涉及逐元素 mul/add 和激活函数 exp/div
- 无矩阵乘法 matmul
- 每个 token 独立计算（除历史依赖外）

### 3.2 Tiling 推导

#### Prefill 模式 Tiling

- **同时驻留 UB 的 Tensor**：

| Tensor | 用途 | shape | dtype | 大小估算 |
|--------|------|-------|-------|---------|
| w_tile | 权重 tile | [width, block_D] | FP32 | width * 512 * 4B = 8KB (width=4) |
| hist_buffer | 历史缓存 | [width-1, block_D] | FP32 | 3 * 512 * 4B = 6KB |
| x_cur | 当前 token | [block_M, block_D] | FP32 | 64 * 512 * 4B = 128KB |
| acc | 累加器 | [block_M, block_D] | FP32 | 64 * 512 * 4B = 128KB |
| neg_acc | silu 中间变量 | [block_M, block_D] | FP32 | 64 * 512 * 4B = 128KB |
| exp_neg | silu 中间变量 | [block_M, block_D] | FP32 | 64 * 512 * 4B = 128KB |
| denom | silu 分母 | [block_M, block_D] | FP32 | 64 * 512 * 4B = 128KB |
| out_fp16 | 输出 tile | [block_M, block_D] | FP16 | 64 * 512 * 2B = 64KB |
| **总计** | | | | **~710KB** |

- **推导步骤**：
  1. 尾轴对齐：FP16 → 16 元素对齐, FP32 → 8 元素对齐
  2. UB 预算：710KB < UB 容量（通常 1-2MB）→ ✓ 满足
  3. 展开约束：`(total_len / 64) * (dim / 512) * 8 ≈ 2048/64 * 2048/512 * 8 = 32 * 4 * 8 = 1024 < 18000` → ✓ 满足

- **最终 tile**：

```python
# Prefill 模式：按 dim 和 seqlen 分块
pypto.set_vec_tile_shapes(64, 512)  # [block_M, block_D]
```

#### Decode 模式 Tiling

- **同时驻留 UB 的 Tensor**：

| Tensor | 用途 | shape | dtype | 大小估算 |
|--------|------|-------|-------|---------|
| w_tile | 权重 tile | [width, block_D] | FP32 | 4 * 512 * 4B = 8KB |
| hist_buffer | 历史缓存 | [width-1, block_D] | FP32 | 3 * 512 * 4B = 6KB |
| x_cur | 当前 token | [seqlen, block_D] | FP32 | 4 * 512 * 4B = 8KB (投机解码 seqlen=4) |
| acc | 累加器 | [seqlen, block_D] | FP32 | 4 * 512 * 4B = 8KB |
| silu 中间变量 | exp/div 操作 | [seqlen, block_D] | FP32 | 3 * 8KB = 24KB |
| out_fp16 | 输出 tile | [seqlen, block_D] | FP16 | 4 * 512 * 2B = 4KB |
| **总计** | | | | **~58KB** |

- **推导步骤**：
  1. 尾轴对齐：FP16 → 16 元素对齐, FP32 → 8 元素对齐
  2. UB 预算：58KB < UB 容量 → ✓ 满足
  3. 展开约束：`(batch * seqlen) * (dim / 512) ≈ 256 * 4 * 8 = 8192 < 18000` → ✓ 满足

- **最终 tile**：

```python
# Decode 模式：按 dim 分块
pypto.set_vec_tile_shapes(4, 512)  # [seqlen_max, block_D]
```

### 3.3 替代方案

| 备选 tile | 否决理由 |
|-----------|---------|
| `set_vec_tile_shapes(1, 512)` (Decode) | seqlen=1 时不支持投机解码扩展 |
| `set_vec_tile_shapes(128, 1024)` (Prefill) | UB 容量不足,silu 中间变量过多 |
| `set_vec_tile_shapes(32, 256)` (Prefill) | 展开次数过多,影响性能 |

---

## 4. Loop 与数据流

### 4.1 维度判定

#### Prefill 模式

| 轴 | 维度大小 | 编译期 / 运行期 | Loop 处理 |
|----|---------|----------------|----------|
| batch | DYNAMIC | 运行期 | `pypto.loop(batch_size, name="batch")` |
| seqlen (每个 batch) | DYNAMIC | 运行期 | 分块处理,`pypto.loop(seqlen_num, name="seq_block")` |
| dim | DYNAMIC | 运行期 | `pypto.loop(dim_num, name="dim_block")` |
| width | 3-6 | 编译期已知 | Python `for` 循环 |
| block_M | 64 | 编译期常量 | Python `for` 或内部循环 |
| block_D | 512 | 编译期常量 | Tile shape 参数 |

#### Decode 模式

| 轴 | 维度大小 | 编译期 / 运行期 | Loop 处理 |
|----|---------|----------------|----------|
| batch | DYNAMIC | 运行期 | `pypto.loop(batch, name="batch")` |
| seqlen | DYNAMIC | 运行期 | Python `for` (seqlen 通常较小 1-4) |
| dim | DYNAMIC | 运行期 | `pypto.loop(dim_num, name="dim_block")` |
| width | 3-6 | 编译期已知 | Python `for` 循环 |

### 4.2 完整伪代码

#### Prefill 模式伪代码

```python
@pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.NPU})
def causal_conv1d_prefill_kernel(
    x: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC], pypto.DT_FP16),           # [total_len, dim]
    weight: pypto.Tensor([width, dim], pypto.DT_FP16),                        # [width, dim], width=4 是编译期常量
    conv_state: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC, pypto.DYNAMIC], pypto.DT_FP16),  # [num_cache, state_len, dim]
    cu_seqlens: pypto.Tensor([pypto.DYNAMIC], pypto.DT_INT32),                 # [batch+1]
    y: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC], pypto.DT_FP16),           # [total_len, dim]
    activation: str = "silu",
    width: int = 4,
):
    # ========== 参数与常量 ========== 
    hist_len = width - 1  # Python int (编译期常量)
    block_M = 64  # Python int (tile size)
    block_D = 512  # Python int (tile size)
    
    # ========== 动态轴提取 ========== 
    total_len = x.shape[0]  # SymbolicScalar
    dim = x.shape[1]  # SymbolicScalar
    batch_size = cu_seqlens.shape[0] - 1  # SymbolicScalar
    
    dim_num = (dim + block_D - 1) // block_D  # Python int (展开次数)
    
    # ========== Tiling 配置 ========== 
    pypto.set_vec_tile_shapes(block_M, block_D)
    
    # ========== 权重预处理 ========== 
    w_fp32 = pypto.cast(weight, pypto.DT_FP32)  # [width, dim], FP32
    
    # ========== Batch 循环 ========== 
    for b_idx in pypto.loop(batch_size, name="batch", idx_name="b_idx"):
        seq_start = cu_seqlens[b_idx]  # SymbolicScalar (从 cu_seqlens view)
        seq_end = cu_seqlens[b_idx + 1]  # SymbolicScalar
        seqlen = seq_end - seq_start  # SymbolicScalar
        
        seqlen_num = (seqlen + block_M - 1) // block_M  # Python int
        
        # ========== Dim 分块循环 ========== 
        for d_idx in pypto.loop(dim_num, name="dim_block", idx_name="d_idx"):
            d_offset = d_idx * block_D  # SymbolicScalar
            
            # 加载权重 tile
            w_tile = pypto.view(w_fp32, [width, block_D], [0, d_offset])  # [width, block_D], FP32
            
            # ========== Seq 分块循环 ========== 
            for s_idx in pypto.loop(seqlen_num, name="seq_block", idx_name="s_idx"):
                s_offset = s_idx * block_M  # SymbolicScalar
                actual_block_m = min(block_M, seqlen - s_offset)  # Python int (尾部块)
                
                # ========== 初始化历史缓冲 ========== 
                if s_idx == 0:  # 首个 block,从 conv_state 加载
                    hist_fp16 = pypto.view(conv_state, [hist_len, block_D], [b_idx, 0, d_offset])  # [hist_len, block_D], FP16
                    hist_fp32 = pypto.cast(hist_fp16, pypto.DT_FP32)  # [hist_len, block_D], FP32
                else:  # 后续 block,从历史 token 加载
                    hist_fp32 = pypto.tensor([hist_len, block_D], pypto.DT_FP32, "hist_buffer")
                    for h in range(hist_len):  # Python for (hist_len 是常量)
                        hist_src_offset = seq_start + s_offset - hist_len + h  # SymbolicScalar
                        hist_token = pypto.view(x, [1, block_D], [hist_src_offset, d_offset])  # [1, block_D], FP16
                        hist_fp32[h, :] = pypto.cast(hist_token, pypto.DT_FP32)  # [1, block_D], FP32
                
                # ========== 处理 block 内的 token ========== 
                acc = pypto.tensor([actual_block_m, block_D], pypto.DT_FP32, "acc")  # 累加器
                
                for t_offset in range(actual_block_m):  # Python for (actual_block_m 是 Python int)
                    t_global = seq_start + s_offset + t_offset  # SymbolicScalar
                    
                    # 加载当前 token
                    x_cur_fp16 = pypto.view(x, [1, block_D], [t_global, d_offset])  # [1, block_D], FP16
                    x_cur_fp32 = pypto.cast(x_cur_fp16, pypto.DT_FP32)  # [1, block_D], FP32
                    
                    # ========== 卷积计算 ========== 
                    acc_token = pypto.zeros([1, block_D], pypto.DT_FP32)  # [1, block_D], FP32
                    
                    for w_idx in range(hist_len):  # Python for (hist_len 是常量)
                        w_val = pypto.view(w_tile, [1, block_D], [w_idx, 0])  # [1, block_D], FP32
                        hist_val = pypto.view(hist_fp32, [1, block_D], [w_idx, 0])  # [1, block_D], FP32
                        prod = pypto.mul(w_val, hist_val)  # [1, block_D], FP32
                        acc_token = pypto.add(acc_token, prod)  # [1, block_D], FP32
                    
                    # 当前 token 权重
                    w_cur = pypto.view(w_tile, [1, block_D], [width - 1, 0])  # [1, block_D], FP32
                    prod_cur = pypto.mul(w_cur, x_cur_fp32)  # [1, block_D], FP32
                    acc_token = pypto.add(acc_token, prod_cur)  # [1, block_D], FP32
                    
                    # ========== silu 激活 ========== 
                    if activation == "silu":
                        neg_acc = pypto.mul(acc_token, -1.0)  # [1, block_D], FP32
                        exp_neg = pypto.exp(neg_acc)  # [1, block_D], FP32
                        ones = pypto.full([1, block_D], 1.0, pypto.DT_FP32)  # [1, block_D], FP32
                        denom = pypto.add(exp_neg, ones)  # [1, block_D], FP32
                        out_fp32 = pypto.div(acc_token, denom)  # [1, block_D], FP32
                    else:
                        out_fp32 = acc_token
                    
                    out_fp16 = pypto.cast(out_fp32, pypto.DT_FP16)  # [1, block_D], FP16
                    
                    # 写回输出
                    pypto.assemble(out_fp16, [t_global, d_offset], y)  # y[t_global, d_offset]
                    
                    # ========== 滚动历史 ========== 
                    for h in range(hist_len - 1):  # Python for
                        hist_fp32[h, :] = hist_fp32[h + 1, :]  # 滚动
                    hist_fp32[hist_len - 1, :] = x_cur_fp32  # 更新最新历史
                
                # ========== 更新 conv_state (最后一个 block) ========== 
                if s_idx == seqlen_num - 1:  # 最后一个 seq block
                    for pos in range(hist_len):  # Python for
                        last_token_offset = seq_start + seqlen - hist_len + pos  # SymbolicScalar
                        last_token_fp16 = pypto.view(x, [1, block_D], [last_token_offset, d_offset])  # [1, block_D], FP16
                        pypto.assemble(last_token_fp16, [b_idx, pos, d_offset], conv_state)  # conv_state[b_idx, pos, d_offset]
```

#### Decode 模式伪代码

```python
@pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.NPU})
def causal_conv1d_decode_kernel(
    x: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC, pypto.DYNAMIC], pypto.DT_FP16),  # [batch, seqlen, dim]
    weight: pypto.Tensor([width, dim], pypto.DT_FP16),  # [width, dim], width=4 是编译期常量
    conv_state: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC, pypto.DYNAMIC], pypto.DT_FP16),  # [num_cache, state_len, dim]
    y: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC, pypto.DYNAMIC], pypto.DT_FP16),  # [batch, seqlen, dim]
    activation: str = "silu",
    width: int = 4,
):
    # ========== 参数与常量 ========== 
    hist_len = width - 1  # Python int
    block_D = 512  # Python int
    
    # ========== 动态轴提取 ========== 
    batch = x.shape[0]  # SymbolicScalar
    seqlen = x.shape[1]  # SymbolicScalar
    dim = x.shape[2]  # SymbolicScalar
    state_len = conv_state.shape[1]  # SymbolicScalar
    
    dim_num = (dim + block_D - 1) // block_D  # Python int
    
    # ========== Tiling 配置 ========== 
    pypto.set_vec_tile_shapes(seqlen, block_D)  # 支持 seqlen=1 或 seqlen=4
    
    # ========== 权重预处理 ========== 
    w_fp32 = pypto.cast(weight, pypto.DT_FP32)  # [width, dim], FP32
    
    # ========== Dim 分块循环 ========== 
    for d_idx in pypto.loop(dim_num, name="dim_block", idx_name="d_idx"):
        d_offset = d_idx * block_D  # SymbolicScalar
        
        # 加载权重 tile
        w_tile = pypto.view(w_fp32, [width, block_D], [0, d_offset])  # [width, block_D], FP32
        
        # ========== Batch 循环 ========== 
        for b_idx in pypto.loop(batch, name="batch", idx_name="b_idx"):
            state_token_offset = seqlen - 1  # SymbolicScalar
            
            # ========== 从 conv_state 加载历史 ========== 
            hist_fp32 = pypto.tensor([hist_len, block_D], pypto.DT_FP32, "hist_buffer")
            for h in range(hist_len):  # Python for
                src_idx = state_token_offset + h  # SymbolicScalar
                hist_src_fp16 = pypto.view(conv_state, [1, block_D], [b_idx, src_idx, d_offset])  # [1, block_D], FP16
                hist_fp32[h, :] = pypto.cast(hist_src_fp16, pypto.DT_FP32)  # [1, block_D], FP32
            
            # ========== 处理每个 token ========== 
            for t_idx in range(seqlen):  # Python for (seqlen 通常 1-4,用静态循环)
                # 加载当前 token
                x_cur_fp16 = pypto.view(x, [1, block_D], [b_idx, t_idx, d_offset])  # [1, block_D], FP16
                x_cur_fp32 = pypto.cast(x_cur_fp16, pypto.DT_FP32)  # [1, block_D], FP32
                
                # ========== 卷积计算 ========== 
                acc_token = pypto.zeros([1, block_D], pypto.DT_FP32)  # [1, block_D], FP32
                
                for w_idx in range(hist_len):  # Python for
                    w_val = pypto.view(w_tile, [1, block_D], [w_idx, 0])  # [1, block_D], FP32
                    hist_val = pypto.view(hist_fp32, [1, block_D], [w_idx, 0])  # [1, block_D], FP32
                    prod = pypto.mul(w_val, hist_val)  # [1, block_D], FP32
                    acc_token = pypto.add(acc_token, prod)  # [1, block_D], FP32
                
                # 当前 token 权重
                w_cur = pypto.view(w_tile, [1, block_D], [width - 1, 0])  # [1, block_D], FP32
                prod_cur = pypto.mul(w_cur, x_cur_fp32)  # [1, block_D], FP32
                acc_token = pypto.add(acc_token, prod_cur)  # [1, block_D], FP32
                
                # ========== silu 激活 ========== 
                if activation == "silu":
                    neg_acc = pypto.mul(acc_token, -1.0)  # [1, block_D], FP32
                    exp_neg = pypto.exp(neg_acc)  # [1, block_D], FP32
                    ones = pypto.full([1, block_D], 1.0, pypto.DT_FP32)  # [1, block_D], FP32
                    denom = pypto.add(exp_neg, ones)  # [1, block_D], FP32
                    out_fp32 = pypto.div(acc_token, denom)  # [1, block_D], FP32
                else:
                    out_fp32 = acc_token
                
                out_fp16 = pypto.cast(out_fp32, pypto.DT_FP16)  # [1, block_D], FP16
                
                # 写回输出
                pypto.assemble(out_fp16, [b_idx, t_idx, d_offset], y)  # y[b_idx, t_idx, d_offset]
                
                # ========== 滚动历史 ========== 
                for h in range(hist_len - 1):  # Python for
                    hist_fp32[h, :] = hist_fp32[h + 1, :]
                hist_fp32[hist_len - 1, :] = x_cur_fp32
            
            # ========== 滚动更新 conv_state ========== 
            # 保留中间状态
            if state_token_offset + 1 < state_len:
                state_keep1 = pypto.view(conv_state, [1, block_D], [b_idx, state_token_offset + 1, d_offset])  # [1, block_D], FP16
                pypto.assemble(state_keep1, [b_idx, 0, d_offset], conv_state)
            if state_token_offset + 2 < state_len:
                state_keep2 = pypto.view(conv_state, [1, block_D], [b_idx, state_token_offset + 2, d_offset])  # [1, block_D], FP16
                pypto.assemble(state_keep2, [b_idx, 1, d_offset], conv_state)
            
            # 写入新 token
            for t_idx in range(seqlen):  # Python for
                write_pos = 2 + t_idx  # Python int
                if write_pos < state_len:
                    new_token_fp16 = pypto.view(x, [1, block_D], [b_idx, t_idx, d_offset])  # [1, block_D], FP16
                    pypto.assemble(new_token_fp16, [b_idx, write_pos, d_offset], conv_state)
```

### 4.3 跨迭代状态

| 状态名 | 初始化 | 更新方式 | submit_before_loop |
|--------|--------|---------|--------------------|
| `hist_fp32` | 从 conv_state 或历史 token 加载 | 每个 token 后滚动更新 | 不需要（block 内部) |
| `acc_token` | 每个 token 初始化为 zeros | 单 token 累加,不跨迭代 | 不需要 |
| `conv_state` | 输入参数 | block 结束后批量更新 | 不需要（最后更新) |

### 4.4 尾块处理

- **Prefill 模式**：
  - seqlen 尾部：`actual_block_m = min(block_M, seqlen - s_offset)`
  - dim 尾部：`actual_block_d = min(block_D, dim - d_offset)`,使用 `valid_shape` 参数
  - 方案：`pypto.view(..., valid_shape=[actual_block_m, block_D])`

- **Decode 模式**：
  - dim 尾部：`actual_block_d = min(block_D, dim - d_offset)`
  - 方案：`valid_shape` 参数

---

## 5. Buffer 规划

### 5.1 Prefill 模式 Buffer 分配

| Buffer | 用途 | shape | dtype | UB/L1 | 大小 |
|--------|------|-------|-------|-------|------|
| `w_fp32` | 权重(全局) | [width, dim] | FP32 | L1 | width * dim * 4B |
| `w_tile` | 权重 tile | [width, block_D] | FP32 | UB | 8KB (width=4) |
| `hist_fp32` | 历史缓存 | [hist_len, block_D] | FP32 | UB | 6KB |
| `acc` | 累加器 | [block_M, block_D] | FP32 | UB | 128KB |
| `neg_acc` | silu 中间 | [block_M, block_D] | FP32 | UB | 128KB |
| `exp_neg` | silu 中间 | [block_M, block_D] | FP32 | UB | 128KB |
| `denom` | silu 分母 | [block_M, block_D] | FP32 | UB | 128KB |
| `out_fp16` | 输出 tile | [block_M, block_D] | FP16 | UB | 64KB |

### 5.2 Decode 模式 Buffer 分配

| Buffer | 用途 | shape | dtype | UB/L1 | 大小 |
|--------|------|-------|-------|-------|------|
| `w_fp32` | 权重(全局) | [width, dim] | FP32 | L1 | width * dim * 4B |
| `w_tile` | 权重 tile | [width, block_D] | FP32 | UB | 8KB |
| `hist_fp32` | 历史缓存 | [hist_len, block_D] | FP32 | UB | 6KB |
| `acc_token` | 单 token 累加 | [1, block_D] | FP32 | UB | 2KB |
| silu 中间变量 | exp/div 操作 | [1, block_D] | FP32 | UB | 6KB (3 个) |
| `out_fp16` | 输出 tile | [seqlen, block_D] | FP16 | UB | 4KB |

---

## 6. 数据流路径

### 6.1 Prefill 模式数据流

```
GM (x, weight, conv_state) → L1 (w_fp32)
                          ↓
                     UB (w_tile, hist_fp32)
                          ↓
               UB (x_cur_fp32) → acc_token
                          ↓
                silu 计算链 (neg_acc, exp_neg, denom)
                          ↓
                     UB (out_fp16)
                          ↓
                    GM (y, conv_state)
```

### 6.2 Decode 模式数据流

```
GM (x, weight, conv_state) → L1 (w_fp32)
                          ↓
                     UB (w_tile, hist_fp32)
                          ↓
               UB (x_cur_fp32) → acc_token
                          ↓
                silu 计算链 (neg_acc, exp_neg, denom)
                          ↓
                     UB (out_fp16)
                          ↓
                    GM (y, conv_state)
```

---

## 7. 状态管理策略

### 7.1 Prefill 模式状态管理

- **初始状态加载**：
  - 首个 block: 从 `conv_state[b_idx, 0..hist_len-1, d_offset]` 加载历史
  - 后续 block: 从 `x[seq_start + s_offset - hist_len .. s_offset - 1, d_offset]` 加载历史 token

- **状态滚动更新**：
  - 每个 token 处理后,历史 buffer 内滚动: `hist[h] = hist[h+1]`
  - 最新 token 加入: `hist[hist_len-1] = x_cur`

- **最终状态写入**：
  - 最后一个 block 结束后,将最后 `hist_len` 个 token 写回 `conv_state[b_idx, 0..hist_len-1, d_offset]`

### 7.2 Decode 模式状态管理

- **投机解码历史加载**：
  - 从 `conv_state[b_idx, state_token_offset + h, d_offset]` 加载历史
  - `state_token_offset = seqlen - 1` (投机解码时提前准备)

- **状态滚动更新**：
  - 每个 token 处理后,历史 buffer 内滚动
  - 处理完所有 seqlen 个 token 后,批量更新 `conv_state`

- **conv_state 滚动更新**：
  - 保留中间状态: `conv_state[b, 0] = conv_state[b, state_token_offset+1]`
  - 写入新 token: `conv_state[b, 2 + t_idx] = x[b, t_idx]`

---

## 8. 边界处理

### 8.1 变长序列边界

- **cu_seqlens 处理**：
  - 每个 batch 的序列边界: `seq_start = cu_seqlens[b_idx]`, `seq_end = cu_seqlens[b_idx+1]`
  - 序列长度: `seqlen = seq_end - seq_start`

- **跨 batch 处理**：
  - packed layout 下,不同 batch 的 token 拼接在 `x[total_len, dim]`
  - 通过 `cu_seqlens` 索引定位每个 batch 的 token 范围

### 8.2 尾部块填充

- **seqlen 尾部**：
  - Prefill: `actual_block_m = min(block_M, seqlen - s_offset)`
  - 使用 `valid_shape` 参数处理非对齐尾部

- **dim 尾部**：
  - `actual_block_d = min(block_D, dim - d_offset)`
  - 使用 `valid_shape` 参数

### 8.3 投机解码支持

- **seqlen > 1**：
  - Decode 模式支持 seqlen=1 (标准解码) 和 seqlen=4 (投机解码)
  - 历史加载 offset 自动调整: `state_token_offset = seqlen - 1`

- **状态管理调整**：
  - 投机解码时,需要保留 conv_state 中间状态
  - 新 token 写入位置: `conv_state[b, 2 + t_idx]`

---

## 9. 约束检查与开放问题

### 9.1 约束自检清单

| # | 约束 | 是否满足 | 备注 |
|---|------|---------|------|
| 1 | 所有累加和 silu 计算使用 FP32 | ✓ | 避免 FP16 精度问题 |
| 2 | 无 matmul 操作,不涉及 cube 配置 | ✓ | Vector 算子 |
| 3 | TileShape 维度数 = 操作数维度数 | ✓ | Prefill: [block_M, block_D], Decode: [seqlen, block_D] |
| 4 | 尾轴满足对齐(FP16: 16, FP32: 8) | ✓ | block_D=512,满足对齐 |
| 5 | 同阶段 UB 占用 ≤ 容量 | ✓ | Prefill ~710KB, Decode ~58KB |
| 6 | 表达式展开 < 18000 | ✓ | Prefill: 1024, Decode: 8192 |
| 7 | 输出经 assemble 显式写回 | ✓ | 使用 pypto.assemble |
| 8 | 无 view/assemble 同张量回环 | ✓ | conv_state 分开 view 和 assemble |
| 9 | 动态轴标 pypto.DYNAMIC | ✓ | total_len, batch, dim, seqlen |
| 10 | 动态 loop 提供 unroll_list | ⚠ | 待添加,建议 unroll_list=[32, 16, 8, 4, 2, 1] |
| 11 | 跨迭代状态用 submit_before_loop | ✓ | hist_fp32 在 block 内更新 |
| 12 | 尾块用 valid_shape 处理 | ✓ | 实际 block 大小用 valid_shape |
| 13 | 无 SymbolicScalar 用作 ** / list index / Python if | ✓ | 使用 pypto.view 和 SymbolicScalar.min() |

### 9.2 开放问题

| # | 问题 | 影响范围 | 待解决方式 |
|---|------|---------|-----------|
| 1 | cu_seqlens 的 view 索引 | Prefill batch 循环 | 需验证 cu_seqlens[b_idx] 返回 SymbolicScalar,用于 view offset |
| 2 | state_len 动态判断 | Decode 状态更新 | 需使用 SymbolicScalar 比较或确保 state_len 足够大 |
| 3 | unroll_list 配置 | 性能优化 | 需根据实际 batch_size 和 dim_num 范围配置 |
| 4 | activation 参数传递 | Kernel 灵活性 | Python if 判断 activation=="silu",编译期分支 |

---

## 10. 验证方案

### 10.1 测试配置

| 用例 | 输入 shape | dtype | 重点验证 |
|------|----------|-------|---------|
| Prefill_P0 | x[2048,2048], weight[4,2048], conv_state[1,3,2048] | FP16 | 核心 Prefill 场景,精度与性能 |
| Decode_P0 | x[1,1,2048], weight[4,2048], conv_state[1,3,2048] | FP16 | 核心 Decode 场景,状态更新 |
| Prefill_Varlen_P0 | x[2048,2048], cu_seqlens=[0,512,1024,1536,2048] | FP16 | 变长序列功能验证 |
| Decode_Speculative | x[1,4,2048], conv_state[1,5,2048] | FP16 | 投机解码支持,seqlen>1 |
| width_variants | x[64,128], weight[w,128], w=3,4,5,6 | FP16 | 不同 width 支持 |
| dim_edge | x[1024,256], x[1024,4096] | FP16 | dim 边界值,256 和 4096 |
| seqlen_edge | x[1,1024], x[8192,1024] | FP16 | seqlen 边界值,1 和 8192 |

### 10.2 精度容忍度

| dtype | rtol | atol |
|-------|------|------|
| FP16 | 0.01 | 0.01 |

### 10.3 验证重点

- **功能验证**：
  - Prefill 和 Decode 两种模式正确性
  - 变长序列支持
  - 投机解码支持
  - 不同 width 支持

- **精度验证**：
  - 与 golden 参考实现对比
  - silu 激活函数精度
  - conv_state 状态更新正确性

- **性能验证**：
  - 达到首跑精度成功性能的 2 倍
  - Prefill 和 Decode 不同场景性能

---

## 11. 实现提示

### 11.1 参考实现路径

| 参考类型 | 路径 | 可复用点 |
|---------|------|---------|
| 状态管理 | `models/qwen3_next/gated_delta_rule_impl.py` | 状态初始化、更新、变长处理 |
| SiLU 激活 | `examples/02_intermediate/basic_nn/ffn/ffn_module.py` | 手动 sigmoid 实现(FP16 支持) |
| 序列循环 | `examples/02_intermediate/controlflow/loop/loop.py` | view/assemble、valid_shape |
| 激活函数 | `examples/02_intermediate/operators/activation/activation.py` | SiLU/SwiGLU 参考 |

### 11.2 关键 API 文档

| API | 文档路径 |
|-----|----------|
| `pypto.loop` | `docs/api/controlflow/pypto-loop.md` |
| `pypto.view` | `docs/api/operation/pypto-view.md` |
| `pypto.assemble` | `docs/api/operation/pypto-assemble.md` |
| `pypto.mul` | `docs/api/operation/pypto-mul.md` |
| `pypto.add` | `docs/api/operation/pypto-add.md` |
| `pypto.exp` | `docs/api/operation/pypto-exp.md` |
| `pypto.div` | `docs/api/operation/pypto-div.md` |
| `pypto.cast` | `docs/api/operation/pypto-cast.md` |
| `pypto.set_vec_tile_shapes` | `docs/api/config/pypto-set_vec_tile_shapes.md` |

---

## 12. 完成报告

```text
设计状态：已收敛

迭代过程：
  第 1 轮：API 调用链 12 步(Prefill) / 13 步(Decode), cast 3 处(FP16→FP32→FP16)
  第 2 轮：Tiling Vector, tile = [64, 512](Prefill) / [4, 512](Decode)
  第 3 轮：Loop 3 层(batch, seq_block, dim_block), 动态轴 [total_len, batch, dim], 跨迭代依赖 无
  第 4 轮：约束检查 13/13 通过,开放问题 4 个

回退记录：无

开放问题：
  · cu_seqlens 的 SymbolicScalar 索引验证 — Prefill batch 循环
  · state_len 动态判断 — Decode 状态更新
  · unroll_list 配置优化 — 性能调优
  · activation 编译期分支 — Kernel 灵活性
```

---
*生成时间: 2026-04-21*
*设计版本: v2.1*