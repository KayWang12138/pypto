# MX Scaled MatMul 介绍

## 1. 背景

在 `pypto` 中，`scaled_mm` 不是直接做 `A @ B`，而是先对输入做分块缩放，再进行矩阵乘：

\[
\text{out} = (A \odot S_A) @ (B \odot S_B) + \text{bias(optional)}
\]

其中：

- `A`、`B`：输入矩阵（通常是低精度，如 FP8）。
- `S_A`、`S_B`：缩放张量（`scale_a`、`scale_b`）。
- `\odot`：逐元素乘法。
- `out`：输出矩阵，数据类型由配置决定（`FP16/BF16/FP32`）。

这种设计能提升低精度计算链路下的数值稳定性和精度表现。

## 2. 接口语义

典型接口如下：

```python
scaled_mm(
    mat_a,
    mat_b,
    out_dtype,
    scale_a,
    scale_b,
    *,
    a_trans=False,
    b_trans=False,
    scale_a_trans=False,
    scale_b_trans=False,
    extend_params=None,
)
```

语义上可理解为：

1. 将压缩存储的 `scale_a` 和 `scale_b` 展开为与 `A`、`B` 对齐的缩放矩阵。
2. 计算 `A_scaled = A * scaleAExpanded`。
3. 计算 `B_scaled = B * scaleBExpanded`。
4. 计算 `out = A_scaled @ B_scaled`。
5. 若提供了 bias，则执行 bias 加法。

## 3. 为什么 scale 是 3D

`scale_a` 和 `scale_b` 使用 3D 压缩格式存储：

- `scale_a`: `[M, K/64, 2]`（或其转置形式）
- `scale_b`: `[K/64, N, 2]`（或其转置形式）

最后一维 `2` 表示：`K` 轴每 64 个元素由 2 个 scale 值描述。
经过 reshape 后会变为 `K/32`，再按 32 重复，恢复到完整 `K` 轴缩放。

## 4. Scale 展开规则（与测试参考对齐）

以下规则参考 `python/tests/st/test_dynamic_mxmatmul_with_onboard.py`：

### 4.1 `scale_a`

- `scale_a_trans=False`:
  - `scale_a_tmp = scale_a.view(M, K/32)`
- `scale_a_trans=True`:
  - `scale_a_tmp = torch.transpose(scale_a, -2, -1).reshape(K/32, M).T`
- 展开到 `[M, K]`：
  - `scale_a_full = repeat(scale_a_tmp, axis=1, times=32)`

### 4.2 `scale_b`

- `scale_b_trans=False`:
  - `scale_b_tmp = torch.transpose(scale_b, -2, -1).reshape(K/32, N)`
- `scale_b_trans=True`:
  - `scale_b_tmp = scale_b.view(N, K/32).T`
- 展开到 `[K, N]`：
  - `scale_b_full = repeat(scale_b_tmp, axis=0, times=32)`

## 5. 与普通 MatMul 的区别

- 普通 matmul：
  - `out = A @ B`
- MX scaled matmul：
  - `out = (A * scale_a_full) @ (B * scale_b_full) [+ bias]`

关键差异不在矩阵乘本身，而在 scale 的正确还原与应用。

## 6. 常见 Shape 约束（2D 场景）

设实际参与 matmul 的矩阵形状为：

- `A`: `[M, K]`
- `B`: `[K, N]`

常见约束包括：

- `K % 64 == 0`
- `scale_a` 与 `scale_b` 必须能一致映射到 `K/64`
- scale 展开后必须满足：
  - `scale_a_full`: `[M, K]`
  - `scale_b_full`: `[K, N]`

## 7. 端到端计算流程

1. 解析 `A/B`、转置标志、`scale_a/scale_b` 及其转置标志。
2. 按 MX 规则展开 `scale_a/scale_b`。
3. 对 `A`、`B` 执行逐元素缩放。
4. 对缩放后的矩阵执行 matmul。
5. 可选执行 bias 加法。
6. 转换/写回目标输出 dtype。

## 8. 常见问题与易错点

- `scale_b_trans=False` 时，`scale_b` 的 transpose/reshape 顺序写错。
- 忽略最后一维 `2` 的含义，未先变成 `K/32` 就直接 repeat。
- 输入矩阵 trans 标志与 scale trans 标志不匹配。
- 调试时误按普通 `A @ B` 语义理解 `scaled_mm`。

## 9. 总结

MX scaled matmul 本质是“带缩放恢复的低精度 matmul”流程：

- 先根据分块 scale 对输入做恢复缩放；
- 再执行常规 matmul；
- scale 展开是否正确，是结果正确性与精度表现的关键。
