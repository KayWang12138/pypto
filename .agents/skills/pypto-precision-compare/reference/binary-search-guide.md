# PyPTO 算子精度问题定位操作指南

## 前提

- PyPTO kernel 与 golden（纯 PyTorch）实现同一计算逻辑
- 最终输出与 golden 对比存在精度差异
- 需要定位是哪个计算模块或哪个操作导致

## 操作步骤

### Step 1：确认问题

在 `main()` 中对比最终输出，确认哪个 tensor 不匹配：

```python
detailed_tensor_compare(pto_output, golden_output, 'output_name')
```

### Step 2：分析代码结构，划分检查点

分析 kernel 中的计算流程，按模块/函数调用边界划分检查点。优先选择有明确语义的位置（如 matmul 后、softmax 后）。

记录每个检查点的：
- 变量名与 shape
- 所属模块
- 在 kernel 中的行号

### Step 3：在 kernel 中添加一个中点检查点

**不要一次加多个检查点**，从中间位置开始，一次加一个。

**kernel 函数修改：**

1. 在参数列表中添加检查点 tensor（放在输入参数后、输出参数前）：

```python
cp_mid: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_FP32),
```

2. 在循环内，目标模块之后写入（使用整数索引，**不要用 range slice**）：

```python
cp_mid[nv_idx] = mid_result  # 整数索引，连续内存写入
```

**关键原则：**
- 检查点 tensor 第一维用循环变量做整数索引（如 `cp[nv_idx]`），确保连续内存写入
- 检查点 shape 需要能容纳每次循环的结果，通常为 `[num_loops, ...]`
- 不要用 `cp[offset:offset+l, idx:idx+1] = val` 这类 range slice 赋值

### Step 4：在 pypto_function 中创建并传递检查点

```python
cp_mid = torch.zeros([Nv, L, D], dtype=torch.float32, device=device)

input_tensors = [
    ...,  # 原有输入
    cp_mid,  # 检查点（插在原有位置）
    ...,  # 原有输出
]
```

### Step 5：编写 golden 检查点计算函数

复用 golden 已有的子函数，为特定 (b=0, h, c) 迭代计算中间值：

```python
def compute_golden_checkpoint(q, k, v, ..., cache, L, ...):
    for h in range(Nv):
        # 切片输入
        qc = q_used[bs_ofs:bs_ofs+L, nqk_idx, :]
        # ... 其他输入

        # 调用 golden 子函数
        mid_result = golden_sub_func(qc, kc, ...)
        golden_cp[h] = mid_result  # shape 与 pypto 检查点一致
    return golden_cp
```

**注意 golden 与 pypto 的 shape 对齐：**
- pypto 中 `[L, 1]` 的结果，golden 中可能是 `[L]`，需要 `unsqueeze(-1)` 对齐
- dtype 统一为 float32 对比

### Step 6：运行并分析

```bash
python3 your_op.py
```

对比检查点：

```python
max_diff = (pto_cp - golden_cp).abs().max().item()
match = torch.allclose(pto_cp, golden_cp, rtol=1e-3, atol=1e-3)
print(f"  [{'PASS' if match else 'FAIL'}] {desc}: max_diff={max_diff:.6e}")
```

### Step 7：二分定位

根据结果决定下一步：

| 检查点结果 | 结论 | 下一步 |
|-----------|------|--------|
| PASS | 该模块及之前正确 | 在更靠后的位置加检查点 |
| FAIL | 该模块或之前有问题 | 在更靠前的位置加检查点 |

重复 Step 3-6，每次缩小范围，直到定位到具体的 op。

### Step 8：定位到 Assemble 阶段时的额外诊断

如果所有模块的检查点都 PASS，但最终输出 FAIL，问题在 Assemble：

```python
# 1. 检查点值 vs 组装后的输出
for h in range(Nv):
    diff = (pto_output[:, h] - pto_cp[h, :, 0]).abs()
    print(f"h={h}: assemble diff = {diff.max():.6e}")

# 2. 手动 torch 组装 vs golden
manual = torch.zeros(T, Nv)
for h in range(Nv):
    manual[:, h] = pto_cp[h, :, 0]
print(f"manual vs golden: {torch.allclose(manual, golden, rtol=1e-3, atol=1e-3)}")
```

若手动组装正确但 pypto 组装错误，则是 **pypto slice 赋值的 strided 写入 bug**。

## 已知问题速查

| 现象 | 可能原因 | 解决方案 |
|------|---------|---------|
| 模块内检查点 FAIL | 该模块计算逻辑有误 | 逐行对比 pypto 与 golden 的计算公式 |
| 所有检查点 PASS，最终输出 FAIL | Assemble 阶段问题 | 用 `cp[idx] = val` + Python 端 reshape 替代 slice 赋值 |
| 检查点 shape 不匹配 | golden 和 pypto 的维度定义不一致 | 注意 `[L]` vs `[L,1]`，用 `unsqueeze/squeeze` 对齐 |
| 检查点全为 0 | 写入位置/时机有误 | 确认检查点在正确循环层级写入 |
| 编译超时 | 检查点 tensor 过多或过大 | 减少检查点数量，每次只加 1 个 |

