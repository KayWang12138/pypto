# Unroll List 编译性能问题分析

本文档分析当 `unroll_list` 只有一个较大的数时，为什么编译会很慢，以及如何优化。

## 问题描述

当使用 `unroll_list=[32]` 这样的单个较大 unroll 因子时，编译时间会显著增加。

## 根本原因

### 1. 循环展开的编译过程

在 `LoopUnroll::ExpandDynamicLoop` 中，编译器会为每个迭代创建操作：

```cpp
// loop_unroll.cpp:230-255
Status LoopUnroll::ExpandDynamicLoop(Operation *callop) {
    // ...
    ScalarImmediateType begin = EvaluateSymbolicScalar(loop->Begin());
    ScalarImmediateType end = EvaluateSymbolicScalar(loop->End());
    ScalarImmediateType step = EvaluateSymbolicScalar(loop->Step());
    
    // 关键：为每个迭代展开循环
    for (ScalarImmediateType idx = begin; idx < end; idx += step) {
        evaluateSymbol_->UpdateSymbolDict(loop->IterSymbolName(), idx);
        Operation *expandCallop = ExecuteFunctionLoopLookupSat(loop);
        // 为每个迭代展开循环体
        if (ExpandDynamicFunction(expandCallop) != SUCCESS) {
            return FAILED;
        }
    }
    return SUCCESS;
}
```

### 2. 操作创建开销

对于每个迭代，编译器需要：

```cpp
// loop_unroll.cpp:283-293
for (auto &op : currFunction->Operations()) {
    // 1. 求值动态参数（符号表达式 → 具体值）
    EvaluateDynamicOpParams(&op, *evaluateSymbol_, opDynOffsetMap, opDynShapeMap);
    
    // 2. 创建全局 tensor（映射局部 tensor 到全局 tensor）
    if (CreateGlobalTensor(opDynOffsetMap, tensorLocal2Global, &op, currFunction) != SUCCESS) {
        return FAILED;
    }
    
    // 3. 添加新操作（为每个迭代创建操作实例）
    if (AddNewOperation(&op, tensorLocal2Global, opDynOffsetMap, opDynShapeMap) != SUCCESS) {
        return FAILED;
    }
}
```

### 3. 时间复杂度分析

假设：
- `total_tokens = 4096`
- `unroll_list = [32]`
- 循环体有 `N` 个操作

**单个大 unroll 因子的情况：**
```
迭代次数 = 4096 / 32 = 128 次
每次迭代展开 = 32 个操作实例
总操作数 = 128 × 32 × N = 4096 × N
编译时间 = O(迭代次数 × unroll_factor × 操作数)
         = O(128 × 32 × N) = O(4096 × N)
```

**多个 unroll 因子的情况（推荐）：**
```
unroll_list = [32, 16, 8, 4, 2, 1]

1. unroll=32: 处理 4096 - (4096 % 32) = 4096 个元素
   迭代次数 = 4096 / 32 = 128 次
   每次展开 32 个操作

2. unroll=16: 处理剩余部分（如果有）
   迭代次数 = (4096 % 32) / 16 = 0 次（因为 4096 % 32 = 0）

3. ... 其他 unroll 因子类似

总操作数 ≈ 4096 × N（相同）
但编译优化更好，因为：
- 大部分迭代使用 unroll=32（最优）
- 剩余部分使用较小的 unroll 因子（避免过度展开）
```

### 4. 为什么单个大 unroll 因子慢？

#### 4.1 所有迭代都使用最大 unroll

当 `unroll_list=[32]` 时：
- 所有迭代都使用 unroll=32
- 即使某些迭代可能不需要那么大的 unroll
- 编译器必须为每个迭代创建 32 个操作实例

#### 4.2 缺少渐进式优化

多个 unroll 因子的优势：
- 编译器可以优先使用最大的 unroll 因子
- 剩余部分使用较小的 unroll 因子
- 避免不必要的过度展开

#### 4.3 操作创建的开销累积

每次迭代都要：
1. **符号求值**：将循环索引相关的符号表达式求值为具体值
2. **Tensor 映射**：将循环内的局部 tensor 映射到全局 tensor
3. **操作克隆**：为每个迭代创建新的操作实例
4. **依赖分析**：分析操作之间的依赖关系

这些开销会随着迭代次数线性增长。

## 解决方案

### 方案 1：使用多个 unroll 因子（推荐）

```python
# 不推荐：单个大 unroll 因子
for idx in pypto.loop(total_tokens, unroll_list=[32]):
    # ...

# 推荐：多个 unroll 因子
for idx in pypto.loop(total_tokens, unroll_list=[32, 16, 8, 4, 2, 1]):
    # ...
```

**优势：**
- 编译器可以智能选择最合适的 unroll 因子
- 避免不必要的过度展开
- 编译时间更短

### 方案 2：使用 `loop_unroll` 代替 `loop`

如果你的循环体可以一次处理多个迭代，使用 `loop_unroll`：

```python
# 使用 loop_unroll，一次处理 unroll 个元素
for idx, unroll_factor in pypto.loop_unroll(total_tokens, unroll_list=[32, 16, 8, 4, 2, 1]):
    # 一次处理 unroll_factor 个元素
    # ...
```

**优势：**
- 减少循环迭代次数
- 减少编译时的操作创建开销

### 方案 3：减小 unroll 因子

如果编译时间仍然太长，可以减小 unroll 因子：

```python
# 从 [32] 改为 [16, 8, 4, 2, 1]
for idx in pypto.loop(total_tokens, unroll_list=[16, 8, 4, 2, 1]):
    # ...
```

### 方案 4：使用 `pypto.is_loop_begin()` 和 `pypto.is_loop_end()`

如果你的循环体中有条件分支，使用这些函数来优化：

```python
for idx in pypto.loop(total_tokens, unroll_list=[32, 16, 8, 4, 2, 1]):
    if pypto.cond(pypto.is_loop_begin(idx)):
        # 只在循环开始时执行
        pass
    elif pypto.cond(pypto.is_loop_end(idx)):
        # 只在循环结束时执行
        pass
    else:
        # 正常处理
        pass
```

**优势：**
- 避免条件分支的指数级展开
- 减少编译时间和代码量

## 实际案例

### 你的代码

```python
for idx in pypto.loop(total_tokens, name="token_loop_{}".format(task_id), unroll_list=[unroll_level]):
    pypto.set_vec_tile_shapes(unroll_level, hidden_size)
    x_tile = pypto.view(x, shape=[unroll_level, hidden_size], offsets=[idx, 0])
    w1_tile = pypto.view(w1, shape=[hidden_size, ffn_hidden_size], offsets=[0, 0])
    pypto.set_cube_tile_shapes([unroll_level, unroll_level], [tile_k, tile_k * 8], [tile_n, tile_n * 2], True, True)
    pypto.set_matrix_size([unroll_level, hidden_size, ffn_hidden_size])
    res = pypto.matmul(x_tile, w1_tile, pypto.DT_FP32)
    pypto.assemble(res, [idx, 0], out)
```

**问题：**
- `unroll_list=[unroll_level]` 只有一个 unroll 因子
- 如果 `unroll_level=32`，所有迭代都使用 unroll=32
- 编译时需要为每个迭代创建大量操作

**优化建议：**

```python
# 方案 1：使用多个 unroll 因子
unroll_list = [unroll_level]
if unroll_level > 1:
    # 添加更小的 unroll 因子
    unroll_list.extend([unroll_level // 2, unroll_level // 4, unroll_level // 8, 4, 2, 1])
    unroll_list = sorted(set(unroll_list), reverse=True)

for idx in pypto.loop(total_tokens, name="token_loop_{}".format(task_id), unroll_list=unroll_list):
    # ... 循环体保持不变
```

或者：

```python
# 方案 2：使用固定的多个 unroll 因子
for idx in pypto.loop(total_tokens, name="token_loop_{}".format(task_id), 
                      unroll_list=[32, 16, 8, 4, 2, 1]):
    # ... 循环体保持不变
```

## 性能对比

| Unroll List | 迭代次数 | 每次展开 | 总操作数 | 编译时间 |
|------------|---------|---------|---------|---------|
| `[32]` | 128 | 32 | 4096 × N | **慢** |
| `[32, 16, 8, 4, 2, 1]` | 128 + 0 + ... | 32 (大部分) | 4096 × N | **快** |
| `[16, 8, 4, 2, 1]` | 256 | 16 (大部分) | 4096 × N | **中等** |

**关键点：**
- 总操作数相同，但编译时间不同
- 多个 unroll 因子允许编译器智能选择
- 避免不必要的过度展开

## 总结

### 为什么单个大 unroll 因子慢？

1. **所有迭代都使用最大 unroll**：即使不需要
2. **缺少渐进式优化**：无法智能选择 unroll 因子
3. **操作创建开销累积**：每次迭代都要创建大量操作

### 推荐做法

1. ✅ **使用多个 unroll 因子**：`[32, 16, 8, 4, 2, 1]`
2. ✅ **让编译器智能选择**：根据实际情况选择最合适的 unroll 因子
3. ✅ **避免过度展开**：只在需要时使用大的 unroll 因子

### 快速修复

将你的代码：
```python
unroll_list=[unroll_level]
```

改为：
```python
unroll_list=[unroll_level, unroll_level // 2, unroll_level // 4, 8, 4, 2, 1]
# 或者
unroll_list=[32, 16, 8, 4, 2, 1]  # 如果 unroll_level 通常是 32
```

这样可以显著减少编译时间，同时保持运行时的性能。

