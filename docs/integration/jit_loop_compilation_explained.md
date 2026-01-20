# PyPTO JIT 和 Loop 编译过程详解

本文档详细解释 `@pypto.jit` 装饰的 kernel 中，输入 tensor 的内存分配、`pypto.loop` 的编译过程，以及 `unroll_list` 的优化机制。

## 1. 输入 Tensor 的内存分配

### 1.1 输入 Tensor 在 NPU GM 上

**是的，传入的 tensor 默认在 NPU 的 GM（全局内存，即 DDR）上。**

在编译过程中，`AssignMemoryType` Pass 会为所有输入 tensor 设置内存类型：

```cpp
// assign_memory_type.cpp:36-50
for (auto &incast : function.inCasts_) {
    /*
    设置INCAST的memory type为DDR
    将INCAST的每个consumer加到其tobeMap中，tobe=DDR
            /--> op1 --> tensor1 -->
    INCAST  ---> op2 --> tensor2 -->
            \--> op3 --> tensor3 -->
    */
    incast->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    for (const auto &consumerOp : incast->GetConsumers()) {
        inserter.UpdateTensorTobeMap(incast, *consumerOp, MemoryType::MEM_DEVICE_DDR);
    }
}
```

**关键点：**
- `INCAST`（输入转换）操作的内存类型被设置为 `MEM_DEVICE_DDR`（即 GM）
- 所有输入 tensor 的消费者操作都知道输入在 DDR 上
- 后续的数据传输操作（如 TLOAD）会将数据从 GM 加载到 L1/L0

### 1.2 运行时内存分配

在运行时，输入 tensor 通过 `OperatorDeviceRunOnceDataFromDevice` 传递：

```python
# runtime.py:156
pypto_impl.OperatorDeviceRunOnceDataFromDevice(
    self._handler,
    in_tensor_data,
    out_tensor_data,
    _current_stream(),
    workspace_tensor.data_ptr())
```

这些 tensor 数据指针指向 NPU 设备上的内存（GM/DDR）。

## 2. pypto.loop 的编译过程

### 2.1 Python 层的 Loop 定义

```python
# _controller.py:511-546
def loop(*args, **kwargs) -> Iterator[SymInt]:
    start, stop, step = _get_loop_range(*args)
    # ...
    with _loop_function(name, loop_name, loop_range, unroll_set, submit_before_loop) as rlf:
        for k in rlf:
            yield k
```

**关键步骤：**
1. **记录 Loop 函数**：`RecordLoopFunc` 创建一个动态循环函数
2. **设置作用域**：`BeginScope` 开始一个新的作用域
3. **迭代生成**：每次迭代生成一个 `SymbolicScalar`，带有 `_loop_begin` 和 `_loop_end` 属性

### 2.2 C++ 层的 Loop 编译

在编译过程中，`LoopUnroll` Pass 会处理循环：

#### 2.2.1 Loop 展开（Loop Unroll）

```cpp
// loop_unroll.cpp:230-255
Status LoopUnroll::ExpandDynamicLoop(Operation *callop) {
    Function *currFunction = nullptr;
    if (GetCallee(callop, currFunction) != SUCCESS) {
        return FAILED;
    }
    auto loop = currFunction->GetDynloopAttribute();
    ScalarImmediateType begin = EvaluateSymbolicScalar(loop->Begin());
    ScalarImmediateType end = EvaluateSymbolicScalar(loop->End());
    ScalarImmediateType step = EvaluateSymbolicScalar(loop->Step());
    
    // 展开循环：为每个迭代创建操作
    for (ScalarImmediateType idx = begin; idx < end; idx += step) {
        evaluateSymbol_->UpdateSymbolDict(loop->IterSymbolName(), idx);
        Operation *expandCallop = ExecuteFunctionLoopLookupSat(loop);
        if (ExpandDynamicFunction(expandCallop) != SUCCESS) {
            return FAILED;
        }
    }
    return SUCCESS;
}
```

**编译过程：**
1. **符号求值**：将循环的 begin、end、step 从符号表达式求值为具体值
2. **循环展开**：为每个迭代创建独立的操作实例
3. **符号替换**：将循环索引符号替换为具体值
4. **函数展开**：递归展开循环体内的所有操作

#### 2.2.2 循环体操作展开

```cpp
// loop_unroll.cpp:257-296
Status LoopUnroll::ExpandDynamicFunction(Operation *callop) {
    // ...
    if (currFunction->GetCallopList().size() > 0) {
        // 如果有嵌套循环，递归展开
        for (auto &op : currFunction->GetCallopList()) {
            if (ExpandDynamicFunction(op) != SUCCESS) {
                return FAILED;
            }
        }
    } else {
        // 展开循环体内的所有操作
        for (auto &op : currFunction->Operations()) {
            EvaluateDynamicOpParams(&op, *evaluateSymbol_, opDynOffsetMap, opDynShapeMap);
            if (CreateGlobalTensor(opDynOffsetMap, tensorLocal2Global, &op, currFunction) != SUCCESS) {
                return FAILED;
            }
            if (AddNewOperation(&op, tensorLocal2Global, opDynOffsetMap, opDynShapeMap) != SUCCESS) {
                return FAILED;
            }
        }
    }
    return SUCCESS;
}
```

**关键操作：**
- **动态参数求值**：将循环索引相关的符号表达式求值为具体值
- **创建全局 Tensor**：将循环内的局部 tensor 映射到全局 tensor
- **添加新操作**：为每个迭代创建新的操作实例

## 3. unroll_list 的优化机制

### 3.1 Python 层的 unroll_list 处理

```python
# _controller.py:549-603
def loop_unroll(*args, **kwargs):
    start, stop, step = _get_loop_range(*args)
    
    unroll_list = kwargs.pop("unroll_list", [1])
    unroll_list = sorted(set(unroll_list), reverse=True)  # 从大到小排序
    if 1 not in unroll_list:
        unroll_list.append(1)  # 确保有 1 作为兜底
    
    nstart = start
    for p in unroll_list:  # 按 unroll 因子从大到小处理
        nstep = step * p
        left = (stop - start) % nstep  # 计算余数
        for idx in loop(nstart, stop - left, nstep, **kwargs):
            yield (idx, p)
        nstart = stop - left  # 处理剩余部分
```

**处理逻辑：**
1. **排序**：将 unroll_list 从大到小排序（如 `[32, 16, 8, 4, 2, 1]`）
2. **分段处理**：按 unroll 因子分段处理循环
   - 先处理能被最大 unroll 因子整除的部分
   - 再处理剩余部分，使用较小的 unroll 因子
3. **生成多个 Loop**：每个 unroll 因子生成一个独立的 loop

**示例：**
```python
# unroll_list=[32, 16, 8, 4, 2, 1], total_tokens=4096
# 1. 处理 0-4096，步长 32：生成 loop_32，处理 4096/32=128 次迭代
# 2. 处理剩余部分（如果有），使用下一个 unroll 因子
```

### 3.2 编译时的优化

#### 3.2.1 创建 Loop Unroll 函数

```cpp
// loop_unroll.cpp:327-363
Status LoopUnroll::CreateLoopUnrollFunc(Function *function) {
    std::string funcName = function->GetRawName() + "_Loop_Unroll";
    auto newFunc = std::make_unique<Function>(...);
    newFunc->SetFunctionType(FunctionType::DYNAMIC_LOOP_PATH);
    newFunc->SetGraphType(GraphType::TENSOR_GRAPH);
    // ...
    return SUCCESS;
}
```

**关键点：**
- 创建一个新的函数，类型为 `DYNAMIC_LOOP_PATH`
- 这个函数包含所有 unroll 因子的展开路径

#### 3.2.2 多路径展开

当有 `unroll_list=[32, 16, 8, 4, 2, 1]` 时，编译器会：

1. **为每个 unroll 因子创建独立的循环路径**
   - `token_loop_32`：步长为 32 的循环
   - `token_loop_16`：步长为 16 的循环
   - `token_loop_8`：步长为 8 的循环
   - ...以此类推

2. **代码生成优化**
   - 更大的 unroll 因子 → 更大的 loop body → 更少的循环开销
   - 编译器会为每个路径生成优化的代码

3. **运行时选择**
   - 运行时根据实际的循环次数选择最合适的路径
   - 优先使用最大的 unroll 因子

### 3.3 实际代码生成示例

对于你的代码：
```python
for idx in pypto.loop(total_tokens, name="token_loop_{}".format(task_id), unroll_list=[unroll_level]):
    # loop body
```

**编译后的结构：**
```cpp
// 伪代码示例
if (total_tokens >= 32) {
    // 使用 unroll=32 的路径
    for (int idx = 0; idx < total_tokens - (total_tokens % 32); idx += 32) {
        // 展开 32 次循环体
        // body(idx), body(idx+1), ..., body(idx+31)
    }
    // 处理剩余部分
    for (int idx = total_tokens - (total_tokens % 32); idx < total_tokens; idx++) {
        // body(idx)
    }
} else {
    // 使用较小的 unroll 因子或默认路径
}
```

### 3.4 unroll_list 的优化效果

**优点：**
1. **减少循环开销**：更大的 unroll 因子 → 更少的循环迭代次数
2. **提高指令级并行**：展开的循环体可以更好地利用流水线
3. **更好的缓存利用**：更大的 loop body 可以更好地利用 L1/L0 缓存

**注意事项：**
1. **代码大小**：更大的 unroll 因子会增加生成的代码大小
2. **编译时间**：多个 unroll 路径会增加编译时间
3. **条件分支**：如果循环体内有 `pypto.cond`，unroll 会指数级增加分支数量（2^unroll_factor）

## 4. 总结

### 4.1 输入 Tensor 内存
- ✅ **输入 tensor 在 NPU 的 GM（DDR）上**
- ✅ 通过 `INCAST` 操作标记为 `MEM_DEVICE_DDR`
- ✅ 后续操作（如 TLOAD）会将数据从 GM 加载到 L1/L0

### 4.2 Loop 编译过程
1. **Python 层**：`RecordLoopFunc` 记录循环信息
2. **编译时**：`LoopUnroll` Pass 展开循环
   - 符号求值：将循环范围求值为具体值
   - 循环展开：为每个迭代创建操作实例
   - 函数展开：递归展开循环体内的所有操作
3. **代码生成**：生成实际的循环代码

### 4.3 unroll_list 优化
1. **分段处理**：按 unroll 因子从大到小分段处理循环
2. **多路径生成**：为每个 unroll 因子生成独立的代码路径
3. **运行时选择**：根据实际循环次数选择最优路径
4. **性能提升**：减少循环开销，提高指令级并行和缓存利用

### 4.4 在你的代码中

```python
for idx in pypto.loop(total_tokens, name="token_loop_{}".format(task_id), unroll_list=[unroll_level]):
    pypto.set_vec_tile_shapes(unroll_level, hidden_size)
    x_tile = pypto.view(x, shape=[unroll_level, hidden_size], offsets=[idx, 0])
    # ...
```

**编译过程：**
1. 输入 `x`, `w1` 在 GM 上
2. Loop 被展开为多个迭代，每个迭代处理 `unroll_level` 个 token
3. 每个迭代中：
   - `view` 操作计算偏移（不传输数据）
   - `matmul` 操作触发实际的数据传输（GM → L1 → L0）
4. 如果 `unroll_level=32`，编译器会生成优化的 32 次展开的循环体

