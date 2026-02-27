---
name: pypto-verify-binary-search
description: PyPTO 算子二分查找调试技能。用于在算子出现精度问题时，通过二分查找方法自动定位导致精度问题的具体 op。当算子输出结果与 golden 不匹配、需要快速定位问题 op 时使用此技能。
license: 完整条款见 LICENSE.txt
---

# PyPTO 算子二分查找调试技能 (PyPTO Verify Binary Search)

此技能提供通过二分查找方法定位 PyPTO 算子中导致精度问题的具体 op 的功能。

## 核心原理

当算子输出结果与 golden 不匹配时,使用二分查找方法快速定位问题:

1. **在 jit 函数和 golden 函数的关键计算位置使用 `pass_verify_print()` 输出中间结果**
2. **从 `pass_verify_print()` 生成的数据文件中读取并对比中间结果**
3. **二分缩小范围**：结果相同往后二分,不同往前二分
4. **定位第一个计算结果不同的 op**

**关键要求**：
- 必须在 `@pypto.jit` 装饰器中设置 `verify_options={"enable_pass_verify": True}`
- `pass_verify_print()` 会自动保存数据到文件，无需手动处理
- 对比时从生成的数据文件中读取数据进行比对

## 核心原则

### 原则 1：先验证输入和最终输出

开始二分查找前，先确认：
- 输入数据是否正确（与 golden 输入一致）
- 最终输出确实不匹配 golden

### 原则 2：插入中间输出点

在 jit 函数的关键计算位置使用精度工具输出中间结果：

```python
verify_options = {"enable_pass_verify": True}

@pypto.jit(run_mode="npu", verify_options=verify_options)
def jit_kernel(inputs, outputs):
    # 计算步骤 1
    temp1 = pypto.compute_op1(inputs[0])
    # 插入中间输出
    pypto.pass_verify_print(temp1)

    # 计算步骤 2
    temp2 = pypto.compute_op2(temp1)
    pypto.pass_verify_print(temp2)

    # ... 后续步骤
```

**重要说明**：
- 必须设置 `verify_options={"enable_pass_verify": True}`
- `pass_verify_print()` 会在精度验证模式下自动打印数据
- 可选使用 `cond` 参数控制打印条件：`pypto.pass_verify_print(tensor, cond=condition)`

### 原则 3：二分查找策略

按照以下策略进行二分查找：

```
输入 [op1] [op2] [op3] ... [opN] 输出
  ↑                              ↑
正确                          不正确

1. 在中间位置插入输出点，对比 golden
2. 如果结果相同 → 问题在中间位置之后 → 往后二分
3. 如果结果不同 → 问题在中间位置之前或此处 → 往前二分
4. 重复直到找到第一个结果不同的 op
```

## 二分查找流程

### 步骤 1：准备 golden 中间结果

在 golden 函数中同样使用 `pass_verify_print()` 输出中间结果：

```python
verify_options = {"enable_pass_verify": True}

@pypto.jit(run_mode="npu", verify_options=verify_options)
def golden_kernel(inputs, outputs):
    # 计算步骤 1
    temp1 = compute_op1(inputs[0])
    pypto.pass_verify_print(temp1)

    # 计算步骤 2
    temp2 = compute_op2(temp1)
    pypto.pass_verify_print(temp2)

    # ... 后续步骤
```

运行 golden 生成中间结果数据。`pass_verify_print()` 会自动将数据保存到文件中。

### 步骤 2：首次二分 - 插入中间点

在 jit 函数的中间位置插入第一个输出点：

```python
verify_options = {"enable_pass_verify": True}

@pypto.jit(run_mode="npu", verify_options=verify_options)
def jit_kernel(inputs, outputs):
    # 前半部分计算
    temp_mid = pypto.compute_op_mid(...)

    # 中间点输出
    pypto.pass_verify_print(temp_mid)

    # 后半部分计算
    ...
```

### 步骤 3：对比中间结果

运行 jit 函数，`pass_verify_print()` 会自动生成数据文件。使用数据读取工具对比中间点结果与 golden：

```python
import numpy as np
import os

# pass_verify_print 生成的数据文件路径
# 文件名格式通常为: tensor_<hash>_<print_index>.bin 或 .dat
# 需要根据实际文件名读取
def read_pass_verify_data(filename):
    """读取 pass_verify_print 生成的数据文件"""
    # 根据实际文件格式读取数据
    # 具体读取方法取决于数据文件格式
    data = np.fromfile(filename, dtype=np.float32)
    return data

# 对比函数
def compare_with_golden(jit_data, golden_data, name, tolerance=1e-3):
    """对比 jit 结果与 golden 结果"""
    diff = np.max(np.abs(jit_data - golden_data))
    max_val = np.max(np.abs(golden_data))
    relative_error = diff / (max_val + 1e-10)

    match = relative_error < tolerance

    status = "✓ PASS" if match else "✗ FAIL"
    print(f"{name}: {status}")
    print(f"  Max diff: {diff}")
    print(f"  Relative error: {relative_error}")

    return match

# 使用示例
# golden_mid = read_pass_verify_data("golden_mid.bin")
# jit_mid = read_pass_verify_data("jit_mid.bin")
# is_match = compare_with_golden(jit_mid, golden_mid, "mid")

if is_match:
    print("Mid point matches golden → Problem in second half")
else:
    print("Mid point differs from golden → Problem in first half or at this op")
```

### 步骤 4：根据结果继续二分

#### 场景 A：中间点匹配 golden

说明问题在中间点之后，在后半部分继续二分：

```python
verify_options = {"enable_pass_verify": True}

@pypto.jit(run_mode="npu", verify_options=verify_options)
def jit_kernel(inputs, outputs):
    # 前半部分（已验证正确）
    temp_mid = pypto.compute_op_mid(...)

    # 在后半部分的中间插入新输出点
    temp_mid2 = pypto.compute_op_mid2(temp_mid)
    pypto.pass_verify_print(temp_mid2)

    # 继续后续计算
    ...
```

#### 场景 B：中间点不匹配 golden

说明问题在中间点之前或此处，在前半部分继续二分：

```python
verify_options = {"enable_pass_verify": True}

@pypto.jit(run_mode="npu", verify_options=verify_options)
def jit_kernel(inputs, outputs):
    # 在前半部分的中间插入新输出点
    temp_mid_early = pypto.compute_op_mid_early(...)
    pypto.pass_verify_print(temp_mid_early)

    # 继续计算到中间点
    temp_mid = pypto.compute_op_mid(temp_mid_early)
    pypto.pass_verify_print(temp_mid)
    ...
```

### 步骤 5：重复二分直到定位

继续执行步骤 3-4，直到：

- 找到两个相邻的输出点：前一个匹配 golden，后一个不匹配
- 问题 op 就在这两个输出点之间

```python
# 最终定位
if previous_match and not current_match:
    print(f"Problem located between op_{i} and op_{i+1}")
    print(f"Check op_{i+1} for issues")
```

## 使用示例

### 示例：定位复杂算子中的问题

假设有以下算子，输出不匹配：

```python
verify_options = {"enable_pass_verify": True}

@pypto.jit(run_mode="npu", verify_options=verify_options)
def jit_kernel(inputs, outputs):
    x = inputs[0]
    y = inputs[1]

    t1 = pypto.add(x, y)
    t2 = pypto.mul(t1, x)
    t3 = pypto.sin(t2)
    t4 = pypto.exp(t3)
    t5 = pypto.add(t4, x)
    t6 = pypto.mul(t5, y)
    t7 = pypto.sqrt(t6)

    outputs[0] = pypto.assemble(t7, to_global=True)
```

#### 第一轮二分

在中间插入输出（t4）：

```python
verify_options = {"enable_pass_verify": True}

@pypto.jit(run_mode="npu", verify_options=verify_options)
def jit_kernel(inputs, outputs):
    x = inputs[0]
    y = inputs[1]

    t1 = pypto.add(x, y)
    t2 = pypto.mul(t1, x)
    t3 = pypto.sin(t2)
    t4 = pypto.exp(t3)

    # 中间点输出
    pypto.pass_verify_print(t4)

    t5 = pypto.add(t4, x)
    t6 = pypto.mul(t5, y)
    t7 = pypto.sqrt(t6)

    outputs[0] = pypto.assemble(t7, to_global=True)
```

对比结果：如果 t4 匹配 golden → 问题在 t4 之后。

#### 第二轮二分

在后半部分中间插入（t6）：

```python
verify_options = {"enable_pass_verify": True}

@pypto.jit(run_mode="npu", verify_options=verify_options)
def jit_kernel(inputs, outputs):
    x = inputs[0]
    y = inputs[1]

    t1 = pypto.add(x, y)
    t2 = pypto.mul(t1, x)
    t3 = pypto.sin(t2)
    t4 = pypto.exp(t3)

    t5 = pypto.add(t4, x)

    # 后半部分中间点输出
    t6 = pypto.mul(t5, y)
    pypto.pass_verify_print(t6)

    t7 = pypto.sqrt(t6)

    outputs[0] = pypto.assemble(t7, to_global=True)
```

对比结果：如果 t6 不匹配 golden → 问题在 t5 或 t6。

#### 第三轮定位

在 t5 后插入输出：

```python
verify_options = {"enable_pass_verify": True}

@pypto.jit(run_mode="npu", verify_options=verify_options)
def jit_kernel(inputs, outputs):
    x = inputs[0]
    y = inputs[1]

    t1 = pypto.add(x, y)
    t2 = pypto.mul(t1, x)
    t3 = pypto.sin(t2)
    t4 = pypto.exp(t3)

    t5 = pypto.add(t4, x)
    pypto.pass_verify_print(t5)

    t6 = pypto.mul(t5, y)

    outputs[0] = pypto.assemble(t7, to_global=True)
```

对比结果：
- 如果 t5 匹配 golden，但 t6 不匹配 → 问题在 t6（pypto.mul）
- 如果 t5 不匹配 golden → 问题在 t5（pypto.add）

## 对比函数

### 数据读取函数

```python
import numpy as np

def read_pass_verify_data(filename, dtype=np.float32):
    """读取 pass_verify_print 生成的数据文件

    Args:
        filename: 数据文件路径
        dtype: 数据类型，默认 np.float32

    Returns:
        numpy array
    """
    data = np.fromfile(filename, dtype=dtype)
    return data
```

### 标准对比函数

```python
def compare_with_golden(jit_result, golden_result, name, tolerance=1e-3):
    """对比 jit 结果与 golden 结果

    Args:
        jit_result: 从 pass_verify_print 文件读取的 jit 数据
        golden_result: 从 pass_verify_print 文件读取的 golden 数据
        name: 检查点名称
        tolerance: 容忍度，默认 1e-3

    Returns:
        bool: 是否匹配
    """
    diff = np.max(np.abs(jit_result - golden_result))
    max_val = np.max(np.abs(golden_result))
    relative_error = diff / (max_val + 1e-10)

    match = relative_error < tolerance

    status = "✓ PASS" if match else "✗ FAIL"
    print(f"{name}: {status}")
    print(f"  Max diff: {diff}")
    print(f"  Relative error: {relative_error}")

    return match
```

### 批量对比函数

```python
def batch_compare(jit_files, golden_files, keys, tolerance=1e-3):
    """批量对比多个中间输出

    Args:
        jit_files: jit 数据文件路径字典 {key: filename}
        golden_files: golden 数据文件路径字典 {key: filename}
        keys: 需要对比的检查点名称列表
        tolerance: 容忍度，默认 1e-3

    Returns:
        dict: 对比结果 {key: bool}
    """
    results = {}

    for key in keys:
        jit_data = read_pass_verify_data(jit_files[key])
        golden_data = read_pass_verify_data(golden_files[key])
        results[key] = compare_with_golden(jit_data, golden_data, key, tolerance)

    return results
```

## 二分查找日志记录

### 日志格式

```python
# Binary Search Log
Round 1: Check mid point
  - Point: t4 (op 4/7)
  - Result: MATCH golden
  - Conclusion: Problem in second half (ops 5-7)

Round 2: Check mid point of second half
  - Point: t6 (op 6/7)
  - Result: MISMATCH golden
  - Conclusion: Problem in first half of second half (ops 5-6)

Round 3: Check op 5
  - Point: t5 (op 5/7)
  - Result: MATCH golden
  - Conclusion: Problem at op 6 (pypto.mul)

Final: Problem located at op 6: pypto.mul(t5, y)
```

### 日志文件

将日志保存到文件便于后续分析：

```python
def log_binary_search(log_file, message):
    """记录二分查找日志"""
    with open(log_file, 'a') as f:
        f.write(message + '\n')
    print(message)
```

## 优化技巧

### 技巧 1：减少输出点

只输出需要对比的中间结果，避免过多输出影响性能：

```python
# 输出当前二分轮次需要的点，其他点注释掉
# pypto.pass_verify_print(temp1)
pypto.pass_verify_print(temp_mid)
```

### 技巧 2：使用条件输出

通过 `cond` 参数控制打印条件：

```python
@pypto.jit(run_mode="npu", verify_options={"enable_pass_verify": True})
def jit_kernel(inputs, outputs):
    # 计算各步骤
    ...

    # 只在特定条件下输出（例如只输出第一个元素）
    cond = (idx == 0)
    pypto.pass_verify_print(temp_mid, cond=cond)
```

### 技巧 3：自动保存数据

`pass_verify_print()` 自动保存数据到文件，无需手动保存。数据文件通常保存在运行目录下，便于后续对比分析。

## 常见问题

### Q1: 中间结果太大无法输出

**解决方法**：
- 使用 `cond` 参数只输出部分元素
- 输出统计信息而不是完整数据

```python
# 只输出第一个元素
pypto.pass_verify_print(tensor, cond=(idx == 0))

# 或者使用条件判断
cond = (pypto.greater(tensor, threshold).sum() > 0)
pypto.pass_verify_print(tensor, cond=cond)
```

### Q2: op 太多，二分效率低

**解决方法**：
- 先根据代码逻辑划分大块，对每个块进行二分
- 优先检查可疑的 op（例如复杂的数学运算、类型转换等）

### Q3: 多个 op 都有问题

**解决方法**：
- 找到第一个有问题的 op 并修复后，重新运行
- 继续二分查找下一个有问题的 op
- 重复直到所有问题解决

## 检查清单

使用二分查找调试时，确保：

- [ ] 验证输入数据与 golden 一致
- [ ] 在 jit 函数装饰器中设置 `verify_options={"enable_pass_verify": True}`
- [ ] 在 golden 函数中print打印数据
- [ ] 每轮只插入必要的中间输出点
- [ ] 使用相对误差而非绝对误差进行对比
- [ ] 正确读取 `pass_verify_print()` 生成的数据文件
- [ ] 记录每轮二分的日志
- [ ] 定位问题后检查相关 op 的实现
- [ ] 修复后重新验证

## 参考资料

- PyPTO API: `docs/api/`
- pass_verify_print API: `docs/api/others/pypto-pass_verify_print.md`
- 测试框架: `python/tests/st/interface/`
