---
name: pypto-verify-binary-search
description: PyPTO 算子二分查找调试技能。利用精度工具通过二分查找方法快速定位算子精度问题。当需要调试 PyPTO 算子精度、定位精度差异来源或进行中间结果对比时使用此技能。
license: 完整条款见 LICENSE.txt
---

# PyPTO 算子二分查找调试技能

通过二分查找方法快速定位 PyPTO 算子中导致精度问题的具体 op。

## 核心原理

二分查找定位精度问题的核心方法：

1. **在jit装饰器中使用 `pypto.pass_verify_save()` 保存中间结果**
2. **在 golden 函数中使用 `numpy.tofile()` 保存中间结果**
3. **对比kernel的 `.data` 文件和 golden 的 `.bin` 文件**
4. **二分缩小范围**：结果相同→往后二分，不同→往前二分
5. **定位第一个计算结果不同的 op**

## 通用对比工具

本技能提供了通用对比脚本 `scripts/verify_binary_search.py`，自动完成检查点扫描和对比。

### 快速使用

```bash
# 在算子目录运行（自动检测检查点）
python3 .opencode/skills/pypto-verify-binary-search/scripts/verify_binary_search.py

# 列出所有检查点
python3 .opencode/skills/pypto-verify-binary-search/scripts/verify_binary_search.py --list

# 显示详细对比
python3 .opencode/skills/pypto-verify-binary-search/scripts/verify_binary_search.py --verbose

# 指定工作目录
python3 .opencode/skills/pypto-verify-binary-search/scripts/verify_binary_search.py -w models/your_operator -v
```

### 命令行参数

```bash
python3 verify_binary_search.py [OPTIONS]

必需参数:
  -w, --work-dir DIR      工作目录（算子所在目录）

可选参数:
  -o, --output-dir DIR    指定 output 目录名（不指定则自动检测）
  --rtol FLOAT            相对误差容忍度（默认 1e-3）
  --atol FLOAT            绝对误差容忍度（默认 1e-3）
  -v, --verbose           显示详细的元素级对比
  -l, --list              只列出检查点，不进行对比
```

### 工具功能

- ✓ 自动检测最新 output 目录
- ✓ 自动扫描所有检查点文件
- ✓ 智能匹配 jit 和 golden 文件
- ✓ 自动分析并给出二分建议
- ✓ 支持详细元素级对比

## 核心原则

### 原则 1：先验证整体结果

**重要**：开始二分查找前，必须先验证整体结果的正确性。

- 执行python3 -m pytest test_operator.py 运行测试，对比算子输出和 golden 输出
- **如果整体结果匹配** → ✓ 结束调试，不需要二分查找
- **如果整体结果不匹配** → ✗ 继续进行二分查找

### 原则 2：插入中间输出点

在关键计算位置使用 `pypto.pass_verify_save()` 输出中间结果：

```python
# 启用验证选项
verify_options = {"enable_pass_verify": True}

@pypto.jit(run_mode="npu", verify_options=verify_options)
def jit_kernel(inputs, outputs):
    temp1 = pypto.compute_op1(inputs[0])
    pypto.pass_verify_save(temp1, "checkpoint1_after_op1")

    temp2 = pypto.compute_op2(temp1)
    pypto.pass_verify_save(temp2, "checkpoint2_after_op2")

    output = pypto.compute_op3(temp2)
    pypto.pass_verify_save(output, "checkpoint3_output")
```

**重要说明**：
- 必须设置 `verify_options={"enable_pass_verify": True}`
- 数据保存路径：`output/output_*/tensor/`
- 文件格式：`{fname}_{number}.data`

### 原则 3：保存 golden 中间结果

在 golden 函数中保存中间结果：

```python
def golden(inputs, outputs):
    temp1 = compute_op1(inputs[0])
    # 保存中间结果
    if isinstance(temp1, torch.Tensor):
        temp1.cpu().float().numpy().tofile("golden_checkpoint1_after_op1.bin")

    temp2 = compute_op2(temp1)
    if isinstance(temp2, torch.Tensor):
        temp2.cpu().float().numpy().tofile("golden_checkpoint2_after_op2.bin")
```

**文件命名约定**：
- golden 文件：`golden_{checkpoint_name}.bin`
- jit 文件：`{checkpoint_name}_{number}.data`（自动生成）
- 名称必须匹配（去掉 `golden_` 前缀和数字后缀）

### 原则 4：二分查找策略

```
输入 [op1] [op2] [op3] ... [opN] 输出
  ↑                              ↑
正确                          不正确

1. 在中间位置插入输出点，对比 golden
2. 如果结果相同 → 问题在中间位置之后 → 往后二分
3. 如果结果不同 → 问题在中间位置之前或此处 → 往前二分
4. 重复直到找到第一个结果不同的 op
```

## 完整工作流程

### 步骤 0：验证整体结果

运行测试，判断是否需要二分查找：

```python
import numpy as np

jit_output = jit_function(...)
golden_output = golden_function(...)

is_match = np.allclose(
    np.array(jit_output.flatten().tolist()),
    np.array(golden_output.flatten().tolist()),
    rtol=0.003, atol=0.003
)

if is_match:
    print("✓ 整体结果正确，调试结束")
else:
    print("✗ 整体结果不匹配，开始二分查找")
```

### 步骤 1：插入检查点

在 jit 和 golden 函数中插入对应的检查点（参考原则 2 和 3）。

### 步骤 2：运行测试生成数据

```bash
# 运行算子测试
python3 test_operator.py
```

### 步骤 3：使用通用工具对比

```bash
# 自动对比所有检查点
python3 .opencode/skills/pypto-verify-binary-search/scripts/verify_binary_search.py -v
```

### 步骤 4：根据建议继续二分

工具会自动分析并给出建议：

```
✗ 检查点 checkpoint1 匹配，但 checkpoint2 不匹配
→ 问题位置：checkpoint1 和 checkpoint2 之间的操作
→ 建议：在这两个检查点之间插入新的检查点
```

### 步骤 5：重复直到定位问题

在问题范围内插入更多检查点，重复步骤 1-4，直到定位到具体的 op。

## 代码示例

### jit 函数（完整示例）

```python
@pypto.jit(..., verify_options={"enable_pass_verify": True})
def kernel(inputs, outputs):
    # 步骤 1
    temp1 = pypto.matmul(inputs[0], inputs[1])
    pypto.pass_verify_save(temp1, "checkpoint1_matmul")

    # 步骤 2
    temp2 = pypto.add(temp1, inputs[2])
    pypto.pass_verify_save(temp2, "checkpoint2_add")

    # 步骤 3
    output = pypto.mul(temp2, 2.0)
    pypto.pass_verify_save(output, "checkpoint3_mul")

    pypto.assemble(output, [0], outputs[0])
```

### golden 函数（完整示例）

```python
def golden(inputs, outputs):
    # 步骤 1
    temp1 = torch.matmul(inputs[0], inputs[1])
    temp1.cpu().float().numpy().tofile("golden_checkpoint1_matmul.bin")

    # 步骤 2
    temp2 = temp1 + inputs[2]
    temp2.cpu().float().numpy().tofile("golden_checkpoint2_add.bin")

    # 步骤 3
    output = temp2 * 2.0
    output.cpu().float().numpy().tofile("golden_checkpoint3_mul.bin")

    return output
```

## 最佳实践

### 1. 检查点命名

使用有意义的名称，反映计算步骤：

```python
# ✓ 推荐
pypto.pass_verify_save(sij, "checkpoint1_after_qk_matmul")
pypto.pass_verify_save(softmax_out, "checkpoint2_after_softmax")

# ✗ 不推荐
pypto.pass_verify_save(sij, "temp1")
pypto.pass_verify_save(softmax_out, "x")
```

### 2. 渐进式二分

```
第1轮：输入 → 中间 → 输出（3个检查点）
  ↓ 发现中间不匹配
第2轮：在中间位置前后插入检查点（5个检查点）
  ↓ 继续缩小范围
第3轮：在问题范围内插入更多检查点
  ↓
定位到具体 op
```

### 3. 数据量控制

对于大数据量的 tensor，可以：

```python
# 只保存部分数据（使用条件判断）
if condition:
    pypto.pass_verify_save(tensor, "checkpoint_name")
```

### 4. 清理调试代码

修复问题后：

```bash
# 移除 golden 中间文件和最终的bin文件
rm -f golden_*.bin

# 移除调试用的 pass_verify_save
# 移除 verify_options 参数
```

## 优化技巧

### 技巧 1：减少输出点

只输出当前二分轮次需要的点：

```python
# 注释掉不需要的检查点
# pypto.pass_verify_save(temp1, "checkpoint1")
pypto.pass_verify_save(temp_mid, "checkpoint_mid")  # 只输出这一个
```

### 技巧 2：使用条件输出

通过 `cond` 参数控制输出条件：

```python
# 只在特定条件下输出
cond = (idx == 0)
pypto.pass_verify_save(temp_mid, "checkpoint_mid", cond=cond)
```

### 技巧 3：清理输出文件

每轮二分后可以清理旧的输出文件：

```bash
# 清理旧的输出data和bin文件
rm -f output/output_*/tensor/checkpoint_*.data
```

## 常见问题

### Q1: 中间结果太大无法输出

**解决方法**：
- 使用 `cond` 参数只输出部分元素
- 只在特定条件下保存

```python
# 只输出第一个元素
pypto.pass_verify_save(tensor, "checkpoint", cond=(idx == 0))
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

### Q4: 找不到检查点文件

**解决方法**：
- 确认 jit 代码中是否使用了 `pypto.pass_verify_save()`
- 确认是否设置了 `verify_options={"enable_pass_verify": True}`
- 检查文件命名是否符合约定

### Q5: 检查点名称不匹配

**解决方法**：
- golden 文件名必须以 `golden_` 开头
- jit 和 golden 的检查点名称必须一致（去掉前缀后）
- 示例：`checkpoint1_op1` 对应 `golden_checkpoint1_op1.bin`

## 检查清单

使用二分查找调试时，确保：

- [ ] **步骤 0**：先验证整体结果
  - [ ] 如果整体结果正确 → 结束调试
  - [ ] 如果整体结果不正确 → 继续后续步骤
- [ ] **步骤 1**：插入检查点
  - [ ] 在 jit 函数装饰器中设置 `verify_options={"enable_pass_verify": True}`
  - [ ] 在 jit 函数中使用 `pypto.pass_verify_save(tensor, fname)`
  - [ ] 在 golden 函数中使用 `numpy.tofile()` 保存中间结果
  - [ ] 文件命名遵循约定
- [ ] **步骤 2**：运行测试生成数据
- [ ] **步骤 3**：使用通用工具对比
  - [ ] 运行对比脚本
  - [ ] 查看对比结果
- [ ] **步骤 4**：根据建议继续二分
  - [ ] 根据对比结果判断问题位置
  - [ ] 在问题范围内插入新的检查点
- [ ] **步骤 5**：定位并修复问题
  - [ ] 定位到具体的 op
  - [ ] 检查相关 op 的实现
  - [ ] 修复后重新验证
  - [ ] 清理调试代码


## 参考资料

- PyPTO API: `docs/api/`
- pass_verify_save API: `docs/api/others/pypto-pass_verify_save.md`
