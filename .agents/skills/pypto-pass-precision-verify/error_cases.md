---
name: pypto-pass-precision-verify-error-cases
description: PyPTO Pass精度验证常见错误案例库，提供问题诊断与解决方案
trigger: Pass错误案例、Pass报错参考、Pass问题排查、Pass错误解决方案
---

## 目录

1. [案例使用说明](#案例使用说明)
2. [Pass验证错误案例](#pass验证错误案例)
   - [案例01：Reshape操作导致Pass验证报错](#案例01reshape操作导致pass验证报错)
   - [案例02：精度相差较小，且前端代码存在数据类型转换导致精度不正确](#案例02精度相差较小且前端代码存在数据类型转换导致精度不正确)
---

## 案例使用说明

**案例结构**：每个案例包含以下标准化章节：
- **场景描述**：问题发生的典型场景
- **问题特征**：错误码、日志特征、现象描述
- **诊断步骤**：如何定位问题（包含具体命令）
- **解决方案**：修复方法（包含代码示例）
- **验证结果**：预期结果与判断标准
- **注意事项**：特殊情况和限制条件


---

## Pass验证错误案例

### 案例01：Reshape操作导致Pass验证报错

#### 场景描述

在使用`pypto.reshape`进行动态shape转换时，Pass验证工具报EXCEPTION错误，但实际运行可能正常。

**典型场景**：
- Tile大小（tile_b, tile_s）大于实际数据边界
- Reshape操作将3D tensor转换为2D tensor
- 使用valid_shape指定有效数据范围

#### 问题特征

**错误码**：无特定错误码，显示EXCEPTION

**日志特征**：
```
[VERIFY:EXCEPTION:OP] Pass_26_InferDynShape, ..., RESHAPE_COPY_IN
<52x32xFP32/52x32xFP32> = RESHAPE_COPY_IN <52x32xFP32/20x32xFP32>
The size of tensor a (52) must match the size of tensor b (20) at non-singleton dimension 0
```

**现象描述**：
- Pass_26_InferDynShape报RESHAPE或RESHAPE_COPY_IN异常
- 错误信息：`shape '[valid_shape]' is invalid for input of size total_size`
- 原因：Pass将动态valid_shape误判为静态tensor大小

**根因分析**：
- 输入数据：`[tile_b, tile_s, 32]` = `[4, 13, 32]` = 1664元素
- Reshape目标：`[tile_b*tile_s, 32]` = `[52, 32]`
- 实际有效数据：`[min(batch-b_idx*tile_b, tile_b), min(seq-s_idx*tile_s, tile_s), 32]` = `[20, 32]` = 640元素
- Pass错误推理：将tensor分配大小 `[52, 32]` 当作有效数据范围 `[20, 32]`

#### 诊断步骤

**步骤1：查看验证日志**
```bash
cat output/output_*/verify_*/verify_graph_result_brief.log | grep EXCEPTION
```

预期输出：
```
[VERIFY:EXCEPTION:OP] Pass_26_InferDynShape, ..., op_magic, RESHAPE_COPY_IN
```

**步骤2：使用get_op_info.py查询报错OP**
```bash
python3 .agents/skills/pypto-pass-error-locator/scripts/get_op_info.py \
    --ir-file output/output_*/Pass_26_InferDynShape/After_*.tifwkgr \
    --op-magic <报错的op_magic>
```

预期输出：
```
=== Operation Information ===
OP Magic: <报错的op_magic>
Opcode: TILE_RESHAPE_COPY_IN
...
```

**步骤3：对比Before/After IR**
```bash
diff -u output/output_*/Pass_26_InferDynShape/Before_*.tifwkgr \
        output/output_*/Pass_26_InferDynShape/After_*.tifwkgr | grep -A2 -B2 "RESHAPE"
```

预期发现：
```diff
-  <52 x 32 x DT_FP32> %tensor_id = !op_magic TILE_RESHAPE_COPY_IN ...
+  <52 x 32 x DT_FP32 / 52 x 32 x DT_FP32> %tensor_id = !op_magic TILE_RESHAPE_COPY_IN ...
```

差异分析：
- Before IR：无显式valid_shape
- After IR：Pass错误添加静态valid_shape `[52, 32]`

**步骤4：查看精度对比结果**
```bash
python3 your_test.py 2>&1 | tail -1
```

判断标准：
- 精度对比PASS → 实际计算正确
- Pass EXCEPTION → Pass框架推理误报

#### 解决方案

**前端代码规避方法**：

在reshape操作后添加 `+ 0.0` 操作：

```python
# 原代码（导致Pass报错）
input_a_view_2d = pypto.reshape(input_a_view, [tile_b*tile_s, input_a_view.shape[-1]], 
            valid_shape=[pypto.min(tile_b, input_tensor_a.shape[0]-b_idx*tile_b) * 
                        pypto.min(tile_s, input_tensor_a.shape[1]-s_idx*tile_s), 
                        input_a_view.shape[-1]])

# 规避代码（Pass不再报错）
input_a_view_2d = pypto.reshape(input_a_view, [tile_b*tile_s, input_a_view.shape[-1]], 
            valid_shape=[pypto.min(tile_b, input_tensor_a.shape[0]-b_idx*tile_b) * 
                        pypto.min(tile_s, input_tensor_a.shape[1]-s_idx*tile_s), 
                        input_a_view.shape[-1]])
input_a_view_2d = input_a_view_2d + 0.0  # 添加 + 0.0 规避Pass推理错误
```

**完整示例**：

```python
# 1111.py 示例（第63-68行）
input_a_view_2d = pypto.reshape(input_a_view, [tile_b*tile_s, input_a_view.shape[-1]], 
                        valid_shape=[pypto.min(tile_b, input_tensor_a.shape[0]-b_idx*tile_b) * 
                                    pypto.min(tile_s, input_tensor_a.shape[1]-s_idx*tile_s), 
                                    input_a_view.shape[-1]])
input_a_view_2d = input_a_view_2d + 0.0  # 添加此行

input_b_view_2d = pypto.reshape(input_b_view, [tile_b*tile_s, input_b_view.shape[-1]],
                        valid_shape=[pypto.min(tile_b, input_tensor_b.shape[0]-b_idx*tile_b) * 
                                    pypto.min(tile_s, input_tensor_b.shape[1]-s_idx*tile_s), 
                                    input_b_view.shape[-1]])
input_b_view_2d = input_b_view_2d + 0.0  # 添加此行
```

**原理说明**：
- `+ 0.0` 操作会生成新tensor，使valid_shape与完整shape对齐
- 不改变数据值（0.0 + x = x）
- Pass验证工具能正确识别完整shape而非动态valid_shape

#### 验证结果

**预期结果**：
- ✅ 精度对比PASS（实际计算正确）
- ❌ Pass验证可能仍报EXCEPTION（误报，不影响实际运行）

**判断标准**：
以精度对比结果为准，Pass验证为误报。

**示例验证输出**：
```python
('PASS', 0.0019057205645367503, 0.0001455510500818491, 0.00048828125, 0.0)
```

#### 注意事项

1. **必须验证精度对比**：规避后需重新运行精度对比验证，确认实际计算正确
2. **Pass误报可忽略**：Pass验证工具可能仍报错（框架推理bug），不影响实际运行
3. **如需修复Pass**：可向Pass框架开发者提交Issue，报告InferDynShape推理bug
4. **不影响数据**：`+ 0.0` 操作不改变任何数据值

#### 相关资源

- **诊断工具**：[get_op_info.py](../scripts/get_op_info.py)
- **对比工具**：[pass_compare.py](../../../tools/verifier/pass_compare.py)

---

### 案例02：精度相差较小，且前端代码存在数据类型转换导致精度不正确

#### 场景描述

在前端代码中存在不同数据类型（如 FP16/BF16 与 FP32）之间的转换操作时，
Pass验证通过但上板结果与torch输出存在较小精度差异。

**典型场景**：
- 输入数据为 FP16/BF16 类型
- 中间计算使用 FP32 类型
- 输出数据再次转换为 FP16/BF16 类型
- 精度误差累积导致最终结果偏差

#### 问题特征

**错误码**：无报错，Pass验证PASS

**日志特征**：
```
[VERIFY]:PASS: ... All passes verified successfully
精度对比结果: ('FAIL', max_diff=0.001, avg_diff=0.0005, ...)
```

**现象描述**：
- Pass验证全部PASS，无EXCEPTION或MISMATCH错误
- 上板执行正常完成
- 输出数据与torch结果存在较小差异（max_diff < 0.01）
- 前端代码中存在 dtype 转换（如 `tensor.to(torch.float16)`）

**根因分析**：
- FP16/BF16 精度范围有限（FP16: ±65504, BF16: 有效位较少）
- 多次类型转换导致精度损失累积
- Pass验证仅检查数据流正确性，不验证精度损失

#### 诊断步骤

**步骤1：检查精度对比结果**
```bash
python3 your_test.py 2>&1 | grep -E "FAIL|PASS|max_diff"
```

预期输出：
```
('FAIL', 0.001234, 0.000567, 0.000488, 0.0)
```

**步骤2：检查前端代码数据类型**
```bash
grep -n "dtype|float16|bfloat16|torch.float" your_test.py
```

预期发现：
```python
input_tensor = torch.tensor(..., dtype=torch.float16)  # FP16 输入
# 或
output = tensor.to(torch.float16)  # 类型转换
```

**步骤3：统一数据类型验证**

将前端代码的所有 Tensor 都采用 FP32 类型，重新执行该用例：

```bash
# 修改前端代码，全部使用 FP32
sed -i 's/dtype=torch.float16/dtype=torch.float32/g' your_test.py
sed -i 's/dtype=torch.bfloat16/dtype=torch.float32/g' your_test.py
sed -i 's/to(torch.float16)/to(torch.float32)/g' your_test.py

# 重新运行测试
python3 your_test.py
```

#### 解决方案

**前端代码修改方法**：

将所有 Tensor 统一使用 FP32 类型：

```python
# 原代码（存在精度损失）
input_tensor_a = torch.tensor(..., dtype=torch.float16)  # FP16 输入
input_tensor_b = torch.tensor(..., dtype=torch.bfloat16)  # BF16 输入
output = result.to(torch.float16)  # 输出转换

# 修改后（统一FP32）
input_tensor_a = torch.tensor(..., dtype=torch.float32)  # FP32 输入
input_tensor_b = torch.tensor(..., dtype=torch.float32)  # FP32 输入
output = result  # 保持FP32，无需转换
```

**完整示例**：

```python
# 原代码示例
import torch
import pypto

def testFunc():
    # FP16 输入 - 可能导致精度损失
    inputA = torch.randn(64, 128, dtype=torch.float16)
    inputB = torch.randn(64, 128, dtype=torch.float16)
    
    # PyPTO计算
    result = pyptoFunc(inputA, inputB)
    
    # 输出转换 - 再次损失精度
    output = result.to(torch.float16)
    return output

# 修改后示例
def testFunc():
    # 统一使用FP32 - 避免精度损失
    inputA = torch.randn(64, 128, dtype=torch.float32)
    inputB = torch.randn(64, 128, dtype=torch.float32)
    
    # PyPTO计算（保持FP32）
    result = pyptoFunc(inputA, inputB)
    
    # 输出保持FP32
    output = result
    return output
```

**原理说明**：
- FP32 提供更高的精度范围（±3.4e38）
- 统一数据类型避免转换损失

#### 验证结果

**预期结果**：
- ✅ Pass验证PASS
- ✅ 精度对比PASS（max_diff < 1e-5）

**判断标准**：
统一FP32后，精度差异应显著减小或消除。

**示例验证输出**：
```python
# 修改前（FP16存在精度损失）
('FAIL', 0.001234, 0.000567, 0.000488, 0.0)

# 修改后（FP32精度正常）
('PASS', 1e-7, 1e-8, 1e-8, 0.0)
```

#### 注意事项

1. **生产环境权衡**：FP32占用更多内存，需根据实际场景权衡精度与性能
2. **部分场景可接受**：某些应用对 FP16/BF16 的精度损失可接受（如推理场景）
3. **验证环境建议**：调试时优先使用 FP32，确认正确后再评估是否需要低精度
4. **PyPTO支持**：PyPTO 支持多种数据类型，但精度验证以 FP32 为基准

#### 相关资源

- **数据类型参考**：PyPTO文档中 DataType 定义（DT_FP16, DT_BF16, DT_FP32）
- **精度对比工具**：torch.allclose(), torch.max_diff()
- **相关案例**：案例01（Reshape操作导致Pass验证报错）
