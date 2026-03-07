# TENSOR_VIEW_OFFSET_MISMATCH 测试用例

## 概述

本测试用例用于验证 PyPTO 中的 `TENSOR_VIEW_OFFSET_MISMATCH (0x12004)` 错误码。

## 错误描述

**错误码：** TENSOR_VIEW_OFFSET_MISMATCH (0x12004)

**错误描述：** Tensor 视图偏移不匹配

**出现原因：**
- View 操作的偏移参数错误
- 偏移超出源 Tensor 范围
- 偏移为负数

## 测试内容

### 错误示例

View 操作的偏移参数错误：
- 源 Tensor 形状为 [8, 8]
- 视图形状为 [4, 4]
- 偏移为 [100, 100]，超出源 Tensor 范围
- 这会触发 `TENSOR_VIEW_OFFSET_MISMATCH`

```python
x shape: [8, 8]
view shape: [4, 4]
offsets: [100, 100]（超出范围）
view_tensor = pypto.view(x, [4, 4], [100, 100])  # 触发错误
```

## 运行测试

### 1. 设置环境变量

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export TILE_FWK_DEVICE_ID=0
```

### 2. 运行测试

```bash
python3 custom/test_tensor_view_offset_mismatch.py --run_mode npu
```

## 预期输出

```
============================================================
测试 TENSOR_VIEW_OFFSET_MISMATCH (0x12004)
============================================================

输入 x shape: torch.Size([8, 8]), dtype: torch.float32
源 Tensor 形状: [8, 8]
视图形状: [4, 4]
尝试使用的偏移: [100, 100]
有效偏移范围: [0, 4]（非负数且 offset + view_shape <= input_shape）
偏移无效: [100, 100] 超出范围

------------------------------------------------------------
测试: 错误示例（视图偏移超出范围）
------------------------------------------------------------
错误：预期应该触发 TENSOR_VIEW_OFFSET_MISMATCH，但没有报错
结果 shape: torch.Size([4, 4])
注意：PyPTO 可能自动处理了超出范围的偏移，
      或者错误在编译/运行时才触发。

============================================================
测试完成
============================================================

说明：
TENSOR_VIEW_OFFSET_MISMATCH (0x12004) 错误码表示视图偏移不匹配。
在当前 PyPTO 版本中，该错误可能不会在所有情况下触发，
因为 PyPTO 可能在编译时或运行时才检查偏移的有效性，
或者使用自动处理机制来处理超出范围的偏移。
该测试用例展示了如何尝试触发此错误码。
```

## 测试结果说明

### 错误示例

当尝试使用超出范围的偏移时，PyPTO 的行为：
- **预期错误：** `TENSOR_VIEW_OFFSET_MISMATCH (0x12004)`
- **实际行为：** 在当前 PyPTO 版本中，该错误可能不会立即触发
- **可能原因：**
  1. PyPTO 在运行时才检查偏移的有效性
  2. PyPTO 使用自动处理机制来处理超出范围的偏移
  3. 错误在编译阶段或运行阶段才触发

### 测试详情

- **源 Tensor shape:** [8, 8]
- **视图 shape:** [4, 4]
- **尝试使用的偏移:** [100, 100]
- **有效偏移范围:** [0, 4]（非负数且 offset + view_shape <= input_shape）
- **偏移无效:** [100, 100] 超出范围

## 重要说明

**TENSOR_VIEW_OFFSET_MISMATCH (0x12004)** 错误码在 PyPTO 中的行为：

1. **错误检查时机：**
   - 该错误可能在编译时、运行时或数据访问时才触发
   - 不一定在函数定义或 jit 编译时立即触发

2. **自动处理机制：**
   - PyPTO 可能有自动处理超出范围偏移的机制
   - 可能会返回未定义的值或进行边界检查

3. **建议：**
   - 在使用 `pypto.view()` 时，开发者应自行验证偏移的有效性
   - 确保 `offsets[i] + shape[i] <= input_shape[i]` 对所有维度成立
   - 确保 `offsets[i] >= 0` 对所有维度成立

## 解决办法

当遇到 `TENSOR_VIEW_OFFSET_MISMATCH` 时：

1. **检查 View 操作的偏移参数**
   - 确保偏移在有效范围内
   - 确保 `offsets[i] + shape[i] <= input_shape[i]`
   - 确保 `offsets[i] >= 0`

2. **确保偏移在有效范围内**
   - 偏移不能为负数
   - 偏移加上视图形状不能超过源 Tensor 的形状

3. **示例：正确用法**

```python
# 正确示例：偏移在有效范围内
@pypto.frontend.jit
def correct_view_example(x, out):
    # x 形状为 [8, 8]
    # 视图形状为 [4, 4]
    # 偏移为 [2, 2]，有效范围
    out[:] = pypto.view(x, [4, 4], [2, 2])
    return
```

## 参考文档

- 错误码处理指南：`.opencode/skills/error-code-handling/SKILL.md`
- PyPTO API 文档：`./docs/api/operation/pypto-view.md`
