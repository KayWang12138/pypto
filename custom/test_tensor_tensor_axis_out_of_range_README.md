# TENSOR_AXIS_OUT_OF_RANGE 测试用例

## 概述

本测试用例用于验证 PyPTO 中的 `TENSOR_AXIS_OUT_OF_RANGE (0x12002)` 错误码。

## 错误描述

**错误码：** TENSOR_AXIS_OUT_OF_RANGE (0x12002)

**错误描述：** Tensor 轴超出范围

**出现原因：**
- 访问的轴索引超出 Tensor 维度
- 轴参数为负数或过大
- 在 reduce 操作中使用了无效的轴参数

## 测试内容

### 错误示例

在 reduce 操作中使用了超出范围的轴参数：
- Tensor 形状为 [4, 4]，有效轴为 0 和 1
- 尝试使用 `dim=2`，这会触发 `TENSOR_AXIS_OUT_OF_RANGE`

```python
x shape: [4, 4]
有效轴范围: 0 到 1
尝试访问的轴: 2（超出范围）
result = pypto.sum(x, dim=2, keepdim=False)  # 触发错误
```

## 运行测试

### 1. 设置环境变量

```bash
export TILE_FWK_DEVICE_ID=0
```

### 2. 运行测试

```bash
python3 custom/test_tensor_tensor_axis_out_of_range.py --run_mode npu
```

## 预期输出

```
============================================================
测试 TENSOR_AXIS_OUT_OF_RANGE (0x12002)
============================================================

输入 x shape: torch.Size([4, 4]), dtype:: torch.float32
Tensor 维度: 2
有效轴范围: 0 到 1
尝试访问的轴: 2（超出范围）

------------------------------------------------------------
测试: 错误示例（轴超出范围）
------------------------------------------------------------
✓ 成功捕获错误: RuntimeError
错误信息: RuntimeError: ASSERTION FAILED: axis >= 0 && axis < shapeSize
✓ 触发了轴相关的错误（与 TENSOR_AXIS_OUT_OF_RANGE 相关）

============================================================
测试完成
============================================================
```

## 测试结果说明

### 错误示例

当尝试访问超出范围的轴时，PyPTO 会触发错误：
- **错误类型：** `RuntimeError`
- **错误信息：** `"`ASSERTION FAILED: axis >= 0 && axis < shapeSize\nAxis is not in the reasonable range!"`
- **关联错误码：** 这个错误与 `TENSOR_AXIS_OUT_OF_RANGE (0x12002)` 相关，因为轴索引超出了 tensor 的维度范围

### 测试详情

- **输入 tensor shape:** [4, 4]
- **Tensor 维度:** 2
- **有效轴范围:** 0 到 1
- **尝试访问的轴:** 2（超出范围）
- **错误捕获:** ✓ 成功

## 解决办法

当遇到 `TENSOR_AXIS_OUT_OF_RANGE` 时：

1. **检查 Tensor 维度数量**
   - 使用 `tensor.dim()` 或 `len(tensor.shape)` 获取维度数量
   - 确保轴索引在 `[0, dim-1]` 范围内

2. **使用正确的轴索引**
   - 正轴：从 0 开始，最大为 `dim-1`
   - 负轴：从 -1 开始，最小为 `-dim`

3. **示例：正确用法**

```python
# 正确示例：访问有效的轴
@pypto.frontend.jit
def correct_axis_example(x):
    # x 形状为 [4, 4]，有效轴为 0 和 1
    out = pypto.sum(x, dim=0, keepdim=False)  # 轴 0 在范围内
    return out
```

## 参考文档

- 错误码处理指南：`.opencode/skills/error-code-handling/SKILL.md`
- PyPTO API 文档：`./docs/api/operation/pypto-sum.md`
