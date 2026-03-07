# TENSOR_VIEW_DIMENSION_MISMATCH 测试用例

## 概述

本测试用例用于验证 PyPTO 中的 `TENSOR_VIEW_DIMENSION_MISMATCH (0x12003)` 错误码。

## 错误描述

**错误码：** TENSOR_VIEW_DIMENSION_MISMATCH (0x12003)

**错误描述：** Tensor 视图维度不匹配

**出现原因：**
- View 操作的维度与源 Tensor 不匹配
- 视图形状参数错误
- 在 cube 操作中，形状维度与 offset 维度不匹配

## 测试内容

### 错误示例

View 操作的维度与源 Tensor 不匹配：
- 源 Tensor 形状为 [8, 8]（2维）
- 视图形状为 [4, 4, 4]（3维）
- offsets 为 [0, 0, 0]（3维）
- 这会触发 `TENSOR_VIEW_DIMENSION_MISMATCH`

```python
x shape: [8, 8]（2维）
view shape: [4, 4, 4]（3维）
offsets: [0, 0, 0]（3维）
view_tensor = pypto.view(x, [4, 4, 4], [0, 0, 0])  # 触发错误
```

## 运行测试

### 1. 设置环境变量

```bash
export TILE_FWK_DEVICE_ID=0
```

### 2. 运行测试

```bash
python3 custom/test_tensor_view_dimension_mismatch.py --run_mode npu
```

## 预期输出

```
============================================================
测试 TENSOR_VIEW_DIMENSION_MISMATCH (0x12003)
============================================================

输入 x shape: torch.Size([8, 8]), dtype: torch.float32
源 Tensor 维度: 2
源 Tensor 形状: [8, 8]
尝试创建视图形状: [4, 4, 4]（3维）
视图维度: 3
维度不匹配: 源 Tensor 是 2 维，但视图要求 3 维

------------------------------------------------------------
测试: 错误示例（视图维度不匹配）
------------------------------------------------------------
✓ 成功捕获错误: RuntimeError
错误信息: ASSERTION FAILED: validShape.size() == viewShape.size()
✓ 触发了视图相关的错误（与 TENSOR_VIEW_DIMENSION_MISMATCH 相关）

============================================================
测试完成
============================================================
```

## 测试结果说明

### 错误示例

当尝试创建维度不匹配的视图时，PyPTO 会触发错误：
- **错误类型：** `RuntimeError`
- **错误信息：** `"ASSERTION FAILED: validShape.size() == viewShape.size() Their size actually are 2and 3"`
- **关联错误码：** 这个错误与 `TENSOR_VIEW_DIMENSION_MISMATCH (0x12003)` 相关，因为视图的维度（3维）与源 Tensor 的维度（2维）不匹配

### 测试详情

- **源 Tensor shape:** [8, 8]
- **源 Tensor 维度:** 2
- **视图 shape:** [4, 4, 4]
- **视图维度:** 3
- **offsets:** [0, 0, 0]
- **维度不匹配:** 源 Tensor 是 2 维，但视图要求 3 维
- **错误捕获:** ✓ 成功

## 解决办法

当遇到 `TENSOR_VIEW_DIMENSION_MISMATCH` 时：

1. **检查 View 操作的维度参数**
   - 确保 view 的 shape 维度与源 Tensor 兼容
   - 确保 offsets 维度与 shape 维度一致

2. **确保视图形状与源 Tensor 兼容**
   - View 的总元素数量应小于等于源 Tensor 的总元素数量
   - View 的维度可以与源 Tensor 不同，但需要合理

3. **示例：正确用法**

```python
# 正确示例：视图形状与源 Tensor 兼容
@pypto.frontend.jit
def correct_view_example(x):
    # x 形状为 [8, 8]
    # 视图形状为 [4, 4]，offsets 为 [2, 2]
    view_tensor = pypto.view(x, [4, 4], [2, 2])
    return view_tensor
```

## 参考文档

- 错误码处理指南：`.opencode/skills/error-code-handling/SKILL.md`
- PyPTO API 文档：`./docs/api/operation/pypto-view.md`
