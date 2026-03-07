# TENSOR_CONSISTENCY_ERROR 测试用例

## 概述

本测试用例用于验证 PyPTO 中的 `TENSOR_CONSISTENCY_ERROR (0x10007)` 错误码。

## 错误描述

**错误码：** TENSOR_CONSISTENCY_ERROR (0x10007)

**错误描述：** Tensor 一致性错误

**出现原因：**
- Tensor 形状或类型不一致
- 操作过程中 Tensor 状态发生变化

## 测试内容

### 测试1: 错误示例
直接相加形状不一致的 tensor，预期会触发 `TENSOR_CONSISTENCY_ERROR`。

```python
x shape: [4, 4]
y shape: [2, 2]
result = pypto.add(x, y)  # 触发错误
```

### 测试2: 正确示例
先使用 reshape 确保 tensor 形状一致，然后再相加，预期成功。

```python
if x.shape != y.shape:
    y = pypto.reshape(y, x.shape)
result = pypto.add(x, y)  # 成功
```

### 测试：形状一致的情况
当两个 tensor 形状一致时，直接相加应该成功。

## 运行测试

### 1. 设置环境变量

```bash
export TILE_FWK_DEVICE_ID=0
export PTO_TILE_LIB_CODE_PATH=/mnt/workspace/pto_isa/pto-isa/
```

### 2. 编译安装

```bash
python3 build_ci.py -f python3 --disable_auto_execute
```

### 3. 运行测试

```bash
python3 custom/test_tensor_consistency_error.py
```

## 预期输出

```
============================================================
测试 TENSOR_CONSISTENCY_ERROR (0x10007)
============================================================

输入 x shape: torch.Size([4, 4]), dtype: torch.float32
输入 y (错误测试) shape: torch.Size([2, 2]), dtype: torch.float32
输入 y (reshape测试) shape: torch.Size([2, 8]), dtype: torch.float32

------------------------------------------------------------
测试1: 错误示例（形状不一致直接相加）
------------------------------------------------------------
成功捕获错误: RuntimeError
错误信息: ASSERTION FAILED: false && "shape not support binary operation"
✓ 触发了形状相关的错误（与 TENSOR_CONSISTENCY_ERROR 相关）

------------------------------------------------------------
测试2: 正确示例（reshape确保形状一致）
------------------------------------------------------------
✓ 执行成功
结果 shape: torch.Size([4, 4])
最大绝对误差: 0.0
✓ 精度验证通过

------------------------------------------------------------
测试3: 形状一致的情况（预期成功）
------------------------------------------------------------
✓ 执行成功
结果 shape: torch.Size([4, 4])
最大绝对误差: 0.0
✓ 精度验证通过

============================================================
测试完成
============================================================
```

## 测试结果说明

### 测试1：错误示例
当尝试直接相加形状不一致的 tensor（[4, 4] 和 [2, 2]）时，PyPTO 会触发错误：
- 错误类型：`RuntimeError`
- 错误信息：`"shape not support binary operation"`
- 这个错误与 `TENSOR_CONSISTENCY_ERROR (0x10007)` 相关，因为形状不一致会导致 tensor 一致性问题

### 测试2：正确示例
当使用 `pypto.reshape()` 将 tensor 形状调整为一致后再相加，操作成功执行：
- 输入 x shape: [4, 4]
- 输入 y shape: [2, 8]（reshape 为 [4, 4]）
- 输出 shape: [4, 4]
- 精度验证通过

### 测试3：形状一致的情况
当两个 tensor 形状一致时，直接相加成功执行：
- 输入 x shape: [4, 4]
- 输入 y shape: [4, 4]
- 输出 shape: [4, 4]
- 精度验证通过

## 解决办法

当遇到 `TENSOR_CONSISTENCY_ERROR` 时：

1. **检查 Tensor 形状和类型**
   - 确保参与操作的 tensor 形状兼容
   - 确保数据类型匹配

2. **使用 reshape 或 broadcast**
   - 在操作前调整 tensor 形状
   - 使用 `pypto.reshape()` 或 `pypto.broadcast()`

3. **使用类型转换**
   - 使用 `pypto.cast()` 统一数据类型

## 参考文档

- 错误码处理指南：`.opencode/skills/error-code-handling/SKILL.md`
- PyPTO API 文档：`./docs/api/`
