# GELU 算子开发断裂点报告

## 算子信息
- **算子名称**: gelu
- **类别**: activation
- **复杂度**: easy
- **开发时间**: 2026-03-27

## 断裂点列表

### FP-1: 文档与实现不一致 - pypto.gelu() 不存在

| 项目 | 值 |
|------|-----|
| **类型** | documentation |
| **严重级别** | fatal |
| **置信度** | high |
| **归因** | framework |
| **描述** | `docs/tutorials/development/tensor_operation.md` 第 79 行列出 `pypto.gelu(x)` API，但实际运行时报错 `module 'pypto' has no attribute 'gelu'` |
| **错误信息** | `AttributeError: module 'pypto' has no attribute 'gelu'. Did you mean: 'relu'?` |
| **影响** | 无法使用文档声称的内置 GELU API，必须手动实现 |
| **建议修复** | 1. 添加 `pypto.gelu()` 实现，或 2. 更新文档移除不存在的 API |

### FP-2: 动态 shape 下 pow/mul 操作失败

| 项目 | 值 |
|------|-----|
| **类型** | api_limitation |
| **严重级别** | fatal |
| **置信度** | high |
| **归因** | framework |
| **描述** | 使用 `pypto.DYNAMIC` 声明 tensor shape 时，`pypto.pow(x, 3)` 和 `pypto.mul(x, x)` 操作会触发 shape size 检查失败 |
| **错误信息** | `RuntimeError: The shape size of tensor must less than or equal to INT32_MAX` |
| **复现条件** | Kernel 使用 `pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_FP32)` 声明，然后调用 `pypto.pow()` 或 tensor 自乘 |
| **影响** | 无法使用 tanh 近似公式手动实现 GELU |
| **建议修复** | 修复动态 shape 编译时的 shape size 计算逻辑 |

## 开发结果

**dev_result**: BLOCKED_API

### 尝试记录

1. **尝试 1**: 使用 `pypto.gelu()` 内置 API
   - 结果: 失败 - API 不存在

2. **尝试 2**: 使用 `x * x * x` 计算 x³
   - 结果: 失败 - shape size 检查错误

3. **尝试 3**: 使用 `pypto.mul(x, x)` 分步计算
   - 结果: 失败 - shape size 检查错误

4. **尝试 4**: 使用 `pypto.pow(x, 3)` 计算 x³
   - 结果: 失败 - shape size 检查错误

## 产出工件

| 文件 | 状态 | 说明 |
|------|------|------|
| `spec.md` | ✅ 完成 | 算子规格文档 |
| `design.md` | ✅ 完成 | 设计方案 |
| `gelu_golden.py` | ✅ 完成 | PyTorch golden 参考 |
| `gelu_impl.py` | ⚠️ 阻塞 | 实现受框架限制 |
| `test_gelu.py` | ✅ 完成 | 测试文件 |
| `README.md` | ✅ 完成 | 算子文档 |

## 建议

1. **短期**: 等待框架修复或添加 `pypto.gelu()` API
2. **中期**: 修复动态 shape 下 pow/mul 的 shape size 检查问题
3. **长期**: 完善文档与实现的一致性检查

---

*报告生成时间: 2026-03-27*
*来源: pypto-op-autodev*
