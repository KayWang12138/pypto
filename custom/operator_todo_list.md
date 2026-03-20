# PyPTO算子翻译待办列表

## 统计

- **已完成**: 13 个
- **待实现**: 30 个

---

## 实现记录

| 序号 | 算子名称 | torch_npu | golden | pypto | 状态 |
|------|----------|-----------|--------|-------|------|
| 1 | npu_fast_gelu | ✅ | ✅ | ✅ | ✅ |
| 2 | npu_silu | ✅ | ✅ | ✅ | ✅ |
| 3 | npu_rms_norm | ✅ | ✅ | ✅ | ⚠️ 2D |
| 4 | npu_linear | ✅ | ✅ | ⚠️ | ⚠️ 无bias |
| 5 | npu_softmax_cross_entropy_with_logits | ✅ | ✅ | ✅ | ✅ |
| 6 | npu_dropout | ✅ | ✅ | ❌ | ❌ |
| 7 | npu_reshape | ✅ | ✅ | ✅ | ✅ |
| 8 | npu_transpose | ✅ | ✅ | ✅ | ✅ |
| 9 | npu_max/min | ✅ | ✅ | ⚠️ | ⚠️ 无索引 |
| 10 | npu_one_hot | ✅ | ✅ | ✅ | ✅ |
| 11 | npu_mish | ✅ | ✅ | ✅ | ✅ |
| 12 | npu_gelu | ✅ | ✅ | ✅ | ⚠️ erf |
| 13 | npu_swiglu | ✅ | ✅ | ❌ | ❌ 编译失败 |

---

## PyPTO能力限制汇总

1. **不支持随机数**: npu_dropout
2. **不支持tanh/gelu**: npu_mish, npu_gelu (用torch替代)
3. **切片操作编译失败**: npu_swiglu, npu_gelu_mul
4. **部分功能缺失**: linear无bias, max/min无索引