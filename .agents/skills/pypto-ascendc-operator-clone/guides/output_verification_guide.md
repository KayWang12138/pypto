# 阶段四：输出验证详细指南

## 1. 验证层次

```
Level 1: PyPTO vs Golden（验证kernel正确性）
    ↓ 通过
Level 2: Golden vs AscendC（验证功能一致性）
    ↓ 通过
Level 3: PyPTO vs AscendC（最终验证）
```

## 2. 验证注意事项 ⭐

### 注意事项1：功能一致性

**要保证验证时执行的PyPTO算子与Ascend C算子的预期实现功能是一致的**

```python
# 错误示例：验证的功能不一致
pypto_out = operator(input)  # 无扩展功能
ascendc_out = npu_operator(input, option=xxx)  # 有扩展功能
# 两个输出不能直接对比！

# 正确示例：功能一致
pypto_out = operator(input, option=xxx)  # 有扩展功能
ascendc_out = npu_operator(input, option=xxx)  # 有扩展功能
# 可以直接对比
```

### 注意事项2：输出规格可能不同

**必要时可以只关注核心输出的规格和精度一致**

不同实现可能输出不同的中间结果：
- 输出shape可能不同（如中间变量的shape）
- 输出数量可能不同（某些实现输出更多中间结果）

**处理方式**：
- 优先验证核心输出（最终计算结果）
- 文档说明中间结果的差异
- 如果差异影响下游使用，需要特殊处理

### 注意事项3：精度容差设置

**根据数据类型选择合适的容差**：

| 数据类型 | atol | rtol | 说明 |
|---------|------|------|------|
| BFLOAT16 | 0.01 | 0.01 | 推荐值，可根据实际情况调整 |
| FLOAT16 | 0.01 | 0.01 | 推荐值 |
| FLOAT32 | 0.001 | 0.001 | 高精度场景 |

**如果验证失败，不要立即放宽容差，先分析原因**：

1. 是否存在计算逻辑差异
2. 中间精度损失累积
3. 输入数据是否有异常值
4. 是否有未处理的边界情况

## 3. 验证脚本模板

详见 `templates/output_verification.py`

```python
def test_pypto_vs_ascendc():
    """PyPTO vs AscendC 对比验证"""
    
    # 1. 固定随机种子
    torch.manual_seed(42)
    
    # 2. 生成测试数据
    input1 = torch.randn(...)
    input2 = torch.randn(...)  # 如果需要
    
    # 3. PyPTO执行
    pypto_out = xxx_operator(input1, input2, ...)
    
    # 4. AscendC执行（相同参数）
    ascendc_out = torch_npu.npu_xxx(
        input1, input2,
        ...  # 相同的可选参数
    )
    
    # 5. 对比
    diff = (pypto_out - ascendc_out).abs()
    print(f"Max diff: {diff.max()}")
    print(f"Mean diff: {diff.mean()}")
    
    assert diff.max() < 阈值
```