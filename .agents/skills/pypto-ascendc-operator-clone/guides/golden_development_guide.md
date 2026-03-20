# 阶段二：Golden开发详细指南

## 1. 为什么先开发Golden

1. **作为正确性基准**：Golden是PyTorch标准实现，可以直接验证
2. **独立验证**：Golden可以单独测试，快速定位问题
3. **与AscendC对比**：验证Golden与AscendC输出一致
4. **加速调试**：PyPTO kernel问题可以用Golden快速定位

## 2. Golden开发流程 ⭐

### 核心原则：Golden必须覆盖所有计划实现的PyPTO功能

**Golden的功能范围 = 阶段一评估后确定的PyPTO实现范围**

这意味着：
- 所有评估为P0/P1的功能，Golden都必须实现
- 所有可选参数，只要PyPTO计划支持，Golden就必须支持
- Golden的功能必须是完整的，不能有缺失

### 步骤1：根据需求分析确定Golden功能范围

回顾阶段一的需求分析结果：

```
| 功能模块 | PyPTO实现优先级 | Golden是否需要实现 |
|---------|---------------|-------------------|
| 核心计算 | P0 | ✓ 必须实现 |
| 扩展功能A | P1 | ✓ 必须实现 |
| 扩展功能B | P2 | ✓ 必须实现 |
| 扩展功能C | P3 | ✗ 可不实现 |
```

**只有评估为P3（暂缓）的功能，Golden可以不实现**

### 步骤2：参考AscendC实现确定计算细节 ⭐⭐⭐

**关键：Golden的计算逻辑必须与AscendC一致！**

**分析方法**：
1. 查看AscendC kernel代码中的计算顺序
2. 特别注意中间结果的数据类型转换
3. 注意softmax等操作的数值稳定性处理
4. 如果AscendC使用了分块计算，分析分块逻辑

**示例**：FusedFloydAttention的softmax处理
```
AscendC使用FlashSoftmax：
- 分块计算score的max和sum
- 逐块更新输出

Golden实现方式：
- 如果验证精度，可以使用标准softmax
- 如果需要严格一致，可以实现分块版本
```

### 步骤3：实现完整功能的Golden函数

```python
def xxx_golden(
    # 必需参数
    input1: torch.Tensor,
    input2: torch.Tensor,
    ...
    # 可选参数（所有计划支持的功能都要有）
    optional_param1: Optional[torch.Tensor] = None,  # P0功能
    optional_param2: Optional[torch.Tensor] = None,  # P1功能
    optional_param3: Optional[torch.Tensor] = None,  # P2功能
    ...
    # 属性参数
    attr1: float = 默认值,
    attr2: int = 默认值,
    ...
) -> Tuple[torch.Tensor, ...]:
    """
    Golden参考实现 - 完整功能版本
    
    必须支持所有计划在PyPTO中实现的功能：
    - 核心计算逻辑
    - 可选参数的处理（None检查、默认值处理）
    - 条件分支逻辑
    """
    
    # 处理可选参数
    if optional_param1 is None:
        optional_param1 = torch.zeros(...)  # 默认值处理
    if optional_param2 is None:
        optional_param2 = torch.ones(...)   # 默认值处理
    
    # 实现完整计算逻辑（包含所有功能分支）
    result = 核心计算(input1, input2)
    
    if optional_param1 is not None:
        result = 扩展功能A处理(result, optional_param1)
    
    if optional_param2 is not None:
        result = 扩展功能B处理(result, optional_param2)
    
    ...
    
    return result
```

### 步骤4：是否需要分块版本？

**判断标准：查看AscendC代码是否使用了分块计算**

| AscendC实现方式 | Golden是否需要分块版本 | 说明 |
|----------------|----------------------|------|
| 标准softmax | 不需要 | 标准实现即可 |
| FlashSoftmax | 可选 | 标准softmax可以验证功能，但数值可能有微小差异 |
| 分块matmul | 可能需要 | 取决于是否影响输出 |

**建议**：
- 优先使用标准实现验证功能正确性
- 如果发现标准实现与AscendC输出差异较大，再实现分块版本

### 步骤5：验证Golden与AscendC一致性

**目的：确保Golden与AscendC算子输出一致**

**验证步骤**：

1. **准备测试数据**
   ```python
   torch.manual_seed(42)  # 固定随机种子
   input = torch.randn(...)
   ```

2. **执行Golden**
   ```python
   golden_out = xxx_golden(input, ...)
   ```

3. **执行AscendC**（通过torch_npu调用）
   ```python
   ascendc_out = torch_npu.npu_xxx(input.npu(), ...)
   ascendc_out = ascendc_out.cpu()  # 移回CPU对比
   ```

4. **对比输出**
   ```python
   diff = (golden_out.float() - ascendc_out.float()).abs()
   max_diff = diff.max().item()
   print(f"Max diff: {max_diff}")
   # 根据数据类型设置合理容差
   assert max_diff < 阈值
   ```

## 3. Golden功能与PyPTO功能的对应关系 ⭐

**核心要求**：Golden和PyPTO必须支持完全相同的功能集

```
┌─────────────────────────────────────────────────────────────┐
│                    功能对应关系                              │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   AscendC算子功能                                            │
│   ├─ 功能1 (P0) ────────► Golden ✓ ──────► PyPTO ✓          │
│   ├─ 功能2 (P1) ────────► Golden ✓ ──────► PyPTO ✓          │
│   ├─ 功能3 (P2) ────────► Golden ✓ ──────► PyPTO ✓          │
│   └─ 功能4 (P3) ────────► Golden ✗ ──────► PyPTO ✗          │
│                                                              │
│   对比验证：Golden ↔ PyPTO（功能完全对应，可以直接对比）      │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## 4. 常见问题处理

### 4.1 数据类型处理

```python
# BF16/F16输入 → FP32中间计算 → BF16/F16输出
input_fp32 = input.float()  # 转FP32计算
result = 计算过程(input_fp32)
result = result.to(input.dtype)  # 转回原类型
```

### 4.2 可选参数处理

```python
def xxx_golden(
    ...,
    atten_mask: Optional[torch.Tensor] = None,
):
    if atten_mask is not None:
        if atten_mask.dtype == torch.bool:
            # bool类型mask：True表示遮蔽
            score = torch.where(atten_mask, float('-inf'), score)
        else:
            # 数值类型mask：直接加
            score = score + atten_mask
```

### 4.3 数值稳定性

```python
# softmax数值稳定性
max_val = score.max(dim=-1, keepdim=True)[0]
score_stable = score - max_val
exp_score = torch.exp(score_stable)
sum_exp = exp_score.sum(dim=-1, keepdim=True)
weights = exp_score / sum_exp
```

## 5. 重要提醒 ⭐⭐⭐

1. **参考AscendC代码**：不确定的计算细节，直接看AscendC怎么实现的
2. **功能完整**：Golden必须覆盖P0和P1功能
3. **数据类型一致**：中间计算使用FP32保证精度
4. **实际运行验证**：必须在NPU上实际运行验证与AscendC的一致性