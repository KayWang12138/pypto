# 错误分析报告模板

## 错误基本信息
- **错误ID**: unique_error_id
- **Pass模块**: ShapeInference
- **错误类型**: ShapeMismatch
- **严重程度**: HIGH

## 根本原因
- **类别**: input_data
- **描述**: 输入张量的shape与期望的不匹配
- **详细信息**:
  - 期望shape: [32, 128]
  - 实际shape: [32, 64]

## 可能原因
1. 输入张量shape计算错误
2. 模型定义中shape配置不当
3. 动态shape推断失败

## 修复建议
1. 检查输入张量的shape
2. 验证模型定义中的shape配置
3. 使用print调试中间结果

## 相关代码位置
- 文件: path/to/file.py:123
- 函数: function_name
