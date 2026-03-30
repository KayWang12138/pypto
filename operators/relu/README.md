# ReLU 算子

## 概述

ReLU（Rectified Linear Unit）是深度学习中常用的激活函数。对输入 tensor 的每个元素执行整流线性单元运算，只保留正数部分，负数变为 0。

## 数学公式

$$res_i = \max(0, input_i)$$

## 目录结构

```
custom/relu/
├── spec.md           # 需求规范
├── api_report.md     # API 掣索报告
├── design.md         # 设计文档
├── relu_golden.py    # Golden 参考实现
├── relu_impl.py      # 算子实现代码
├── test_relu.py      # 测试代码
├── README.md         # 本文件
```

## 运行方式

### 环境准备
```bash
# 设置 NPU 设备
export TILE_FWK_DEVICE_ID=0
```

### 运行测试
```bash
# 运行所有测试
python custom/relu/test_relu.py

# 运行特定测试
python custom/relu/test_relu.py relu::test_relu_perf_p0
python custom/relu/test_relu.py relu::test_relu_func_p0
python custom/relu/test_relu.py relu::test_relu_func_p1
python custom/relu/test_relu.py relu::test_relu_func_p2

# 使用模拟模式
python custom/relu/test_relu.py --run_mode sim
```

## 验证入口

- **性能_P0**: [1024, 1024] - 核心性能场景
- **功能_P0**: [32, 64] - 核心功能验证
- **功能_P1**: [2, 128, 256] - 3维输入验证
- **功能_P2**: [1, 1, 64, 64] - 4维输入验证

## 已知限制

1. **Shape 维度**: 仅支持 2-4 维
2. **Shape Size**: 不大于 INT32_MAX (2147483647)
3. **空 Tensor**: 不支持
4. **特殊值**: 不支持 nan/inf
5. **数据类型**: DT_FP16, DT_FP32, DT_BF16

6. **Contiguous**: 输入 tensor 必须是连续的

## 碰到问题先看这里

| 问题 | 可能原因 | 解决方案 |
|------|----------|----------|
| TILE_FWK_DEVICE_ID 未设置 | 环境变量未配置 | `export TILE_FWK_DEVICE_ID=0` |
| 编译失败 | TileShape 配置错误 | 检查 `set_vec_tile_shapes` 参数 |
| 运行时 shape 不匹配 | 输入 shape 超出约束 | 检查输入 shape 是否为 2-4 维 |
| 精度不通过 | 数据类型或容差问题 | 检查 dtype 和 atol/rtol 设置 |
