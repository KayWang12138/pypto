# sinh 算子开发计划

## 算子概述

- **算子名称**: sinh (双曲正弦函数)
- **数学公式**: sinh(x) = (e^x - e^(-x)) / 2
- **功能**: 计算输入张量的双曲正弦函数

## API 映射关系

根据文档搜索结果，sinh 算子需要组合以下 PyPTO API：

| 数学操作 | PyPTO API | 文档位置 |
|---------|----------|---------|
| e^x | `pypto.exp(x)` | docs/api/operation/pypto-exp.md |
| -x | `pypto.neg(x)` | docs/api/operation/pypto-neg.md |
| a - b | `pypto.sub(a, b)` | docs/api/operation/pypto-sub.md |
| a / b | `pypto.div(a, b)` | docs/api/operation/pypto-div.md |

### 实现公式

```python
def sinh(x: Tensor) -> Tensor:
    # 计算 e^x
    exp_x = pypto.exp(x)

    # 计算 e^(-x)
    neg_x = pypto.neg(x)
    exp_neg_x = pypto.exp(neg_x)

    # 计算 e^x - e^(-x)
    diff = pypto.sub(exp_x, exp_neg_x)

    # 除以 2
    result = pypto.div(diff, 2.0)

    return result
```

## NPU 性能优化方案

当前版本不进行深度 NPU 性能优化，优先保证功能正确性。

## 目录结构

```
custom/sinh/
├── sinh.py          # 包含 golden 和测试用例
├── README.md        # 算子文档
```

## 开发步骤

### 步骤 1: 创建测试及 golden 文件
- 创建 `sinh.py` 文件
- 实现 `sinh_golden()` 函数用于验证
- 实现 `test_sinh()` 测试函数，包含 Level 0~N 多级用例

### 步骤 2: 创建算子实现
- 在 `sinh.py` 文件中实现 `sinh_activation()` 函数
- 使用 `@pypto.frontend.jit` 装饰器
- 按照 API 映射关系实现核心逻辑

### 步骤 3: 构建和测试
- 编译 whl 包: `python3 build_ci.py -f python3 --disable_auto_execute`
- 执行测试: `python3 custom/sinh/sinh.py --run_mode npu`

### 步骤 4: 编写 README.md
- 记录数学公式和 API 映射关系
- 包含编译运行指南
- 记录测试结果
- 记录已知限制

## 验证检查点

1. **编译检查点**: 编译通过，无错误无警告
2. **功能检查点**: Level 0 用例（8-16 元素）通过
3. **精度检查点**: 精度测试通过（相对误差 < 1e-3）
4. **泛化检查点**: 多种输入规模测试通过

## 开发进度

- [x] 需求分析
- [x] 环境检查
- [x] API 搜索和验证
- [x] 创建测试文件
- [x] 实现算子
- [x] 编译验证
- [x] 运行测试
- [x] 编写文档

## 测试结果总结

### Level 0: 小规模测试 (8x8)
- 输入形状: (8, 8)
- 最大误差: 0.000002
- 平均误差: 0.000000
- 状态: ✅ 通过

### Level 1: 中等规模测试 (32x128)
- 输入形状: (32, 128)
- 最大误差: 0.000002
- 平均误差: 0.000000
- 状态: ✅ 通过

### Level 2: 大规模测试 (64x256)
- 输入形状: (64, 256)
- 最大误差: 0.000004
- 平均误差: 0.000000
- 状态: ✅ 通过

### 结论
所有测试用例均通过，精度符合要求（最大误差 < 1e-3），功能验证成功。

---

*计划创建时间: Thu Feb 12 2026*
*最后更新时间: Thu Feb 12 2026*
*开发状态: ✅ 已完成*
