---
name: pypto-ascendc-operator-via-golden
description: 基于AscendC文档和代码，通过PyTorch Golden驱动的方式开发PyPTO算子。先生成PyTorch golden代码，确保计算逻辑正确，再让PyPTO生成计算流完全一致的算子代码，解决功能和精度问题，完成算子开发。包含需求分析、Golden开发、算子实现、精度调试、验证交付的完整流程。
tag: [PyPTO, Ascend C, PyTorch Golden, 算子开发, 精度调试]
---

# 基于PyTorch Golden驱动的PyPTO算子开发流程

## 核心理念

> **PyTorch Golden是真理来源，PyPTO算子必须与Golden计算流完全一致**

### 核心原则

1. **Golden优先**：先验证Golden正确，再开发PyPTO kernel
2. **计算流一致**：PyPTO实现严格遵循Golden的每个计算步骤
3. **问题必解决**：功能问题用DFX工具，精度问题用二分法skill
4. **实际验证**：所有验证在NPU上实际运行，禁止编造数据

---

## 工作流程

```
需求理解 → Golden开发验证 → PyPTO开发 → 功能问题解决 → 精度问题解决 → 验证交付
```

---

## 阶段一：需求理解

### 步骤
1. 阅读AscendC文档和代码
2. 根据以上读取的内容，仿照pypto下的.agents/user_in.md中的需求3提取算子需求。内容要简明概要聚焦重点，篇幅跟需求3差不多即可，着重注意隐藏功能/可选参数/扩展功能/计算公式等
3. 提取完需求后，展示出来，让用户确认一下细节，确认是否符合要求！提醒用户检查：是否有类似“online softmax”的隐藏需求没有被发现或提及？

### 输出
- `custom/{算子名}/needs_analysis.md`

---

## 阶段二：PyTorch Golden开发验证 ⭐

**核心阶段：Golden必须先于PyPTO完成并验证**

### 步骤
1. 参考 `pypto-golden-generator` skill开发Golden
2. 确实支持所有算子功能，处理边界情况
3. 创建测试用例：小规模(8-16元素)、中等(1K)、边界
4. 运行验证确保Golden正确，所有功能都确实实现了
5. 生成并运行完毕后，展示出来，让用户确认一下细节，确认是否符合要求！提醒用户检查：是否有类似“online softmax”的要求没有在golden中确实实现？

### 代码规范
```python
def {算子名}_golden(x: torch.Tensor, param1: float) -> torch.Tensor:
    """
    Golden参考实现
    计算流程：
        1. 步骤1
        2. 步骤2
    """
    # Step 1: ...
    # Step 2: ...
    return output
```

**重要原则**：
- Golden专注于计算流程的准确、完整复现，不添加错误检查/泛化性判断
- 去掉边界条件判断（如 `if sp == 1`），保持计算流的一致性
- 确保计算步骤清晰、准确，便于 PyPTO 对照实现

### 输出
- `custom/{算子名}/{算子名}_golden.py`
- 所有测试用例通过

---

## 阶段三：PyPTO算子开发

**严格遵循Golden的计算流**

### 步骤
1. 分析Golden的每个计算步骤，参考 `pypto-operator-develop-workflow` skill 开发PyPTO算子，逐步骤实现与Golden计算流完全一致的PyPTO kernel

### 输出
- `custom/{算子名}/{算子名}.py`

---

## 阶段四：功能问题解决

完成算子开发后，在空闲NPU上运行PyPTO算子，可能会遇见除精度错误外的报错，即功能问题。

### 常见问题
- 编译错误：API使用、类型不匹配
- 运行时错误：内存越界、设备错误
- 逻辑错误：Shape错误、参数错误

### 解决方法
0. 根据报错解决问题
1. 查阅 `docs/api/` 文档
2. 参考 `examples/` 示例
3. 对比Golden实现，保持计算流一致
4. **禁止简化代码或绕过问题**

---

## 阶段五：精度问题解决

完成算子开发并解决所有功能问题后，在空闲NPU上运行PyPTO算子，可能会发现精度问题。

**使用二分法skill精确定位**

### 步骤
1. 运行精度测试确认差异
2. 调用 `pypto-binary-search-verify` skill定位问题
3. 对比Golden和PyPTO在该步骤的实现
4. 修复后重新验证

## 阶段六：验证交付

### 验证层次
- Level 0: 8-16元素（基础功能）
- Level 1: 1K元素（典型场景）
- Level 2: 极值/零值（边界情况）

### 交付物
```
custom/{算子名}/
├── {算子名}.py              # PyPTO实现
├── {算子名}_golden.py       # Golden实现
├── spec.md                  # 需求规格
└── README.md                # 算子文档
```

### README包含
1. 功能描述和数学公式
2. Golden计算流程
3. PyPTO实现要点
4. 编译运行指南
5. 测试结果

---

## 检查清单

### ✅ Golden开发（核心）
- [ ] Golden函数已实现
- [ ] 计算流程清晰注释
- [ ] 多个测试用例已创建
- [ ] 所有测试通过

### ✅ PyPTO开发
- [ ] 计算流与Golden一致
- [ ] 有Golden对照注释
- [ ] 编译通过

### ✅ 问题解决
- [ ] 功能问题已解决
- [ ] 精度问题已解决（使用二分法skill）

### ✅ 验证交付
- [ ] 多规模测试通过
- [ ] 文档完整
- [ ] 交付物齐全

---

## 关键提示

⚠️ **不验证Golden，不开发PyPTO**
⚠️ **问题必须解决，不逃避不简化**
⚠️ **所有验证必须实际运行**

## 相关Skills

- `pypto-golden-generator` - Golden生成
- `pypto-operator-develop-workflow` - PyPTO开发
- `pypto-binary-search-verify` - 精度调试