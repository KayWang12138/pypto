# 仓库 examples/ 全景速览

> **目标**：把仓库 `examples/` 的学习路径与各层级重点“压缩成一张地图”，方便你按目标选示例、按能力点补齐知识。

---

## 1. 分层结构（从入门到工程化）

- **01_beginner（初级）**：张量创建/基础算子/tiling/变换操作
- **02_intermediate（中级）**：自定义复杂算子、神经网络组件、运行时控制流与动态形状
- **03_advanced（高级）**：attention、组合模式、多函数协作、系统级优化（如 cost model / 图捕获）
- **models（模型）**：真实模型算子实现（形状约束/量化/融合路径）

---

## 2. 初级（01_beginner）：你应该学会什么

### 2.1 basic

覆盖要点：
- 张量创建与属性访问
- 逐元素算术运算、矩阵乘法、激活函数
- View/Assemble 的基础模式

### 2.2 compute

覆盖要点：
- elementwise（add/sub/mul/div/exp/log/abs/sqrt/rsqrt/clip 等）
- matmul 的多种配置
- reduce（sum/max/min 等）

### 2.3 tiling

覆盖要点：
- Vector/Cube tiling 的设置方式
- 不同 tiling 对结果一致性与性能的影响

### 2.4 transform

覆盖要点：
- assemble/gather/concat/view 等“形状与布局相关”的核心算子

---

## 3. 中级（02_intermediate）：从“算子”到“模块”

### 3.1 operators

- **activation**：用基础 op 组合 SiLU/GELU/SwiGLU/GeGLU
- **softmax**：数值稳定写法 + loop + tiling 的综合案例（最推荐反复精读）

### 3.2 nn

- **layer_normalization**：LayerNorm / RMSNorm；归约与精度控制
- **ffn**：完整 FFN 模块；动态 batch；工程化配置类

### 3.3 controflow（运行时特性）

- dynamic shape（dynamic_axis 标注）
- condition（cond/分支）
- loop（循环展开、op 放置规则、编译期 print 行为）

---

## 4. 高级（03_advanced）：系统级能力

### 4.1 nn/attention

关注点：
- transpose/reshape 的复杂组合
- 动态 batch/动态序列长度

### 4.2 patterns/function

关注点：
- 多函数组合与残差连接
- “小函数拼大模块”的工程组织方式

### 4.3 cost_model

关注点：
- 使用代价模型评估与优化执行效率
- 常见用于仿真/策略选择阶段

**建议阅读：**
- [Codegen 模块](../02-core/10-codegen.md) - 代码生成与代价模型的关系
- [Machine 模块](../02-core/08-machine.md) - 执行调度与性能优化
- [性能优化指南](../07-features/01-performance-optimization.md) - 性能分析与优化方法

### 4.4 aclgraph（图捕获模式）

关注点：
- 减少 Host 侧开销
- 在高频调用/服务化推理场景更常见

**建议阅读：**
- [Codegen 模块](../02-core/10-codegen.md) - 代码生成机制
- [Machine 模块](../02-core/08-machine.md) - 执行调度与图捕获
- [性能优化指南](../07-features/01-performance-optimization.md) - Host 侧优化

---

## 5. models：真实模型算子文档你要怎么读

模型样例文档通常包含：
- shape/格式/dtype/连续性约束（非常关键）
- 公式与融合点（用于对齐实现与精度）
- 函数原型与参数说明（用于接入）
- 调用示例脚本路径（用于最小复现）

建议把这些信息"抽出来"放到你的工程 README 或接入文档里，避免团队成员重复踩坑。

## 6. 示例→对应 note 专题映射表

| 示例 | 对应 note 专题 | 说明 |
|------|--------------|------|
| **softmax** | [性能优化](../07-features/01-performance-optimization.md)、[精度调试](../05-debugging/02-precision-debugging.md)、[控制流编译](../03-mechanisms/06-controlflow.md) | 数值稳定写法、loop、tiling 综合案例 |
| **layer_normalization** | [精度调试](../05-debugging/02-precision-debugging.md)、[性能优化](../07-features/01-performance-optimization.md) | 归约与精度控制 |
| **attention** | [控制流编译](../03-mechanisms/06-controlflow.md)、[性能优化](../07-features/01-performance-optimization.md) | 动态形状、复杂控制流 |
| **cost_model** | [Codegen 模块](../02-core/10-codegen.md)、[性能优化](../07-features/01-performance-optimization.md) | 代价模型与性能评估 |
| **aclgraph** | [Codegen 模块](../02-core/10-codegen.md)、[Machine 模块](../02-core/08-machine.md) | 图捕获与执行优化 |


