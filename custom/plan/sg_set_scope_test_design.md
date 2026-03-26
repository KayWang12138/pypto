# sg_set_scope 开关测试思路文档

## 测试目标

验证 `sg_set_scope` 功能增强的两个新开关：
1. **开关1（allowParallelMerge）**：允许并行分支合并
2. **开关2（allowCrossScopeMerge）**：允许含有 scope 的 supernode 和其他 supernode 合并

## 测试框架

基于现有测试模式，使用 `ComputationalGraphBuilder` 构建测试图，通过 `IsoPartitioner` 进行分区，验证子图生成结果。

---

## 开关1测试思路：allowParallelMerge

### 测试场景

创建一个具有多个并行分支的计算图，每个分支包含一组独立的操作。

**图结构设计**：
```
输入1 → 操作1（scopeId=1）→ 输出1
输入2 → 操作2（scopeId=1）→ 输出2
输入3 → 操作3（scopeId=1）→ 输出3
输入4 → 操作4（scopeId=1）→ 输出4
```

**关键特征**：
- 四个操作具有相同的 scopeId=1
- 操作之间没有直接的数据依赖关系（并行分支）
- 每个操作都是独立的计算单元

### 测试用例

#### 用例1：开关1开启（allowParallelMerge=True）

**设置**：
- 四个操作的 scopeId=1
- allowParallelMerge=True
- 其他参数保持默认值

**预期行为**：
- 分区过程中，系统识别所有 scopeId=1 的操作
- 即使它们没有直接连接，也会被合并到同一个 supernode 中
- 所有操作应该分配到同一个子图

**验证点**：
- 验证生成的子图数量为 1（只有一个 supernode 包含所有 scopeId=1 的操作）
- 验证所有操作是否都在同一个子图中

#### 用例2：开关1关闭（allowParallelMerge=False）

**设置**：
- 四个操作的 scopeId=1
- allowParallelMerge=False（保持原有行为）
- 其他参数保持默认值

**预期行为**：
- 分区过程中，只合并直接连接的操作
- 由于四个操作之间没有直接连接，它们不会被合并
- 每个操作保持独立

**验证点**：
- 验证生成的子图数量为 4（每个操作独立成一个子图）
- 验证每个操作是否在不同的子图中

---

## 开关2测试思路：allowCrossScopeMerge

### 测试场景

创建两个 supernode，每个 supernode 内部有相同 scopeId 的操作，supernode 之间有数据依赖。

**图结构设计**：
```
[Supernode A]
  输入 → 操作1（scopeId=1）→ 中间结果
  输入 → 操作2（scopeId=1）→ 中间结果

[Supernode B]
  中间结果 → 操作3（scopeId=2）→ 输出
  中间结果 → 操作4（scopeId=2）→ 输出
```

**关键特征**：
- Supernode A 包含 scopeId=1 的操作
- Supernode B 包含 scopeId=2 的操作
- Supernode A 的输出作为 Supernode B 的输入
- 两个 supernode 都有 scopeId

### 测试用例

#### 用例3：开关2开启（allowCrossScopeMerge=True）

**设置**：
- 操作1、操作2的 scopeId=1，allowCrossScopeMerge=True
- 操作3、操作4的 scopeId=2，allowCrossScopeMerge=True
- 其他参数保持默认值

**预期行为**：
- 分区过程中，当评估 Supernode A 和 Supernode B 是否应该合并时
- 检查两个 supernode 的 allowCrossScopeMerge 属性
- 由于两个 supernode 都允许跨 scope 合并，它们应该被合并

**验证点**：
- 验证 Supernode A 和 Supernode B 合并后的子图数量
- 验证操作1、2、3、4 最终是否在同一个子图中

#### 用例4：开关2关闭（allowCrossScopeMerge=False）

**设置**：
- 操作1、操作2的 scopeId=1，allowCrossScopeMerge=False
- 操作3、操作4的 scopeId=2，allowCrossScopeMerge=False
- 其他参数保持默认值

**预期行为**：
- 分区过程中，当评估 Supernode A 和 Supernode B 是否应该合并时
- 检查两个 supernode 的 allowCrossScopeMerge 属性
- 由于至少有一个不允许跨 scope 合并，它们不应该被合并

**验证点**：
- 验证 Supernode A 和 Supernode B 保持分离
- 验证操作1、2在一个子图中，操作3、4在另一个子图中

---

## 综合测试用例

### 用例5：两个开关组合测试

**场景**：
- 多个并行分支（测试开关1）
- 同时有跨 supernode 的依赖关系（测试开关2）

**图结构设计**：
```
[Scope=1, allowParallelMerge=True, allowCrossScopeMerge=False]
  输入 → 操作1 → 输出
  输入 → 操作2 → 输出

[Scope=2, allowParallelMerge=True, allowCrossScopeMerge=True]
  输入 → 操作3 → 输出
  输入 → 操作4 → 输出

操作1、2 的输出 → 操作5（无 scopeId）→ 操作3、4 的输入
```

**预期行为**：
- Scope=1 的两个操作（1、2）合并（因为 allowParallelMerge=True）
- Scope=2 的两个操作（3、4）合并（因为 allowParallelMerge=True）
- Scope=1 和 Scope=2 的 supernode 不合并（因为 Scope=1 的 allowCrossScopeMerge=False）

**验证点**：
- 验证操作1、2在同一个子图中
- 验证操作3、4在同一个子图中
- 验证两个子图保持分离

---

## 测试执行流程

1. **构建计算图**：使用 ComputationalGraphBuilder 按照上述场景创建图
2. **设置 ScopeInfo**：为操作设置 scopeId 和开关属性
3. **运行分区器**：使用 IsoPartitioner 进行图分区
4. **验证结果**：检查子图数量、操作分布是否符合预期

## 预期测试覆盖范围

| 测试用例 | 测试开关 | 验证点 | 预期子图数 |
|---------|---------|--------|-----------|
| 用例1 | 开关1=True | 并行分支合并 | 1 |
| 用例2 | 开关1=False | 并行分支不合并 | 4 |
| 用例3 | 开关2=True | 跨 supernode 合并 | 1 |
| 用例4 | 开关2=False | 跨 supernode 不合并 | 2 |
| 用例5 | 两开关组合 | 组合行为 | 2 |

---

## 注意事项

1. **向后兼容性**：测试应该确保原有功能不受影响
2. **边界条件**：测试应该覆盖空图、单节点图等边界情况
3. **参数验证**：测试应该验证非法参数的处理（如错误的开关值）
4. **性能考虑**：测试应该在大规模图上验证性能影响
