---
name: pypto-operator-perf-tuning
description: PyPTO 算子性能调优技能。提供系统化的性能问题诊断、根因分析和优化策略选择流程。当遇到算子性能瓶颈、需要优化执行效率或降低控制开销时使用此技能。
---

# PyPTO 算子性能调优

## 概述

此技能提供系统化的性能调优工作流程，包括性能问题诊断、根因分析、优化策略选择和效果验证。

## 核心优化思路

**通用优化原则**：
1. **数据局部性优化**：将高频访问的数据放在靠近 CPU 的位置（栈、缓存、寄存器）
2. **减少内存访问开销**：避免频繁的 page fault 和缓存未命中
3. **降低控制开销**：减少不必要的同步、分支和函数调用
4. **提高并行度**：充分利用多核和多线程能力

## 工作流程

### 阶段 1：性能问题诊断

#### 步骤 1.1：启用性能数据采集

在算子实现文件中启用性能调试：

```python
@pypto.frontend.jit(
    runtime_options={"run_mode": "npu"},
    debug_options={"runtime_debug_mode": 1}
)
def kernel_function(...):
    # 算子实现
    return result
```

#### 步骤 1.2：生成性能数据

```bash
# 编译
python3 build_ci.py -f python3 --disable_auto_execute

# 运行并生成性能数据
python3 custom/operator_name/operator.py --run-mode npu
```

生成的文件：
- `merged_swimlane.json` - 泳道图数据
- `machine_runtime_operator_trace.json` - 性能追踪数据
- `bubble_analysis.log` - 气泡分析报告

#### 步骤 1.3：使用分析脚本诊断

```bash
python3 .opencode/skills/pypto-operator-perf-tuning/scripts/diagnose.py <output_dir>
```

脚本会自动分析并输出：
- 性能瓶颈识别
- 根因分析
- 优化建议

### 阶段 2：根因分析

#### 2.1 控制开销分析

**症状**：控制开销占比 > 50%

**可能原因**：
1. **频繁的内存访问**
   - 高频函数中访问堆上数据
   - 数据局部性差，缓存导致 page fault

2. **过多的同步操作**
   - 不必要的原子操作
   - 频繁的线程间通信

3. **复杂的控制流**
   - 深层嵌套的条件判断
   - 频繁的函数调用

**诊断方法**：
- 查看泳道图中 AICPU-CTRL 阶段耗时
- 检查 `DEV_TASK_SCHED_EXEC` 阶段是否异常
- 分析任务下发和解依赖的时间分布

#### 2.2 计算效率分析

**症状**：计算时间占比高，但吞吐量低

**可能原因**：
1. **数据传输瓶颈**
   - 频繁的内存拷贝
   - 数据对齐问题

2. **指令流水线停顿**
   - 数据依赖导致流水线阻塞
   - 分支预测失败

3. **资源利用率低**
   - 向量化不足
   - 并行度不够

**诊断方法**：
- 查看任务执行时间分布
- 分析 AIC/AIV 核心利用率
- 检查是否有空闲时间片

#### 2.3 内存访问分析

**症状**：内存访问延迟高，缓存未命中率高

**可能原因**：
1. **访问模式不连续**
   - 跨步访问数组
   - 随机访问模式

2. **数据结构设计不合理**
   - 链表等间接访问结构
   - 数据结构过大导致缓存失效

3. **频繁的动态分配**
   - 循环内频繁 new/delete
   - 小内存块分配

**诊断方法**：
- 分析内存访问模式
- 检查是否有频繁的堆分配
- 评估数据结构的缓存友好性

### 阶段 3：优化策略选择

根据根因分析结果，选择合适的优化策略：

#### 策略 1：数据局部性优化

**适用场景**：
- 高频函数中频繁访问某些数据
- 控制开销占比高
- 诊断显示大量 page fault

**优化方法**：

1. **栈上分配替代堆分配**
   ```cpp
   // 修改前
   SchduleContext* context = new SchduleContext();
   // ... 使用 context
   delete context;

   // 修改后
   SchduleContext localContext;
   // ... 使用 localContext
   // 自动销毁，无需手动释放
   ```

2. **使用引用减少解引用**
   ```cpp
   // 修改前
   void function(Data* ptr) {
       for (int i = 0; i < n; i++) {
           value = ptr->data[i];  // 每次解引用
       }
   }

   // 修改后
   void function(Data* ptr) {
       Data& ref = *ptr;  // 一次解引用
       for (int i = 0; i < n; i++) {
           value = ref.data[i];  // 直接访问
       }
   }
   ```

3. **数据结构紧凑化**
   - 将频繁访问的字段放在结构体前面
   - 减少结构体填充（padding）
   - 使用数组替代链表

**预期效果**：
- 减少 page fault 次数
- 提高缓存命中率
- 降低控制开销 20-50%

#### 策略 2：内存访问模式优化

**适用场景**：
- 内存访问延迟高
- 缓存未命中率高
- 访问模式不连续

**优化方法**：

1. **循环重排**
   ```cpp
   // 修改前：缓存不友好
   for (int i = 0; i < rows; i++) {
       for (int j = 0; j < cols; j++) {
           sum += matrix[j][i];  // 跨步访问
       }
   }

   // 修改后：缓存友好
   for (int i = 0; i < rows; i++) {
       for (int j = 0; j < cols; j++) {
           sum += matrix[i][j];  // 连续访问
       }
   }
   ```

2. **数据预取**
   ```cpp
   // 手动预取
   for (int i = 0; i < n; i++) {
       __builtin_prefetch(&data[i + PREFETCH_DISTANCE]);
       process(data[i]);
   }
   ```

3. **内存对齐**
   - 使用 alignas 确保数据对齐
   - 避免跨缓存行访问
   - 使用 SIMD 友好的数据布局

**预期效果**：
- 提高缓存命中率
- 减少内存访问延迟
- 提升计算效率 10-30%

#### 策略 3：控制流优化

**适用场景**：
- 复杂的条件判断
- 频繁的函数调用
- 分支预测失败率高

**优化方法**：

1. **分支预测优化**
   ```cpp
   // 修改前：分支不可预测
   for (int i = 0; i < n; i++) {
       if (data[i] > threshold) {
           process(data[i]);
       }
   }

   // 修改后：分支可预测
   int* less = temp;
   int* greater = temp + n;
   for (int i = 0; i < n; i++) {
       if (data[i] > threshold) {
           *greater++ = data[i];
       } else {
           *less++ = data[i];
       }
   }
   // 分别处理两个数组
   ```

2. **函数内联**
   - 将小函数标记为 inline
   - 减少函数调用开销
   - 避免上下文切换

3. **查表法替代计算**
   ```cpp
   // 修改前：复杂计算
   int result = expensive_computation(x);

   // 修改后：查表
   static const int lookup_table[] = {0, 1, 4, 9, 16, ...};
   int result = lookup_table[x];
   ```

**预期效果**：
- 减少分支预测失败
- 降低函数调用开销
- 提升执行效率 5-20%

#### 策略 4：并行度优化

**适用场景**：
- 核心利用率低
- 任务串行执行
- 有明显的并行机会

**优化方法**：

1. **任务并行化**
   - 识别独立的任务
   - 使用线程池并行执行
   - 避免锁竞争

2. **数据并行化**
   - 使用 SIMD 指令
   - 向量化循环
   - 批量处理数据

3. **流水线化**
   - 重叠计算和通信
   - 双缓冲技术
   - 异步执行

**预期效果**：
- 提高核心利用率
- 减少等待时间
- 提升吞吐量 2-10 倍

### 阶段 4：代码修改指南

#### 步骤 4.1：定位优化点

使用以下方法定位需要优化的代码：

1. **性能热点分析**
   - 查看泳道图中耗时最长的阶段
   - 识别高频调用的函数
   - 定位循环中的瓶颈

2. **代码审查**
   - 查找频繁的内存访问
   - 识别不必要的拷贝
   - 检查可以优化的数据结构

3. **编译器优化建议**
   - 启用编译器优化选项
   - 查看编译器警告
   - 使用 PGO（Profile-Guided Optimization）

#### 步骤 4.2：应用优化策略

根据选择的优化策略，修改代码：

1. **修改前备份**
   - 使用版本控制系统
   - 记录修改原因
   - 保留原始实现

2. **渐进式修改**
   - 一次应用一个优化
   - 每次修改后验证
   - 避免引入多个变量

3. **保持功能正确性**
   - 添加单元测试
   - 验证输出结果
   - 检查边界条件

#### 步骤 4.3：代码示例

**示例 1：高频数据访问优化**

```cpp
// 场景：在循环中频繁访问类成员
class Processor {
    std::vector<int> data_;
    std::atomic<int> counter_;

    void process() {
        // 修改前：每次访问都通过 this 指针
        for (size_t i = 0; i < data_.size(); i++) {
            data_[i] = counter_.load() * 2;
        }
    }

    void process_optimized() {
        // 修改后：使用局部变量
        std::vector<int>& data = data_;
        int counter = counter_.load();

        for (size_t i = 0; i < data.size(); i++) {
            data[i] = counter * 2;
        }
    }
};
```

**示例 2：内存分配优化**

```cpp
// 场景：循环内频繁分配小对象
void process_loop() {
    // 修改前：每次循环都分配
    for (int i = 0; i < n; i++) {
        TempBuffer* buf = new TempBuffer();
        process(buf);
        delete buf;
    }
}

void process_loop_optimized() {
    // 修改后：循环外分配，循环内复用
    TempBuffer buf;
    for (int i = 0; i < n; i++) {
        buf.reset();
        process(&buf);
    }
}
```

**示例 3：数据结构优化**

```cpp
// 场景：频繁访问的结构体
// 修改前：字段分散，缓存不友好
struct DataBad {
    char flag;           // 1 byte
    double value;        // 8 bytes
    int id;             // 4 bytes
    // padding: 7 bytes
    // total: 20 bytes
};

// 修改后：字段紧凑，缓存友好
struct DataGood {
    double value;        // 8 bytes
    int id;             // 4 bytes
    char flag;           // 1 byte
    // padding: 3 bytes
    // total: 16 bytes
};
```

### 阶段 5：效果验证

#### 步骤 5.1：性能对比

1. **生成优化后的性能数据**
   ```bash
   python3 custom/operator_name/operator.py --run-mode npu
   ```

2. **对比关键指标**
   - 控制开销占比
   - 计算时间
   - 吞吐量
   - 延迟

3. **使用对比脚本**
   ```bash
   python3 .opencode/skills/pypto-operator-perf-tuning/scripts/compare.py <before_dir> <after_dir>
   ```

#### 步骤 5.2：功能验证

1. **单元测试**
   - 验证输出正确性
   - 检查边界条件
   - 测试异常情况

2. **回归测试**
   - 对比优化前后的输出
   - 确保精度满足要求
   - 检查内存泄漏

#### 步骤 5.3：稳定性测试

1. **多次运行测试**
   - 验证性能稳定
   - 检查是否有性能抖动
   - 测试长时间运行

2. **不同规模测试**
   - 小规模：验证功能正确性
   - 中规模：验证典型场景
   - 大规模：验证性能可扩展性

## 优化检查清单

在应用性能优化时，检查以下项目：

- [ ] **问题诊断**：使用分析脚本识别性能瓶颈
- [ ] **根因分析**：确定性能问题的根本原因
- [ ] **策略选择**：根据根因选择合适的优化策略
- [ ] **代码备份**：保留优化前的代码
- [ ] **渐进式修改**：一次应用一个优化
- [ ] **功能验证**：确保优化后功能正确
- [ ] **性能对比**：量化优化效果
- [ ] **稳定性测试**：验证性能稳定性
- [ ] **文档更新**：记录优化原因和效果

## 常见问题

### Q1: 如何判断应该使用哪种优化策略？

A: 根据性能诊断结果：
- **控制开销高** → 数据局部性优化
- **内存访问延迟高** → 内存访问模式优化
- **分支预测失败多** → 控制流优化
- **核心利用率低** → 并行度优化

### Q2: 优化后性能反而下降怎么办？

A: 检查以下方面：
- 是否引入了新的性能瓶颈
- 是否破坏了编译器优化
- 是否增加了不必要的拷贝
- 是否引入了锁竞争
- 回退到优化前的代码，重新分析

### Q3: 如何平衡性能和代码可读性？

A: 遵循以下原则：
- 优先选择算法级优化
- 使用注释解释优化原因
- 保持接口简洁
- 复杂优化单独封装
- 提供未优化版本用于调试

### Q4: 优化效果如何量化？

A: 使用以下指标：
- 吞吐量提升倍数
- 延迟降低百分比
- 控制开销降低比例
- 核心利用率提升
- 功耗变化（如可测量）

### Q5: 如何确保优化不影响其他场景？

A: 进行全面测试：
- 不同输入规模
- 不同数据分布
- 边界条件
- 异常情况
- 多线程并发

## 参考资料

- [泳道图分析文档](https://pypto.gitcode.com/tools/swimlane_graph/index.html)
- [性能分析技能](../pypto-operator-perf-autotune/SKILL.md)
- [内存优化最佳实践](references/memory_optimization.md)
- [C++ 性能优化指南](references/cpp_performance.md)
