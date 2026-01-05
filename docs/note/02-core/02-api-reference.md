# PyPTO API 使用总结

> **适用对象：** 想要系统了解PyPTO API的开发者  
> **学习时间：** 40-60分钟  
> **前置知识：** 已完成[上手指南](../00-getting-started/00-quick-start.md)  
> **学习目标：** 全面掌握PyPTO Python API的使用方法

## 概述

本文档总结 PyPTO 的 Python API 体系，帮助开发者快速了解和使用各项功能。PyPTO 提供了直观的 Tensor 级别抽象，基于 PTO 编程范式，支持从简单计算到复杂模型开发的完整功能。

**API分类：**
- 🎯 **核心API**：`@pypto.jit`、`pypto.Tensor`、Tiling配置
- 🧮 **数学运算**：加减乘除、指数对数、矩阵运算
- 🔄 **形状操作**：reshape、transpose、squeeze/unsqueeze
- ⚙️ **配置选项**：Pass配置、运行模式、日志级别

### 实现细节速查（API ≠ 仅文档：以导出与源码为准）

**PyPTO 的"导出面"在哪看：**
- `python/pypto/__init__.py` 负责导出 `jit` / `RunMode` / `Tensor` / `from_torch` 等，并在 import 时加载后端共享库（`_load_shared_libs()`）。
- **重要：** 本文档中提到的模块名（如 `pypto.jit`、`pypto.Tensor` 等）以实际导出为准，请参考 `python/pypto/__init__.py` 和 `python/pypto/frontend/__init__.py` 中的实际导出内容。

**两条 JIT API（务必区分）：**
- `pypto.jit`（旧版）：实现位于 `python/pypto/runtime.py`，核心类是 `_JIT`
- `pypto.frontend.jit`（新版前端）：实现位于 `python/pypto/frontend/parser/entry.py`，核心类是 `JitCallableWrapper`

**配置 API 的真实实现位置：**
- `pypto.set_pass_options / set_runtime_options / set_codegen_options / set_host_options` 均在 `python/pypto/config.py`
  - 底层通过 `pypto_impl.GetOptions/SetOption` 与后端 options 交互
  - 是否生效以 `framework/src/interface/configs/tile_fwk_config_schema.json` 为准（不在 schema 的 key 不会被设置）

**相关文档：**
- [框架总览](00-overview.md) - PyPTO 框架技术文档总览
- [核心概念详解](10-concepts.md) - PyPTO 核心概念和术语解释
- [架构设计总结](11-architecture-design.md) - PyPTO 架构设计理念

---

## 目录

- [API 概览](#api-概览)
- [核心 API 分类](#核心-api-分类)
- [使用模式](#使用模式)
- [最佳实践](#最佳实践)
- [API 版本兼容性](#api-版本兼容性)
- [总结](#总结)

---

## API 概览

### 主要模块

基于 `python/pypto/__init__.py` 和 `python/pypto/frontend/__init__.py` 的分析：

| 模块 | 功能 | 主要 API | 文档位置 |
|------|------|---------|---------|
| `pypto` | 核心功能入口 | `jit`, `Tensor`, `set_*` | `docs/api/README.md` |
| `pypto.op` | 张量操作 | `add`, `mul`, `matmul` 等 | `docs/api/operation/` |
| `pypto.nn` | 神经网络组件 | `Linear`, `Conv2d` 等 | `docs/api/` |
| `pypto.utils` | 工具函数 | `shape`, `dtype` 等 | `docs/api/others/` |

### API 层次结构

PyPTO API 采用分层设计，从基础到高级：

```python
import pypto

# 1. 张量创建和操作
tensor = Tensor(shape, dtype)  # 基础创建
result = tensor.add(other)        # 基础操作

# 2. 函数定义和编译
@pypto.jit                           # JIT 装饰器
def my_function(x, y):
    return x + y                    # 函数定义

# 3. 配置设置
pypto.set_vec_tile_shapes(64, 512)  # Tile 配置
pypto.set_cube_tile_shapes([128, 128], [128, 128], [128, 128])  # Cube 配置
```

### API 设计原则

**Pythonic 设计：**
- 使用标准的 Python 语法和约定
- 支持操作符重载（`+`, `-`, `*`, `/`）
- 提供链式调用接口

**类型安全：**
- 强类型检查和推断
- 支持静态类型注解
- 运行时类型验证

**性能优化：**
- 零拷贝设计（尽可能避免数据拷贝）
- 延迟执行模型
- 编译时优化

---

## 核心 API 分类

### 1. 张量操作 API

#### 创建类 API

基于 `docs/api/tensor/pypto-Tensor构造函数.md`：

```python
# 从 NumPy 数组创建
tensor = Tensor([1, 2, 3], DataType.DT_FP32)

# 创建特定形状的张量
zeros = pypto.zeros((3, 4))          # 全零张量
ones = pypto.ones((2, 3))            # 全一张量
full = pypto.full((2, 2), 5.0)       # 填充特定值

# 从已有张量创建
cloned = tensor.clone()               # 深拷贝
view = tensor.view((6,))             # 视图（零拷贝）
reshaped = tensor.reshape((2, 3))    # 重塑形状
```

#### 数学运算 API

基于 `docs/api/tensor/` 下的各种操作文档：

```python
# 基本运算
result = a + b     # 或 a.add(b)
result = a - b     # 或 a.sub(b)
result = a * b     # 或 a.mul(b)
result = a / b     # 或 a.div(b)

# 矩阵运算
result = a.matmul(b)                 # 矩阵乘法
result = a.transpose(0, 1)           # 转置

# 逐元素运算
result = a.exp()                      # 指数
result = a.log()                      # 对数
result = a.sqrt()                     # 平方根
result = a.sin()                      # 正弦
result = a.cos()                      # 余弦

# 规约运算
sum_result = a.sum(dim=0)            # 求和
max_result = a.max(dim=1)            # 最大值
mean_result = a.mean(dim=-1)         # 均值
```

#### 形状操作 API

```python
# 维度操作
expanded = tensor.unsqueeze(0)        # 增加维度
squeezed = tensor.squeeze(1)          # 移除维度

# 切片操作
sliced = tensor[1:3, :]              # 行切片
indexed = tensor[[0, 2, 4]]          # 索引选择

# 拼接和分割
concatenated = a.concat(b, dim=0)  # 拼接
split_tensors = pypto.split(tensor, 2, dim=0)  # 分割
```

### 2. 函数编译 API

#### JIT 装饰器

基于 `docs/api/pypto-frontend-jit.md`：

```python
# 基本使用
@pypto.jit
def simple_add(x, y):
    return x + y

# 带配置参数（参考 @docs/api/config/pypto-jit.md）
@pypto.jit(
    codegen_options={"support_dynamic_aligned": True},
    host_options={"only_codegen": False}
)
def configured_function(x):
    return x * 2
```

#### 配置 API

基于 `docs/api/config/` 下的配置文档：

```python
# Tile 形状设置
pypto.set_vec_tile_shapes(height=64, width=512)  # Vector 计算 Tile
pypto.set_cube_tile_shapes([128, 128], [128, 128], [128, 128])  # Matrix 计算 Tile

# Pass 配置
# Pass 配置（参考 @docs/api/config/pypto-set_pass_config.md）
from pypto import PassConfigKey
pypto.set_pass_config("default", "all", PassConfigKey.KEY_DUMP_GRAPH, True)

# 编译选项
# 编译选项通过 jit 的 codegen_options、host_options 等参数配置
```

### 3. 控制流 API

#### 条件分支

```python
@pypto.jit
def conditional_op(x, y, condition):
    # 使用 where 函数实现条件选择
    return pypto.where(condition, x, y)

@pypto.jit
def if_else_example(x, threshold):
    # 简单的条件逻辑
    mask = x > threshold
    return pypto.where(mask, x * 2, x / 2)
```

#### 循环操作

```python
@pypto.jit
def loop_sum(x, n):
    result = x
    for i in range(n):
        result = result + x
    return result

@pypto.jit
def dynamic_loop(x, lengths):
    # 基于动态形状的循环（使用 pypto.loop）
    # 注意：动态 shape 应使用 from_torch 的 dynamic_axis 参数标记
    batch_size = x.shape[0]  # 使用 Tensor.shape 属性
    for idx in pypto.loop(batch_size):  # 使用 pypto.loop 创建动态循环
        # 循环处理逻辑
        pass
    return x
```

### 4. 高级特性 API

#### 动态形状

动态 shape 应使用 `pypto.from_torch` 的 `dynamic_axis` 参数标记（参考 @docs/api/others/pypto-from_torch.md）：

```python
import torch
import pypto

# 创建 PyTorch 张量
x_torch = torch.randn(32, 128)

# 使用 from_torch 标记动态维度（第 0 维为动态）
x_pto = pypto.from_torch(x_torch, dynamic_axis=[0])  # 标记第 0 维为动态

@pypto.jit
def dynamic_reshape(x):
    # 获取动态维度
    batch_size = x.shape[0]  # 使用 Tensor.shape 属性

    # 动态重塑
    reshaped = x.reshape((batch_size, -1))
    return reshaped
```

#### 量化操作

```python
# 量化操作（参考 @docs/api/operation/pypto-quantize.md）
# 注意：dequantize 接口请参考 API 文档确认是否存在
@pypto.jit
def quantized_computation(x, scale, zero_point):
    # 量化（参考 API 文档确认参数格式）
    # quantized = pypto.quantize(x, scale, zero_point)
    # 注意：请参考实际 API 文档使用
    return x
```

#### 内存管理

```python
# 缓存策略设置
tensor.set_cache_policy('l1_cache')   # L1 缓存优先
tensor.set_cache_policy('l2_cache')   # L2 缓存优先

# 内存布局控制
# 注意：PyPTO Tensor 不提供 contiguous() 方法
# 内存布局由框架自动管理，from_torch 要求输入 torch.Tensor 是连续的
```

---

## 使用模式

### 1. 基本计算模式

最简单的 PyPTO 使用模式，适合快速原型开发：

```python
import pypto
import numpy as np

# 数据准备
a_np = np.random.randn(1024, 1024).astype(np.float32)
b_np = np.random.randn(1024, 1024).astype(np.float32)

# 张量创建
a = pypto.zeros(a_np.shape)
b = pypto.zeros(b_np.shape)

# 计算定义
@pypto.jit
def matrix_multiply(x, y):
    return x.matmul(y)

# 执行计算
result = matrix_multiply(a, b)
# 注意：PyPTO Tensor 不提供 to_numpy() 方法
# 如需获取数据，请使用 torch.Tensor 作为输入输出，通过 from_torch 转换
```

### 2. 模型开发模式

适合构建复杂的神经网络模型：

```python
import pypto
import numpy as np

class SimpleMLP:
    def __init__(self, input_size, hidden_size, output_size):
        # 参数初始化
        # 初始化为零张量，实际使用时需要合适的初始化方法
        self.w1 = pypto.zeros((input_size, hidden_size))
        self.b1 = pypto.zeros((hidden_size,))
        self.w2 = pypto.zeros((hidden_size, output_size))
        self.b2 = pypto.zeros((output_size,))

    @pypto.jit
    def forward(self, x):
        # 前向传播
        h = (x.matmul(self.w1) + self.b1).relu()
        return h.matmul(self.w2) + self.b2

    def __call__(self, x):
        return self.forward(x)

# 使用模型
model = SimpleMLP(784, 256, 10)
input_tensor = pypto.zeros((32, 784))  # 简化为零张量作为示例
output = model(input_tensor)
```

### 3. 性能优化模式

针对高性能计算的优化配置：

```python
# Tile 形状优化
pypto.set_vec_tile_shapes(64, 512)    # Vector 计算优化
pypto.set_cube_tile_shapes([128, 128], [128, 128], [128, 128])  # Matrix 计算优化

# Pass 优化配置
# Pass 配置（参考 @docs/api/config/pypto-set_pass_config.md）
from pypto import PassConfigKey
pypto.set_pass_config("default", "all", PassConfigKey.KEY_DUMP_GRAPH, True)

# 编译优化
@pypto.jit(
    codegen_options={"support_dynamic_aligned": True},
    pass_options={}
)
def optimized_computation(x, y):
    # 复杂的计算逻辑
    temp1 = x.matmul(y)
    temp2 = temp1.relu()
    temp3 = temp2.sum(dim=-1)
    return temp3.softmax(dim=-1)

# 执行优化计算
result = optimized_computation(input1, input2)
```

### 4. 调试和分析模式

用于开发和调试阶段：

```python
# 启用调试模式
# 调试选项通过 jit 的 host_options 配置

# 详细日志
# 日志级别通过环境变量 GLOBAL_LOG_LEVEL 设置

# 性能分析
@pypto.jit  # 参考 @docs/api/config/pypto-jit.md，jit 不支持 profile 参数
def profiled_function(x):
    return complex_computation(x)

# 执行并分析
result = profiled_function(input_data)
# 性能分析使用系统工具（perf、gprof 等）
```

---

## 最佳实践

### 1. 张量操作优化

#### 使用原地操作
```python
# 推荐：原地操作节省内存
tensor.add_(other)  # 而不是 tensor = tensor + other

# 推荐：原地运算符
tensor *= 2         # 而不是 tensor = tensor * 2
tensor += other     # 而不是 tensor = tensor + other
```

#### 合理使用视图
```python
# 推荐：使用 view 而非 reshape（当可能时）
view_tensor = tensor.view((new_shape,))  # 零拷贝

# 推荐：连续内存布局
contiguous_tensor = tensor.contiguous()  # 确保内存连续
```

#### 批量操作
```python
# 推荐：批量处理而非循环
# 注意：pypto 不存在 stack 接口，需要使用其他方式处理批次
batch_results = [process(x) for x in batch]
```

### 2. 函数编译优化

#### 合理设置 Tile 形状
```python
# 根据数据形状和硬件特性设置
pypto.set_vec_tile_shapes(64, 512)    # Vector 计算
pypto.set_cube_tile_shapes([128, 128], [128, 128], [128, 128])  # Matrix 计算

# 经验规则：
# - Vector Tile: 16KB-64KB 范围，尾轴 32B 对齐
# - Cube Tile: 根据矩阵维度选择合适的 Tile 大小
```

#### 使用合适的数据类型
```python
# 推理使用 FP16，训练使用 FP32
weights_fp16 = weights.to(pypto.DT_FP16)  # (@docs/api/datatype/DataType.md)
inputs_fp16 = inputs.to(pypto.DT_FP16)

# 计算完成后转换回原始类型
result_fp32 = result.to(pypto.DT_FP32)  # (@docs/api/datatype/DataType.md)
```

#### 控制编译范围
```python
# 大函数拆分编译
@pypto.jit
def small_function(x):
    return x * 2

@pypto.jit
def large_function(x, y, z):
    temp1 = small_function(x)  # 编译为独立函数
    temp2 = small_function(y)
    return temp1 + temp2 + z   # 编译为组合函数
```

### 3. 内存管理

#### 及时释放资源
```python
# 手动释放临时张量
del intermediate_result

# 使用上下文管理器
with pypto.no_grad():
    result = compute_heavy_operation(x)
```

#### 复用缓冲区
```python
# 框架会自动优化，但可以显式控制
@pypto.jit
def reuse_memory(a, b):
    # 编译器会自动优化内存复用
    temp = a + b
    result = temp * 2
    return result  # temp 的内存可以被复用
```

#### 监控内存使用
```python
# 获取内存信息
memory_info = pypto.get_memory_info()
print(f"Total memory: {memory_info.total}")
print(f"Used memory: {memory_info.used}")
print(f"Free memory: {memory_info.free}")
```

### 4. 错误处理

#### 类型检查
```python
# 运行时类型检查
assert isinstance(tensor, pypto.Tensor)
assert tensor.dtype == pypto.DT_FP32  # (@docs/api/datatype/DataType.md)
assert tensor.shape == (batch_size, feature_dim)
```

#### 形状验证
```python
@pypto.jit
def validate_shapes(x, y):
    # 编译时形状检查
    assert x.shape[1] == y.shape[0], "矩阵乘法维度不匹配"  # 使用 Tensor.shape 属性
    return x.matmul(y)
```

### 5. 性能监控

#### 编译时间监控
```python
import time

start_time = time.time()
@pypto.jit
def compiled_function(x):
    return complex_computation(x)

compile_time = time.time() - start_time
print(f"编译时间: {compile_time:.3f} 秒")
```

#### 执行性能分析
```python
# 使用性能分析工具
# 性能分析使用系统工具（perf、gprof 等）
result = function(input_data)

print(prof.summary())  # 输出性能摘要
```

---

## API 版本兼容性

### 当前版本：v1.0

**主要特性：**
- 完整的张量操作 API（数学运算、形状操作等）
- JIT 编译支持（同步/异步执行）
- NPU 执行支持（CloudNPU 平台）
- 动态形状支持（运行时确定形状）
- 量化操作支持（INT8 计算）

### 兼容性保证

#### 向后兼容
- 已发布的稳定 API 保持不变
- 配置文件格式保持兼容
- 序列化格式保持兼容
- 错误码和异常类型保持稳定

#### 升级建议
- **渐进式升级**：分阶段升级，避免一次性变更过多
- **测试先行**：升级前进行充分的回归测试
- **备份策略**：保留旧版本作为降级方案

### 版本差异说明

#### v1.0 新特性
```python
# 动态形状支持（使用 from_torch 的 dynamic_axis 参数）
import torch
x_torch = torch.randn(32, 128)
x_pto = pypto.from_torch(x_torch, dynamic_axis=[0])  # 标记第 0 维为动态

@pypto.jit
def dynamic_function(x):
    batch_size = x.shape[0]  # 运行时获取形状（使用 Tensor.shape 属性）
    return x.reshape((batch_size, -1))
```

#### 废弃功能
- 无（v1.0 为首个稳定版本）

#### 未来规划
- **v1.1**：增强的自动调优功能
- **v2.0**：多硬件平台支持
- **v2.x**：分布式计算支持

---

## 总结

PyPTO API 设计遵循以下核心原则：

### 1. Pythonic 设计
- 直观的 Python 接口风格
- 丰富的操作符重载
- 链式调用支持
- 符合 Python 开发习惯

### 2. 性能优先
- 零拷贝设计理念
- 编译时优化策略
- 运行时效率优化
- 硬件特性充分利用

### 3. 类型安全
- 强类型检查机制
- 静态类型推断
- 运行时类型验证
- 错误信息清晰

### 4. 易于扩展
- 插件化架构设计
- 配置驱动机制
- 自定义操作支持
- 多后端适配能力

通过合理使用这些 API，开发者可以：
- **快速原型**：使用直观的 Tensor API 快速实现算法
- **性能优化**：通过配置和最佳实践获得高性能
- **生产就绪**：利用类型安全和错误处理构建可靠应用

### API 学习路径

**入门阶段：**
1. 掌握基础张量操作
2. 学习 JIT 编译用法
3. 理解基本配置选项

**进阶阶段：**
1. 深入性能优化技巧
2. 掌握内存管理最佳实践
3. 学习高级特性使用

**专家阶段：**
1. 自定义操作开发
2. 编译器扩展开发
3. 多硬件平台适配

---

**相关文档：**
- [Hello World 示例](../01-examples/00-hello-world.md) - 基础 API 使用示例
- [Softmax 示例](../01-examples/03-softmax.md) - 进阶 API 使用示例
- [架构设计总结](11-architecture-design.md) - 理解 API 设计背景
