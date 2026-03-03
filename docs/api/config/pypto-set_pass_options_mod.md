# pypto.set\_pass\_options

## 产品支持情况

| 产品             | 是否支持 |
|:-----------------|:--------:|
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 |    √     |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 |    √     |

## 功能说明

修改Pass优化参数信息。其主要功能是在编译流程中，针对特定的优化策略和具体的Pass，动态修改其运行时参数配置，从而实现精细化的控制和调试。

## 函数原型

```python
set_pass_options(*,
                     sg_set_scope: Optional[int] = None,
                     pg_upper_bound: Optional[int] = None,
                     pg_lower_bound: Optional[int] = None,
                     vec_nbuffer_setting: Optional[Dict[int, int]] = None,
                     cube_l1_reuse_setting: Optional[Dict[int, int]] = None,
                     cube_nbuffer_setting: Optional[Dict[int, int]] = None,
                     )
```

## 参数说明

| 参数名                  | 输入/输出 | 说明                                                                 |
|-------------------------|-----------|----------------------------------------------------------------------|
| sg_set_scope            | 输入      | 含义：该参数是手动合图参数。将operation赋予特定的scopeId，若相邻的operation具有相同的非-1的scopeId，则会被强制合并在一个子图之中，并且这个子图不会与其他子图合并。 <br> 类型：int <br> 取值范围：-1~2147483647 <br> 默认值：-1 <br> 影响Pass范围：GraphPartition |
| pg_upper_bound          | 输入      | 含义：合图参数，用于配置子图大小上界。 <br> 说明：当子图大小达到上界不允许与其他子图合并。 <br> 类型：int <br> 取值范围：0~2147483647 <br> 默认值：10000 <br> 影响Pass范围： GraphPartition |
| pg_lower_bound          | 输入      | 含义：合图参数，用于配置子图大小下界。 <br> 说明：当子图大小小于下界时尝试与其他子图合并。 <br> 类型：int <br> 取值范围：0~2147483647 <br> 默认值：512 <br> 影响Pass范围： GraphPartition |
| vec_nbuffer_setting     | 输入      | 含义：该参数是 VECTOR 子图的手动合并参数，开发者显式指定同构子图组ID(hash order)的合并粒度N，将同构子图组内N个子图合并为一个新的子图。<br> 类型：dict[int, int] <br> 取值范围：key: -1 ~ pass计算得到的vector同构子图组数量；value: 1 ~ 2147483647 (INT_MAX) <br> 生效条件：仅在vec_nbuffer_mode = 2时生效 <br> 默认值：{}(空字典) <br> 影响Pass范围： NBufferMerge |
| cube_l1_reuse_setting   | 输入      | 含义：该参数是 L1 缓存复用的手动配置参数，能够通过指定同构子图组ID(hash order)精确控制合并粒度，消除冗余的 GM 到 L1 数据搬运。<br> 类型： dict[int, int] <br> 取值范围：key: -1 ~ pass计算得到的cube同构子图组数量；value: 1 ~ 2147483647 (INT_MAX) <br> 默认值：{}(空字典) <br> 影响Pass范围：L1ReuseMerge |
| cube_nbuffer_setting    | 输入      | 含义：该参数是 CUBE 子图的手动合并参数，开发者显式指定同构子图组ID(hash order)的合并粒度，将同构子图组内N个子图合并为一个新的子图。 <br> 类型：dict[int, int] <br> 取值范围：key: -1 ~ pass计算得到的vector同构子图组数量；value: 1 ~ 2147483647 (INT_MAX) <br> 默认值：{}(空字典) <br> 影响Pass范围： L1ReuseMerge |

## 返回值说明

无。

## 约束说明

- 设置时机：不要求在图编译开始前调用，可以在任何时候进行设置。
- 类型安全：必须确保传入的value的类型与参数定义的类型完全一致，否则可能导致未定义行为或运行时错误。
- 作用范围：参数设置是局部的，只会影响当前jit或者loop内的编译过程，若未设置，则继承上层作用域。

## 调用示例

```python
   pypto.set_pass_options(
                       pg_upper_bound=10000,
                       pg_lower_bound=512,
                       vec_nbuffer_setting={1:2},
                       cube_l1_reuse_setting={0:8},
                       cube_nbuffer_setting={1:2})
```

### dict配置说明
#### 键值对含义
Key (hashorder): 同构子图组id。<br>
- 值 M: 匹配 hashorder 为 M 的特定子图组。<br>
- 值 -1: 匹配所有未显式指定的子图组。<br>

Value (N): 表示合并粒度。即：同构子图组内每N个子图合并为一个新子图执行。<br>
#### 配置行为
Pass 在处理子图合并时，遵循 “完全匹配 > 默认配置 > 不处理” 的逻辑：<br>
- 精确匹配: 若 hashorder 命中字典中的特定 Key，则按其对应的 Value N 进行合并。<br>
- 默认覆盖: 若未精确命中，但字典中存在 -1，则按 -1 对应的 Value 执行合并。<br>
- 不处理: 若既未精确命中也无 -1 配置，则该子图不进行合并优化。<br>
#### 配置示例
| 配置                  | 说明                                                                 |
|---------------------- |----------------------------------------------------------------------|
|{0: 5, 2: 8, -1: 2}    |hashorder为0的同构子图组，每5个子图合并为一个子图；<br>hashorder为2的同构子图组，每8张子图合并为一个子图；<br>其他的同构子图组使用-1对应的默认合并粒度，即每2张子图合并为一个子图。<br> |
|{0: 5}|对于hashorder为0的同构子图组，每5个子图合并为一个子图；<br>其他同构子图组不做处理。|

#### hashorder寻找方式
打印DEBUG日志

## 合图说明

### 概述
合图是指将计算图中多个逻辑上独立的Operation合并为一个逻辑子图，并由该子图最终生成一个物理计算内核（Kernel）的过程。深度学习模型的计算图往往由大量粒度较小的Operation构成，传统逐Operation 执行模式下，每个 Operation 都会独立触发一次内核启动，计算完毕后将中间结果写回全局内存（GM）。这种执行方式在实际硬件上会引入显著的内核启动开销和冗余内存访问，难以充分发挥计算单元的全部计算能力。合图优化通过Operation逻辑聚合，使多个 Operation 在同一Kernel中协同执行。计算的中间结果得以保存在片上高速缓冲中，供下游Operation直接读取，同构消除冗余的GM读写，显著提升计算–访存比并改善整体执行效率。

在 PyPTO 编程模型中，开发者通过 Tensor 和 Tensor Operation 构建计算图。合图过程由编译器内部的优化 Pass 自动完成，用户无需手工编写融合Operation代码。合图 Pass 会在保证计算结果正确性的前提下，对计算图进行分析与重写，将原始计算图划分并重组为更适合目标硬件执行的子图。PyPTO 的合图优化主要分为深度方向合图和广度方向合图两类，分别针对不同的性能瓶颈场景。<br>

#### 深度方向合图

深度方向合图基于计算图中的生产者–消费者关系，沿数据依赖路径将前后相邻的 Operation 进行融合。该方式通过消除中间结果的写回操作，直接优化数据流路径，使原本受限于带宽的算子链得以在单个内核内一气呵成地完成计算。在 PyPTO 中，深度方向的算子融合由 GraphPartition Pass 负责执行。
![](../figures/pypto.set_pass_options_1.png)

#### 广度方向合图
广度方向合图针对计算图中处于同一层级、可并行执行的 Operation，通过将多个并行 Operation 合并到同一 Kernel 中执行以增强单次 Kernel 的计算规模。在核内指令编排阶段，多分支融合能更充分地填充硬件流水，实现更优的多pipe并发。在访存层面，通过归并同源访存，将多次重复的GM到L1搬运整合为单次加载，在节省内存带宽的同时，有效摊薄了 Kernel 启动开销，最终提升了硬件计算单元的整体吞吐率。
![](../figures/pypto.set_pass_options_2.png)

针对昇腾硬件的 CUBE / VECTOR 双计算单元架构，PyPTO 提供了不同的广度方向合图 Pass：
* NBufferMerge Pass：面向 VECTOR 计算单元的并行算子合并。
* L1ReuseMerge Pass：面向 CUBE 计算单元的算子合并。

### 其他概念
#### 同构子图
在计算图中，拓扑结构、算子类型完全一致的局部片段。
#### 同构子图组
互为同构子图的片段，形成的集合被称为同构子图组。在广度方向合图 Pass 处理过程中，会为每一组同构子图生成一个唯一的特征标识，即 hashorder。