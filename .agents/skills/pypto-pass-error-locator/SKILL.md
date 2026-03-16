---
name: pypto-pass-error-locator
description: PyPTO Pass 模块问题定位技能。用于解析 PyPTO 运行日志，提取错误信息、堆栈跟踪等关键信息，识别错误是哪个 pass 模块导致的，分析错误产生的可能原因，为用户自行修复和通过skill自动修复提供参考。当需要分析 PyPTO pass 模块错误时使用此技能。
license: 完整条款见 LICENSE.txt
---

# PyPTO Pass Error locator Skill

## 概述

本技能提供 PyPTO Pass 模块错误的完整分析能力，包括日志解析、错误识别、原因分析和修复建议。

## 使用场景

当需要根据执行异常日志，分析定位某pass异常原因以及如何修复该问题时使用此技能。

## 功能概述

- 解析 PyPTO 运行日志文件，提取错误信息
- 识别错误相关的 pass 模块
- 分析错误产生的根本原因
- 提供错误分类和严重程度评估
- 生成详细的错误分析报告
- 提供问题修复策略建议

## 触发机制

当用户输入包含以下关键字时，自动触发此技能：
- **执行xx用例失败，分析pass失败原因**：根据用例执行日志信息，识别错误是哪个 pass 模块导致的，分析错误产生的可能原因
- **根据xx日志信息，分析pass失败原因**：根据用户提供的错误日志信息，识别错误是哪个 pass 模块导致的，分析错误产生的可能原因

## 工作流程

### 步骤 1：获取日志

1. 如果用户提供了具体的错误信息，直接使用用户提供的日志内容
2. 如果用户提供了具体的错误日志路径，从用户指定的错误文件中获取日志内容
3. 如果用户提供了具体的错误日志目录，从用户指定的日志目录下查找日志文件并获取日志内容
4. 如果用户没提供具体错误信息或日志路径，从如下位置获取日志文件
    - `$HOME/ascend/log/` 目录（默认）
    - `$ASCEND_PROCESS_LOG_PATH/` 目录（如果设置）
    - `$ASCEND_WORK_PATH/` 目录（如果设置且未设置 ASCEND_PROCESS_LOG_PATH）

### 步骤 1：解析日志

**日志格式示例**：
```
[INFO ] PYPTO(592313):2026-03-13 14:26:37.767 [l1_copy_reuse.cpp:642][PASS]:[L1CopyInReuseMerge.Operation]:Op 704 feature: [63, 128, 0, 32, 512, 148].
```

**解析模式**：

1. **日志级别识别**
    - 模式：`\[(DEBUG|INFO|WARNING|ERROR)\s*\]`
    - DEBUG (0 ASCEND_GLOBAL_LOG_LEVEL): 调试信息
    - INFO (1 ASCEND_GLOBAL_LOG_LEVEL): 一般信息
    - WARNING (2 ASCEND_GLOBAL_LOG_LEVEL): 警告信息
    - ERROR (3 ASCEND_GLOBAL_LOG_LEVEL): 致命错误

2. **模块名提取**
   - 模式：`PYPTO\((\d+)\):`
   - 模块名：`PYPTO`
   - 进程ID：`592313`

3. **时间戳提取**
   - 模式：`\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3}`
   - 示例：`2026-03-13 14:26:37.767`

4. **文件位置提取**
   - 模式：`\[([a-zA-Z0-9_]+\.(cpp|py|h|cc)):(\d+)\]`
   - 文件名：`l1_copy_reuse.cpp`
   - 行号：`642`

5. **所属模块识别**
   - 模式：`\[(FUNCTION|PASS|CODEGEN|MACHINE|DISTRIBUTED|SIMULATION|VERIFY)\]:`
   - FUNCTION: 表示这是函数调用相关的日志，记录函数入口、出口、参数等信息
   - PASS: 表示这是pass模块的日志，记录各个pass阶段的执行情况
   - CODEGEN: 表示这是代码生成相关的日志，记录中间表示(IR)生成、优化等信息
   - MACHINE: 表示这是机器码生成相关的日志，记录汇编代码生成、指令调度等信息
   - DISTRIBUTED: 表示这是分布式相关的日志，记录多设备通信、数据分发等信息
   - SIMULATION: 表示这是仿真相关的日志，记录NPU仿真执行、内存模拟等信息
   - VERIFY: 表示这是验证相关的日志，记录结果验证、精度检查等信息
   - 示例：`[PASS]:`

6. **Pass 模块信息提取**
   - 模式：`\[([a-zA-Z0-9_]+\.[a-zA-Z0-9_]+)\]:`
   - Pass模块名：`L1CopyInReuseMerge.Operation`

7. **消息内容提取**
    - 提取最后一个冒号后的所有内容
    - 示例：`Op 704 feature: [63, 128, 0, 32, 512, 148].`

8. **堆栈跟踪提取**（Python异常）
    - 识别 `Traceback (most recent call last):` 开始的堆栈
    - 提取每个堆栈帧的文件、行号、函数名

### 步骤 3：Pass 模块识别

**识别策略**：

1. **直接识别**：从错误日志解析获取的关键信息中提取 pass 名称
   - 模式：`Pass\[([^\]]+)\]`
   - 模式：`in pass ([a-zA-Z0-9_]+)`

2. **堆栈分析**：从堆栈跟踪中推断 pass 模块
   - 检查文件路径：`framework/passes/`
   - 检查函数名：`run_pass`, `apply_pass`, `execute`

3. **错误模式匹配**：根据错误特征推断 pass 模块
   - 冗余消除 → RemoveUndrivenView、RemoveRedundantOp、CommonOperationEliminate
   - 切图合图 → GraphPartition、ReduceCopyMerge、NBufferMerge、L1CopyInReuseMerge
   - 数据通路 → IntraSubgraphAdapter、GenerateMoveOp
   - 动态推导 → InferDynShape、InferParamIndex
   - 内存管理 → InferMemoryConflict、SrcDstBufferMerge、AddAlloc、OoOSchedule、RemoveAlloc、GlobalMemoryReuse
   - 调度优化 → OoOSchedule

### 步骤 4：错误原因分析

**分析维度**：

1. **输入数据问题**
   - Shape 不对齐
   - Dtype 不匹配
   - 数据范围异常
   - 空张量/零维度

2. **API 使用问题**
   - 参数错误
   - 不支持的配置
   - 缺少必需参数
   - 参数冲突

3. **环境问题**
   - NPU 设备不可用
   - 内存不足
   - CANN 版本不兼容
   - 驱动问题

4. **代码逻辑问题**
   - 算子实现错误
   - 边界条件处理不当
   - 类型转换错误
   - 内存访问越界

### 步骤 5：Pass 模块知识库

**常见 Pass 模块及错误模式**：

| Pass 模块 | 功能 | 常见错误 | 可能原因 |
|----------|------|---------|---------|
| **RemoveRedundantReshape** | 删除冗余的reshape操作（当所有消费者都是reshape时） | 输入输出操作数无效、消费者为空指针、操作已删除标记错误 | 输入输出操作数数量不为1、输入输出tensor为空指针、消费者为空指针、操作已被其他pass标记为删除 |
| **AutoCast** | 自动插入类型转换操作（BF16/FP16/INT32-FP16）并优化冗余CAST链 | 获取InOutCast连接tensor失败、插入BF16 CAST失败、插入FP16 CAST失败、插入INT32-FP16 CAST失败、冗余CAST链消除失败 | 架构不支持某些数据类型、INCAST/OUTCAST节点连接问题、CAST链优化过程中图结构问题、类型转换路径不合法 |
| **InferMemoryConflict** | 推断内存冲突并插入register copy | 初始化失败、前向传播失败、后向传播失败、插入Copy失败、更新前向tensor失败、更新后向tensor失败、设置默认shape失败、推断tile shape失败、获取reshape tile失败、插入前置copy失败、插入后置copy失败、获取tile shape失败、ViewType处理失败、检查原始shape冲突失败、检查冲突失败、检查传输失败、验证tile shape失败 | ViewType输入输出tensor的数据类型不在viewType表中、vecTypeTile的tile维度n不是偶数、输入输出tensor的raw shape中有负数（动态shape）、输入输出tensor的raw size不相等且不匹配特定的reshape pattern（4D<->2D, 3D<->2D的batch matmul优化pattern）、tile shape维度与输入shape维度不等、tile shape大小为0、无法推导tile shape、推导reshape tile shape失败 |
| **RemoveUndrivenView** | 删除无驱动的view操作（将OP_ASSEMBLE_SSA降级为OP_ASSEMBLE并删除无驱动的OP_VIEW） | 缺少INPLACE_IDX属性错误、生产者数量无效错误、生产者类型错误 | OP_ASSEMBLE_SSA缺少INPLACE_IDX属性、输入操作数的生产者数量不为1、生产者类型不是OP_VIEW |
| **ExpandFunction** | 将tensor graph扩展为tile graph | 操作验证失败、更新IO操作数失败、操作展开失败、Scope混合错误 | 操作验证失败、输入输出操作数为空指针、操作不支持展开、scope配置错误、在CV分离平台上混合Cube和Vector操作 |
| **MergeViewAssemble** | 合并连续的view/assemble操作链 | 初始化失败、处理操作失败、清理失败、追加合并的view操作失败、追加合并的assemble操作失败、合并view链失败、合并assemble链失败、处理消费者链失败、处理assemble消费者失败、处理链尾失败、处理assemble链尾失败、计算合并offset失败、计算assemble offset失败、删除冗余assemble失败 | 创建ViewOpAttribute/AssembleOpAttribute失败、获取view/assemble属性失败、操作链中存在空指针、链首操作没有输入操作数、链尾操作没有输出操作数、输入/输出tensor为空指针、offset计算失败、tensor offset添加失败、获取view valid shape失败 |
| **SplitReshape** | 拆分reshape操作，将一个reshape拆分为多个reshape并添加view/assemble | 初始化失败、收集CopyOut失败、检查CopyIn失败、添加操作失败、删除reshape失败、消除死操作失败、设置内存类型失败、获取变化轴失败、检查动态状态失败、更新动态shape失败、分组reshape偏移失败 | 对齐shape不正确（alignedShape中有0）、输出维度与动态shape维度不匹配、动态shape涉及变化的轴（合轴、分轴、部分合轴）、offset与dynShape维度不同、偏移值维度不一致 |
| **SplitRawTensor** | 拆分原始tensor | Tensor拆分失败、内存分配失败、Offset计算错误 | Tensor size过大、内存不足、offset边界错误 |
| **SplitLargeFanoutTensor** | 拆分大扇出tensor | LCM shape计算失败、Offset生成失败、操作创建失败 | Shape维度不匹配、GCD/LCM计算溢出、内存类型冲突 |
| **DuplicateOp** | 复制操作 | 操作复制失败、Tensor克隆失败、消费者替换失败 | 内存不足、属性复制不完整、消费者链断裂 |
| **AssignMemoryType** | 为tensor分配内存类型（original和tobe），处理特殊操作（reshape/view/assemble/nop/reduce_acc/shmem_wait_until），处理cube级联场景，插入convert操作 | 插入convert失败、处理未知内存类型失败、处理小tile到大tile失败、处理大tile到小tile失败 | 操作属性获取失败、offset与rawshape维度不匹配、offset计算失败、内存对齐检查失败（32字节对齐）、内存大小超限（UB/L1阈值）、维度倍数检查失败、tobeMap更新失败、视图属性为空、输入输出操作数索引越界 |
| **InferDiscontinuousInput** | 推断不连续输入并插入copy操作 | 从INCAST推断失败、插入copy操作失败 | 图结构复杂、内存类型不匹配、动态shape处理、offset重叠检查失败、view冲突检测失败 |
| **RemoveRedundantOp** | 删除冗余的view-assemble操作对和reshape操作 | 处理DummyOps失败、处理DummyOp失败、处理Reshape失败、处理ViewAssemble失败、合并ViewAssemble失败 | 输入输出shape和dynshape不相等、ASSEMBLE输出有多个生产者、View输入非同源、Assemble数据重排（view offset与assemble offset不匹配）、图连接更新失败 |
| **InsertOpForViewAssemble** | 为view-assemble插入操作 | 操作插入失败、属性设置失败、图连接更新失败 | 内存类型不匹配、offset计算错误、操作顺序依赖 |
| **SplitK** | 拆分K维度，消除Reduce_Acc操作 | Loop检查失败、Reduce_Acc消除失败、操作验证失败 | 存在Loop结构、OP_A_MUL_B/OP_A_MULACC_B输出数量不为1、输出tensor不是L0C类型或消费者数量不为1、OP_REDUCE_ACC输入数量小于1、OP_REDUCE_ACC输出数量不为1、OP_REDUCE_ACC输入输出tensor不是DDR类型 |
| **GraphPartition** | 图切分（将图划分为多个子图以支持并行执行） | 设置参数失败、图切分失败、构建操作图失败、构建超级节点图失败、构建hash值失败、构建同构组失败、合并非同构组失败、合并同构组失败、更新切分结果失败 | 切分参数未初始化、图结构复杂、循环依赖、资源约束冲突、同构组扩展失败 |
| **ReduceCopyMerge** | 减少copy合并（在DAV_3510平台上处理CV混合图并优化copy操作） | ReduceCopy失败、内部子图ID检查失败、图构建失败、合并策略应用失败、循环检测失败、混合图验证失败 | 操作不属于任何内部子图、图结构复杂、CV混合图权重计算失败、合并阈值不满足、存在循环依赖、混合图比例不合法 |
| **NBufferMerge** | 向量子图合并 | 合并策略失败 | 参数配置错误、子图成环 |
| **L1CopyInReuseMerge** | L1拷贝输入重用合并 | 重用分析失败、L1缓存策略失败、依赖关系错误 | 缓存容量限制、访问模式不规律、数据依赖复杂 |
| **IntraSubgraphAdapter** | 处理边界tensor，插入view/assemble打通子图间数据通路 | 内存类型冲突、无效内存类型、生产者类型错误、生产者输入错误、跨核move操作错误 | 边界tensor的original和tobe内存类型不一致、内存类型不在[UB, L1, DDR, L0C]范围内、tensor有多个生产者但不是OP_ASSEMBLE/OP_COPY_OUT、OP_ASSEMBLE输入操作数不为1、跨核move操作出现在子图开头 |
| **GenerateMoveOp** | 将view/assemble/convert/duplicate操作转换为具体的移动操作 | 内存路径查找失败、UB2L1转换失败、L0C2L1属性设置失败 | 内存路径不支持、ND到NZ格式转换失败、内存层次结构复杂 |
| **CommonOperationEliminate** | 通用操作消除 | 公共操作检测失败、消除策略失败、图一致性错误 | 操作等价判断错误、副作用分析不完整、依赖关系遗漏 |
| **AxisCombine** | 轴组合 | 轴组合轴组合失败、Shape变换失败、内存布局错误 | 轴依赖关系复杂、stride不连续、内存对齐要求 |
| **PadLocalBuffer** | 填充本地缓冲区 | 填充计算失败、缓冲区重分配失败、对齐验证失败 | 对齐要求复杂、缓冲区大小限制、填充策略冲突 |
| **RemoveUnalignedReshape** | 删除未对齐的reshape | 对齐检测失败、Reshape优化失败、边界处理错误 | 对齐要求不满足、边界条件复杂、内存布局冲突 |
| **ReplaceTensor** | 替换tensor | Tensor替换失败、引用更新失败、内存一致性错误 | Tensor类型不匹配、引用链断裂、内存类型冲突 |
| **PreGraphProcess** | 预图处理 | 预处理初始化失败、状态设置失败、依赖分析错误 | 初始化顺序错误、状态冲突、依赖关系不完整 |
| **InferDynShape** | 推断动态shape | 动态shape推断失败、符号表达式错误、约束求解失败 | 动态维度关系复杂、符号计算溢出、约束不一致 |
| **SubgraphToFunction** | 子图转函数 | 子图转换失败、函数创建失败、接口映射错误 | 子图结构复杂、接口不匹配、调用约定错误 |
| **InferParamIndex** | 推断参数索引 | 参数索引推断失败、访问模式分析失败、边界检查错误 | 参数数量不匹配、索引越界、访问模式不规则 |
| **SrcDstBufferMerge** | 源目标缓冲区合并 | 缓冲区分析失败、合并策略失败、传输优化错误 | 缓冲区大小不匹配、传输方向冲突、内存重叠 |
| **AddAlloc** | 添加分配操作 | 分配操作插入失败、内存大小计算失败、生命周期分析失败 | 内存不足、分配策略错误、生命周期管理复杂 |
| **OoOSchedule** | OOO调度 | 调度分析失败、依赖关系处理失败、资源分配冲突 | 依赖关系复杂、资源约束严格、调度窗口限制 |
| **TuneTileOpSeqForVF** | 为VF调整tile操作序列 | 序列分析失败、调整策略失败、性能预测错误 | 操作依赖复杂、向量资源限制、性能模型不准确 |
| **RemoveAlloc** | 删除分配操作 | 分配操作分析失败、冗余检测失败、生命周期验证错误 | 分配操作链复杂、生命周期重叠、使用分析不完整 |
| **CopyOutResolve** | Copy输出解析 | Copy-out分析失败、传输路径优化失败、输出处理错误 | 输出依赖复杂、传输路径阻塞、内存类型转换 |
| **InsertSync** | 插入同步操作 | 同步点分析失败、同步操作插入失败、一致性验证失败 | 同步依赖复杂、一致性要求严格、性能开销过大 |
| **TuneSyncForVF** | 为VF调整同步 | 同步分析失败、优化策略失败、正确性验证失败 | 同步粒度不当、依赖关系遗漏、性能-正确性权衡 |
| **MixSubgraphSplit** | 混合子图拆分 | 子图拆分失败、混合模式分析失败、资源分配错误 | 子图耦合紧密、资源竞争、拆分边界不当 |
| **GlobalMemoryReuse** | 全局内存重用 | 内存使用分析失败、重用策略失败、生命周期冲突 | 内存访问模式复杂、生命周期重叠、重用机会识别困难 |
| **LoopaxesProc** | 循环轴处理 | 循环分析失败、轴处理失败、边界条件错误 | 循环结构复杂、边界条件不规则、依赖关系嵌套 |
| **CodegenPreproc** | 代码生成预处理 | 预处理验证失败、代码生成准备失败、最终一致性检查失败 | 图状态不一致、资源分配冲突、生成约束不约足 |

### 步骤 6：错误严重程度评估

**评估标准**：

- **CRITICAL**: 导致程序崩溃、无法继续执行
- **HIGH**: 影响核心功能、结果错误
- **MEDIUM**: 影响性能、功能受限
- **LOW**: 警告信息、可忽略

### 步骤 7：生成分析报告

**报告格式**（Markdown）：

报告模板参见 `references/error_report_template.md` 文件。

## 错误模式匹配规则

### 1. Pass 模块相关错误

**模式识别**：
```
[INFO ] PYPTO(592313):2026-03-13 14:26:37.767 [l1_copy_reuse.cpp:642][PASS]:[L1CopyInReuseMerge.Operation]:Op 704 feature: [63, 128, 0, 32, 512, 148].
[ERROR] PYPTO(592314):2026-03-13 14:26:38.123 [shape_inference.cpp:123][ERROR]:[ShapeInference]:Shape mismatch at dim 2.
[ERROR] PYPTO(592315):2026-03-13 14:26:38.456 [constant_folding.cpp:78][ERROR]:[ConstantFolding]:Folding failed for op 123.
```

**提取策略**：
- 使用正则表达式 `\[([a-zA-Z0-9_]+\.[a-zA-Z0-9_]+)\]:` 提取 pass 名称
- 识别 pass 相关的关键词：`folding`, `inference`, `lowering`, `optimization`, `reuse`, `merge`

### 2. Shape 相关错误

**正则模式**：
```python
SHAPE_ERROR_PATTERNS = [
    r"Shape mismatch",
    r"Invalid shape",
    r"Dimension mismatch",
    r"Shape not compatible"
]
```

**分析策略**：
- 提取期望的 shape 和实际的 shape
- 检查是否是动态 shape 问题
- 分析 shape 不匹配的位置

### 3. 类型相关错误

**正则模式**：
```python
TYPE_ERROR_PATTERNS = [
    r"Type mismatch",
    r"Invalid dtype",
    r"Unsupported type",
    r"Type conversion failed"
]
```

**分析策略**：
- 提取期望的 dtype 和实际的 dtype
- 检查类型转换是否支持
- 分析类型不兼容的原因

### 4. 内存相关错误

**正则模式**：
```python
MEMORY_ERROR_PATTERNS = [
    r"Out of memory",
    r"Allocation failed",
    r"Buffer overflow",
    r"Memory access violation"
]
```

**分析策略**：
- 估算所需内存大小
- 检查可用内存
- 分析内存分配失败的原因

### 5. 设备相关错误

**模式识别**：
```
Invalid Device
NPU not available
Device not found
```

## 修复策略建议

### 策略 1：输入数据修复

**适用场景**：Shape 不匹配、类型不匹配

**建议**：
1. 检查输入张量的 shape 和 dtype
2. 使用 `reshape()` 或 `view()` 调整 shape
3. 使用 `to()` 转换 dtype
4. 添加 shape 和 dtype 的断言

### 策略 2：API 参数修复

**适用场景**：参数错误、不支持的配置

**建议**：
1. 检查 API 文档，确认参数正确性
2. 使用默认参数
3. 添加参数验证
4. 打印参数值进行调试

### 策略 3：环境配置修复

**适用场景**：设备不可用、内存不足

**建议**：
1. 检查 NPU 设备状态：`npu-smi info`
2. 设置正确的设备 ID：`export TILE_FWK_DEVICE_ID=0`
3. 减少批处理大小
4. 释放不必要的内存

### 策略 4：代码逻辑修复

**适用场景**：算子实现错误、边界条件问题

**建议**：
1. 使用 golden 函数验证结果
2. 添加中间结果打印
3. 检查边界条件处理
4. 使用单元测试覆盖边界情况

## 常见问题

常见问题详见 `references/faq.md` 文件。

## 输出验证

**验证检查点**：
- [ ] 错误信息完整提取
- [ ] 堆栈跟踪准确解析
- [ ] Pass 模块名称正确识别
- [ ] 时间戳格式正确
- [ ] 错误原因分析准确
- [ ] 修复建议合理可行

## 参考文件

| 文件 | 内容 |
|------|----------|
| `scripts/parse_log.py` | 日志解析脚本 |
| `scripts/analyze_pass_error.py` | Pass 模块错误分析脚本 |
| `tests/test_log_parser.py` | 日志解析单元测试 |
| `tests/test_pass_analyzer.py` | Pass 分析单元测试 |
| `knowledge_base/pass_errors.json` | Pass 模块错误知识库 |
