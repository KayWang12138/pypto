# PyPTO Pass 模块功能总结文档

本文档汇总了 PyPTO 所有 Pass 模块的功能介绍，按执行顺序排列。每个 Pass 都包含详细的实现方式、关键函数和核心逻辑。

---

## 目录

1. [Tensor Graph Pass (00-14)](#1-tensor-graph-pass-00-14)
2. [Tile Graph Pass (15-30)](#2-tile-graph-pass-15-30)
3. [Block Graph Pass (31-41)](#3-block-graph-pass-31-41)

---

## 1. Tensor Graph Pass (00-14)

### 00 RemoveRedundantReshape

**简要描述**：删除冗余的Reshape操作，当Reshape的输入输出shape相同时，或所有消费者都是Reshape时，该Reshape被视为冗余并删除。

**实现方式**：
- 遍历Function中的所有操作
- 对每个Reshape操作检查：
  - 输入输出操作数是否为1
  - 输入输出shape是否包含-1（动态shape跳过）
  - 所有消费者是否都是Reshape操作且shape未改变
- 标记冗余操作并调用EraseOperations清理

**关键函数**：
- `RemoveReshape`：遍历识别冗余Reshape
- `CheckIOOperands`：检查输入输出有效性

**数据结构**：
- `std::unordered_set<Operation *> redundantResapes`：存储冗余操作

---

### 01 AutoCast

**简要描述**：自动插入类型转换(Cast)操作，处理BF16/FP16/FP32之间的转换，根据NPU架构调整支持的Cast对。

**实现方式**：
- PreCheck阶段：调用AutoCastChecker验证Function状态
- RunOnFunction阶段：
  1. 根据NPU架构设置合法Cast转换对
  2. 获取InOutCast连接的tensor信息
  3. 为不支持BF16的操作插入FP32 Cast
  4. 为不支持FP16的操作插入FP32 Cast
  5. DAV_3510架构处理Int32到FP16的中间转换
  6. 删除冗余Cast链（ShortenChain/RemoveRedundantCastChain）

**关键函数**：
- `InsertBF16Cast`：插入BF16到FP32的Cast
- `InsertFP16Cast`：插入FP16到FP32的Cast
- `InsertInt32Fp16Cast`：处理Int32到FP16的转换
- `GetCastChain`：获取Cast操作链
- `ShortenChain`：缩短Cast链

**数据结构**：
- `std::set<std::pair<DataType, DataType>> legalCastPair`：合法Cast转换对
- `std::unordered_set<Operation *> addedCast_`：新插入的Cast操作

---

### 02 InferMemoryConflict

**简要描述**：通过前向和后向传播分析tensor的内存使用情况，检测冲突并在需要时插入Register Copy。

**实现方式**：
- 初始化：从InCast和OutCast tensor建立memoryInfo映射
- 前向传播：从InCast开始BFS遍历，通过非计算节点（View/Assemble/Reshape/ViewType）传播内存信息
- 后向传播：从OutCast开始反向传播
- 冲突检测：检查Symbol和memoryId是否相同
- 特殊处理：Batch MatMul优化的Reshape pattern（MatchReshapePattern）
- 插入Copy：为冲突的tensor插入OP_REGISTER_COPY

**关键函数**：
- `ForwardPropagation`/`BackwardPropagation`：BFS传播内存信息
- `CheckConflict`：检查内存冲突
- `CheckRawShapeConflict`：检查RawShape冲突
- `MatchReshapePattern`：识别Batch MatMul的Reshape pattern
- `InsertPrecededCopys`/`InsertPostCopys`：插入前置/后置Copy

**数据结构**：
- `std::unordered_map<LogicalTensorPtr, LogicalTensorPtr> memoryInfo`：内存信息映射
- `std::set<Operation*> preregcopys`/`postregcopys`：需要Copy的操作

---

### 03 RemoveUndrivenView

**简要描述**：删除未被驱动的View操作，将OP_ASSEMBLE_SSA降级为普通OP_ASSEMBLE。

**实现方式**：
- 遍历Function中的所有操作
- 识别OP_ASSEMBLE_SSA操作
- 检查INPLACE_IDX属性获取inplaceIdx
- 检查inplaceIdx来源的View操作是否被驱动（生产者数量=1，生产者为OP_VIEW）
- 如果View未被驱动：
  - 删除该View操作
  - 将OP_ASSEMBLE_SSA降级为OP_ASSEMBLE
- 调用EraseOperations清理

---

### 04 ExpandFunction

**简要描述**：将Tensor Graph展开为Tile Graph，是从高层抽象到底层实现的关键转换步骤。

**实现方式**：
- PreCheck：验证Function在Pass执行前的状态
- 检查是否为Tensor Graph（不是则跳过）
- 验证所有操作的有效性
- 将Function类型从TENSOR_GRAPH转换为TILE_GRAPH
- 保存所有原始操作，重置操作列表
- 更新操作的输入输出关系
- 遍历原始操作：
  - 跳过OP_PRINT操作
  - 调用CheckAssembleNeedCopy检查是否需要复制
  - 不需要展开的操作（View/Assemble/Nop）直接添加
  - 需要展开的操作调用ExpandOperationInto展开
- PostCheck验证执行后状态

**关键函数**：
- `Expandfunction`：展开Function
- `CheckAssembleNeedCopy`：检查Assemble是否需要复制
- `NotNeedExpand`：判断是否需要展开
- `ExpandOperation`：展开单个操作

**数据结构**：
- `std::unordered_map<int, std::unordered_set<CoreType>> scopeMap_`：记录每个scope的核心类型

---

### 05 MergeViewAssemble

**简要描述**：合并连续的View和Assemble操作链，减少操作数量。

**实现方式**：
- 遍历Function中的所有操作
- 对View操作调用MergeViewChain：
  - 遍历View的消费者链
  - 累加offset计算合并后的offset
  - 合并动态offset和dynamic valid shape
- 对Assemble操作调用MergeAssembleChain
- 添加合并后的操作
- 删除已访问的操作

**关键函数**：
- `MergeViewChain`：合并View操作链
- `MergeAssembleChain`：合并Assemble操作链
- `CalculateMergedOffsets`：计算合并后的offset

---

### 06 SplitReshape

**简要描述**：当Reshape的输入输出存在重叠时，将Reshape拆分为多个View和Assemble操作。

**实现方式**：
- 收集CopyOut信息：记录输入tensor和后续Reshape
- 检查CopyIn信息：检查Reshape输出是否被CopyIn消费
- 遍历Reshape操作检查重叠状态：
  - PERFECTLY_MATCH：完全匹配，一对一
  - ONE_TO_ONE：一对一
  - BE_COVERED：被覆盖，多对一
  - ONE_TO_MULTI：一对多
  - MULTI_TO_ONE：多对一
  - PERFECTLY_MATCH_WITH_ALL：完全匹配所有
- 根据重叠状态调用相应处理函数
- 删除原始Reshape操作，设置内存类型

**关键函数**：
- `ProcessPerfectlyMatch`/`ProcessOnetoMulti`等：处理不同重叠状态
- `CalcTileInfo`：计算重叠区域的shape和offset
- `CheckDynStatus`：检查动态shape状态

---

### 07 SplitRawTensor

**简要描述**：拆分RawTensor以匹配LogicalTensor的shape，减少内存浪费。

**实现方式**：
- 遍历所有LogicalTensor
- 检查是否需要拆分：
  - RawShape和Shape相等：不拆分
  - InCast或OutCast：不拆分
- 创建新的RawTensor匹配LogicalTensor的shape
- 更新LogicalTensor的RawTensor引用
- 更新相关View操作的fromOffset
- 更新相关Assemble操作的toOffset
- 清理旧的RawTensor

**关键函数**：
- `ShouldProcessTensor`：检查是否需要拆分
- `UpdateConsumerView`：更新View的offset
- `UpdateProducerAssemble`：更新Assemble的offset

---

### 08 SplitLargeFanoutTensor

**简要描述**：处理大扇出tensor的拆分，当Assemble被多个消费者消费且每个消费者只消费部分数据时拆分。

**实现方式**：
- 收集大tensor信息：检查Assemble是否有多个消费者
- 尝试拆分大tensor：
  - 检查是否被完全覆盖（IsBeCovered）
  - 检查是否有重复tile（HasDuplicateToTile）
- 计算LCM shape：最小公倍数shape
- 计算GCD shape：最大公约数shape
- 创建一对一或多对多操作
- 删除冗余的Assemble和View操作
- 消除死操作

**关键函数**：
- `TryToSplitLargeTensor`：尝试拆分
- `CreateOpFor1toM`/`CreateOpForMtoM`：创建操作
- `MoreSplit`：更复杂拆分
- `GCD`/`LCM`：计算最大公约数/最小公倍数

---

### 09 DuplicateOp

**简要描述**：复制View和GatherIn操作，为每个消费者创建新操作，避免共享问题。

**实现方式**：
- 从OutCast开始DFS遍历
- 对每个操作：
  - 如果是OP_GATHER_IN_L1：
    - 遍历输出tensor
    - 对每个消费者（除第一个）：
      - 克隆输出tensor
      - 创建新的GatherIn操作
      - 设置startOffset属性
  - 如果是OP_VIEW：
    - 跳过toType为L1/BT/FIX_QUANT_PRE的View
    - 遍历输出tensor
    - 对每个消费者（除View操作）：
      - 克隆输出tensor
      - 创建新的View操作
      - 复制属性（offset/dynOffset/dynValidShape）
- 消除死操作

---

### 10 AssignMemoryType

**简要描述**：为操作和tensor分配内存类型，根据操作需求和硬件限制分配合适的内存。

**实现方式**：
- 遍历操作分配输入输出内存类型
- 设置InCast/OutCast为DDR
- 处理未知内存类型（设为DDR）
- 处理特殊操作：
  - Reshape：调用AssignMemtypeForSplitReshape
  - ViewType：检查输入是否来自View
  - Nop：输入输出不同则设为DDR
  - ReduceAcc：输入设为DDR
  - ShmemWaitUntil：输出设为DDR
- 处理L0C2L1通路
- 处理超大本地buffer（超过阈值设为DDR）
- 调用ConvertInserter插入Convert操作

**关键函数**：
- `RunOnOperation`：分配操作输入输出内存类型
- `ProcessViewwithSpecificMem`：处理View内存类型
- `ProcessAssemblewithSpecificMem`：处理Assemble内存类型
- `AssignSpecialOpMemtype`：处理特殊操作
- `UpdateOverSizedLocalBuffer`：处理超大buffer

---

### 11 InferDiscontinuousInput

**简要描述**：从InCast出发前向传播，检测tensor的内存访问冲突并插入Copy。

**实现方式**：
- 初始化：记录操作输入度和tensor生产者数量
- 从InCast开始BFS遍历：
  - 获取inplace tensor（View/Assemble/Reshape/IndexOutCast）
  - 检查内存访问冲突
  - 存在冲突则标记插入Copy
- 插入Copy操作：
  - 创建新的RawTensor和LogicalTensor
  - 内存类型设为MEM_UB
  - 插入View/Assemble/Copy操作

**关键函数**：
- `InferFromIncast`：BFS前向传播
- `GetInplacedTileTensors`：获取inplace tensor
- `InsertTensorCopy`：插入Copy操作
- `GetInputTileConflict`：获取输入tile冲突

---

### 12 RemoveRedundantOp

**简要描述**：消除重复的计算操作，通过哈希识别相同计算并合并。

**实现方式**：
- 遍历所有tensor
- 对每个tensor：
  - 获取生产者操作
  - 计算操作哈希（包含操作码、输入、属性等）
  - 检查哈希是否已存在
  - 如果存在且tensor信息匹配，替换消费者输入
  - 如果不存在，缓存哈希
- 清理已删除操作
- 消除死操作

**关键函数**：
- `TensorProducersMerge`：检查是否可以合并
- `TensorHashExist`：检查哈希是否存在
- `UpdateConnection`：更新连接关系
- `ComputeHash`：计算操作哈希

---

### 13 InsertOpForViewAssemble

**简要描述**：在View和Assemble操作之间插入Copy操作，处理内存类型转换。

**实现方式**：
- 遍历所有Assemble操作
- 检查是否需要插入Copy：
  - 检查Assemble的生产者是否为View
  - 检查View和Assemble的offset是否匹配
  - 检查动态shape是否匹配
  - 检查内存类型是否不同
- 如果需要：
  - 如果Assemble输出是DDR：调用InsertViewAssemble创建View+Assemble
  - 如果Assemble输出是UB：插入UB到DDR的Copy

**关键函数**：
- `JudgedViewAssemble`：判断是否需要插入
- `NeedInsertCopy`：检查需要插入的条件
- `InsertCopy`：插入Copy操作

---

### 14 SplitK

**简要描述**：消除ReduceAcc操作，优化K轴归约计算，将多个A_MUL_B的CopyOut直接连接到最终GM。

**实现方式**：
- PreCheck：
  - 检查不存在Loop结构
  - 验证A_MUL_B输出tensor（必须是L0C类型，只有一个consumer）
  - 验证REDUCE_ACC输入输出（输入>=1，输出=1，都是DDR类型）
- RunOnFunction：
  - 遍历所有操作查找REDUCE_ACC
  - 清除输出tensor的producer列表
  - 遍历输入tensor的producer（CopyOut）
  - 将CopyOut的输出替换为REDUCE_ACC的输出
  - 设置CopyOut的atomic_add属性
  - 删除REDUCE_ACC操作
- 消除死操作

**转换前后图结构**：
```
Before: A_MUL_B → L0C → Copy_Out → Gm → Reduce_Acc → Gm(Final)
After:  A_MUL_B → L0C → Copy_Out ──────────────────────────────→ Gm(Final)
```

---

## 2. Tile Graph Pass (15-30)

### 15 GraphPartition

**简要描述**：复合Pass，整合图划分和多个子优化Pass。

**实现方式**：
- 包含以下子Pass：
  1. IsoPartitioner：同构子图划分
  2. NBufferMerge：NBuffer合并
  3. L1CopyReuse：L1 Copy复用
  4. CommonOperationEliminate：公共操作消除
  5. ReduceCopy：Reduce Copy优化

**子Pass详细**：
- NBufferMerge：通过着色算法将相似子图分组合并
- L1CopyReuse：优化L1缓存的copy操作
- CommonOperationEliminate：识别并消除重复计算

---

### 16 ReduceCopyMerge

**简要描述**：合并Reduce Copy操作，使用并查集(DSU)算法。

**实现方式**：
- 检查平台是否支持CV混合（DAV_3510）
- 构建并查集：
  - 初始化color和cycle
  - 构建依赖图
  - 设置合并阈值
- 执行合并：
  - 检查图是否可以合并
  - 执行合并操作
- 更新子图ID

**关键函数**：
- `MergePrepare`：准备合并，构建并查集
- `MergeLoop`：执行合并循环

---

### 17 NBufferMerge

**简要描述**：NBuffer合并优化，通过着色算法将相似子图分组。

**实现方式**：
- Init：统计子图subgraphID，初始化着色数据结构
- 计算子图哈希：
  - 遍历操作计算哈希值
  - 特殊处理：单独reshape不合并，AIC操作不合并
  - 构建哈希映射表
- 合并处理：
  - 遍历哈希组
  - 根据hashMergeNum确定合并数量
  - 更新子图subgraphID
  - ping-pong优化
- 拓扑排序确保执行顺序

**关键函数**：
- `GetColorHash`：计算子图哈希
- `MergeProcess`：执行合并
- `MergePingPong`：ping-pong优化
- `ColorTopo`：拓扑排序

**支持模式**：noMerge、autoMerge、manualMerge、autoMulityInOutMerge、manualMulityInOutMerge

---

### 18 L1CopyInReuseMerge

**简要描述**：L1 Copy复用优化。

**实现方式**：
- 识别可以复用的L1 Copy操作
- 分析输入输出依赖关系
- 合并可复用的操作

---

### 19 IntraSubgraphAdapter

**简要描述**：适配子图间的边界tensor，处理跨子图的tensor传递。

**实现方式**：
- 收集所有边界tensor（跨子图的tensor）
- 检查边界tensor合法性：
  - 内存类型是否合法
  - original类型和tobe类型是否一致
- 处理边界tensor：
  - 如果producer和consumer在同一个子图：直接处理
  - 如果producer和consumer在不同子图：
    - common colors = 0：调用ProcessBoundaryTensor处理
    - common colors = 1：调用SplitBoundaryTensor拆分
    - common colors > 1：报错
- 插入ASSEMBLE（在producer侧）和VIEW（在consumer侧）
- 设置tensor类型为DDR

**关键函数**：
- `CollectBoundaryTensors`：收集边界tensor
- `CheckBoundaryTensor`：检查合法性
- `ProcessBoundaryTensor`：处理边界tensor
- `SplitBoundaryTensor`：拆分边界tensor

---

### 20 GenerateMoveOp

**简要描述**：将VIEW/ASSEMBLE/CONVERT/DUPLICATE转换为具体的搬运操作。

**实现方式**：
- 遍历所有操作根据类型转换：
  - OP_ASSEMBLE → OP_COPY_OUT
  - OP_VIEW → 根据输出内存类型转换为相应操作：
    - GM输入 → COPY_IN
    - L0A输出 → L1_TO_L0A/L1_TO_L0AT（有转置）
    - L0B输出 → L1_TO_L0B/L1_TO_L0BT（有转置）
    - 其他 → 根据内存路径转换
  - OP_CONVERT → 根据转换路径转换
  - OP_DUPLICATE → COPY_OUT
- 特殊处理：
  - UB到L1的NZ格式转换（ProcessUB2L1）
  - L0C到L1的转换（SetL0C2L1CopyAttr）

**内存路径映射**：
- DDR → L1/UB: OP_COPY_IN
- L1 → L0A: OP_L1_TO_L0A
- L1 → L0B: OP_L1_TO_L0B
- L0C → DDR: OP_COPY_OUT
- UB → L1: OP_UB_COPY_L1
- L0C → L1: OP_L0C_TO_L1

---

### 21 CommonOperationEliminate

**简要描述**：消除重复的计算操作，通过哈希操作特征识别相同计算。

**实现方式**：
- 获取每个tensor的生产者映射
- 遍历tensor检查是否可以合并：
  - 计算操作哈希（操作码、输入tensor信息、属性、子图ID）
  - 检查哈希是否已存在
  - 如果存在且tensor信息匹配，更新连接
- 清理已删除操作和死操作

**排除条件**：
- L1到L0的搬运操作
- VIEW操作
- 有dontTouch属性的操作
- 不同shape或数据类型的tensor

---

### 22 AxisCombine

**简要描述**：对齐广播操作的输入轴，插入BRCB或EXPAND操作。

**实现方式**：
- 检查combineAxis配置
- 调用axisCombineMarker标记需要轴合并的tensor
- 遍历操作对广播操作对齐输入：
  - 获取两个输入tensor
  - 如果输入shape相同，跳过
  - 对每个最后一维为1的输入：
    - 获取对齐值（根据数据类型）
    - 调用AlignedIfNeed对齐最后一维
    - 插入BRCB或EXPAND操作
    - 更新操作输入

**关键函数**：
- `AlignBroadCastOpInputs`：对齐广播输入
- `AlignedIfNeed`：计算对齐后的维度
- `GetPaddingValue`：获取对齐值

---

### 23 PadLocalBuffer

**简要描述**：对齐本地缓冲区的tensor shape，满足硬件对齐要求。

**实现方式**：
- 设置combineAxis和forceCombineAxis配置
- 如果是combineAxis模式：
  - 调用axisCombineMarker标记
  - 调用DoPadding执行padding
- 否则：
  - 遍历操作处理特殊场景：
    - 尾轴reduce：设置reduceAxisCombined属性
    - copyin：设置属性
    - broadcast：设置padding值
  - 调用DoPadding执行padding
  - 处理transpose操作

**对齐值**：
- B8数据类型：32B对齐
- B4数据类型：64B对齐
- 其他：16B对齐

**关键函数**：
- `PadMatmul`：对齐Matmul tensor
- `PadVector`：对齐Vector tensor
- `ProcessTranspose`：处理transpose

---

### 24 RemoveUnalignedReshape

**简要描述**：移除未对齐的Reshape操作。

**实现方式**：
- 检查Reshape操作的对齐属性
- 移除不符合对齐要求的Reshape

---

### 25 ReplaceTensor

**简要描述**：替换tensor，识别inplace操作减少内存使用。

**实现方式**：
- 构建tensor索引映射
- 使用UnionFind合并可inplace的tensor：
  - 遍历inplace操作（View/Assemble/Reshape/A_MULACC_B等）
  - 合并输入输出对
- 对每个组查找base tensor（优先选择InCast/OutCast）
- 前向传播替换：从base tensor向前遍历，更新消费者输入
- 后向传播替换：从base tensor向后遍历，更新生产者输出
- 重构VIEW连接：
  - 设置inplaceIdx属性
  - 插入NOP操作如果需要
- 处理HUB操作
- 标记部分内存tensor

**关键函数**：
- `UniteTensor`：合并tensor
- `FindBaseTensor`：查找base tensor
- `ForwardProcess`/`BackwardProcess`：前向/后向传播
- `RefactorViewConnectForReplace`：重构VIEW连接

**Inplace操作集合**：
- OP_VIEW, OP_ASSEMBLE, OP_RESHAPE, OP_A_MULACC_B, OP_INDEX_OUTCAST, OP_VIEW_TYPE

---

### 26 PreGraphProcess

**简要描述**：图预处理，设置子图颜色、边界、Cube属性，优化View和Assemble。

**实现方式**：
1. 子图拓扑排序（ColorGraph.PreColorSort）
2. 初始化tensor子图ID（InitializeTensorColor）
3. 更新Copy操作isCube属性（UpdateCopyOpIsCube）
4. 设置tensor边界（SetBoundary.SetTensorBoundary）
5. 处理特殊Copy操作：
   - CopyOut：调用ProcessSpecialMTEOperation
   - CopyIn：调用ProcessMoveInOperation
6. 删除冗余Assemble（RemoveRedundantAssemble）
7. 更新Cube操作属性（CubeProcess.UpdateCubeOp）
8. 合并View和Assemble（MergeViewAssembleUtils）

**关键函数**：
- `PreColorSort`：子图拓扑排序
- `InitializeTensorColor`：初始化tensor颜色
- `SetTensorBoundary`：设置边界
- `UpdateCubeOp`：更新Cube属性

---

### 27 InferDynShape

**简要描述**：推断动态shape，通过拓扑排序遍历操作并调用每个操作的infer shape函数。

**实现方式**：
- 构建操作到索引的映射
- 构建操作图的输入输出关系
- 调用TopoProgram进行拓扑排序
- 按拓扑顺序遍历操作：
  - 对每个操作调用对应的infer shape函数
  - 传播动态shape

**关键函数**：
- `InferShape`：推断shape
- `TopoProgram`：拓扑排序

---

### 28 SubgraphToFunction

**简要描述**：将Tile Graph子图转换为Execute Graph函数。

**实现方式**：
1. 将View转换为CopyIn（用于COA记录）
2. 添加GetTensorData依赖
3. 如果是静态流程：
   - 构建图
   - 记录Incast和Outcast
4. 记录Incast和Outcast信息
5. 构建参数映射
6. 将子图转换为函数：
   - 创建root function
   - 遍历子图处理每个
   - 哈希计算实现子图复用
   - 符号化处理
7. 清除GetTensorData依赖
8. 将CopyIn恢复为View

**关键函数**：
- `IslandToFunction`：子图转函数
- `ProcessSubgraph`：处理单个子图
- `RecordIncastOutcast`：记录Incast/Outcast
- `SymbolizeFunction`：符号化

---

### 29 InferParamIndex

**简要描述**：推断子函数的参数索引，处理动态形状。

**实现方式**：
- 遍历所有子程序
- 对每个子函数：
  - 重置动态有效形状（ResetDynValidShape）
  - 推断形状（InferShape）
  - 更新有效形状（UpdateValidShape）
  - 设置子函数有效形状（SetSubValidShape）

**关键函数**：
- `UpdateParamIndex`：更新参数索引
- `ResetDynValidShape`：重置动态形状
- `InferShape`：推断形状

---

### 30 SrcDstBufferMerge

**简要描述**：合并源目标buffer，实现内存复用。

**实现方式**：
- 遍历所有子程序
- 对每个子程序：
  - 初始化操作列表
  - 遍历操作尝试内存复用：
    - 检查是否可以忽略（CheckIgnoreScene）
    - 检查是否有inplace语义（CheckHasInplaced）
    - 查找可复用tensor（FindReplaced）
    - 执行复用

**关键函数**：
- `ProcessInplaceReuse`：处理inplace复用
- `ProcessL0MemoryReuse`：处理L0内存复用
- `CanSrcDstReuse`：检查是否可以复用

---

## 3. Block Graph Pass (31-41)

### 31 AddAlloc

**简要描述**：添加内存分配操作。

**实现方式**：
- 遍历所有子程序
- 对每个子程序：
  - 生成tensor分配消息映射
  - 生成alloc节点
  - 重新排序操作

**内存类型到Alloc操作映射**：
- MEM_L0A → OP_L0A_ALLOC
- MEM_L0B → OP_L0B_ALLOC
- MEM_L0C → OP_L0C_ALLOC
- MEM_UB → OP_UB_ALLOC

---

### 32 OoOSchedule

**简要描述**：乱序调度，优化操作执行顺序提升并行度。

**实现方式**：
- 判断是否为混合图（AIC+AIV）
- 如果是混合图：调用MixSchedule
- 如果不是混合图：调用NonMixSchedule
- 调度过程：
  - 创建任务分割器
  - 分割任务图
  - 排序任务列表
  - 估计延迟
  - 创建核心调度器
  - 调度任务
  - 合并任务
  - 构建操作列表
  - 修改边界操作顺序

**关键函数**：
- `MixSchedule`/`NonMixSchedule`：混合/非混合图调度
- `Schedule`：执行调度
- `SortAndLatencyEstimate`：排序和延迟估计
- `ModifyBoundaryOrder`：修改边界顺序

---

### 33 TuneTileOpSeqForVF

**简要描述**：Vector Fusion优化，调整TileOp序列。

**实现方式**（仅DAV_3510+ENABLE_VF）：
- 检查ENABLE_VF配置
- 遍历programs构建Pipe操作映射
- 分别对AIV0和AIV1调用ChangeOpSeq：
  - 获取Pipe V上的操作索引
  - 遍历相邻操作对
  - 判断是否可合并（IsMergeable）
  - 判断是否有性能收益
  - 执行调整（MoveOpsForMerge）
- 刷新操作列表

**关键函数**：
- `ChangeOpSeq`：改变操作序列
- `IsMergeable`：判断可合并性
- `MoveOpsForMerge`：移动操作合并

---

### 34 RemoveAlloc

**简要描述**：移除alloc操作。

**实现方式**：
- 遍历所有子程序
- 遍历操作
- 如果操作名称包含"ALLOC"：标记删除
- 删除所有已标记操作

---

### 35 CopyOutResolve

**简要描述**：解析CopyOut，插入AICPU_CALL触发resolve。

**实现方式**：
- 检查copyOutResolveCoalescing配置
- 查找每个Outcast对应的最后一个CopyOut
- 在CopyOut位置插入AICPU_CALL
- 使用coalescing机制合并多个AICPU_CALL：
  - 在指定距离内合并
  - 重新排序操作
  - 删除冗余AICPU_CALL
- 维护outcastCopyOutResolveCounterList

**关键函数**：
- `LookupOutcastLastCopyOut`：查找最后一个CopyOut
- `InsertCopyOutResolveForLeaf`：插入resolve操作

---

### 36 InsertSync

**简要描述**：在Operation之间插入SYNC_SRC和SYNC_DST同步操作。

**实现方式**：
- 构建TensorRangeMap
- 进行管道调度（PipeDispatch）：
  - 遍历操作构建DepOp
  - 查找数据依赖（RAW/WAW/WAR）
  - 构建setPipe和waitPipe列表
  - 调整特殊操作的pipe配置
- 发射操作（IssueOp）：
  - 发射普通操作
  - 发射同步操作
  - 死锁检测
  - Event ID管理

**关键函数**：
- `InsertSync`：插入同步
- `PipeDispatch`：管道调度
- `IssueOp`：发射操作

**特殊处理**：
- Reshape：pipe设为MTE3
- CopyIn：根据目标内存类型设置
- CopyOut：根据源内存类型设置

---

### 37 TuneSyncForVF

**简要描述**：Vector Fusion同步优化。

**实现方式**（仅DAV_3510+ENABLE_VF）：
- 检查ENABLE_VF配置
- 构建Pipe操作映射
- 分别对AIV0和AIV1调用ChangeOpSeq：
  - 获取Pipe V操作索引
  - 遍历相邻操作对
  - 判断是否可合并（全是SYNC_SRC/SYNC_DST）
  - 使用性能模型计算收益
  - 执行调整
- 刷新操作列表

**关键函数**：
- `IsMergeable`：判断可合并性
- `NeedAdjustSetFlag`：计算性能收益
- `AdjustSetWaitFlag`：执行调整

---

### 38 MixSubgraphSplit

**简要描述**：将Mix子图拆分为独立的Cube和Vector子图。

**实现方式**（仅DAV_3510）：
- 收集需要拆分的Mix子图：
  - 遍历CallOp按哈希分组
  - 查找cacheFunction
  - 判断是否有internalSubgraphID
- 分析内部组件（MixInternalComponentsAnalyzer）
- 计算programID重映射
- 执行拆分：
  - 为每个组件创建新Function
  - 应用依赖关系
  - 创建新CallOp
  - 更新programID映射

**关键函数**：
- `GatherSubGraphInfo`：收集子图信息
- `CalculateSplit`：计算拆分
- `ExecuteSplit`：执行拆分

---

### 39 GlobalMemoryReuse

**简要描述**：全局内存复用，通过TensorBucket机制。

**实现方式**：
- 初始化：
  - 生成连接矩阵
  - 初始化root casts
  - 初始化leaf内存复用
- 处理leaf function：
  - 收集输出tensor信息
  - 收集输入tensor信息
  - BFS寻找可复用输入
- 处理操作：
  - 遍历操作
  - 检查是否已有存储分配
  - 复用已有存储或分配新存储
- 分配：
  - 预处理需要分配的tensor
  - 找到最佳复用bucket
  - 更新bucket offset

**关键函数**：
- `InitializeLeafGlobalMemoryReuse`：初始化leaf复用
- `ProcessLeafGlobalMemoryReuse`：处理leaf
- `Allocate`：执行分配
- `GetBestFitBucket`：找到最佳bucket

---

### 40 LoopaxesProc

**简要描述**：处理Loop Axes，标记loopGroup和loopAxes属性。

**实现方式**（仅ENABLE_VF或VF_OPT_MARK_FOR）：
- 检查配置
- 遍历subProgram：
  - 重置group状态
  - 遍历操作：
    - 检查是否需要清除状态
    - 提取loopAxes（没有则从shape推导）
    - 比较loopAxes（支持动态参数）
    - 不一致则创建新group
    - 标记loopGroup和loopAxes属性

**关键函数**：
- `UpdateOpLoopAxes`：更新操作loop axes
- `SameLoopAxes`：比较loop axes
- `NeedClearStatus`：判断是否需要清除状态

---

### 41 CodegenPreproc

**简要描述**：代码生成前预处理。

**实现方式**：
- 将OP_VIEW_TYPE转换为OP_VIEW
- 保存GM tensor参数索引（DYNAMIC_LOOP_PATH场景）
- 强制合并axis：
  - combineAxis模式：ForceCombineAxisForAxisCombine
  - 其他模式：ForceCombineAxis
- 设置needAlloc属性：
  - 遍历输出tensor
  - 跳过DDR tensor
  - 首次出现的memId设置needAlloc=true

**关键函数**：
- `SaveGmTensorParamIdxToOp`：保存GM tensor索引
- `ForceCombineAxisForAxisCombine`：强制合并axis
- `SetNeedAllocAttr`：设置needAlloc属性

---

## Pass 执行流程图

```
Tensor Graph Pass (00-14)
    │
    ▼
ExpandFunction (04) ──────────────────────────────────────────► Tile Graph Pass (15-30)
    │                                                               │
    │  RemoveRedundantReshape (00)                                  │
    │  AutoCast (01)                                                 │
    │  InferMemoryConflict (02)                                      │
    │  RemoveUndrivenView (03) ─────────────────────────────────────►│ SubgraphToFunction (28)
    │                                                               │ InferDynShape (27)
    │                                                               │ ReplaceTensor (25)
    │                                                               │ PadLocalBuffer (23)
    │                                                               │ PreGraphProcess (26)
    │                                                               ▼
    └─────────────────────────────────────────────────────────────► Block Graph Pass (31-41)
                                                                         │
                                                                         │ AddAlloc (31)
                                                                         │ OoOSchedule (32)
                                                                         │ InsertSync (36)
                                                                         │ GlobalMemoryReuse (39)
                                                                         │ CodegenPreproc (41)
                                                                         ▼
                                                                    代码生成
```

---

## 附录：源码目录结构

```
framework/src/passes/
├── tensor_graph_pass/      # Tensor Graph Pass (00-14)
│   ├── auto_cast.*
│   ├── expand_function.*
│   ├── infer_memory_conflict.*
│   ├── remove_redundant_reshape.*
│   └── remove_undriven_view.*
│
├── tile_graph_pass/         # Tile Graph Pass (15-30)
│   ├── graph_constraint/
│   │   ├── axis_combine.*
│   │   ├── infer_dyn_shape.*
│   │   ├── pad_local_buffer.*
│   │   ├── replace_tensor.*
│   │   └── pre_graph/
│   ├── graph_optimization/
│   │   ├── duplicate_op.*
│   │   ├── merge_view_assemble.*
│   │   ├── split_k.*
│   │   ├── split_reshape.*
│   │   └── split_large_fanout_tensor.*
│   ├── graph_partition/
│   │   ├── common_operation_eliminate.*
│   │   ├── graph_partition.*
│   │   ├── n_buffer_merge.*
│   │   └── reduce_copy.*
│   ├── data_path/
│   │   ├── assign_memory_type.*
│   │   ├── generate_move_op.*
│   │   └── intra_subgraph_adapter.*
│   └── subgraph_to_function.*
│
├── block_graph_pass/        # Block Graph Pass (31-41)
│   ├── schedule_ooo/
│   │   ├── add_alloc.*
│   │   ├── ooo_schedule.*
│   │   └── remove_alloc.*
│   ├── memory_reuse/
│   │   └── global_memory_reuse.*
│   ├── tune_tileopseq_for_vf.*
│   ├── tune_sync_for_vf.*
│   ├── insert_sync.*
│   ├── mix_subgraph_split.*
│   ├── loopaxes_proc.*
│   └── codegen_preproc.*
│
├── pass_mgr/                # Pass管理
├── pass_interface/         # Pass接口
├── pass_utils/              # Pass工具
└── pass_check/             # Pass检查
```
