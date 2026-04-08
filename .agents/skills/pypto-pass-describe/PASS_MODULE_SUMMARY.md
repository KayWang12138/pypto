# PyPTO Pass 模块功能总结

PyPTO 编译器使用 Pass 进行图优化，将高层 Tensor Graph 逐步转换为可执行的 Block Graph。

---

## 一、Pass 总体介绍

PyPTO 编译器使用一系列 Pass（优化遍）进行图优化，这些 Pass 负责将高层抽象的 Tensor Graph 逐步转换为可执行的 Block Graph。整个转换过程分为三个阶段，每个阶段处理不同抽象级别的计算图，完成特定的优化任务。

**第一阶段：Tensor Graph Pass**

该阶段处理高层抽象的 Tensor Graph，使用 LogicalTensor 和 Operation 表示计算图。这一阶段的主要任务包括类型转换（处理 BF16/FP16/FP32 之间的转换）、图优化（删除冗余的 Reshape 和 View 操作）、内存冲突检测（插入 Register Copy 解决冲突），以及关键的图展开操作（将 Tensor Graph 展开为 Tile Graph）。

**第二阶段：Tile Graph Pass**

该阶段处理 Tile 级别的图，使用具体的 Tile 操作。主要任务包括图划分（将大图划分为多个子图）、子图合并（NBuffer 合并、Reduce 合并、L1 Copy 复用）、数据布局优化（Reshape 拆分、RawTensor 拆分、Buffer 对齐）、内存管理（分配内存类型、处理非连续输入），以及调度准备（拓扑排序、边界设置、子图转函数）。

**第三阶段：Block Graph Pass**

该阶段处理可执行的 Block Graph，包含具体的操作和调度信息。主要任务包括内存分配（添加 Alloc 操作）、乱序调度（优化操作执行顺序，提升并行度）、同步插入（插入 SYNC 操作确保数据依赖正确）、Vector Fusion 优化（调整 TileOp 和同步操作位置），以及代码生成准备（设置 needAlloc 属性、强制合并 axis）。

---

## 二、各阶段介绍

### Tensor Graph Pass

**处理对象**：高层抽象的 Tensor Graph，使用 LogicalTensor 和 Operation 表示。

**主要功能**：
- 类型转换：自动插入 Cast 操作处理 BF16/FP16/FP32 转换
- 图优化：删除冗余 Reshape、View 操作
- 内存处理：检测并解决内存冲突，插入 Register Copy
- 图展开：将 Tensor Graph 展开为 Tile Graph

---

### Tile Graph Pass

**处理对象**：Tile 级别的图，使用具体的 Tile 操作。

**主要功能**：
- 图划分：将大图划分为多个子图
- 子图合并：NBuffer 合并、Reduce 合并、L1 Copy 复用
- 数据布局：Reshape 拆分、RawTensor 拆分、Buffer 对齐
- 内存管理：分配内存类型、处理非连续输入
- 调度准备：拓扑排序、边界设置、子图转函数

---

### Block Graph Pass

**处理对象**：可执行的 Block Graph，包含具体操作和调度信息。

**主要功能**：
- 内存分配：添加 Alloc 操作
- 乱序调度：优化操作执行顺序，提升并行度
- 同步插入：插入 SYNC 操作确保数据依赖
- Vector Fusion 优化：调整 TileOp 和同步操作位置
- 代码生成准备：设置 needAlloc 属性、强制合并 axis

---

## 三、各 Pass 模块简要说明

### Tensor Graph Pass

| Pass 名称 | 功能说明 |
|:---|:---|
| RemoveRedundantReshape | 删除冗余的 Reshape 操作。当 Reshape 的输入输出 shape 完全相同，或所有消费者都是 Reshape 且 shape 未改变时，该 Reshape 被视为冗余并删除 |
| AutoCast | 自动插入 Cast 操作进行类型转换。处理 BF16/FP16/FP32 之间的转换，根据 NPU 架构调整支持的 Cast 对，删除冗余的 Cast 链 |
| InferMemoryConflict | 检测并解决内存冲突。通过前向和后向传播分析 tensor 的内存使用情况，当检测到冲突时插入 Register Copy 操作 |
| RemoveUndrivenView | 删除未被驱动的 View 操作。检查 View 操作的消费者是否被正确驱动，将 OP_ASSEMBLE_SSA 降级为普通 OP_ASSEMBLE |
| ExpandFunction | 将 Tensor Graph 展开为 Tile Graph。这是编译流程中的关键转换步骤，将高层抽象的计算图转换为 Tile 级别的图结构 |
| MergeViewAssemble | 合并连续的 View 和 Assemble 操作链。通过累加 offset 计算合并后的 offset，减少操作数量 |
| SplitReshape | 将 Reshape 拆分为 View 和 Assemble 操作。当 Reshape 的输入输出存在重叠时，拆分处理数据重叠的场景 |
| SplitRawTensor | 拆分 RawTensor 以匹配 LogicalTensor 的 shape。当 RawShape 和 Shape 不同时，创建新的 RawTensor 减少内存浪费 |
| SplitLargeFanoutTensor | 拆分大扇出 tensor。当一个 tensor 被多个消费者消费且每个消费者只消费部分数据时进行拆分，提高并行度 |
| DuplicateOp | 复制 View 和 GatherIn 操作。为每个消费者创建新操作，避免共享问题导致的数据错误 |
| AssignMemoryType | 为操作和 tensor 分配内存类型。根据操作需求和硬件限制，将 tensor 分配到 DDR/UB/L1/L0C 等不同内存 |
| InferDiscontinuousInput | 推断非连续输入的内存访问模式。从 InCast 出发进行前向传播，检测 tensor 的内存访问冲突并插入 Copy 操作 |
| RemoveRedundantOp | 消除重复的计算操作。通过哈希操作特征（操作码、输入、属性等）识别相同计算，用一个操作替换多个重复操作 |
| InsertOpForViewAssemble | 在 View 和 Assemble 之间插入 Copy 操作。当 View 和 Assemble 的内存类型不同时，插入必要的 Copy 操作处理内存类型转换 |
| SplitK | 消除 ReduceAcc 操作，优化 K 轴归约计算。将多个 A_MUL_B 的 CopyOut 直接连接到最终 GM，删除 ReduceAcc 节点减少操作数量 |

---

### Tile Graph Pass

| Pass 名称 | 功能说明 |
|:---|:---|
| GraphPartition | 复合 Pass，负责图划分和子图优化。整合了 IsoPartitioner、NBufferMerge、L1CopyReuse、CommonOperationEliminate、ReduceCopy 等子 Pass |
| ReduceCopyMerge | 合并 Reduce Copy 操作。使用并查集（DSU）算法分析 Reduce Acc 操作的输入输出关系，将多个连续的 Reduce Acc 合并为一个 |
| NBufferMerge | NBuffer 合并优化。通过着色算法将相似子图分组合并，支持 autoMerge、manualMerge 等多种模式，减少子图切换开销 |
| L1CopyInReuseMerge | L1 Copy 复用优化。识别可以复用的 L1 Copy 操作，分析输入输出依赖关系，合并可复用的操作 |
| IntraSubgraphAdapter | 适配子图间的边界 tensor。收集跨子图的 tensor，插入 ASSEMBLE（在 producer 侧）和 VIEW（在 consumer 侧）操作处理数据传递 |
| GenerateMoveOp | 将抽象操作转换为具体搬运操作。将 VIEW/ASSEMBLE/CONVERT/DUPLICATE 转换为 COPY_IN、COPY_OUT、L1_TO_L0A 等具体的数据搬运操作 |
| CommonOperationEliminate | 消除重复的计算操作。通过哈希操作特征识别相同计算，更新消费者输入引用并清理死操作 |
| AxisCombine | 对齐广播操作的输入轴。为输入最后一维为 1 的 tensor 插入 BRCB 或 EXPAND 操作，确保广播操作高效执行 |
| PadLocalBuffer | 对齐本地缓冲区的 tensor shape。对 Matmul 和 Vector 操作的输入输出进行 padding，满足硬件对齐要求（B8 对齐、B4 对齐等） |
| RemoveUnalignedReshape | 移除未对齐的 Reshape 操作。检查 Reshape 操作的对齐属性，移除不符合对齐要求的操作 |
| ReplaceTensor | 替换 tensor 实现内存复用。使用 UnionFind 将可以 inplace 的 tensor 分组，识别 inplace 操作减少内存分配 |
| PreGraphProcess | 图预处理。执行子图拓扑排序、初始化 tensor 子图 ID、设置边界属性、更新 Cube 操作属性、合并 View 和 Assemble |
| InferDynShape | 推断动态 shape。通过拓扑排序遍历操作，调用每个操作的 infer shape 函数，正确推断所有 tensor 的动态 shape |
| SubgraphToFunction | 将子图转换为 Execute Graph 函数。将 Tile Graph 的每个子图转换为可执行的函数，构建子图调用关系，支持子图哈希复用 |
| InferParamIndex | 推断子函数的参数索引。遍历子函数，重置动态有效形状、推断形状、更新有效形状，为代码生成提供参数信息 |
| SrcDstBufferMerge | 合并源目标 buffer 实现内存复用。遍历操作识别可以复用输入内存的输出 tensor，复用内存减少分配 |

---

### Block Graph Pass

| Pass 名称 | 功能说明 |
|:---|:---|
| AddAlloc | 添加内存分配操作。根据 tensor 的内存类型（L0/L1/UB），插入相应的 Alloc 操作 |
| OoOSchedule | 乱序调度优化。通过分析操作间的依赖关系构建任务图，使用调度算法优化操作执行顺序，提升并行度和性能 |
| TuneTileOpSeqForVF | Vector Fusion 优化：调整 TileOp 序列。分析 Pipe V 上的操作序列，识别可融合的 Vector 操作，调整操作顺序减少同步开销（仅 DAV_3510） |
| RemoveAlloc | 移除 alloc 操作。清理不需要的内存分配操作，简化图结构 |
| CopyOutResolve | 解析 CopyOut 操作。为每个 Outcast 插入 AICPU_CALL 操作触发 copy out resolve，支持 coalescing 优化减少调用开销 |
| InsertSync | 插入同步操作。分析操作间的 RAW/WAW/WAR 依赖关系，在不同 Pipe 和 Core 之间插入 SYNC_SRC 和 SYNC_DST 操作确保数据正确 |
| TuneSyncForVF | Vector Fusion 优化：调整同步操作位置。使用性能模型计算调整收益，优调 SetFlag 和 WaitFlag 的位置提升性能（仅 DAV_3510） |
| MixSubgraphSplit | 拆分 Mix 子图。将同时包含 Cube 和 Vector 操作的 Mix 子图拆分为独立的子图，为每个组件创建新的 Function 和 CallOp（仅 DAV_3510） |
| GlobalMemoryReuse | 全局内存复用。通过 TensorBucket 机制实现跨操作的内存复用，支持 leaf 内和 leaf 间的复用，减少总内存占用 |
| LoopaxesProc | 处理 Loop Axes。为支持 Vector Fusion 的操作标记 loopGroup 和 loopAxes 属性，将具有相同 loopAxes 的操作划分为同一个 group（仅 VF） |
| CodegenPreproc | 代码生成前预处理。将 OP_VIEW_TYPE 转换为 OP_VIEW，保存 GM tensor 参数索引，强制合并 axis，设置 needAlloc 属性 |