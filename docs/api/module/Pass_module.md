# PyPTO Pass 模块功能总结

本文档总结了 PyPTO 编译流程中所有 Pass 模块的功能，按照执行阶段组织，以表格形式呈现。

---

## 1. Pass 总体介绍

PyPTO 编译流程通过一系列 Pass 模块将用户代码转换为可执行的 NPU 代码。这些 Pass 模块按照执行顺序分为三个主要阶段：

### 1.1 Pass 阶段说明

| 阶段 | 说明 | 主要目标 |
|------|------|----------|
| Tensor Graph Pass | 处理 Tensor Graph 层面的优化 | 冗余操作消除、类型转换、内存冲突推断、图展开 |
| Tile Graph Pass | 处理 Tile Graph 层面的优化 | 操作合并、内存分配、图划分、子图转换 |
| Block Graph Pass | 处理 Block Graph 层面的优化 | 参数推断、内存复用、调度优化、同步插入、代码生成准备 |

---

## 2. Pass 三个阶段介绍

### 2.1 Tensor Graph Pass 阶段

Tensor Graph Pass 阶段处理 Tensor Graph 层面的优化，主要关注：

- **冗余操作消除**: 删除不必要的 Reshape、View 等操作
- **类型转换优化**: 自动插入和优化 Cast 操作
- **内存冲突处理**: 推断和解决内存访问冲突
- **图结构转换**: 将 Tensor Graph 展开为 Tile Graph

### 2.2 Tile Graph Pass 阶段

Tile Graph Pass 阶段处理 Tile Graph 层面的优化，主要关注：

- **操作合并**: 合并 View、Assemble、Reduce Copy 等操作
- **内存优化**: 分配内存类型、复用内存、拆分 tensor
- **图划分**: 将计算图划分为多个子图
- **子图优化**: 子图合并、子图复用、子图转换
- **数据路径优化**: 生成搬运操作、处理边界 tensor
- **形状优化**: 推断动态 shape、对齐 tensor shape

### 2.3 Block Graph Pass 阶段

Block Graph Pass 阶段处理 Block Graph 层面的优化，主要关注：

- **参数管理**: 推断参数索引、处理动态形状
- **内存复用**: 合并源目标 buffer、复用内存
- **调度优化**: 乱序调度、优化操作执行顺序
- **同步管理**: 插入同步操作、优化同步开销
- **代码生成准备**: 内存分配、copy out resolve、代码生成预处理

---

## 3. Pass 详细介绍

### 3.1 Tensor Graph Pass 阶段

| Pass 名称 | 简要描述 | 主要功能 |
|-----------|----------|----------|
| RemoveRedundantReshape | 删除冗余的 Reshape 操作 | 识别并删除输入输出形状相同或消费者都是 Reshape 的冗余操作 |
| AutoCast | 自动插入和优化类型转换操作 | 根据硬件架构和操作数据类型支持，自动插入必要的 Cast 操作并优化冗余 Cast 链 |
| InferMemoryConflict | 推断和解决内存冲突 | 通过前向和后向传播分析 tensor 内存使用，检测冲突并插入 Register Copy 操作 |
| RemoveUndrivenView | 删除未被驱动的 View 操作 | 处理 OP_ASSEMBLE_SSA，删除未驱动的 View 并降级为 OP_ASSEMBLE |
| ExpandFunction | 将 Tensor Graph 展开为 Tile Graph | 将高层操作展开为具体的 Tile 操作，从抽象到底层实现的关键转换 |

### 3.2 Tile Graph Pass 阶段

| Pass 名称 | 简要描述 | 主要功能 |
|-----------|----------|----------|
| MergeViewAssemble | 合并连续的 View 和 Assemble 操作 | 将多个连续的 View/Assemble 合并为一个，减少操作数量 |
| SplitReshape | 拆分 Reshape 操作 | 当输入输出存在重叠时，拆分为多个 View 和 Assemble 操作 |
| SplitRawTensor | 拆分 RawTensor | 当 LogicalTensor shape 小于 RawTensor shape 时，创建新的 RawTensor 以匹配 |
| SplitLargeFanoutTensor | 处理大扇出 tensor 的拆分 | 将被多个消费者消费的大 tensor 拆分为多个小 tensor，提高并行度 |
| DuplicateOp | 复制 View 和 GatherIn 操作 | 为多个消费者创建新的操作，避免操作共享问题 |
| AssignMemoryType | 为操作和 tensor 分配内存类型 | 根据操作需求和硬件限制，分配合适的内存类型并插入 Convert 操作 |
| InferDiscontinuousInput | 推断非连续输入的内存访问模式 | 从 InCast 出发前向传播，检测冲突并插入 Copy 操作 |
| InsertOpForViewAssemble | 在 View 和 Assemble 之间插入 Copy 操作 | 处理内存类型差异，插入必要的 Copy 操作 |
| RemoveRedundantOp | 消除重复的操作 | 识别并删除重复的操作，减少计算量 |
| SplitK | 消除 ReduceAcc 操作 | 优化 K 轴归约计算，将多个 A_MUL_B 的 CopyOut 直接连接到最终 GM |
| GraphPartition | 图划分和优化 | 复合 Pass，整合图划分、NBuffer 合并、L1 Copy 复用等子 Pass |
| ReduceCopyMerge | 合并 Reduce Copy 操作 | 将多个连续的 Reduce Acc 操作合并为一个，减少操作数量 |
| NBufferMerge | NBuffer 合并优化 | 通过着色算法将子图分组合并，减少子图切换开销 |
| L1CopyInReuseMerge | L1 Copy In 复用优化 | 合并重复的 L1 Copy In 操作，减少内存访问 |
| IntraSubgraphAdapter | 适配子图间的边界 tensor | 处理跨子图的 tensor 传递，插入必要的 ASSEMBLE 和 VIEW 操作 |
| GenerateMoveOp | 生成搬运操作 | 将 VIEW/ASSEMBLE/CONVERT/DUPLICATE 转换为具体的搬运操作 |
| CommonOperationEliminate | 消除重复的计算操作 | 通过哈希操作特征识别相同计算，用一操作替换多个相同操作 |
| AxisCombine | 对齐广播操作的输入 | 插入 BRCB 或 EXPAND 操作确保输入最后一维对齐 |
| PadLocalBuffer | 对齐本地缓冲区的 tensor shape | 通过 padding 确保 tensor 满足硬件对齐要求 |
| RemoveUnalignedReshape | 删除未对齐的 Reshape 操作 | 删除输入输出不对齐的 Reshape 操作以避免性能问题 |
| ReplaceTensor | 替换 tensor | 识别 inplace 操作并用一个 tensor 替换多个 tensor，减少内存使用 |
| PreGraphProcess | 预处理图结构 | 设置子图颜色、边界、Cube 操作属性，优化 View 和 Assemble 操作 |
| InferDynShape | 推断动态 shape | 通过拓扑排序遍历操作并调用 infer shape 函数 |
| SubgraphToFunction | 将子图转换为 Execute Graph | 构建子图调用关系，处理 Incast/Outcast 和符号化 |

### 3.3 Block Graph Pass 阶段

| Pass 名称 | 简要描述 | 主要功能 |
|-----------|----------|----------|
| InferParamIndex | 推断参数索引 | 为子函数推断参数索引，处理动态形状，更新子函数参数信息 |
| SrcDstBufferMerge | 合并源目标 buffer | 通过 inplace 语义和 L0 内存复用，减少内存分配 |
| AddAlloc | 添加内存分配操作 | 为需要分配内存的 tensor 插入 alloc 操作 |
| OoOSchedule | 执行乱序调度 | 分析操作依赖关系，使用调度算法优化执行顺序，提升并行度 |
| TuneTileOpSeqForVF | 针对 Vector Fusion 优化 TileOp 序列 | 调整 Pipe V 操作的执行顺序，优化同步开销 |
| RemoveAlloc | 移除 alloc 操作 | 清理不需要的内存分配操作，简化图结构 |
| CopyOutResolve | 解析 CopyOut 操作 | 为每个 Outcast 插入 AICPU_CALL 操作以触发 copy out resolve |
| InsertSync | 插入同步操作 | 在 Operation 之间插入 SYNC_SRC 和 SYNC_DST 操作，确保数据依赖正确 |
| TuneSyncForVF | 针对 Vector Fusion 优化同步操作 | 调整 SetFlag 和 WaitFlag 的位置，优化同步开销 |
| MixSubgraphSplit | 混合子图拆分 | 将 Mix 子图拆分为多个独立的 Cube 和 Vector 子图，并重新分配 subgraphID |
| GlobalMemoryReuse | 全局内存复用 | 通过 TensorBucket 机制实现跨操作的内存复用，支持 leaf 内和 leaf 间的复用 |
| LoopAxesProc | 循环轴处理 | 处理 Loop Axes，为支持 Vector Fusion 的操作标记 loopGroup 和 loopAxes 属性 |
| CodegenPreproc | 代码生成预处理 | 为代码生成阶段进行预处理和准备，包括类型转换、参数索引保存、axis 合并等 |

---

## 4. 总结

PyPTO 编译流程通过 Pass 模块实现了从用户代码到可执行 NPU 代码的完整转换。这些 Pass 模块按照 Tensor Graph、Tile Graph、Block Graph 三个阶段组织，每个阶段专注于不同层次的优化：

- **Tensor Graph 阶段**: 专注于高层图结构的优化，包括冗余操作消除、类型转换、内存冲突处理、图展开
- **Tile Graph 阶段**: 专注于中层图结构的优化，包括操作合并、内存分配、图划分、子图转换等
- **Block Graph 阶段**: 专注于底层执行优化的优化，包括调度、同步、内存复用、代码生成准备

通过这些 Pass 的协同工作，PyPTO 编译器能够生成高效、正确的 NPU 可执行代码。
