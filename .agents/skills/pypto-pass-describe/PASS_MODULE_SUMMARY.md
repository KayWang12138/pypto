# PyPTO Pass 模块功能总结文档

本文档汇总了 PyPTO 所有 Pass 模块的功能介绍。

---

## 一、Pass 总体介绍

PyPTO Pass 是编译器图优化 pass，负责将高层 Tensor Graph 逐步转换为可执行的 Block Graph，并完成各类优化。共包含 **41 个 Pass**，分为三个阶段：

| 阶段 | Pass 数量 | 处理对象 | 主要功能 |
|:---|:---:|:---|:---|
| Tensor Graph Pass | 15 | Tensor Graph | 类型转换、内存冲突检测、Reshape优化 |
| Tile Graph Pass | 16 | Tile Graph | 图划分、子图合并、数据布局优化、调度准备 |
| Block Graph Pass | 11 | Block Graph | 内存分配、乱序调度、同步插入、代码生成准备 |

---

## 二、各阶段介绍

### 2.1 Tensor Graph Pass (00-14)

**处理对象**：高层抽象的 Tensor Graph，使用 LogicalTensor 和 Operation 表示。

**主要功能**：
- **类型转换**：自动插入 Cast 操作处理 BF16/FP16/FP32 转换
- **图优化**：删除冗余 Reshape、View 操作
- **内存处理**：检测并解决内存冲突，插入 Register Copy
- **图展开**：将 Tensor Graph 展开为 Tile Graph

**执行顺序**：00 → 01 → 02 → 03 → 04（ExpandFunction 转换为 Tile Graph）

---

### 2.2 Tile Graph Pass (15-30)

**处理对象**：Tile 级别的图，使用具体的 Tile 操作。

**主要功能**：
- **图划分**：将大图划分为多个子图
- **子图合并**：NBuffer 合并、Reduce 合并、L1 Copy 复用
- **数据布局**：Reshape 拆分、RawTensor 拆分、Buffer 对齐
- **内存管理**：分配内存类型、处理非连续输入
- **调度准备**：拓扑排序、边界设置、子图转函数

**执行顺序**：15（GraphPartition 复合）→ 子图优化 → 子图转函数 (28) → 参数推断 (29) → 内存复用 (30)

---

### 2.3 Block Graph Pass (31-41)

**处理对象**：可执行的 Block Graph，包含具体操作和调度信息。

**主要功能**：
- **内存分配**：添加 Alloc 操作
- **乱序调度**：优化操作执行顺序，提升并行度
- **同步插入**：插入 SYNC 操作确保数据依赖
- **Vector Fusion 优化**：调整 TileOp 和同步操作位置
- **代码生成准备**：设置 needAlloc 属性、强制合并 axis

**执行顺序**：31 → 32 → 33 → 34 → 35 → 36 → 37 → 38 → 39 → 40 → 41 → 代码生成

---

## 三、各 Pass 模块简要说明

### 3.1 Tensor Graph Pass (00-14)

| 序号 | Pass 名称 | 功能说明 |
|:---:|:---|:---|
| 00 | RemoveRedundantReshape | 删除冗余的 Reshape 操作（输入输出 shape 相同或所有消费者都是 Reshape） |
| 01 | AutoCast | 自动插入 Cast 操作，处理 BF16/FP16/FP32 类型转换 |
| 02 | InferMemoryConflict | 检测内存冲突，插入 Register Copy 解决冲突 |
| 03 | RemoveUndrivenView | 删除未被驱动的 View 操作，将 OP_ASSEMBLE_SSA 降级为 OP_ASSEMBLE |
| 04 | ExpandFunction | 将 Tensor Graph 展开为 Tile Graph（关键转换步骤） |
| 05 | MergeViewAssemble | 合并连续的 View 和 Assemble 操作链 |
| 06 | SplitReshape | 将 Reshape 拆分为 View 和 Assemble 操作（处理数据重叠） |
| 07 | SplitRawTensor | 拆分 RawTensor 以匹配 LogicalTensor 的 shape |
| 08 | SplitLargeFanoutTensor | 拆分大扇出 tensor，提高并行度 |
| 09 | DuplicateOp | 复制 View 和 GatherIn 操作，为每个消费者创建新操作 |
| 10 | AssignMemoryType | 为操作和 tensor 分配内存类型（DDR/UB/L1/L0C 等） |
| 11 | InferDiscontinuousInput | 推断非连续输入的内存访问模式，插入 Copy |
| 12 | RemoveRedundantOp | 消除重复的计算操作（通过哈希识别） |
| 13 | InsertOpForViewAssemble | 在 View 和 Assemble 之间插入 Copy 操作 |
| 14 | SplitK | 消除 ReduceAcc 操作，优化 K 轴归约计算 |

---

### 3.2 Tile Graph Pass (15-30)

| 序号 | Pass 名称 | 功能说明 |
|:---:|:---|:---|
| 15 | GraphPartition | 复合 Pass：图划分、子图合并（包含 NBufferMerge、L1CopyReuse 等） |
| 16 | ReduceCopyMerge | 合并 Reduce Copy 操作，使用并查集算法 |
| 17 | NBufferMerge | NBuffer 合并优化，通过着色算法将相似子图分组 |
| 18 | L1CopyInReuseMerge | L1 Copy 复用优化 |
| 19 | IntraSubgraphAdapter | 适配子图间的边界 tensor，插入 ASSEMBLE 和 VIEW |
| 20 | GenerateMoveOp | 将 VIEW/ASSEMBLE 转换为具体搬运操作（COPY_IN/COPY_OUT 等） |
| 21 | CommonOperationEliminate | 消除重复的计算操作 |
| 22 | AxisCombine | 对齐广播操作的输入轴，插入 BRCB 或 EXPAND |
| 23 | PadLocalBuffer | 对齐本地缓冲区的 tensor shape（Matmul/Vector 对齐） |
| 24 | RemoveUnalignedReshape | 移除未对齐的 Reshape 操作 |
| 25 | ReplaceTensor | 替换 tensor，识别 inplace 操作减少内存使用 |
| 26 | PreGraphProcess | 图预处理：拓扑排序、边界设置、Cube 属性更新 |
| 27 | InferDynShape | 推断动态 shape |
| 28 | SubgraphToFunction | 将子图转换为 Execute Graph 函数（支持子图复用） |
| 29 | InferParamIndex | 推断子函数的参数索引 |
| 30 | SrcDstBufferMerge | 合并源目标 buffer，实现内存复用 |

---

### 3.3 Block Graph Pass (31-41)

| 序号 | Pass 名称 | 功能说明 |
|:---:|:---|:---|
| 31 | AddAlloc | 添加内存分配操作（L0/L1/UB Alloc） |
| 32 | OoOSchedule | 乱序调度，优化操作执行顺序提升并行度 |
| 33 | TuneTileOpSeqForVF | Vector Fusion 优化：调整 TileOp 序列（仅 DAV_3510） |
| 34 | RemoveAlloc | 移除 alloc 操作 |
| 35 | CopyOutResolve | 解析 CopyOut，插入 AICPU_CALL 触发 resolve |
| 36 | InsertSync | 插入 SYNC_SRC 和 SYNC_DST 同步操作 |
| 37 | TuneSyncForVF | Vector Fusion 优化：调整同步操作位置（仅 DAV_3510） |
| 38 | MixSubgraphSplit | 将 Mix 子图拆分为独立的 Cube 和 Vector 子图（仅 DAV_3510） |
| 39 | GlobalMemoryReuse | 全局内存复用，通过 TensorBucket 机制 |
| 40 | LoopaxesProc | 处理 Loop Axes，标记 loopGroup 和 loopAxes 属性（仅 VF） |
| 41 | CodegenPreproc | 代码生成前预处理（保存参数索引、设置 needAlloc） |

---

## 四、Pass 执行流程图

```
┌─────────────────────────────────────────────────────────────┐
│                    Tensor Graph Pass (00-14)                 │
│  RemoveRedundantReshape → AutoCast → InferMemoryConflict    │
│  → RemoveUndrivenView → [ExpandFunction] ↓                   │
└─────────────────────────────────────────────────────────────┘
                              ↓ 展开为 Tile Graph
┌─────────────────────────────────────────────────────────────┐
│                    Tile Graph Pass (15-30)                   │
│  GraphPartition → 子图优化 → IntraSubgraphAdapter            │
│  → GenerateMoveOp → PreGraphProcess → SubgraphToFunction    │
│  → InferParamIndex → SrcDstBufferMerge ↓                    │
└─────────────────────────────────────────────────────────────┘
                              ↓ 转换为 Block Graph
┌─────────────────────────────────────────────────────────────┐
│                    Block Graph Pass (31-41)                 │
│  AddAlloc → OoOSchedule → InsertSync → CopyOutResolve        │
│  → GlobalMemoryReuse → CodegenPreproc → 代码生成             │
└─────────────────────────────────────────────────────────────┘
```

---

## 五、关键 Pass 说明

### 5.1 类型转换关键 Pass

| Pass | 功能 |
|:---|:---|
| ExpandFunction (04) | 将 Tensor Graph 转换为 Tile Graph，是图转换的关键入口 |
| GenerateMoveOp (20) | 将抽象的 VIEW/ASSEMBLE 转换为具体搬运操作 |
| SubgraphToFunction (28) | 将 Tile Graph 子图转换为可执行函数 |

### 5.2 内存优化关键 Pass

| Pass | 功能 |
|:---|:---|
| AssignMemoryType (10) | 为 tensor 分配合适的内存类型 |
| ReplaceTensor (25) | 识别 inplace 操作，减少内存分配 |
| GlobalMemoryReuse (39) | 全局内存复用，减少总内存占用 |

### 5.3 调度优化关键 Pass

| Pass | 功能 |
|:---|:---|
| OoOSchedule (32) | 乱序调度，提升并行度 |
| InsertSync (36) | 插入同步操作，确保数据依赖正确 |
| TuneSyncForVF (37) | Vector Fusion 同步优化 |

### 5.4 Vector Fusion 相关 Pass

| Pass | 功能 | 适用平台 |
|:---|:---|:---|
| TuneTileOpSeqForVF (33) | 调整 TileOp 序列 | DAV_3510 |
| TuneSyncForVF (37) | 调整同步操作位置 | DAV_3510 |
| LoopaxesProc (40) | 标记 loopGroup/loopAxes | 启用 VF |

---

## 六、源码目录速查

```
framework/src/passes/
├── tensor_graph_pass/           # 00-14
├── tile_graph_pass/             # 15-30
│   ├── graph_constraint/
│   ├── graph_optimization/
│   ├── graph_partition/
│   └── data_path/
├── block_graph_pass/            # 31-41
│   ├── schedule_ooo/
│   └── memory_reuse/
└── pass_mgr/                    # Pass 管理器
```
