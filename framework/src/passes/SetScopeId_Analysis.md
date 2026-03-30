# AddRawOperation SetScopeId 配置分析与修复报告

## 1. 背景机制

### 1.1 AddRawOperation 的默认行为

`function.cpp:1481-1494`：

```cpp
Operation &Function::AddRawOperation(const Opcode opCode, const LogicalTensors &iOperands,
                                     const LogicalTensors &oOperands, bool updateTensorMap) {
    auto &op = operations_.emplace_back(std::make_shared<Operation>(*this, opCode, iOperands, oOperands, updateTensorMap));
    opPosition_.emplace(op.get(), operations_.size() - 1);
    operations_.back()->SetScopeId(config::GetPassOption<int>(SG_SET_SCOPE));  // 默认为 -1
    return *operations_.back();
}
```

`AddRawOperation` 自动从全局配置项 `SG_SET_SCOPE` 读取 scopeId 并设置。该配置项仅在 `ExpandFunction` 中被临时设置，其余时刻均为 `-1`。

### 1.2 SG_SET_SCOPE 的设置时机

`expand_function.cpp:165-181`：

```cpp
Status ExpandFunction::ExpandOperation(Function &function, Operation &op) const {
    int scopeIdx = op.GetScopeId();
    config::SetPassOption(SG_SET_SCOPE, scopeIdx);
    ExpandOperationInto(function, ...);
    config::SetPassOption(SG_SET_SCOPE, -1);
}
```

### 1.3 scopeId 的作用

`supernode_graph_builder.cpp` 中 `GraphPartition` 使用 scopeId：
- 第 311-314 行：将 scopeId 传播到 supernode
- 第 330-332 行：有 scopeId 的 node 不可合并（`isMergeable = false`）
- 第 608-625 行 `UpdateScopeId`：将 scopeId 向相邻的 ASSEMBLE/VIEW op 传播
- 第 636-645 行：相同 scopeId 的相邻 op 会被合并到同一个 subgraph

### 1.4 Pass 执行顺序（PVC2_OOO 策略）

| 序号 | Pass 名 | AddRawOperation 使用 | SetScopeId 修复前状态 |
|------|---------|---------------------|---------------------|
| 1 | RemoveRedundantReshape | 不创建新 op | N/A |
| 2 | AutoCast | 有 (OP_CAST) | **已修复** |
| 3 | InferMemoryConflict | 有 (OP_REGISTER_COPY) | **已修复** |
| 4 | RemoveUndrivenView | 不创建新 op | N/A |
| 5 | ExpandFunction | AddOperation | 原本正确（通过 SG_SET_SCOPE） |
| 6 | MergeViewAssemble | 有 (OP_VIEW, OP_ASSEMBLE) | **已修复** |
| 7 | SplitReshape | 通过 GraphUtils | 原本正确（GraphUtils 中已设置） |
| 8 | SplitRawTensor | 不创建新 op | N/A |
| 9 | SplitLargeFanoutTensor | 有 (AddOperation) | **已修复** |
| 10 | DuplicateOp | 有 (OP_GATHER_IN_L1, OP_VIEW) | **已修复** |
| 11 | AssignMemoryType | 不创建新 op | N/A |
| 12 | InferDiscontinuousInput | 有 (OP_VIEW, OP_ASSEMBLE) | **已修复** |
| 13 | RemoveRedundantOp | 有 (AddOperation: OP_VIEW) | **已修复** |
| 14 | InsertOpForViewAssemble | 有 (OP_ASSEMBLE, OP_VIEW) | **已修复** |
| 15 | SplitK | 不创建新 op | N/A |
| 16 | **GraphPartition** | - | 分配 scopeId |

---

## 2. 已修复的问题详情

### 2.1 DuplicateOp — `duplicate_op.cpp:82,120`

**问题**：克隆 OP_GATHER_IN_L1 和 OP_VIEW 时未继承原始 op 的 scopeId。

**修复**：

```diff
 // ProcessGatherIn (line 82)
 auto &newOp = function.AddRawOperation(Opcode::OP_GATHER_IN_L1, operation.GetIOperands(), {dst});
 newOp.SetAttribute(OpAttributeKey::startOffset, operation.GetIntAttribute(OpAttributeKey::startOffset));
+newOp.SetScopeId(operation.GetScopeId());

 // ProcessView (line 120)
 auto &newOp = function.AddRawOperation(Opcode::OP_VIEW, {iOperand}, {dst});
+newOp.SetScopeId(operation.GetScopeId());
 auto oriViewAttr = dynamic_cast<ViewOpAttribute *>(operation.GetOpAttribute().get());
```

---

### 2.2 InsertOpForViewAssemble — `insert_op_for_viewassemble.cpp:32,37`

**问题**：在 VIEW-ASSEMBLE 对之间插入新的 ASSEMBLE-DDR-VIEW 序列时未继承 scopeId。

**修复**：

```diff
 Operation &assemble = function.AddRawOperation(Opcode::OP_ASSEMBLE, {moveOutTensorPtr}, {ddrTensorPtr});
+assemble.SetScopeId(assembleOp->GetScopeId());
 assemble.SetOpAttribute(std::make_shared<AssembleOpAttribute>(...));

 Operation &view = function.AddRawOperation(Opcode::OP_VIEW, {ddrTensorPtr}, {moveInTensorPtr});
+view.SetScopeId(viewOp->GetScopeId());
 view.SetOpAttribute(std::make_shared<ViewOpAttribute>(...));
```

---

### 2.3 MergeViewAssemble — `merge_view_assemble_utils.h` + `.cpp:96,113`

**问题**：合并 VIEW/ASSEMBLE 链后创建新 op 未继承原始 scopeId。需要在数据结构中新增字段。

**修复**：

头文件 `merge_view_assemble_utils.h` — 结构体新增 `scopeId` 字段：

```diff
 struct ViewOp {
     std::shared_ptr<LogicalTensor> input;
     std::shared_ptr<LogicalTensor> output;
     std::vector<int64_t> offset;
     std::vector<SymbolicScalar> dynOffset;
     std::vector<SymbolicScalar> dynValidShape;
     MemoryType toType = MemoryType::MEM_UNKNOWN;
     bool hasCopyInMode;
     npu::tile_fwk::Any copyInModeValue;
+    int scopeId = -1;
 };

 struct AssembleOp {
     std::shared_ptr<LogicalTensor> input;
     std::shared_ptr<LogicalTensor> output;
     std::vector<int64_t> offset;
     std::vector<SymbolicScalar> dynOffset;
+    int scopeId = -1;
 };
```

实现文件 — Record 时保存 scopeId：

```diff
 // RecordMergedViewOperation (line 298)
-viewOpToAppend_.emplace_back(ViewOp{startTensor, endTensor, newOffset, newDynOffset, newDynValidShape, lastViewAttr->GetTo(), hasCopyInMode, std::move(copyInModeValue)});
+viewOpToAppend_.emplace_back(ViewOp{startTensor, endTensor, newOffset, newDynOffset, newDynValidShape, lastViewAttr->GetTo(), hasCopyInMode, std::move(copyInModeValue), lastViewOp->GetScopeId()});

 // RecordAssembleOperation (line 426)
-assembleOpToAppend_.emplace_back(AssembleOp{input, output, offset, dynOffset});
+assembleOpToAppend_.emplace_back(AssembleOp{input, output, offset, dynOffset, scopeId});
```

函数签名新增 `scopeId` 参数：

```diff
-void RecordAssembleOperation(const std::shared_ptr<LogicalTensor> &input,
-    const std::shared_ptr<LogicalTensor> &output, const std::vector<int64_t> &offset,
-    const std::vector<SymbolicScalar> &dynOffset);
+void RecordAssembleOperation(const std::shared_ptr<LogicalTensor> &input,
+    const std::shared_ptr<LogicalTensor> &output, const std::vector<int64_t> &offset,
+    const std::vector<SymbolicScalar> &dynOffset, int scopeId);
```

调用处传入 `operation.GetScopeId()`：

```diff
-RecordAssembleOperation(startTensor, endTensor, newOffset, newDynOffset);
+RecordAssembleOperation(startTensor, endTensor, newOffset, newDynOffset, operation.GetScopeId());
```

AppendMerged 时设置 scopeId：

```diff
 // AppendMergedViewOperations
 auto &mergedViewOp = function.AddRawOperation(Opcode::OP_VIEW, {viewOp.input}, {viewOp.output});
+mergedViewOp.SetScopeId(viewOp.scopeId);
 mergedViewOp.SetOpAttribute(attr);

 // AppendMergedAssembleOperations
 auto &mergedAssembleOp = function.AddRawOperation(Opcode::OP_ASSEMBLE, {assembleOp.input}, {assembleOp.output});
+mergedAssembleOp.SetScopeId(assembleOp.scopeId);
 mergedAssembleOp.SetOpAttribute(attr);
```

---

### 2.4 InferDiscontinuousInput — `infer_discontinuous_input.cpp:268,275`

**问题**：插入 VIEW/ASSEMBLE 处理不连续输入时未继承 scopeId。

**修复**：从 input tensor 的 producer 继承 scopeId：

```diff
 void InferDiscontinuousInput::InsertViewOp(Function &function, LogicalTensorPtr iOperand, LogicalTensorPtr oOperand) {
     auto &insertViewOp = function.AddRawOperation(Opcode::OP_VIEW, {iOperand}, {oOperand});
+    auto &producers = iOperand->GetProducers();
+    if (!producers.empty()) {
+        insertViewOp.SetScopeId((*producers.begin())->GetScopeId());
+    }
     insertViewOp.SetOpAttribute(...);
 }

 void InferDiscontinuousInput::InsertAssembleOp(Function &function, LogicalTensorPtr iOperand, LogicalTensorPtr oOperand) {
     auto &insertAssembleOp = function.AddRawOperation(Opcode::OP_ASSEMBLE, {iOperand}, {oOperand});
+    auto &producers = iOperand->GetProducers();
+    if (!producers.empty()) {
+        insertAssembleOp.SetScopeId((*producers.begin())->GetScopeId());
+    }
     insertAssembleOp.SetOpAttribute(...);
 }
```

---

### 2.5 InferMemoryConflict — `infer_memory_conflict.cpp:474,501`

**问题**：在 RESHAPE 前后插入 REGISTER_COPY 解决内存冲突时未继承 scopeId。

**修复**：从被保护的 op 继承 scopeId：

```diff
 // InsertPrecededCopys (line 474)
 auto &copyOp = function.AddRawOperation(Opcode::OP_REGISTER_COPY, {inputTensor}, {newTensor});
+copyOp.SetScopeId(op->GetScopeId());

 // InsertPostCopys (line 501)
 auto &copyOp = function.AddRawOperation(Opcode::OP_REGISTER_COPY, {newTensor}, {outputTensor});
+copyOp.SetScopeId(op->GetScopeId());
```

---

### 2.6 SplitLargeFanoutTensor — `split_large_fanout_tensor.cpp:265,304`

**问题**：进一步拆分大 fanout tensor 时创建的新 ASSEMBLE/VIEW 未继承 scopeId。

**修复**：`CreateOpForMoreSplit` 新增 `scopeId` 参数并传入：

```diff
 // MoreSplit 调用处 (line 257)
-CreateOpForMoreSplit(function, largeTensor, overlaps, gcdShape, dualOverlap, gcdTileOffsets, viewOpOffset);
+CreateOpForMoreSplit(function, largeTensor, overlaps, gcdShape, dualOverlap, gcdTileOffsets, viewOpOffset, viewOp->GetScopeId());

 // 函数签名
-void CreateOpForMoreSplit(Function &function, LogicalTensorPtr largeTensor, LogicalTensors overlaps,
-    Shape gcdShape, LogicalTensorPtr dualOverlap, std::vector<Shape> gcdTileOffsets, Offset viewOpOffset);
+void CreateOpForMoreSplit(Function &function, LogicalTensorPtr largeTensor, LogicalTensors overlaps,
+    Shape gcdShape, LogicalTensorPtr dualOverlap, std::vector<Shape> gcdTileOffsets, Offset viewOpOffset, int scopeId);

 // 新 ASSEMBLE (line 265)
 auto &newAssembleOp = function.AddOperation(Opcode::OP_ASSEMBLE, {newGcdTensor}, {dualOverlap});
+newAssembleOp.SetScopeId(scopeId);

 // 新 VIEW (line 304)
 auto &newViewOp = function.AddOperation(Opcode::OP_VIEW, {overlapGcdTile}, {newGcdTensor});
+newViewOp.SetScopeId(scopeId);
```

---

### 2.7 RemoveRedundantOp — `remove_redundant_op.cpp:402`

**问题**：合并冗余 VIEW-ASSEMBLE 链后创建新 VIEW 时未继承 scopeId。

**修复**：

```diff
 auto &newViewOp = function.AddOperation(Opcode::OP_VIEW, {startTensor}, {newViewTensor});
+newViewOp.SetScopeId(op.GetScopeId());
```

---

### 2.8 AutoCast — `auto_cast.cpp:149` + `auto_cast.h:40`

**问题**：插入 CAST 类型转换 op 时未继承相关 op 的 scopeId。

**修复**：`InsertCastOp` 函数新增 `scopeId` 参数：

```diff
 // auto_cast.h
-void InsertCastOp(Function &function, LogicalTensorPtr src, LogicalTensorPtr tgt, const TileShape &tileShape);
+void InsertCastOp(Function &function, LogicalTensorPtr src, LogicalTensorPtr tgt, const TileShape &tileShape, int scopeId);

 // auto_cast.cpp 实现
 void AutoCast::InsertCastOp(Function &function, LogicalTensorPtr src, LogicalTensorPtr tgt,
-                                       const TileShape &tileShape) {
+                                       const TileShape &tileShape, int scopeId) {
     Operation &newCast = function.AddRawOperation(Opcode::OP_CAST, {src}, {tgt});
+    newCast.SetScopeId(scopeId);
     ...
 }
```

6 个调用点全部更新为传入 `op->GetScopeId()`（或 `castChain[i]->GetScopeId()`）：

```diff
-InsertCastOp(function, srcTensor, fp32Tensor, op->GetTileShape());
+InsertCastOp(function, srcTensor, fp32Tensor, op->GetTileShape(), op->GetScopeId());

-InsertCastOp(function, iop, newInput, op->GetTileShape());
+InsertCastOp(function, iop, newInput, op->GetTileShape(), op->GetScopeId());

-InsertCastOp(function, newOutput, oop, op->GetTileShape());
+InsertCastOp(function, newOutput, oop, op->GetTileShape(), op->GetScopeId());
```

---

## 3. 已正确实现的参考案例（无需修改）

| 文件 | 位置 | 说明 |
|------|------|------|
| ConvertOpInserter | `convert_op_inserter.cpp:510` | `convertOp.SetScopeId(producerScopeId)` 从 producer 继承 |
| GraphUtils::AddAssembleOperation | `graph_utils.cpp:53` | `newOp.SetScopeId(assemble.originOp->GetScopeId())` 从原始 op 继承 |
| GraphUtils::AddReshapeOperation | `graph_utils.cpp:65` | `newOp.SetScopeId(reshapeOp.originOpPtr->GetScopeId())` 从原始 op 继承 |
| ExpandFunction | `expand_function.cpp:178` | 通过 `SG_SET_SCOPE` 全局配置传递 |

---

## 4. 修改文件清单

| 文件 | 修改类型 |
|------|----------|
| `tile_graph_pass/graph_optimization/duplicate_op.cpp` | 新增 2 处 SetScopeId |
| `tile_graph_pass/graph_optimization/insert_op_for_viewassemble.cpp` | 新增 2 处 SetScopeId |
| `pass_utils/merge_view_assemble_utils.h` | ViewOp/AssembleOp 结构体新增 scopeId 字段，RecordAssembleOperation 签名变更 |
| `pass_utils/merge_view_assemble_utils.cpp` | Record/Append 各新增 SetScopeId，共 5 处 |
| `tile_graph_pass/graph_optimization/infer_discontinuous_input.cpp` | InsertViewOp/InsertAssembleOp 各新增 SetScopeId |
| `tensor_graph_pass/infer_memory_conflict.cpp` | InsertPrecededCopys/InsertPostCopys 各新增 SetScopeId |
| `tile_graph_pass/graph_optimization/split_large_fanout_tensor.h` | CreateOpForMoreSplit 签名变更 |
| `tile_graph_pass/graph_optimization/split_large_fanout_tensor.cpp` | 新增 scopeId 参数传递 + 2 处 SetScopeId |
| `tile_graph_pass/graph_optimization/remove_redundant_op.cpp` | 新增 1 处 SetScopeId |
| `tensor_graph_pass/auto_cast.h` | InsertCastOp 签名变更 |
| `tensor_graph_pass/auto_cast.cpp` | 函数实现 + 6 个调用点传入 scopeId |

---

## 5. 修复模式总结

**统一修复模式**：任何通过 `AddRawOperation` / `AddOperation` 创建新 op 来替代、复制或补充现有 op 的场景，都应显式调用 `SetScopeId` 继承原始 op 或关联 op 的 scopeId。

```cpp
// 模式 1：从原始 op 继承（用于复制/合并场景）
auto &newOp = function.AddRawOperation(opcode, iOperands, oOperands);
newOp.SetScopeId(originalOp.GetScopeId());

// 模式 2：从 producer 继承（用于插入中间 op 场景）
auto &newOp = function.AddRawOperation(opcode, {input}, {output});
auto &producers = input->GetProducers();
if (!producers.empty()) {
    newOp.SetScopeId((*producers.begin())->GetScopeId());
}
```

---

## 6. 编译验证

```
Build[CI] Finish, Duration 54 secs.
Successfully built pypto-0.2.1.tar.gz and pypto-0.2.1-cp311-cp311-linux_aarch64.whl
```

编译通过，无新增 error/warning。

---

## 7. 补充说明：GraphPartition 后的 Pass

以下 Pass 在 GraphPartition 之后运行，操作在单个 subgraph function 内，scopeId 影响较小（由 `UpdateSubgraphID` 管理）：

- GenerateMoveOp — 通过 `UpdateSubgraphID` 正确设置
- ReplaceTensor — 通过 `UpdateSubgraphID` 正确设置
- RemoveUnalignedReshape — 通过 `UpdateSubgraphID` 正确设置
- PreGraphProcess/SetBoundary — 通过 `UpdateSubgraphID` 正确设置
- AxisCombine — 通过 `UpdateSubgraphID` 设置
- PreGraphProcess/RemoveRedundantAssemble — 未设置 scopeId 也未 UpdateSubgraphID
- IntraSubgraphAdapter — 未设置 scopeId
- InsertSync — 未设置 scopeId
- InplaceProcess — 未设置 scopeId
