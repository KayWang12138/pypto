# combine_axis 合轴优化开关实现指南

## 一、概述

### 1.1 功能定义

`combine_axis` 是 PyPTO 编译框架提供的**尾轴 broadcast inline 优化开关**，通过减少不必要的内存扩展操作，提升 NPU 计算效率。

### 1.2 核心目标

**典型场景**：双目运算 `(32,1) + (32,128)`

- **未启用优化**：
  ```
  (32,1) → EXPAND → (32,128) → 计算
  ```
  需要将 shape 为 `(32,1)` 的 tensor 完整 broadcast 到 `(32,128)`，消耗内存带宽

- **启用优化**：
  ```
  (32,1) → BRCB → (32,8) → 计算
  ```
  通过 BRCB 指令直接扩展到 `(32,8)`，再进行 `(32,8) + (32,128)` 运算

### 1.3 性能收益

- **内存带宽节省**：避免全量 broadcast，减少数据搬运
- **UB 空间节省**：尾轴合并后 shape 变小，占用更少的 UB 缓冲区
- **计算效率提升**：BRCB 指令在硬件层面优化，比 EXPAND 更高效

### 1.4 支持产品

| 产品             | 是否支持 |
|:-----------------|:--------:|
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 |    √     |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 |    √     |

---

## 二、Python API 接口

### 2.1 接口定义

**文件位置**：`python/pypto/experimental.py:135`

```python
def set_operation_options(*, force_combine_axis: Optional[bool] = None,
                         combine_axis: Optional[bool] = None):
    """
    Set operation options.

    Parameters
    ---------
    force_combine_axis : bool
        Codegen forced axis fusion optimization, Not recommended.
    combine_axis : bool
        Codegen forced axis fusion optimization.
    """
    options_dict = {k: v for k, v in locals().items() if v is not None}
    set_options(operation_options=options_dict)
```

### 2.2 参数说明

| 参数名 | 类型 | 默认值 | 说明 |
|-------|------|--------|------|
| `combine_axis` | bool | False | 尾轴 broadcast inline 优化开关（推荐使用） |
| `force_combine_axis` | bool | False | 早期版本接口，有很多功能约束，后续会逐步下线 |

### 2.3 使用示例

```python
import pypto

# 开启合轴优化
pypto.experimental.set_operation_options(combine_axis=True)

@pypto.jit
def kernel_func(x, y):
    # 形状为 (batch, 1) 的 tensor
    bias = pypto.full((x.shape[0], 1), 0.5)
    # 自动触发合轴优化，bias 会被 BRCB 扩展而不是 EXPAND
    return x + bias
```

### 2.4 典型应用场景

**文件位置**：`models/glm_v4_5/glm_attention.py:299`

```python
def attention(q, k, v, ...):
    pypto.experimental.set_operation_options(combine_axis=True)

    # Attention 计算，包含大量广播操作
    # (batch, seq_len, 1) + (batch, seq_len, seq_len)
    scale = (q @ k.transpose(-1, -2)) / (d_k ** 0.5)
    attn = scale + bias  # bias shape: (batch, seq_len, 1)

    return attn
```

---

## 三、编译框架实现架构

### 3.1 整体流程图

```
用户调用
    ↓
set_operation_options(combine_axis=True)
    ↓
TileGraphPass 阶段
    ├── AxisCombineMarker::Run()
    │   ├── Init() - 构建算子依赖图
    │   ├── ForwardVisit() - 前向标记支持合轴的 tensor
    │   └── BackwardVisit() - 后向传播合轴状态
    │
    └── AxisCombine::Process()
        └── AlignBroadCastOpInputs() - 插入 BRCB 算子
    ↓
PadLocalBuffer 阶段
    └── PadVectorForAxisCombine() - 合轴模式下的 padding 策略
    ↓
CodegenPreproc 阶段
    ├── ForceCombineAxisForAxisCombine() - 设置合轴属性
    └── CombineTailAxis() - 执行尾轴合并
    ↓
代码生成
```

### 3.2 关键 Pass 协作关系

| Pass | 阶段 | 职责 |
|------|------|------|
| **AxisCombineMarker** | TileGraphPass | 标记哪些 tensor 支持合轴优化 |
| **AxisCombine** | TileGraphPass | 在 broadcast 操作中插入 BRCB 算子 |
| **PadLocalBuffer** | TileGraphPass | 处理合轴模式下的内存对齐 |
| **CodegenPreproc** | BlockGraphPass | 执行尾轴合并，设置合轴属性 |

---

## 四、详细实现说明

### 4.1 AxisCombineMarker Pass

**文件位置**：`framework/src/passes/tile_graph_pass/graph_constraint/axis_combine_marker.cpp`

**职责**：标记计算图中哪些 tensor 支持轴组合优化

#### 4.1.1 执行流程

```cpp
void AxisCombineMarker::Run(Function &function) {
    Init(function);      // 阶段1：初始化
    ForwardVisit();      // 阶段2：正向遍历
    BackwardVisit();    // 阶段3：反向遍历
    PrintTensorStatus();
}
```

#### 4.1.2 状态定义

```cpp
enum class AxisReorderStatus {
    ENABLE,    // 支持合轴
    DISABLE,   // 不支持合轴
    UNKNOWN    // 待确定
};
```

#### 4.1.3 各算子类型的合轴判断逻辑

| 算子类型 | 判断函数 | 状态规则 |
|---------|---------|---------|
| **COPY_IN** | `UpdateCopyinStatus()` | 输出尾轴=1 且与输入尾轴相同 → ENABLE；否则 → DISABLE |
| **VIEW** | `UpdateViewStatus()` | 输入输出尾轴不同 → DISABLE；输出尾轴=1 → 继承输入状态 |
| **ASSEMBLE** | `UpdateAssembleStatus()` | 输入 ENABLE 且尾轴不变 → ENABLE；否则 → DISABLE |
| **EXPAND** | `UpdateExpandStatus()` | 非 尾轴 expand → ENABLE；尾轴 expand → DISABLE |
| **REDUCE** | `UpdateReduceStatus()` | reduce 倒数第二轴 → DISABLE；否则 → 输入状态 |
| **ELMWISE** | `UpdateElewiseStatus()` | 任一输入 DISABLE → DISABLE；输出尾轴=1 → ENABLE |

#### 4.1.4 正向遍历逻辑

```cpp
void AxisCombineMarker::ForwardVisit() {
    std::queue<size_t> procOpQueue;
    std::vector<size_t> inDegree(opList_.size(), 0);

    // 从入度为 0 的算子开始
    for (size_t j = 0; j < opInGraph_.size(); ++j) {
        if (opInGraph_[j].empty()) {
            procOpQueue.push(j);
            UpdateOpACEnableForward(j);
        }
        inDegree[j] = opInGraph_[j].size();
    }

    // BFS 拓扑排序
    while (!procOpQueue.empty()) {
        auto opIdx = procOpQueue.front();
        procOpQueue.pop();
        for (auto outIdx : opOutGraph_[opIdx]) {
            inDegree[outIdx]--;
            if (inDegree[outIdx] == 0) {
                procOpQueue.push(outIdx);
                UpdateOpACEnableForward(outIdx);
            }
        }
    }
}
```

#### 4.1.5 反向遍历逻辑

```cpp
void AxisCombineMarker::BackwardVisit() {
    std::queue<size_t> procOpQueue;
    std::vector<size_t> outDegree(opList_.size(), 0);

    // 从出度为 0 的算子开始
    for (size_t j = 0; j < opOutGraph_.size(); ++j) {
        if (opOutGraph_[j].empty()) {
            procOpQueue.push(j);
            UpdateOpACEnableBackward(j);
        }
        outDegree[j] = opOutGraph_[j].size();
    }

    // BFS 反向拓扑排序
    while (!procOpQueue.empty()) {
        auto opIdx = procOpQueue.front();
        procOpQueue.pop();
        for (auto outIdx : opInGraph_[opIdx]) {
            outDegree[outIdx]--;
            if (outDegree[outIdx] == 0) {
                procOpQueue.push(outIdx);
                UpdateOpACEnableBackward(outIdx);
            }
        }
    }
}
```

---

### 4.2 AxisCombine Pass

**文件位置**：`framework/src/passes/tile_graph_pass/graph_constraint/axis_combine.cpp`

**职责**：在 broadcast 操作中插入 BRCB 算子

#### 4.2.1 支持的算子类型

```cpp
const std::unordered_set<Opcode> NEED_BRC_OPS{
    Opcode::OP_ADD,
    Opcode::OP_SUB,
    Opcode::OP_MUL,
    Opcode::OP_DIV,
    Opcode::OP_MAXIMUM,
    Opcode::OP_MINIMUM,
    Opcode::OP_EXPANDEXPDIF,
};
```

#### 4.2.2 核心逻辑

```cpp
Status AxisCombine::AlignBroadCastOpInputs(Function &function, Operation &op) {
    auto inputTensor = op.GetIOperands();
    auto inTensor0 = inputTensor[0];
    auto inTensor1 = inputTensor[1];

    if (inTensor0->GetShape() == inTensor1->GetShape()) {
        return SUCCESS;  // 形状相同，无需处理
    }

    for (size_t idx = 0; idx < inputTensor.size(); ++idx) {
        auto srcTensor = inputTensor[idx];
        auto alignedShape = srcTensor->GetShape();

        if (alignedShape.back() == 1) {
            int64_t padValue = 0;
            GetPaddingValue(srcTensor, padValue);

            // 判断是否支持合轴
            if (!axisCombineMarker.IsTensorEnableAxisCombine(srcTensor)) {
                padValue = inputTensor[idx ^ 1]->GetShape().back();
            }

            // 对齐维度
            AlignedIfNeed(alignedShape.back(), padValue);

            // 创建对齐后的 tensor
            auto alignedTensor = std::make_shared<LogicalTensor>(
                function, srcTensor->Datatype(), alignedShape, srcTensor->Format());
            alignedTensor->SetMemoryTypeBoth(MemoryType::MEM_UB, true);

            // 插入 BRCB 或 EXPAND 算子
            auto &brcb = function.AddRawOperation(Opcode::OP_BRCB,
                {srcTensor}, {alignedTensor});

            if (!axisCombineMarker.IsTensorEnableAxisCombine(srcTensor)) {
                brcb.SetOpCode(Opcode::OP_EXPAND);
                brcb.SetAttribute(OP_ATTR_PREFIX + "EXPANDDIM",
                    GetExpandDim(srcTensor->GetShape(),
                        inputTensor[idx ^ 1]->GetShape()));
                brcb.SetAttribute(OP_ATTR_PREFIX + "validShape",
                    inputTensor[idx ^ 1]->GetDynValidShape());
            }

            brcb.UpdateSubgraphID(op.GetSubgraphID());
            srcTensor->RemoveConsumer(op);
            op.ReplaceIOperand(idx, alignedTensor);
            inputTensor[idx] = alignedTensor;

            // 标记需要 BRCB 的输入
            op.SetAttribute(OpAttributeKey::brcbIdx, static_cast<int64_t>(idx + 1));
        }
    }
    return SUCCESS;
}
```

#### 4.2.3 示例说明

**场景**：`(32, 1) + (32, 128)`

**支持合轴**：
```
输入 A: (32, 1) [ENABLE]
输入 B: (32, 128) [UNKNOWN]

↓ 插入 BRCB

(32, 1) → BRCB → (32, 8) → 计算 → (32, 128)
```

**不支持合轴**：
```
输入 A: (32, 1) [DISABLE]
输入 B: (32, 128) [UNKNOWN]

↓ 插入 EXPAND

(32, 1) → EXPAND → (32, 128) → 计算 → (32, 128)
```

---

### 4.3 PadLocalBuffer Pass

**文件位置**：`framework/src/passes/tile_graph_pass/graph_constraint/pad_local_buffer.cpp`

**职责**：处理合轴模式下的内存对齐和 padding

#### 4.3.1 对齐策略

| 操作类型 | 对齐逻辑 |
|---------|---------|
| **REDUCE** | 对齐到尾轴或倒数第二个非1的轴 |
| **BRCB** | 倒数第二轴对齐到8，尾轴对齐到Block粒度 |
| **BROADCAST** | 支持合轴 → 倒数第二轴对齐；否则 → 尾轴对齐 |
| **ELMWISE/VIEW** | 支持合轴且尾轴=1 → 倒数第二轴对齐；否则 → 尾轴对齐 |
| **EXPAND** | 始终尾轴对齐 |
| **RESHAPE** | 尾轴=1 → 倒数第二轴对齐 |

#### 4.3.2 核心函数

```cpp
void PadLocalBuffer::PadVectorForAxisCombine(
    Operation &op, LogicalTensorPtr &in,
    std::unordered_set<std::shared_ptr<RawTensor>> &visitedRaw) {

    if (in->shape.empty()) {
        return;
    }

    if (visitedRaw.count(in->tensor)) {
        return;
    }
    visitedRaw.emplace(in->tensor);

    OpCalcType calcType = OpcodeManager::Inst().GetOpCalcType(op.GetOpcode());
    size_t paddingValue = GetPaddingValue(in);
    size_t lastIdx = in->shape.size() - 1;

    in->oriShape = in->shape;
    in->tensor->oriRawshape = in->tensor->rawshape;

    auto producerOp = *(in->GetProducers().begin());

    // BRCB 算子特殊处理
    if (producerOp != nullptr && producerOp->GetOpcode() == Opcode::OP_BRCB) {
        if (lastIdx == 0 && in->tensor->rawshape[lastIdx] != 1) {
            return;
        }
        AlignedRawTensorIfNeed(in, lastIdx - 1, BRCB_SECOND_LAST_BASE);
    }

    // Reduce 操作处理
    if (calcType == OpCalcType::REDUCE) {
        ProcessReduceForAxisCombine(op, in, paddingValue);
        return;
    }

    // Broadcast 操作处理
    if (calcType == OpCalcType::BROADCAST) {
        auto dimIdx = lastIdx;
        if (Platform::Instance().GetSoc().GetNPUArch() != NPUArch::DAV_3510 &&
            axisCombineMarker.IsTensorEnableAxisCombine(in)) {
            dimIdx = ProcessBroadcastForAxisCombine(in);
        }
        AlignedRawTensorIfNeed(in, dimIdx, paddingValue);
        return;
    }

    // 其他操作处理
    if (calcType == OpCalcType::ELMWISE ||
        calcType == OpCalcType::MOVE_IN ||
        calcType == OpCalcType::MOVE_OUT ||
        op.GetOpcode() == Opcode::OP_VIEW) {

        if (op.GetOpcode() == Opcode::OP_EXPAND ||
            !axisCombineMarker.IsTensorEnableAxisCombine(in)) {
            AlignedRawTensorIfNeed(in, lastIdx, paddingValue);
            return;
        }

        if (op.GetOpcode() == Opcode::OP_INDEX_OUTCAST &&
            op.GetIOperandIndex(in) == 0) {
            AlignedRawTensorIfNeed(in, lastIdx, paddingValue);
            return;
        }

        if (lastIdx > 0 && in->tensor->rawshape[lastIdx] == 1) {
            AlignedRawTensorIfNeed(in, lastIdx - 1, paddingValue);
            return;
        }
    }

    AlignedRawTensorIfNeed(in, lastIdx, paddingValue);
}
```

---

### 4.4 CodegenPreproc Pass

**文件位置**：`framework/src/passes/block_graph_pass/codegen_preproc.cpp`

**职责**：代码生成前的预处理，实现尾轴合并

#### 4.4.1 合轴逻辑

```cpp
void CodegenPreproc::CombineTailAxis(
    std::vector<int64_t> &shape, size_t shapeSize) const {
    shape[shapeSize - 1] = shape[shapeSize - 1] * shape[shapeSize - NUM2];
    shape[shapeSize - NUM2] = 1;
}
```

**效果示例**：
```
原始 shape: [32, 128, 1]
合轴后 shape: [4096, 1]  (32*128=4096)
```

#### 4.4.2 强制合轴逻辑

```cpp
Status CodegenPreproc::ForceCombineAxisForAxisCombine(Function &func) const {
    const std::set<Opcode> skipInputCombineOps = {
        Opcode::OP_BRCB, Opcode::OP_EXPAND
    };

    for (auto &subProgram : func.rootFunc_->programs_) {
        for (auto &op : subProgram.second->Operations(false)) {
            if (OpcodeManager::Inst().GetCoreType(op.GetOpcode()) !=
                OpCoreType::AIV && !IsUBCopy(op)) {
                continue;
            }

            // 设置输入合轴属性
            std::vector<bool> inputCombineAxis;
            LogicalTensors inputs = op.GetIOperands();

            for (size_t i = 0; i < inputs.size(); ++i) {
                if (inputs[i]->tensor->rawshape.back() == 1 &&
                    skipInputCombineOps.count(op.GetOpcode()) == 0) {
                    inputCombineAxis.push_back(true);
                } else {
                    inputCombineAxis.push_back(false);
                }
            }
            op.SetAttr(OpAttributeKey::inputCombineAxis, inputCombineAxis);

            // 设置输出合轴属性
            std::vector<bool> outputCombineAxis;
            auto outputs = op.GetOOperands();

            for (size_t i = 0; i < outputs.size(); ++i) {
                if (outputs[i]->tensor->rawshape.back() == 1 &&
                    ReduceNeedCombineAxis(op)) {
                    outputCombineAxis.push_back(true);

                    // 修正 EXPANDDIM 属性
                    FixExpandDimForAxisCombine(op,
                        static_cast<int>(outputs[i]->tensor->rawshape.size()));
                } else {
                    outputCombineAxis.push_back(false);
                }
            }
            op.SetAttr(OpAttributeKey::outputCombineAxis, outputCombineAxis);
        }
    }
    return SUCCESS;
}
```

#### 4.4.3 执行尾轴合并

```cpp
Status CodegenPreproc::ProcessAxis(
    Operation &op, std::vector<bool> attr, bool isInput) const {

    LogicalTensors operands = isInput ? op.GetIOperands() : op.GetOOperands();

    for (size_t i = 0; i < operands.size(); ++i) {
        if (attr[i]) {
            size_t shapeSize = operands[i]->shape.size();
            CombineTailAxis(operands[i]->shape, shapeSize);
            CombineTailAxis(operands[i]->oriShape, shapeSize);
            CombineTailAxis(operands[i]->tensor->rawshape, shapeSize);

            if (forceCombineAxis) {
                CombineLastAxis(operands[i]->dynValidShape_, shapeSize);
            }
        }
    }
    return SUCCESS;
}
```

---

## 五、配置管理

### 5.1 配置文件

**文件位置**：`framework/src/interface/configs/tile_fwk_config.json`

```json
{
  "force_combine_axis": false,
  "combine_axis": false
}
```

### 5.2 配置读取

```cpp
// pad_local_buffer.cpp:645
combineAxis = function.paramConfigs_.combineAxis;
forceCombineAxis = function.paramConfigs_.forceCombineAxis;

// codegen_preproc.cpp:268
combineAxis = function.paramConfigs_.combineAxis;
forceCombineAxis = function.paramConfigs_.forceCombineAxis;
```

---

## 六、典型应用案例

### 6.1 Attention 机制

**场景描述**：Self-Attention 计算中，mask tensor 的形状为 `(batch, seq_len, 1)`，需要与 score tensor `(batch, seq_len, seq_len)` 进行相加。

**未启用优化**：
```python
# mask shape: (batch, seq_len, 1)
# score shape: (batch, seq_len, seq_len)
# 需要将 mask 广播到 (batch, seq_len, seq_len)
score = score + mask  # mask 会通过 EXPAND 扩展
```

**启用优化**：
```python
import pypto

pypto.experimental.set_operation_options(combine_axis=True)

@pypto.jit
def attention(q, k, v, mask):
    # mask shape: (batch, seq_len, 1)
    # score shape: (batch, seq_len, seq_len)

    scale = (q @ k.transpose(-1, -2)) / (d_k ** 0.5)
    score = scale + mask  # mask 会通过 BRCB 扩展到 (batch, seq_len, 8)

    attn = softmax(score, dim=-1)
    output = attn @ v
    return output
```

### 6.2 FFN + MoE 场景

**文件位置**：`models/glm_v4_5/glm_moe_fusion.py:144`

```python
def moe_fusion(hidden_states, router_logits, ...):
    pypto.experimental.set_operation_options(combine_axis=True)

    # Router logits shape: (batch, seq_len, num_experts)
    # Expert mask shape: (batch, seq_len, 1)
    # 通过 BRCB 扩展，避免全量广播
    expert_mask = router_logits > threshold
    expert_weights = expert_mask * hidden_states

    return expert_weights
```

### 6.3 稀疏 Attention 场景

**文件位置**：`models/deepseek_v32_exp/sparse_flash_attention_quant_impl.py:505`

```python
def sparse_flash_attention(q, k, v, mask):
    pypto.experimental.set_operation_options(combine_axis=True)

    # Sparse mask shape: (batch, seq_len, 1)
    # Attention score shape: (batch, num_heads, seq_len, seq_len)
    # BRCB 扩展节省大量 UB 空间
    attn_score = q @ k.transpose(-1, -2)
    attn_score = attn_score + mask

    return softmax(attn_score, dim=-1) @ v
```

---

## 七、测试要点

### 7.1 功能验证

| 测试项 | 验证内容 | 预期结果 |
|-------|---------|---------|
| **BRCB 插入** | 尾轴为1且支持合轴的 tensor | 正确插入 OP_BRCB |
| **EXPAND 插入** | 尾轴为1但不支持合轴的 tensor | 正确插入 OP_EXPAND |
| **尾轴合并** | 合轴属性的 tensor | shape 正确合并 |
| **Reduce 排除** | Reduce 倒数第二轴的场景 | 正确排除合轴优化 |

### 7.2 边界场景

| 场景 | 验证要点 |
|------|---------|
| **多级广播** | 多个广播操作连续的场景，状态正确传递 |
| **前序 Reduce** | 前序节点是尾轴 reduce 的场景，支持合轴 |
| **前序 COPY_IN** | 前序节点是 COPY_IN 的场景，GM 连续性要求 |
| **混合操作** | ELMWISE 和 BROADCAST 混合的场景，状态正确传播 |
| **非尾轴 Expand** | 非 尾轴 expand 的场景，状态保持 ENABLE |

### 7.3 配置验证

| 配置 | 验证内容 |
|------|---------|
| `combine_axis=False` | 功能关闭，不插入 BRCB |
| `force_combine_axis=True` | 与 `combine_axis=True` 行为一致 |
| 混合使用 | `combine_axis` 优先级高于 `force_combine_axis` |

### 7.4 性能验证

| 指标 | 验证方法 |
|------|---------|
| **内存带宽** | 对比开启/关闭前后的 GM→L1 数据量 |
| **UB 空间** | 验证尾轴合并后的 UB 缓冲区大小 |
| **计算时间** | Profile 记录整体计算时间 |
| **指令效率** | 统计 BRCB vs EXPAND 指令数量 |

---

## 八、已知约束和限制

### 8.1 功能约束

1. **连续性要求**：尾轴 broadcast 输入尾轴必须是连续的
   - 如果前序节点是尾轴 reduce，reduce 接口能够保证
   - 如果前序节点是 COPY_IN，需要在前端保证在 GM 连续

2. **算子限制**：仅支持特定双目算子
   - ADD、SUB、MUL、DIV、MAXIMUM、MINIMUM、EXPANDEXPDIF

3. **内存类型**：主要针对 UB 上的操作
   - AIV 核上的 Vector 操作
   - UB Copy 操作

4. **平台限制**：
   - 非非 DAV_3510 平台才支持完整功能
   - DAV_3510 平台会跳过合轴优化

### 8.2 性能约束

1. **尾轴约束**：只处理尾轴为 1 的张量
   - 避免不必要的分析
   - 减少编译时间

2. **Block 对齐**：必须满足 Block 粒度对齐要求
   - FP16: 16 元素对齐（32B）
   - FP32: 8 元素对齐（32B）

3. **UB 空间限制**：某些场景可能因 UB 空间不足而失效

### 8.3 开发约束

1. **Pass 执行顺序**：
   ```
   AxisCombineMarker → AxisCombine → PadLocalBuffer → CodegenPreproc
   ```
   顺序错误会导致功能失效

2. **状态一致性**：正向和反向遍历的结果应该一致
   - 否则可能导致优化决策错误

3. **属性传递**：合轴属性必须正确传递到代码生成阶段
   - `inputCombineAxis`
   - `outputCombineAxis`

---

## 九、调试指南

### 9.1 日志输出

**启用详细日志**：
```bash
export PTO_LOG_LEVEL=DEBUG
```

**关键日志**：
```
========== AxisCombineMarker Tensor Status ==========
Tensor ID: 1234, Shape: [32, 1], Status: ENABLE
Tensor ID: 5678, Shape: [32, 128], Status: UNKNOWN
======================================================
```

### 9.2 调试检查点

| 检查点 | 检查内容 |
|-------|---------|
| **AxisCombineMarker** | tensorStatus_ 是否正确标记 |
| **AxisCombine** | BRCB/EXPAND 算子是否正确插入 |
| **PadLocalBuffer** | padding 策略是否正确应用 |
| **CodegenPreproc** | 尾轴合并是否正确执行 |

### 9.3 常见问题

| 问题 | 可能原因 | 解决方法 |
|------|---------|---------|
| **BRCB 未插入** | tensor 状态为 DISABLE | 检查前序算子是否修改了尾轴 |
| **Shape 不正确** | 尾轴合并逻辑错误 | 检查 CombineTailAxis 函数 |
| **性能未提升** | 连续性不满足 | 检查 COPY_IN 的输入是否连续 |
| **编译失败** | Block 对齐不满足 | 检查 padding 值计算 |

---

## 十、总结

### 10.1 核心价值

1. **性能提升**：减少内存带宽消耗，提升计算效率
2. **空间节省**：尾轴合并后占用更少的 UB 空间
3. **透明优化**：用户只需设置开关，编译框架自动优化

### 10.2 使用建议

1. **推荐场景**：
   - 大规模 broadcast 操作（Attention、MoE、稀疏场景）
   - 尾轴为 1 的 tensor 广播
   - GM 连续的 COPY_IN 输入

2. **不推荐场景**：
   - 尾轴不为 1 的广播
   - 小规模计算（优化收益不明显）
   - DAV_3510 平台

3. **注意事项**：
   - 确保 COPY_IN 的输入在 GM 上连续
   - 避免在尾轴上进行 ASSEMBLE 操作
   - Reduce 倒数第二轴的场景会自动排除

### 10.3 未来演进

1. **force_combine_axis 下线**：早期接口逐步下线，统一使用 `combine_axis`
2. **更多算子支持**：扩展支持更多广播算子
3. **自动优化**：编译框架自动判断是否启用合轴优化

---

## 十一、相关文档

- [AxisCombineMarker Pass 详细说明](./graph_constraint/axis_combine_marker.md)
- [set_operation_options API 文档](../../api/operation/pypto-experimental-set_operation_options.md)
- [性能调优指南](../../tutorials/debug/performance.md)

---

## 十二、参考实现

- **Python API**：`python/pypto/experimental.py:135`
- **AxisCombineMarker**：`framework/src/passes/tile_graph_pass/graph_constraint/axis_combine_marker.cpp`
- **AxisCombine**：`framework/src/passes/tile_graph_pass/graph_constraint/axis_combine.cpp`
- **PadLocalBuffer**：`framework/src/passes/tile_graph_pass/graph_constraint/pad_local_buffer.cpp`
- **CodegenPreproc**：`framework/src/passes/block_graph_pass/codegen_preproc.cpp`
