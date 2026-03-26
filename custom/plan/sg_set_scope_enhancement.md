# sg_set_scope 功能增强实现计划

## 需求概述

增强 `sg_set_scope` 功能，从单一 scopeid 参数扩展为包含多个开关的复合配置，支持：
1. 允许并行分支合并
2. 允许含有 scope 的 supernode 和其他 supernode 合并
3. 预留 int 值接口 (mixId)

## 配置方式

```python
pypto.set_pass_options(sg_set_scope=(scopeid, allow_parallel_merge, allow_cross_scope_merge, mix_id))
# 示例：
# pypto.set_pass_options(sg_set_scope=(1, True, True, 123))
```

## 验证命令

- 基本编译：`python3 build_ci.py --generator Ninja`
- UT测试：`python3 build_ci.py --generator Ninja -u --utest=xxx.py`

## 实现步骤

### 步骤1：修改数据结构

#### 1.1 修改 `framework/src/interface/operation/operation.h`

**文件位置**: `framework/src/interface/operation/operation.h`

**修改内容**:
1. 在 Operation 类中添加 `ScopeInfo` 结构体定义（在 MixSubgraphFields 附近）
2. 将 `int scopeId_{-1};` (338行) 改为 `ScopeInfo scopeInfo_;`
3. 修改 `SetScopeId` 接口签名
4. 保持 `GetScopeId` 接口向后兼容
5. 添加新的 Get 接口

```cpp
// 在 MixSubgraphFields 结构体附近添加
struct ScopeInfo {
    int scopeId{-1};                          // scope ID
    bool allowParallelMerge{false};              // 开关1：允许并行分支合并
    bool allowCrossScopeMerge{false};           // 开关2：允许含有 scope 的 supernode 和其他 supernode 合并
    int mixId{-1};                           // 开关3：预留的 int 值 (mixId)

    // 向后兼容的构造函数
    ScopeInfo() = default;
    explicit ScopeInfo(int id) : scopeId(id) {}
};

// 替换原有字段
// int scopeId_{-1};  // 删除
ScopeInfo scopeInfo_;  // 新增

// 更新接口
void SetScopeId(int scopeId) {
    scopeInfo_.scopeId = scopeId;
}

// 新增接口：设置完整的 ScopeInfo
void SetScopeInfo(const ScopeInfo &info) {
    scopeInfo_ = info;
}

// 保持向后兼容
int GetScopeId() const {
    return scopeInfo_.scopeId;
}

// 新增 Get 接口
bool GetAllowParallelMerge() const {
    return scopeInfo_.allowParallelMerge;
}

bool GetAllowCrossScopeMerge() const {
    return scopeInfo_.allowCrossScopeMerge;
}

int GetMixId() const {
    return scopeInfo_.mixId;
}
```

#### 1.2 修改配置文件

**文件位置**:
- `framework/src/interface/configs/tile_fwk_config.json`
- `framework/src/interface/configs/tile_fwk_config_schema.json`

**修改内容**:
将 `sg_set_scope` 的默认值从 `-1` 改为 `[-1, false, false, -1]`（支持列表/元组）

```json
// tile_fwk_config.json
"sg_set_scope": [-1, false, false, -1],
// 格式：[scopeid, allow_parallel_merge, allow_cross_scope_merge, mix_id]
```

### 步骤2：完成 Python 调用 C++ 的 pybind 相关修改

#### 2.1 修改 `python/pypto/config.py`

**文件位置**: `python/pypto/config.py`

**修改内容**:
1. 更新 `set_pass_options` 函数中 `sg_set_scope` 参数定义
2. 支持接收 tuple/list 类型
3. 在 Python 层进行参数校验和解析

```python
def set_pass_options(*,
                     pg_skip_partition: Optional[bool] = None,
                     pg_upper_bound: Optional[int] = None,
                     vec_nbuffer_setting: Optional[Dict[int, int]] = None,
                     cube_l1_reuse_setting: Optional[Dict[int, int]] = None,
                     cube_nbuffer_setting: Optional[Dict[int, int]] = None,
                     sg_set_scope: Optional[Union[int, tuple]] = None,
                     ) -> None:
    """
    Set pass options.

    Parameters
    ---------
    ...

    sg_set_scope : Union[int, tuple]
        Merged graph parameter, used to manually control graph merging.
        - If int: only set scopeid (backward compatible)
        - If tuple: (scopeid, allow_parallel_merge, allow_cross_scope_merge, mix_id)
          * scopeid: int, scope ID
          * allow_parallel_merge: bool, enable parallel branch merging
          * allow_cross_scope_merge: bool, allow supernode with scope to merge with others
          * mix_id: int, reserved value for future use
    """
    # 处理 sg_set_scope 参数
    if sg_set_scope is not None:
        if isinstance(sg_set_scope, int):
            # 向后兼容：仅设置 scopeid
            processed_sg_set_scope = [sg_set_scope, False, False, -1]
        elif isinstance(sg_set_scope, (tuple, list)):
            # 新格式：解析元组
            if len(sg_set_scope) != 4:
                raise ValueError(f"sg_set_scope must be a tuple of 4 elements, got {len(sg_set_scope)}")
            if not isinstance(sg_set_scope[0], int):
                raise ValueError(f"sg_set_scope[0] (scopeid) must be int, got {type(sg_set_scope[0])}")
            if not isinstance(sg_set_scope[1], bool):
                raise ValueError(f"sg_set_scope[1] (allow_parallel_merge) must be bool, got {type(sg_set_scope[1])}")
            if not isinstance(sg_set_scope[2], bool):
                raise ValueError(f"sg_set_scope[2] (allow_cross_scope_merge) must be bool, got {type(sg_set_scope[2])}")
            if not isinstance(sg_set_scope[3], int):
                raise ValueError(f"sg_set_scope[3] (mix_id) must be int, got {type(sg_set_scope[3])}")
            processed_sg_set_scope = list(sg_set_scope)
        else:
            raise TypeError(f"sg_set_scope must be int or tuple, got {type(sg_set_scope)}")

        # 将处理后的值放入 options_dict
        locals_dict = {k: v for k, v in locals().items() if v is not None and k != 'sg_set_scope'}
        locals_dict['sg_set_scope'] = processed_sg_set_scope
    else:
        locals_dict = {k: v for k, v in locals().items() if v is not None}

    # 调用 set_options
    options_dict = {k: v for k, v in locals_dict.items() if v is not None}
    set_options(pass_options=options_dict)
```

**注意**: 由于 ConfigManager 不直接支持 list 类型传给 `SetScopeId`，我们需要在 C++ 层面进行适配。

#### 2.2 修改 `framework/src/interface/function/function.cpp`

**文件位置**: `framework/src/interface/function/function.cpp`

**修改内容**:
更新 1492 行的 `SetScopeId` 调用，解析配置并构造 ScopeInfo

```cpp
Operation &Function::AddRawOperation(const Opcode opCode, const LogicalTensors &iOperands,
                                     const LogicalTensors &oOperands, bool updateTensorMap) {
    // ... 原有代码 ...
    auto &op =
        operations_.emplace_back(std::make_shared<Operation>(*this, opCode, iOperands, oOperands, updateTensorMap));
    opPosition_.emplace(op.get(), operations_.size() - 1);

    // 修改 sg_set_scope 读取逻辑
    auto sgSetScope = config::GetPassOption<std::vector<int64_t>>(SG_SET_SCOPE);
    if (sgSetScope.size() == 4) {
        // 新格式：[scopeid, allow_parallel_merge, allow_cross_scope_merge, mix_id]
        ScopeInfo info;
        info.scopeId = static_cast<int>(sgSetScope[0]);
        info.allowParallelMerge = sgSetScope[1] != 0;
        info.allowCrossScopeMerge = sgSetScope[2] != 0;
        info.mixId = static_cast<int>(sgSetScope[3]);
        operations_.back()->SetScopeInfo(info);
    } else if (sgSetScope.size() == 1) {
        // 向后兼容：仅设置 scopeid
        operations_.back()->SetScopeId(static_cast<int>(sgSetScope[0]));
    } else {
        // 无效格式，使用默认值
        operations_.back()->SetScopeId(-1);
    }

    return *operations_.back();
}
```

### 步骤3：修改 pass 代码，完成上述适配

#### 3.1 修改 `framework/src/passes/tile_graph_pass/graph_partition/supernode_graph_builder.cpp`

**文件位置**: `framework/src/passes/tile_graph_pass/graph_partition/supernode_graph_builder.cpp`

**修改点 1**: `UpdateScopeId` 函数 (608-625行)
- 需要读取和传播 ScopeInfo 中的开关1

```cpp
inline void UpdateScopeId(std::vector<Operation*> &opList) {
    for (size_t i = 0; i < opList.size(); i++) {
        int targetScope = opList[i]->GetScopeId();
        if (targetScope == DEFAULT_SCOPE_ID) {
            continue;
        }
        bool allowParallelMerge = opList[i]->GetAllowParallelMerge();

        // 如果允许并行合并，不仅处理直接连接的 consumer，还需要处理并行分支
        if (allowParallelMerge) {
            // TODO: 实现并行分支合并逻辑
            // 这需要找到所有具有相同 scopeId 的 op，即使它们没有直接连接
        } else {
            // 原有逻辑：只处理直接连接的 consumer 和 producer
            for (auto &consumer : opList[i]->ConsumerOps()) {
                if (consumer->GetScopeId() == -1 && consumer->GetOpcode() == Opcode::OP_ASSEMBLE) {
                    consumer->SetScopeInfo(opList[i]->scopeInfo_);
                }
            }
            for (auto &producer : opList[i]->ProducerOps()) {
                if (producer->GetScopeId() == -1 && producer->GetOpcode() == Opcode::OP_VIEW) {
                    producer->SetScopeInfo(opList[i]->scopeInfo_);
                }
            }
        }
    }
}
```

**修改点 2**: `BuildSuperNodeGraph` 函数 (627-677行)
- 开关2 需要在此处应用，影响 supernode 的合并行为
- 当前在 641-645 行只合并直接连接的相同 scopeId op

```cpp
Status SuperNodeGraphBuilder::BuildSuperNodeGraph()
{
    std::vector<Operation*> &opList = operationInfo_->opList_;
    if (opList.size() != operationInfo_->inGraph_.size() || opList.size() != operationInfo_->outGraph_.size()) {
        APASS_LOG_ERROR_F(Elements::Function, "Operation inGraph and outGraph have not been initialized.");
        return FAILED;
    }
    std::vector<std::pair<int32_t, int32_t>> mergePair;
    UpdateScopeId(opList);

    for (size_t i = 0; i < opList.size(); i++) {
        auto targetScope = opList[i]->GetScopeId();
        if (targetScope == -1) {
            continue;
        }
        bool allowParallelMerge = opList[i]->GetAllowParallelMerge();

        if (allowParallelMerge) {
            // 开关1：允许并行分支合并
            // 收集所有具有相同 scopeId 的 op
            std::vector<int32_t> sameScopeOps;
            for (size_t j = 0; j < opList.size(); j++) {
                if (opList[j]->GetScopeId() == targetScope) {
                    sameScopeOps.push_back(j);
                }
            }

            // 将所有相同 scopeId 的 op 合并到一个 supernode
            if (sameScopeOps.size() > 1) {
                for (size_t k = 1; k < sameScopeOps.size(); k++) {
                    mergePair.emplace_back(sameScopeOps[k], sameScopeOps[0]);
                    APASS_LOG_DEBUG_F(Elements::Operation, "Merge parallel ops %d and %d (scope=%d) due to allowParallelMerge",
                        opList[sameScopeOps[k]]->GetOpMagic(),
                        opList[sameScopeOps[0]]->GetOpMagic(), targetScope);
                }
            }
        } else {
            // 原有逻辑：只合并直接连接的 op
            for (auto outputNode : operationInfo_->outGraph_[i]) {
                if (opList[outputNode]->GetScopeId() == targetScope) {
                    mergePair.emplace_back(outputNode, i);
                }
            }
        }
    }

    // ... 原有的其他合并逻辑 (ConvertCombine, L1CopyInCombine 等) ...
    for (size_t i = 0; i < opList.size(); i++) {
        if (ConvertCombine(operationInfo_, opList, i, mergePair)) {
            continue;
        }
        // ... 其他合并逻辑 ...
    }

    superNodeInfo_ = std::make_shared<NodeGraphInfo>();
    if (superNodeInfo_ == nullptr) {
        APASS_LOG_ERROR_F(Elements::Function, "Create SuperNodeInfo failed.");
        return FAILED;
    }
    if (superNodeInfo_->Build(operationInfo_, mergePair, !useCVMixPartition_) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Function, "Build SuperNodeInfo Failed.");
        return FAILED;
    }
    return SUCCESS;
}
```

#### 3.2 修改 `framework/src/passes/tile_graph_pass/graph_partition/iso_partitioner.cpp`

**文件位置**: `framework/src/passes/tile_graph_pass/graph_partition/iso_partitioner.cpp`

**修改点**: `SuitableForMergeCheck` 函数 (491-535行)
- 开关2 需要在此处应用
- 当前 494-501 行不允许含有 scopeId 的 supernode 合并

```cpp
bool IsoPartitioner::SuitableForMergeCheck(int32_t currColor, int32_t mergeColor, bool nonIsoGraphsMerge) const
{
    // ... 原有代码 ...

    // 检查开关2：是否允许跨 scope 合并
    bool currHasScope = false;
    bool mergeHasScope = false;
    bool currAllowCrossMerge = false;
    bool mergeAllowCrossMerge = false;

    for (auto graphPtr : isoSubGroups_[currColor]->isoGraphs_) {
        if (graphPtr->scopeId_ != -1) {
            currHasScope = true;
            // 获取当前 op 的 scopeInfo（需要扩展 SubGraph 类存储 scopeInfo）
            currAllowCrossMerge = true; // 默认允许，实际需要从 Operation 获取
        }
    }
    for (auto graphPtr : isoSubGroups_[mergeColor]->isoGraphs_) {
        if (graphPtr->scopeId_ != -1) {
            mergeHasScope = true;
            mergeAllowCrossMerge = true; // 默认允许，实际需要从 Operation 获取
        }
    }

    // 如果两个 supernode 都有 scopeId，且至少有一个不允许跨 scope 合并，则不允许合并
    if (currHasScope && mergeHasScope) {
        if (!currAllowCrossMerge || !mergeAllowCrossMerge) {
            APASS_LOG_INFO_F(Elements::Operation, "Cannot merge supernodes with scopeId (currAllowCrossMerge=%d, mergeAllowCrossMerge=%d)",
                currAllowCrossMerge, mergeAllowCrossMerge);
            return false;
        }
    }

    // ... 原有代码 ...
    bool coreTypeMergable = operationInfo_->CoreTypeMergeable(opcoreTypes);
    int32_t latencyMerged = 0;
    int32_t currColorSize = static_cast<int32_t>(isoSubGroups_[currColor]->Size());
    int32_t mergeColorSize = static_cast<int32_t>(isoSubGroups_[mergeColor]->Size());
    if (currColorSize == 0 || mergeColorSize == 0) {
        return false;
    }
    latencyMerged = (currColorSize <= mergeColorSize) ?
                        isoSubGroups_[currColor]->GetLatency() +
                            isoSubGroups_[mergeColor]->GetLatency() * (mergeColorSize / currColorSize) :
                        isoSubGroups_[currColor]->GetLatency() * (currColorSize / mergeColorSize) +
                            isoSubGroups_[mergeColor]->GetLatency();
    bool cycleMergable = latencyMerged <= cycleUB_;
    if (nonIsoGraphsMerge) {
        bool shouldMerge = coreTypeMergable && cycleMergable;
        APASS_LOG_DEBUG_F(Elements::Operation, "Try merge current group: %d [%s]\n\t with: %d [%s], is suitable for merge: %d.",
                     currColor, isoSubGroups_[currColor]->GetSubGraph(0)->DumpStr().c_str(),
                     mergeColor, isoSubGroups_[mergeColor]->GetSubGraph(0)->DumpStr().c_str(), shouldMerge);
        return shouldMerge;
    }
    bool isSuitableForMerge = (currColorSize == mergeColorSize);
    isSuitableForMerge = isSuitableForMerge || (std::min(currColorSize, mergeColorSize) >= parallelNum_);
    isSuitableForMerge = isSuitableForMerge ||
                         (std::min(isoSubGroups_[currColor]->GetLatency(), isoSubGroups_[mergeColor]->GetLatency()) <=
                          cycleLB_);
    isSuitableForMerge = coreTypeMergable && isSuitableForMerge && cycleMergable;
    APASS_LOG_DEBUG_F(Elements::Operation, "Try merge current group: %d [%s]\n\t with: %d [%s], is suitable for merge: %d.",
                 currColor, isoSubGroups_[currColor]->GetSubGraph(0)->DumpStr().c_str(),
                 mergeColor, isoSubGroups_[mergeColor]->GetSubGraph(0)->DumpStr().c_str(), isSuitableForMerge);
    return isSuitableForMerge;
}
```

#### 3.3 修改 `framework/src/passes/tile_graph_pass/graph_partition/iso_partitioner.h`

**文件位置**: `framework/src/passes/tile_graph_pass/graph_partition/iso_partitioner.h`

**修改内容**:
在 `SubGraph` 类中添加 `allowCrossScopeMerge` 字段

```cpp
class SubGraph {
public:
    SubGraph(std::shared_ptr<OperationGraphInfo> operationInfo, std::shared_ptr<NodeGraphInfo> superNodeInfo)
        : operationInfo_(operationInfo), superNodeInfo_(superNodeInfo)
        {}
    int32_t GetExpandCandidate(size_t expandNodeIdx, size_t expandLinkIdx, GraphExtendResult &res);
    void AddNode(int32_t nodeIdx);
    void Merge(SubGraph &sg);
    bool HasNode(int32_t nodeIdx) const;
    void BuildInOutSet();
    int32_t GetLatency() const;
    void Clear();
    std::string DumpStr();
    const std::vector<int32_t> &GetNodeList();
    std::vector<Operation*> GetOpList();

    // 新增接口：设置 allowCrossScopeMerge
    void SetAllowCrossScopeMerge(bool allow) {
        allowCrossScopeMerge_ = allow;
    }

    // 新增接口：获取 allowCrossScopeMerge
    bool GetAllowCrossScopeMerge() const {
        return allowCrossScopeMerge_;
    }

    std::shared_ptr<OperationGraphInfo> operationInfo_;
    std::shared_ptr<NodeGraphInfo> superNodeInfo_;
    std::vector<int32_t> nodeList_;
    std::unordered_set<int32_t> nodeSet_;
    std::unordered_set<int32_t> inNodes_;
    std::unordered_set<int32_t> outNodes_;
    std::set<std::pair<int32_t, int32_t>> mergeHistoryIsoSub_;
    int32_t cycle_{0};
    OpCoreType coreType_{OpCoreType::ANY};
    bool mergeable_{true};
    int32_t scopeId_{-1};
    bool allowCrossScopeMerge_{false};  // 新增字段
};
```

#### 3.4 修改 `BuildGraphGroup` 函数

**文件位置**: `framework/src/passes/tile_graph_pass/graph_partition/iso_partitioner.cpp`

**修改点**: 在 `BuildGraphGroup` 函数 (156-183行) 中，从第一个 op 获取 `allowCrossScopeMerge` 值

```cpp
Status IsomorphismGraphGroup::BuildGraphGroup(std::shared_ptr<OperationGraphInfo> operationInfo,
                                              std::shared_ptr<NodeGraphInfo> superNodeInfo,
                                              std::vector<int32_t> &expandCandidate,
                                              std::unordered_set<int32_t> &currentNodeSet,
                                              std::vector<int32_t> &idxInLinkNum, std::deque<int32_t> &zeroInQueue)
{
    operationInfo_ = operationInfo;
    superNodeInfo_ = superNodeInfo;
    subVisitedNodeSet_.clear();
    subVisitedNodeSet_.insert(expandCandidate.begin(), expandCandidate.end());
    currentNodeSet.insert(expandCandidate.begin(), expandCandidate.end());

    // 获取 allowCrossScopeMerge 值（从第一个 op）
    bool allowCrossScopeMerge = false;
    if (!expandCandidate.empty()) {
        int32_t firstNodeIdx = expandCandidate[0];
        // 从第一个节点对应的 operation 获取 allowCrossScopeMerge
        for (int32_t opIdx : superNodeInfo->node2Op_[firstNodeIdx]) {
            allowCrossScopeMerge = operationInfo->opList_[opIdx]->GetAllowCrossScopeMerge();
            if (allowCrossScopeMerge) {
                break;
            }
        }
    }

    for (int32_t nodeIdx : expandCandidate) {
        if (InLinkCountDelete(nodeIdx, idxInLinkNum, zeroInQueue) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Function, "In-link count delete failed.");
            return FAILED;
        }
        std::shared_ptr<SubGraph> sgPtr = std::make_shared<SubGraph>(operationInfo, superNodeInfo);
        if (sgPtr == nullptr) {
            APASS_LOG_ERROR_F(Elements::Function, "Create SubGraph failed.");
            return FAILED;
        }
        sgPtr->AddNode(nodeIdx);
        sgPtr->scopeId_ = superNodeInfo->nodeScope_[nodeIdx];
        sgPtr->SetAllowCrossScopeMerge(allowCrossScopeMerge);  // 新增：设置跨 scope 合并开关
        isoGraphs_.push_back(sgPtr);
    }
    mergeable_ = superNodeInfo_->nodeMergeable_[expandCandidate[0]];
    return SUCCESS;
}
```

#### 3.5 更新 `SuitableForMergeCheck` 函数

**文件位置**: `framework/src/passes/tile_graph_pass/graph_partition/iso_partitioner.cpp`

**修改点**: 使用 `GetAllowCrossScopeMerge()` 接口获取配置

```cpp
bool IsoPartitioner::SuitableForMergeCheck(int32_t currColor, int32_t mergeColor, bool nonIsoGraphsMerge) const
{
    // ... 原有代码 ...

    // 检查开关2：是否允许跨 scope 合并
    bool currHasScope = false;
    bool mergeHasScope = false;
    bool currAllowCrossMerge = false;
    bool mergeAllowCrossMerge = false;

    for (auto graphPtr : isoSubGroups_[currColor]->isoGraphs_) {
        if (graphPtr->scopeId_ != -1) {
            currHasScope = true;
            // 使用新的 GetAllowCrossScopeMerge 接口
            currAllowCrossMerge = graphPtr->GetAllowCrossScopeMerge();
        }
    }
    for (auto graphPtr : isoSubGroups_[mergeColor]->isoGraphs_) {
        if (graphPtr->scopeId_ != -1) {
            mergeHasScope = true;
            // 使用新的 GetAllowCrossScopeMerge 接口
            mergeAllowCrossMerge = graphPtr->GetAllowCrossScopeMerge();
        }
    }

    // 如果两个 supernode 都有 scopeId，且至少有一个不允许跨 scope 合并，则不允许合并
    if (currHasScope && mergeHasScope) {
        if (!currAllowCrossMerge || !mergeAllowCrossMerge) {
            APASS_LOG_INFO_F(Elements::Operation, "Cannot merge supernodes with scopeId (currAllowCrossMerge=%d, mergeAllowCrossMerge=%d)",
                currAllowCrossMerge, mergeAllowCrossMerge);
            return false;
        }
    }

    // ... 原有代码 ...
}
```

### 步骤4：添加新的配置常量

在 `framework/src/interface/configs/config_manager_ng.h` 中，可能需要添加新的配置项：

```cpp
constexpr const char *SG_SET_SCOPE_PARALLEL_MERGE = "sg_set_scope_parallel_merge";
constexpr const char *SG_SET_SCOPE_CROSS_MERGE = "sg_set_scope_cross_merge";
constexpr const char *SG_SET_SCOPE_RESERVED = "sg_set_scope_reserved";
```

### 步骤5：添加 Python 测试用例

在 `python/tests/ut/interface/test_config_options.py` 中添加新的测试用例：

```python
def test_sg_set_scope_new_format():
    # 测试新格式：tuple
    pypto.set_pass_options(sg_set_scope=(1, True, True, 123))
    pass_option = pypto.get_pass_options()
    assert pass_option["sg_set_scope"] == [1, True, True, 123]

    # 测试向后兼容：int
    pypto.set_pass_options(sg_set_scope=48)
    pass_option = pypto.get_pass_options()
    assert pass_option["sg_set_scope"] == [48, False, False, -1]

    # 测试默认值
    pypto.reset_options()
    pass_option = pypto.get_pass_options()
    assert pass_option["sg_set_scope"] == [-1, False, False, -1]

    # 测试参数校验
    try:
        pypto.set_pass_options(sg_set_scope=(1, True))  # 元素不足
        assert False, "Should raise ValueError"
    except ValueError as e:
        assert "tuple of 4 elements" in str(e)

    try:
        pypto.set_pass_scope(sg_set_scope=(1, "True", True, 123))  # 类型错误
        assert False, "Should raise ValueError"
    except ValueError as e:
        assert "must be bool" in str(e)
```

### 步骤6：添加 C++ 单元测试

**测试文件位置**: `framework/tests/ut/passes/src/test_graph_partition.cpp`

#### 测试设计思路

测试目标：验证两个新开关的功能正确性

**测试框架**：
- 使用 `ComputationalGraphBuilder` 构建测试图
- 使用 `IsoPartitioner` 进行分区
- 验证子图数量和操作分布

#### 测试用例设计

**用例1：测试开关1（allowParallelMerge=True）**

测试场景：
- 创建4个并行分支的操作，每个操作有相同的 scopeId=1
- 操作之间没有直接连接关系
- 设置 allowParallelMerge=True

预期结果：
- 所有操作应该合并到同一个 supernode
- 子图数量为 1

图结构示意：
```
输入1 → 操作1(scopeId=1) → 输出1
输入2 → 操作2(scopeId=1) → 输出2
输入3 → 操作3(scopeId=1) → 输出3
输入4 → 操作4(scopeId=1) → 输出4
```

**用例2：测试开关1（allowParallelMerge=False）**

测试场景：
- 相同的4个并行分支操作，scopeId=1
- 设置 allowParallelMerge=False（保持原有行为）

预期结果：
- 操作不合并，保持独立
- 子图数量为 4

**用例3：测试开关2（allowCrossScopeMerge=True）**

测试场景：
- 创建两个 supernode，每个内部有相同 scopeId 的操作
- supernode 之间有数据依赖
- 两个 supernode 都设置 allowCrossScopeMerge=True

预期结果：
- 两个 supernode 应该合并
- 子图数量为 1

图结构示意：
```
[Supernode A: scopeId=1, allowCrossScopeMerge=True]
  操作1 → 中间结果
  操作2 → 中间结果

[Supernode B: scopeId=2, allowCrossScopeMerge=True]
  中间结果 → 操作3 → 输出
  中间结果 → 操作4 → 输出
```

**用例4：测试开关2（allowCrossScopeMerge=False）**

测试场景：
- 相同的两个 supernode 结构
- 两个 supernode 都设置 allowCrossScopeMerge=False

预期结果：
- 两个 supernode 不合并
- 子图数量为 2

**用例5：两个开关组合测试**

测试场景：
- 创建多个并行分支（测试开关1）
- 同时有跨 supernode 的依赖关系（测试开关2）
- 组合不同的开关值

预期结果：
- 根据开关组合验证分区结果
- 验证两种开关的交互行为

#### 测试覆盖矩阵

| 测试用例 | 测试开关 | 预期子图数 | 验证点 |
|---------|---------|-----------|-------|
| 用例1 | 开关1=True | 1 | 并行分支合并 |
| 用例2 | 开关1=False | 4 | 并行分支不合并 |
| 用例3 | 开关2=True | 1 | 跨 supernode 合并 |
| 用例4 | 开关2=False | 2 | 跨 supernode 不合并 |
| 用例5 | 两开关组合 | 2 | 组合行为 |

#### 测试实现要点

1. **设置 ScopeInfo**：
   - 使用 `SetScopeInfo()` 方法设置完整的 ScopeInfo 对象
   - 或分别设置 scopeId 和开关属性

2. **验证子图数量**：
   - 使用 `function->GetTotalSubGraphCount()` 获取子图总数
   - 验证是否符合预期

3. **验证操作分布**：
   - 遍历所有操作，检查其 `GetSubgraphID()`
   - 验证相关操作是否在同一子图中

4. **参数配置**：
   - 设置合适的 cycleUB、parallelTH、cycleLB
   - 确保不会因为其他约束影响测试结果

## 实现顺序

1. **数据结构修改** (步骤1)
    - 修改 `operation.h`
    - 修改配置 JSON 文件

2. **Python 接口修改** (步骤2)
    - 修改 `config.py`
    - 修改 `function.cpp`

3. **Pass 逻辑修改** (步骤3)
    - 修改 `supernode_graph_builder.cpp`
    - 修改 `iso_partitioner.cpp`

4. **测试验证**
    - 步骤5：添加 Python 测试用例（`test_config_options.py`）
    - 步骤6：添加 C++ 单元测试（`test_graph_partition.cpp`）
    - 运行测试验证功能

## 注意事项

1. **向后兼容性**:
   - 保留 `GetScopeId()` 接口
   - 保留 `SetScopeId(int)` 接口
   - 支持 `sg_set_scope=48` 的旧格式

2. **类型转换**:
   - Python 的 tuple/list 传递到 C++ 需要使用 `std::vector<int64_t>`
   - bool 值需要转换为 int (0/1)

3. **SubGraph 类扩展**:
    - 已在 `iso_partitioner.h` 中添加 `allowCrossScopeMerge_` 字段
    - 已添加 `SetAllowCrossScopeMerge()` 和 `GetAllowCrossScopeMerge()` 接口

4. **日志输出**:
   - 添加适当的日志输出，便于调试
   - 在关键决策点输出配置信息

5. **默认值**:
    - 开关1 默认为 False（保持原有行为）
    - 开关2 默认为 False（保持原有行为）
    - mixId 默认为 -1

6. **测试要点**:
    - 测试应该覆盖开关开启和关闭两种情况
    - 测试应该验证向后兼容性（原有 scopeId 功能）
    - 测试应该覆盖边界情况（空图、单节点等）
    - 测试用例应该足够独立，便于调试

## 待确认问题

1. **Python 接口实现**:
    - 需要确认 Python 到 C++ 的参数传递机制
    - ConfigManager 是否支持列表类型传给 C++
    - 可能需要在 C++ 层面进行适配

2. **测试场景细化**:
    - 需要根据实际测试结果调整预期子图数量
    - 可能需要调整参数配置（cycleUB、parallelTH 等）

3. **向后兼容性验证**:
    - 需要确保旧版本的 `sg_set_scope=48` 格式仍然正常工作
    - 需要验证默认值（-1, false, false, -1）的行为

## 验证说明

### 编译验证

在完成代码修改后，执行以下命令进行编译验证：

```bash
python3 build_ci.py --generator Ninja
```

该命令会：
1. 执行 CMake 配置
2. 执行 CMake 编译
3. 生成 whl 包

### Python 单元测试验证

在编译成功后，执行以下命令运行 Python 单元测试：

```bash
python3 build_ci.py --generator Ninja -u --utest=test_config_options.py
```

该命令会：
1. 执行 CMake 编译（如果需要）
2. 运行指定的 Python UT 测试文件
3. 验证配置解析和接口调用是否正确

### C++ 单元测试验证

在编译成功后，执行以下命令运行 C++ 单元测试：

```bash
# 测试开关1和开关2的功能
python3 build_ci.py -f=cpp --generator Ninja -u=GraphPartitionTest.TestAllowParallelMerge
python3 build_ci.py -f=cpp --generator Ninja -u=GraphPartitionTest.TestAllowCrossScopeMerge
python3 build_ci.py -f=cpp --generator Ninja -u=GraphPartitionTest.TestCombinedScopeSwitches

# 测试原有的 scopeId 功能（向后兼容性）
python3 build_ci.py -f=cpp --generator Ninja -u=GraphPartitionTest.TestScopeId
```

该命令会：
1. 执行 C++ 编译（如果需要）
2. 运行指定的 C++ UT 测试用例
3. 验证开关功能的正确性

### 完整验证流程

推荐的验证顺序：

1. **编译验证**：`python3 build_ci.py --generator Ninja`
2. **Python UT测试**：`python3 build_ci.py --generator Ninja -u --utest=test_config_options.py`
3. **C++ UT测试**：运行新增的测试用例
4. **功能验证**：验证 sg_set_scope 的各个开关是否生效
5. **gdb调试**（如需要）：`python3 build_ci.py -f=cpp --generator Ninja -u=GraphPartitionTest.TestAllowParallelMerge --disable_auto_execute`

### 测试文档

详细的测试设计思路和测试用例说明，请参考：
- 测试设计文档：`custom/plan/sg_set_scope_test_design.md`
