# 代码检视报告
**项目名称**：NPU算子检视报告
**检视模块**：/mnt/workspace/gitCode/cann/mce/ops-transformer/attention/sparse_flash_attention/op_host/op_api/aclnn_sparse_flash_attention.cpp
**检视人**：Turing Team
**检视日期**：2026-03-20


## 检视概览
| 统计项 | 数值 |
| ---- | ---- |
| 发现问题总数 | 6 个 |
| 严重级（CRITICAL）问题 | 2 个 |
| 中等级（HIGH）问题 | 1 个 |
| 轻微级（MEDIUM）问题 | 2 个 |
| 低级（LOW）问题 | 1 个 |

**核心结论**：代码存在1处严重的Use-After-Free内存安全问题（TensorHolder析构释放tensor后指针仍被传递给内部函数使用），1处必需输入指针未做空指针校验，需优先修复；另有1处资源申请未检查、1处错误码返回不一致、2处死代码逻辑异常、1处参数范围校验缺失。

---

## 问题详情及修改建议

### 问题ID：ISSUE-001 | 严重级别：CRITICAL（严重）

#### 假设检验过程
**代码段**：`aclnnSparseFlashAttentionGetWorkspaceSize` 函数中 TensorHolder 的使用（第114~133行）
**假设**：H0: 该代码段的内存使用是安全的

| 证据序号 | 证据类型 | 规范ID | 证据描述 | 分值增量 | 累计自信值 |
|---------|---------|--------|---------|---------|-----------|
| 1 | 红线规范违反 | SEC-2.5 | TensorHolder构造函数创建tensor并赋值给softmaxMax/softmaxSum，析构函数调用aclDestroyTensor销毁该tensor。但TensorHolder生命周期在if块内结束，而softmaxMax/softmaxSum在return语句中仍被传递给aclnnInnerSparseFlashAttentionGetWorkspaceSize，形成Use-After-Free | +50% | 50% |
| 2 | 上下文确认 | SEC-2.12 | softmaxMaxHolder和softmaxSumHolder的析构在第127行if块结束时触发，而第128行的return语句使用了已销毁的指针，资源释放与使用不匹配 | +30% | 80% |

**结论**：自信值 **80%** > 60%，**推翻原假设H0**，该代码段存在Use-After-Free风险。

---

**关联红线条款**：SEC-2.5（指向资源句柄的变量在资源释放后应立即赋予新值）、SEC-2.12（资源泄露/释放与使用不匹配）
**代码路径**：aclnn_sparse_flash_attention.cpp:114-133
**问题类型**：Use-After-Free（释放后使用）
**问题描述**：当 `returnSoftmaxLse` 为 false 且 `softmaxMax`/`softmaxSum` 均为 nullptr 时，`TensorHolder` 在 `if` 块内创建 placeholder tensor 并通过引用赋值给 `softmaxMax`/`softmaxSum`。但 `TensorHolder` 的析构函数会在 `if` 块结束时调用 `aclDestroyTensor` 销毁该 tensor，而 `softmaxMax`/`softmaxSum` 仍指向已销毁的对象。随后这些悬空指针被传递给 `aclnnInnerSparseFlashAttentionGetWorkspaceSize`，构成 Use-After-Free，违反资源释放后应立即置空的规范。

#### 修改建议
**修改前代码**：
```cpp
} else {
    if (softmaxMax == nullptr && softmaxSum == nullptr) {
        auto softmaxMaxHolder = TensorHolder(softmaxMax, aclDataType::ACL_FLOAT, std::string("softmaxMax"));
        auto softmaxSumHolder = TensorHolder(softmaxSum, aclDataType::ACL_FLOAT, std::string("softmaxSum"));
        if (softmaxMax == nullptr) {
            OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "Failed to create the holder of tensor softmaxMax!");
            return ge::GRAPH_FAILED;
        }
        if (softmaxSum == nullptr) {
            OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "Failed to create the holder of tensor softmaxSum!");
            return ge::GRAPH_FAILED;
        }
    }
}
return aclnnInnerSparseFlashAttentionGetWorkspaceSize(
    ..., softmaxMax, softmaxSum, ...);
```
**修改后代码**：
```cpp
} else {
    if (softmaxMax == nullptr && softmaxSum == nullptr) {
        std::vector<int64_t> shape = {0};
        int64_t addr = 0xff;
        softmaxMax = aclCreateTensor(shape.data(), shape.size(),
            aclDataType::ACL_FLOAT, shape.data(), 0, ACL_FORMAT_ND,
            shape.data(), shape.size(), static_cast<void *>(&addr));
        softmaxSum = aclCreateTensor(shape.data(), shape.size(),
            aclDataType::ACL_FLOAT, shape.data(), 0, ACL_FORMAT_ND,
            shape.data(), shape.size(), static_cast<void *>(&addr));
        if (softmaxMax == nullptr) {
            OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "Failed to create placeholder tensor softmaxMax!");
            return ge::GRAPH_FAILED;
        }
        if (softmaxSum == nullptr) {
            OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "Failed to create placeholder tensor softmaxSum!");
            return ge::GRAPH_FAILED;
        }
    }
}
aclnnStatus ret = aclnnInnerSparseFlashAttentionGetWorkspaceSize(
    ..., softmaxMax, softmaxSum, ...);
// 释放临时创建的placeholder tensor
if (!returnSoftmaxLse && softmaxMax != nullptr) { aclDestroyTensor(softmaxMax); }
if (!returnSoftmaxLse && softmaxSum != nullptr) { aclDestroyTensor(softmaxSum); }
return ret;
```
**修改说明**：移除 TensorHolder 的 RAII 模式，改为在函数作用域内直接创建 placeholder tensor。在使用完毕后（inner函数调用返回后）再显式释放，确保 tensor 在使用期间始终有效，符合资源释放后立即赋新值（或确保不再使用）的规范。

---

### 问题ID：ISSUE-002 | 严重级别：CRITICAL（严重）

#### 假设检验过程
**代码段**：`aclnnSparseFlashAttentionGetWorkspaceSize` 函数入口（第84~133行）
**假设**：H0: 该代码段对必需的外部输入指针进行了充分的空指针校验

| 证据序号 | 证据类型 | 规范ID | 证据描述 | 分值增量 | 累计自信值 |
|---------|---------|--------|---------|---------|-----------|
| 1 | 红线规范违反 | SEC-2.11 | 函数对softmaxMax/softmaxSum做了空指针校验（第109~126行），但query、key、value、sparseIndices、attentionOut、workspaceSize、executor等必需输入指针完全未做空指针校验，直接传递给inner函数 | +45% | 45% |
| 2 | 上下文防御缺失 | SEC-2.8 | 作为对外暴露的aclnn API函数，缺少对必需输入的空指针防御，如果外部调用者传入nullptr将导致inner函数内部崩溃 | +25% | 70% |

**结论**：自信值 **70%** > 60%，**推翻原假设H0**，该代码段缺少必需输入的空指针校验。

---

**关联红线条款**：SEC-2.8（指针使用前必须判空）、SEC-2.11（外部输入数据需要做合法性校验）
**代码路径**：aclnn_sparse_flash_attention.cpp:84-133
**问题类型**：空指针未保护
**问题描述**：函数 `aclnnSparseFlashAttentionGetWorkspaceSize` 作为对外暴露的 API 入口，仅对 `softmaxMax`/`softmaxSum` 做了空指针校验，但对 `query`、`key`、`value`、`sparseIndices`、`attentionOut`、`workspaceSize`（uint64_t*）、`executor`（aclOpExecutor**）等必需输入/输出参数均未做空指针校验，直接传递给 `aclnnInnerSparseFlashAttentionGetWorkspaceSize`，违反"外部传入指针需要判空后使用"和"外部输入数据需要做合法性校验"规范。

#### 修改建议
**修改前代码**：
```cpp
aclnnStatus aclnnSparseFlashAttentionGetWorkspaceSize(
    const aclTensor *query, const aclTensor *key, const aclTensor *value,
    const aclTensor *sparseIndices, ... , uint64_t *workspaceSize, aclOpExecutor **executor)
{
    if (returnSoftmaxLse) { ... }
    return aclnnInnerSparseFlashAttentionGetWorkspaceSize(...);
}
```
**修改后代码**：
```cpp
aclnnStatus aclnnSparseFlashAttentionGetWorkspaceSize(
    const aclTensor *query, const aclTensor *key, const aclTensor *value,
    const aclTensor *sparseIndices, ... , uint64_t *workspaceSize, aclOpExecutor **executor)
{
    // 校验必需输入指针
    CHECK_NULL_AND_RETURN(query, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_NULL_AND_RETURN(key, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_NULL_AND_RETURN(value, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_NULL_AND_RETURN(sparseIndices, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_NULL_AND_RETURN(attentionOut, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_NULL_AND_RETURN(workspaceSize, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_NULL_AND_RETURN(executor, ACLNN_ERR_PARAM_NULLPTR);

    if (returnSoftmaxLse) { ... }
    return aclnnInnerSparseFlashAttentionGetWorkspaceSize(...);
}
```
**修改说明**：在函数入口处对所有必需输入/输出指针参数做空指针校验，与已有的 softmaxMax/softmaxSum 校验保持一致，符合"外部传入指针需要判空后使用"的规范。

---

### 问题ID：ISSUE-003 | 严重级别：HIGH（高）

#### 假设检验过程
**代码段**：`TensorHolder` 构造函数（第47~58行）
**假设**：H0: 该代码段对资源申请结果进行了充分的成功性检查

| 证据序号 | 证据类型 | 规范ID | 证据描述 | 分值增量 | 累计自信值 |
|---------|---------|--------|---------|---------|-----------|
| 1 | 红线规范违反 | SEC-2.9 | aclCreateTensor的返回值直接赋给inner_，未做显式的非空判断。如果分配失败返回nullptr，后续output=inner_会将nullptr传给外部，依赖调用方的间接检查而非本函数内的显式校验 | +40% | 40% |
| 2 | 上下文防御缺失 | SEC-2.9 | 虽然aclnnSparseFlashAttentionGetWorkspaceSize中后续有if(softmaxMax==nullptr)的检查，但这是间接依赖调用方而非申请方自身的校验，不符合"资源申请后必须判断是否成功"的规范要求 | +25% | 65% |

**结论**：自信值 **65%** > 60%，**推翻原假设H0**，该代码段缺少资源申请成功性检查。

---

**关联红线条款**：SEC-2.9（资源申请后必须判断是否成功）
**代码路径**：aclnn_sparse_flash_attention.cpp:47-58
**问题类型**：资源申请失败未检查
**问题描述**：`TensorHolder` 构造函数中调用 `aclCreateTensor` 后，未对其返回值进行显式的成功性检查。如果 `aclCreateTensor` 分配失败返回 nullptr，`inner_` 将为 nullptr，`output` 也会被赋值为 nullptr。虽然调用方有间接的空指针检查，但违反了"资源申请后必须判断是否成功"的规范。

#### 修改建议
**修改前代码**：
```cpp
TensorHolder(const aclTensor *&output, aclDataType dataType, std::string varName) {
    inner_ = nullptr;
    name_ = varName;
    if (output == nullptr) {
        std::vector<int64_t> shape = {0};
        int64_t addr = 0xff;
        inner_ = aclCreateTensor(shape.data(), shape.size(),
            dataType, shape.data(), 0, ACL_FORMAT_ND,
            shape.data(), shape.size(), static_cast<void *>(&addr));
        output = inner_;
    }
}
```
**修改后代码**：
```cpp
TensorHolder(const aclTensor *&output, aclDataType dataType, std::string varName) {
    inner_ = nullptr;
    name_ = varName;
    if (output == nullptr) {
        std::vector<int64_t> shape = {0};
        int64_t addr = 0xff;
        inner_ = aclCreateTensor(shape.data(), shape.size(),
            dataType, shape.data(), 0, ACL_FORMAT_ND,
            shape.data(), shape.size(), static_cast<void *>(&addr));
        if (inner_ == nullptr) {
            OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "Failed to create tensor %s!", varName.c_str());
            return;
        }
        output = inner_;
    }
}
```
**修改说明**：在 `aclCreateTensor` 调用后立即检查返回值，失败时不赋值给 output（保持 nullptr），并输出错误日志，符合"资源申请后必须判断是否成功"规范。

---

### 问题ID：ISSUE-004 | 严重级别：MEDIUM（中等）

#### 假设检验过程
**代码段**：`aclnnSparseFlashAttentionGetWorkspaceSize` 函数中的错误返回（第111~126行）
**假设**：H0: 该代码段的错误码使用是一致的

| 证据序号 | 证据类型 | 规范ID | 证据描述 | 分值增量 | 累计自信值 |
|---------|---------|--------|---------|---------|-----------|
| 1 | 代码逻辑不一致 | - | OP_LOGE宏使用ACLNN_ERR_PARAM_NULLPTR作为错误码，但return语句返回ge::GRAPH_FAILED。函数签名返回aclnnStatus类型，应返回ACLNN错误码而非ge错误码 | +35% | 35% |
| 2 | 上下文确认 | - | 同一函数内多处错误返回均使用ge::GRAPH_FAILED，与OP_LOGE中的ACLNN_ERR_PARAM_NULLPTR语义不匹配 | +20% | 55% |

**结论**：自信值 **55%** < 60%，**未推翻原假设H0**。但该不一致性值得存疑，供开发者自行判断。

---

**关联红线条款**：无直接关联条款（代码质量问题）
**代码路径**：aclnn_sparse_flash_attention.cpp:111-126
**问题类型**：错误码返回不一致
**问题描述**：函数 `aclnnSparseFlashAttentionGetWorkspaceSize` 返回类型为 `aclnnStatus`，但在错误处理中使用 `ge::GRAPH_FAILED` 作为返回值，而 `OP_LOGE` 日志中使用的错误码是 `ACLNN_ERR_PARAM_NULLPTR`。日志中的错误码与实际返回值不一致，可能影响上层调用者的错误处理逻辑。

#### 修改建议
**修改前代码**：
```cpp
OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "softmaxMax and softmaxSum cannot be nullptr.");
return ge::GRAPH_FAILED;
```
**修改后代码**：
```cpp
OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "softmaxMax and softmaxSum cannot be nullptr.");
return ACLNN_ERR_PARAM_NULLPTR;
```
**修改说明**：将错误返回值从 `ge::GRAPH_FAILED` 改为与日志一致的 `ACLNN_ERR_PARAM_NULLPTR`，确保日志与返回值的语义一致。

---

### 问题ID：ISSUE-005 | 严重级别：MEDIUM（中等）

#### 假设检验过程
**代码段**：`TensorHolder::IsTensorNotNull` 和 `CheckTensorConditionalNotNull` 方法（第67~77行）
**假设**：H0: 这两个方法的逻辑与命名语义一致

| 证据序号 | 证据类型 | 规范ID | 证据描述 | 分值增量 | 累计自信值 |
|---------|---------|--------|---------|---------|-----------|
| 1 | 代码逻辑异常 | SEC-2.4 | IsTensorNotNull()方法名暗示"tensor不为空"，但实现返回inner_==nullptr（即tensor为空时返回true），语义反转 | +30% | 30% |
| 2 | 代码逻辑异常 | SEC-2.4 | CheckTensorConditionalNotNull()中，当inner_非空且conditional为true时报"!= nullptr failed"，当inner_为空且conditional为false时报"== nullptr failed"，条件判断与日志消息语义相反 | +25% | 55% |
| 3 | 上下文确认 | - | 两个方法在当前代码中均未被调用，属于死代码 | +10% | 65% |

**结论**：自信值 **65%** > 60%，**推翻原假设H0**，这两个方法逻辑存在异常。

---

**关联红线条款**：SEC-2.4（禁止使用未初始化的变量 / 逻辑正确性）
**代码路径**：aclnn_sparse_flash_attention.cpp:67-77
**问题类型**：方法逻辑与命名语义不一致（死代码）
**问题描述**：
1. `IsTensorNotNull()` 方法命名表示"tensor不为空"，但返回 `inner_ == nullptr`，当 holder 未创建 tensor（即原始 tensor 不为空）时返回 true。逻辑与命名语义相反。
2. `CheckTensorConditionalNotNull()` 方法的条件分支判断与日志消息含义不一致：当 `inner_` 非空（holder创建了tensor，说明原始tensor为空）且 `conditional` 为 true 时，日志却报告"!= nullptr failed"。
3. 两个方法均为死代码（未在当前文件中被调用）。

#### 修改建议
**修改前代码**：
```cpp
void CheckTensorConditionalNotNull(bool conditional) const {
    if (inner_ && conditional) {
        OP_LOGW("Check %s != nullptr failed!", name_.c_str());
    } else if (!inner_ && !conditional) {
        OP_LOGW("Check %s == nullptr failed!", name_.c_str());
    }
}

bool IsTensorNotNull() const {
    return inner_ == nullptr;
}
```
**修改后代码**：
```cpp
// 建议：如果确认两个方法无需使用，直接删除以避免后续误用
// 如果需要保留，修正逻辑如下：

bool WasTensorCreated() const {
    // 返回true表示holder创建了一个新tensor（即原始output为nullptr）
    return inner_ != nullptr;
}
```
**修改说明**：建议直接删除这两个未使用的死代码方法，避免后续维护者误用。如需保留，应修正方法命名使其与实际语义一致。

---

### 问题ID：ISSUE-006 | 严重级别：LOW（轻微）

#### 假设检验过程
**代码段**：`aclnnSparseFlashAttentionGetWorkspaceSize` 函数的数值参数（第94~101行）
**假设**：H0: 该代码段对外部传入的数值参数进行了充分的范围校验

| 证据序号 | 证据类型 | 规范ID | 证据描述 | 分值增量 | 累计自信值 |
|---------|---------|--------|---------|---------|-----------|
| 1 | 缺少校验 | SEC-2.11 | scaleValue（double）、sparseBlockSizeOptional（int64_t）、sparseMode（int64_t）、preTokens（int64_t）、nextTokens（int64_t）、attentionMode（int64_t）等外部传入的数值参数均未做范围/合法性校验 | +25% | 25% |

**结论**：自信值 **25%** < 60%，**未推翻原假设H0**。该问题为 LOW 级别，供开发者参考。

---

**关联红线条款**：SEC-2.11（外部输入数据需要做合法性校验）
**代码路径**：aclnn_sparse_flash_attention.cpp:94-101
**问题类型**：数值参数未做范围校验
**问题描述**：函数接收多个外部传入的数值参数（`scaleValue`、`sparseBlockSizeOptional`、`sparseMode`、`preTokens`、`nextTokens`、`attentionMode`），但未对这些参数的合法范围做校验。如果上层调用者传入非法值（如负数的 block size、非法的 sparse mode 等），将直接传递给 inner 函数，可能在内部触发未定义行为或非预期结果。

#### 修改建议
**修改前代码**：
```cpp
aclnnStatus aclnnSparseFlashAttentionGetWorkspaceSize(
    ...,
    double scaleValue,
    int64_t sparseBlockSizeOptional,
    ...,
    int64_t sparseMode, int64_t preTokens, int64_t nextTokens, int64_t attentionMode,
    ...) {
    // 无参数范围校验
```
**修改后代码**：
```cpp
aclnnStatus aclnnSparseFlashAttentionGetWorkspaceSize(
    ...,
    double scaleValue,
    int64_t sparseBlockSizeOptional,
    ...,
    int64_t sparseMode, int64_t preTokens, int64_t nextTokens, int64_t attentionMode,
    ...) {
    // 校验数值参数合法性
    if (scaleValue < 0.0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "scaleValue(%f) must be non-negative.", scaleValue);
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (sparseBlockSizeOptional <= 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "sparseBlockSizeOptional(%ld) must be positive.", sparseBlockSizeOptional);
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (sparseMode < 0 || sparseMode > 2) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "sparseMode(%ld) is out of valid range [0, 2].", sparseMode);
        return ACLNN_ERR_PARAM_INVALID;
    }
    // ... 其他参数校验
```
**修改说明**：在函数入口处对外部传入的数值参数做合法性范围校验，尽早拦截非法输入，符合"外部输入数据需要做合法性校验"规范。

---

## 检视类别总结

| 检视类别 | 发现问题数 | 严重级 | 高级 | 中等级 | 低级 |
|---------|-----------|-------|-----|-------|-----|
| 数值运算安全 | 0 | 0 | 0 | 0 | 0 |
| 内存与指针安全 | 1 | 1 (ISSUE-001) | 0 | 0 (见ISSUE-005) | 0 |
| 资源管理 | 1 | 0 | 1 (ISSUE-003) | 0 | 0 |
| 输入验证 | 3 | 1 (ISSUE-002) | 0 | 1 (ISSUE-004) | 1 (ISSUE-006) |
| 并发安全 | 0 | 0 | 0 | 0 | 0 |
| **合计** | **6** | **2** | **1** | **2** | **1** |

## 报告生成时间
2026-03-20
## 报告状态
已完成检视，待修复验证
