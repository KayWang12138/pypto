# 代码检视报告
**项目名称**：NPU算子检视报告
**检视模块**：/mnt/workspace/gitCode/cann/mce/ops-transformer/attention/sparse_flash_attention/op_host/sparse_flash_attention_infershape.cpp
**检视人**：Turing Team
**检视日期**：2026-03-20


## 检视概览
| 统计项 | 数值 |
| ---- | ---- |
| 发现问题总数 | 5 个 |
| 严重级（CRITICAL）问题 | 2 个 |
| 中等级（MEDIUM）问题 | 1 个 |
| 轻微级（LOW）问题 | 2 个 |
| 误报数量 | 0 个 |

**核心结论**：发现2处Critical级安全问题（除零未保护、空指针未保护）需优先修复，1处Medium级输入校验缺失需补充，2处Low级代码质量问题建议优化。

## 问题详情及修改建议

### 问题ID：ISSUE-001 | 严重级别：CRITICAL（严重）

#### 假设检验过程
**代码段**：`InferShapeSparseFlashAttention` 函数中 softmaxMaxShape / softmaxSumShape 的 SetDim 除法运算（行69、74、80、86）
**假设**：H0: 该代码段是安全的

| 证据序号 | 证据类型 | 规范ID | 证据描述 | 分值增量 | 累计自信值 |
|---------|---------|--------|---------|---------|-----------|
| 1 | 红线规范违反 | 2.3 | 4处除法运算的除数来自外部输入 `keyShape->GetDim()` ，未做除零校验，违反"确保除法和余数运算不会导致除以零的错误" | +40% | 40% |
| 2 | 上下文防御缺失 | 2.3 | 代码中无任何对除数是否为0的检查或保护逻辑 | +30% | 70% |

**结论**：自信值 **70%** > 60%，**推翻原假设H0**，该代码段存在风险。

---

**关联红线条款**：2.3 确保除法和余数运算不会导致除以零的错误
**代码路径**：sparse_flash_attention_infershape.cpp:69, 74, 80, 86
**问题类型**：除零未保护
**问题描述**：4处除法运算的除数 `keyShape->GetDim(DIM_INDEX_1)` 和 `keyShape->GetDim(DIM_INDEX_2)` 均来自外部输入的 shape 维度值。当 keyShape 的对应维度值为 0 时，将触发除零错误导致未定义行为（程序崩溃）。经 API 源码确认（`shape.h` 行158-163），`GetDim()` 对 `idx < kMaxDimNum` 的合法索引直接返回 `dims_[idx]`，不做零值过滤，因此除数为 0 是可触发的场景。

#### 修改建议
**修改前代码**：
```cpp
softmaxMaxShape->SetDim(DIM_INDEX_2, queryShape->GetDim(DIM_INDEX_1) / keyShape->GetDim(DIM_INDEX_1));
// ... 同样的模式在行74、80、86
```
**修改后代码**：
```cpp
int64_t keyDim1 = keyShape->GetDim(DIM_INDEX_1);
OP_CHECK_IF(keyDim1 == 0,
    OP_LOGE(context->GetNodeName(), "keyShape dim[1] is zero, cannot divide"),
    return ge::GRAPH_FAILED);
softmaxMaxShape->SetDim(DIM_INDEX_2, queryShape->GetDim(DIM_INDEX_1) / keyDim1);
```
**修改说明**：在除法运算前对除数进行零值校验，违反时返回 `GRAPH_FAILED`，符合规范 2.3 要求。4处除法运算均需添加相同保护。

---

### 问题ID：ISSUE-002 | 严重级别：CRITICAL（严重）

#### 假设检验过程
**代码段**：`InferShapeSparseFlashAttention` 函数中 `attrs` 指针的获取与使用（行59-60）
**假设**：H0: 该代码段是安全的

| 证据序号 | 证据类型 | 规范ID | 证据描述 | 分值增量 | 累计自信值 |
|---------|---------|--------|---------|---------|-----------|
| 1 | 红线规范违反 | 2.8 | `context->GetAttrs()` 返回值未做 nullptr 判空，直接通过 `attrs->GetAttrPointer<bool>(...)` 解引用，违反"指针操作，使用前必须要判空" | +40% | 40% |
| 2 | API源码佐证 | 2.8 | 经查阅 `extended_kernel_context.h` 行132-138，`GetAttrs()` 在 `compute_node_info == nullptr` 时返回 `nullptr`，证明返回值为nullptr是可触发的场景 | +30% | 70% |

**结论**：自信值 **70%** > 60%，**推翻原假设H0**，该代码段存在风险。

---

**关联红线条款**：2.8 指针操作，使用前必须要判空
**代码路径**：sparse_flash_attention_infershape.cpp:59-60
**问题类型**：空指针未保护
**问题描述**：`auto attrs = context->GetAttrs()` 的返回值直接被解引用调用 `GetAttrPointer` 方法，未做 nullptr 判空。经 API 源码确认（`extended_kernel_context.h` 行132-138），`GetAttrs()` 在 `compute_node_info` 为 nullptr 时返回 `nullptr`。如果触发该路径，`attrs->GetAttrPointer(...)` 将导致空指针解引用，产生未定义行为。

#### 修改建议
**修改前代码**：
```cpp
auto attrs = context->GetAttrs();
const bool *lse_flag = attrs->GetAttrPointer<bool>(RETURN_SOFTMAX_LSE_INDEX);
```
**修改后代码**：
```cpp
auto attrs = context->GetAttrs();
OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
const bool *lse_flag = attrs->GetAttrPointer<bool>(RETURN_SOFTMAX_LSE_INDEX);
```
**修改说明**：在 `GetAttrs()` 返回后立即使用 `OP_CHECK_NULL_WITH_CONTEXT` 宏进行空指针校验，与文件中其他指针的检查风格保持一致，符合规范 2.8 要求。

---

### 问题ID：ISSUE-003 | 严重级别：MEDIUM（中等）

#### 假设检验过程
**代码段**：`InferShapeSparseFlashAttention` 函数中 keyShape 维度数量校验缺失及 queryShape 非法维度数未处理（行65-87）
**假设**：H0: 该代码段是安全的

| 证据序号 | 证据类型 | 规范ID | 证据描述 | 分值增量 | 累计自信值 |
|---------|---------|--------|---------|---------|-----------|
| 1 | 一般规范违反 | 2.11 | keyShape 的维度数（GetDimNum）未做校验，直接通过 GetDim 访问 DIM_INDEX_1（需≥2维）和 DIM_INDEX_2（需≥3维），违反"外部输入数据需要做合法性校验" | +20% | 20% |
| 2 | API行为分析 | 2.6 | 经查阅 `shape.h` 行158-163，`GetDim()` 仅检查 `idx >= kMaxDimNum`，对 idx 在 [0, kMaxDimNum) 内但不小于实际维度数的情况不报错，返回未初始化或默认值0 | +25% | 45% |
| 3 | 上下文防御缺失 | 2.11 | else 分支隐式假设 queryShape 为4维，未对非法维度数（如1、2、5维）做错误处理 | +30% | 75% |

**结论**：自信值 **75%** > 60%，**推翻原假设H0**，该代码段存在风险。

---

**关联红线条款**：2.11 外部输入数据需要做合法性校验
**代码路径**：sparse_flash_attention_infershape.cpp:65-87
**问题类型**：外部输入校验缺失
**问题描述**：
1. **keyShape 维度数未校验**：代码根据 `queryShape->GetDimNum()` 的值分两路处理，但未对 `keyShape` 的维度数做任何校验。在 DIM_NUM_3 分支中访问 `keyShape->GetDim(DIM_INDEX_1)` 需要 keyShape 至少有 2 个维度；在 else 分支中访问 `keyShape->GetDim(DIM_INDEX_2)` 需要 keyShape 至少有 3 个维度。若 keyShape 维度数不足，`GetDim()` 将返回未初始化的内存值或0（经 API 源码确认，GetDim 不检查 idx 是否在 GetDimNum 范围内）。
2. **queryShape 非法维度数未处理**：`if(queryShape->GetDimNum() == DIM_NUM_3){...} else {...}` 的 else 分支隐式假设为4维情况，若 queryShape 维度数既非3也非4（如1、2、5维），else 分支会按4维逻辑执行，可能导致越界读取 `GetDim(DIM_INDEX_2)` 或计算结果错误。

#### 修改建议
**修改前代码**：
```cpp
if(queryShape->GetDimNum() == DIM_NUM_3){
    softmaxMaxShape->SetDim(DIM_INDEX_0, keyShape->GetDim(DIM_INDEX_1));
    // ...
}else {
    softmaxMaxShape->SetDim(DIM_INDEX_1, keyShape->GetDim(DIM_INDEX_2));
    // ...
}
```
**修改后代码**：
```cpp
size_t queryDimNum = queryShape->GetDimNum();
if (queryDimNum == DIM_NUM_3) {
    OP_CHECK_IF(keyShape->GetDimNum() < DIM_NUM_3,
        OP_LOGE(context->GetNodeName(), "keyShape dimNum must be >= 3 when queryShape is 3D"),
        return ge::GRAPH_FAILED);
    // ... 原有3维逻辑
} else if (queryDimNum == DIM_NUM_4) {
    OP_CHECK_IF(keyShape->GetDimNum() < DIM_NUM_4,
        OP_LOGE(context->GetNodeName(), "keyShape dimNum must be >= 4 when queryShape is 4D"),
        return ge::GRAPH_FAILED);
    // ... 原有4维逻辑
} else {
    OP_LOGE(context->GetNodeName(), "queryShape dimNum must be 3 or 4, but got %zu", queryDimNum);
    return ge::GRAPH_FAILED;
}
```
**修改说明**：1) 在访问 keyShape 维度前校验其维度数是否满足最低要求；2) 将 else 隐式假设改为显式 `else if (queryDimNum == DIM_NUM_4)` 并添加非法维度数的错误处理分支。符合规范 2.11 "外部输入数据需要做合法性校验"要求。

---

### 问题ID：ISSUE-004 | 严重级别：LOW（轻微）

**关联规范条款**：代码质量 / 日志准确性
**代码路径**：sparse_flash_attention_infershape.cpp:100
**问题类型**：日志信息不准确
**问题描述**：`InferDataTypeSparseFlashAttention` 函数中的 null 检查日志消息写的是 `"InferShapeContext is nullptr"`，但该函数接收的参数类型是 `gert::InferDataTypeContext`，日志描述与实际上下文类型不匹配，可能影响问题排查效率。

#### 修改建议
**修改前代码**：
```cpp
OP_CHECK_IF(context == nullptr, OP_LOGE("SparseFlashAttention", "InferShapeContext is nullptr"),
```
**修改后代码**：
```cpp
OP_CHECK_IF(context == nullptr, OP_LOGE("SparseFlashAttention", "InferDataTypeContext is nullptr"),
```
**修改说明**：将日志中的 `"InferShapeContext"` 修正为 `"InferDataTypeContext"`，与函数实际参数类型一致，便于问题定位。

---

### 问题ID：ISSUE-005 | 严重级别：LOW（轻微）

**关联规范条款**：代码质量 / 冗余逻辑
**代码路径**：sparse_flash_attention_infershape.cpp:62
**问题类型**：冗余空指针判断
**问题描述**：行61的 `OP_CHECK_NULL_WITH_CONTEXT(context, lse_flag)` 宏已保证 lse_flag 为非 nullptr（否则直接 return `GRAPH_FAILED`），行62的三元表达式 `(lse_flag != nullptr)? *lse_flag : false` 中的空指针判断是冗余的。虽不影响安全性，但增加了代码阅读复杂度。

#### 修改建议
**修改前代码**：
```cpp
OP_CHECK_NULL_WITH_CONTEXT(context, lse_flag);
bool return_softmax_lse = (lse_flag != nullptr)? *lse_flag : false;
```
**修改后代码**：
```cpp
OP_CHECK_NULL_WITH_CONTEXT(context, lse_flag);
bool return_softmax_lse = *lse_flag;
```
**修改说明**：移除冗余的 nullptr 判断，直接解引用 lse_flag，代码更简洁且语义更清晰。

---

## 附录：完整修复代码示例

```cpp
ge::graphStatus InferShapeSparseFlashAttention(gert::InferShapeContext *context)
{  
    OP_CHECK_IF(context == nullptr, OP_LOGE("SparseFlashAttention", "InferShapeContext is nullptr"),
               return ge::GRAPH_FAILED);
    const gert::Shape *queryShape = context->GetInputShape(QUERY_INPUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, queryShape);

    const gert::Shape *keyShape = context->GetInputShape(KEY_INPUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, keyShape);
    
    gert::Shape *attentionOutShape = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, attentionOutShape);
    *attentionOutShape = *queryShape;

    gert::Shape *softmaxMaxShape = context->GetOutputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, softmaxMaxShape);

    gert::Shape *softmaxSumShape = context->GetOutputShape(2);
    OP_CHECK_NULL_WITH_CONTEXT(context, softmaxSumShape);
    
    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);  // [ISSUE-002] 新增：attrs空指针检查

    const bool *lse_flag = attrs->GetAttrPointer<bool>(RETURN_SOFTMAX_LSE_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, lse_flag);
    bool return_softmax_lse = *lse_flag;  // [ISSUE-005] 移除冗余三元运算符

    if (return_softmax_lse) {
        size_t queryDimNum = queryShape->GetDimNum();
        if (queryDimNum == DIM_NUM_3) {
            // [ISSUE-003] 新增：keyShape维度数校验
            OP_CHECK_IF(keyShape->GetDimNum() < DIM_NUM_3,
                OP_LOGE(context->GetNodeName(), "keyShape dimNum must be >= 3 when queryShape is 3D"),
                return ge::GRAPH_FAILED);
            // [ISSUE-001] 新增：除零检查
            int64_t keyDim1 = keyShape->GetDim(DIM_INDEX_1);
            OP_CHECK_IF(keyDim1 == 0,
                OP_LOGE(context->GetNodeName(), "keyShape dim[1] is zero, cannot divide"),
                return ge::GRAPH_FAILED);

            softmaxMaxShape->SetDimNum(DIM_NUM_3);
            softmaxMaxShape->SetDim(DIM_INDEX_0, keyDim1);
            softmaxMaxShape->SetDim(DIM_INDEX_1, queryShape->GetDim(DIM_INDEX_0));
            softmaxMaxShape->SetDim(DIM_INDEX_2, queryShape->GetDim(DIM_INDEX_1) / keyDim1);

            softmaxSumShape->SetDimNum(DIM_NUM_3);
            softmaxSumShape->SetDim(DIM_INDEX_0, keyDim1);
            softmaxSumShape->SetDim(DIM_INDEX_1, queryShape->GetDim(DIM_INDEX_0));
            softmaxSumShape->SetDim(DIM_INDEX_2, queryShape->GetDim(DIM_INDEX_1) / keyDim1);
        } else if (queryDimNum == DIM_NUM_4) {
            // [ISSUE-003] 新增：keyShape维度数校验 + 非法维度数错误处理
            OP_CHECK_IF(keyShape->GetDimNum() < DIM_NUM_4,
                OP_LOGE(context->GetNodeName(), "keyShape dimNum must be >= 4 when queryShape is 4D"),
                return ge::GRAPH_FAILED);
            // [ISSUE-001] 新增：除零检查
            int64_t keyDim2 = keyShape->GetDim(DIM_INDEX_2);
            OP_CHECK_IF(keyDim2 == 0,
                OP_LOGE(context->GetNodeName(), "keyShape dim[2] is zero, cannot divide"),
                return ge::GRAPH_FAILED);

            softmaxMaxShape->SetDimNum(DIM_NUM_4);
            softmaxMaxShape->SetDim(DIM_INDEX_0, queryShape->GetDim(DIM_INDEX_0));
            softmaxMaxShape->SetDim(DIM_INDEX_1, keyDim2);
            softmaxMaxShape->SetDim(DIM_INDEX_2, queryShape->GetDim(DIM_INDEX_1));
            softmaxMaxShape->SetDim(DIM_INDEX_3, queryShape->GetDim(DIM_INDEX_2) / keyDim2);

            softmaxSumShape->SetDimNum(DIM_NUM_4);
            softmaxSumShape->SetDim(DIM_INDEX_0, queryShape->GetDim(DIM_INDEX_0));
            softmaxSumShape->SetDim(DIM_INDEX_1, keyDim2);
            softmaxSumShape->SetDim(DIM_INDEX_2, queryShape->GetDim(DIM_INDEX_1));
            softmaxSumShape->SetDim(DIM_INDEX_3, queryShape->GetDim(DIM_INDEX_2) / keyDim2);
        } else {
            // [ISSUE-003] 新增：非法维度数量错误处理
            OP_LOGE(context->GetNodeName(), "queryShape dimNum must be 3 or 4, but got %zu", queryDimNum);
            return ge::GRAPH_FAILED;
        }
    } else {
        softmaxMaxShape->SetDimNum(DIM_NUM_1);
        softmaxMaxShape->SetDim(DIM_INDEX_0, 0);
        softmaxSumShape->SetDimNum(DIM_NUM_1);
        softmaxSumShape->SetDim(DIM_INDEX_0, 0);
    }

    return GRAPH_SUCCESS;
}

ge::graphStatus InferDataTypeSparseFlashAttention(gert::InferDataTypeContext *context)
{
    // [ISSUE-004] 修正日志信息
    OP_CHECK_IF(context == nullptr, OP_LOGE("SparseFlashAttention", "InferDataTypeContext is nullptr"),
               return ge::GRAPH_FAILED);
    const auto inputDataType = context->GetInputDataType(QUERY_INPUT_INDEX);
    context->SetOutputDataType(OUTPUT_INDEX_0, inputDataType);
    context->SetOutputDataType(OUTPUT_INDEX_1, ge::DT_FLOAT);
    context->SetOutputDataType(OUTPUT_INDEX_2, ge::DT_FLOAT);
    return ge::GRAPH_SUCCESS;
}
```

## 修复优先级建议

| 优先级 | 问题编号 | 问题描述 | 建议修复时间 |
|--------|---------|---------|------------|
| P0 | ISSUE-001 | 除零风险（4处） | 立即修复 |
| P0 | ISSUE-002 | GetAttrs()空指针未保护 | 立即修复 |
| P1 | ISSUE-003 | keyShape维度数未校验 + queryShape非法维度数未处理 | 尽快修复 |
| P3 | ISSUE-004 | 日志信息不准确 | 可选修复 |
| P3 | ISSUE-005 | 冗余空指针判断 | 可选修复 |

## 报告生成时间
2026-03-20
## 报告状态
已完成检视，待修复验证
