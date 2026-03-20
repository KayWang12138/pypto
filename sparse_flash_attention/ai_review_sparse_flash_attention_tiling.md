# 代码检视报告
**项目名称**：NPU算子检视报告
**检视模块**：attention/sparse_flash_attention/op_host/sparse_flash_attention_tiling.cpp
**检视人**：Turing Team
**检视日期**：2026-03-20


## 🔍 检视概览
| 统计项 | 数值 |
| ---- | ---- |
| 发现问题总数 | 4 个 |
| 严重级（CRITICAL）问题 | 1 个 |
| 中等级（HIGH）问题 | 2 个 |
| 轻微级（MEDIUM）问题 | 1 个 |
| 误报数量 | 0 个 |

**核心结论**：整体代码结构清晰，输入校验覆盖面广。但存在1处CRITICAL级属性指针未判空问题需优先修复（4个属性指针获取后未做空指针校验即被解引用），2处HIGH级uint32_t乘法溢出风险，1处MEDIUM级map迭代器未校验问题。


## ❌ 问题详情及修改建议

---

### 问题ID：ISSUE-001 | 严重级别：CRITICAL（严重）

#### 🔬 假设检验过程
**代码段**：`GetAttrParaInfo()` 函数及 `CheckSingleParaPreTokens/CheckSingleParaNextTokens/CheckRopeExistence/CheckSoftmaxMaxShape/GenerateInfo` 等多个函数中对 `preTokens`、`nextTokens`、`attentionMode`、`returnSoftmaxLse` 指针的解引用
**假设**：H0: 该代码段是安全的

| 证据序号 | 证据类型 | 规范ID | 证据描述 | 分值增量 | 累计自信值 |
|---------|---------|--------|---------|---------|-----------|
| 1 | 红线规范违反 | 2.8 | `preTokens`、`nextTokens`、`attentionMode`、`returnSoftmaxLse` 通过 `GetAttrPointer` 获取后未做空指针校验，即被直接解引用（`*opParamInfo_.preTokens`等），违反"指针操作使用前必须判空"规范 | +40% | 40% |
| 2 | 上下文防御缺失 | 2.8 | `CheckRequiredAttrExistence()` 仅检查了 `layoutQuery`、`layoutKV`、`sparseBlockSize`、`scaleValue`、`sparseMode` 共5个属性的空指针，遗漏了上述4个属性 | +30% | 70% |
| 3 | 数据流追踪风险 | 2.8 | 结构体定义中4个属性指针默认初始化为 nullptr（sparse_flash_attention_tiling.h:138-141），若 `GetAttrPointer` 返回 nullptr，后续解引用将导致程序崩溃 | +25% | 95% |

**结论**：自信值 **95%** > 60%，**推翻原假设H0**，该代码段存在风险。

---

**关联红线条款**：规范 2.8（指针操作，使用前必须要判空）
**代码路径**：sparse_flash_attention_tiling.cpp:1647-1650（属性获取）, :733/:741/:767/:770/:992/:1008/:1037/:1059/:1767/:1980-1983（解引用点）
**问题类型**：空指针未保护/未判空解引用
**问题描述**：`GetAttrParaInfo()` 函数中通过 `attrs->GetAttrPointer()` 获取了 `preTokens`、`nextTokens`、`attentionMode`、`returnSoftmaxLse` 四个属性指针（行1647-1650），但 `CheckRequiredAttrExistence()` 函数（行1518-1530）仅对 `layoutQuery`、`layoutKV`、`sparseBlockSize`、`scaleValue`、`sparseMode` 进行了空指针校验，遗漏了上述4个属性。这4个指针在后续多个函数中被直接解引用（`*opParamInfo_.preTokens` 等），若 `GetAttrPointer` 因属性未定义返回 nullptr，将导致程序崩溃（coredump），违反规范2.8"指针操作，使用前必须要判空"。

#### 修改建议
**修改前代码**：
```cpp
// sparse_flash_attention_tiling.cpp:1518-1530
ge::graphStatus SFAInfoParser::CheckRequiredAttrExistence() const
{
    OP_CHECK_IF(opParamInfo_.layoutQuery == nullptr, ..., return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.layoutKV == nullptr, ..., return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.sparseBlockSize == nullptr, ..., return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.scaleValue == nullptr, ..., return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.sparseMode == nullptr, ..., return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}
```
**修改后代码**：
```cpp
ge::graphStatus SFAInfoParser::CheckRequiredAttrExistence() const
{
    OP_CHECK_IF(opParamInfo_.layoutQuery == nullptr, ..., return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.layoutKV == nullptr, ..., return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.sparseBlockSize == nullptr, ..., return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.scaleValue == nullptr, ..., return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.sparseMode == nullptr, ..., return ge::GRAPH_FAILED);
    // 补充以下4个属性指针的空指针校验
    OP_CHECK_IF(opParamInfo_.preTokens == nullptr, OP_LOGE(opName_, "attr preTokens is nullptr"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.nextTokens == nullptr, OP_LOGE(opName_, "attr nextTokens is nullptr"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.attentionMode == nullptr, OP_LOGE(opName_, "attr attentionMode is nullptr"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.returnSoftmaxLse == nullptr, OP_LOGE(opName_, "attr returnSoftmaxLse is nullptr"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}
```
**修改说明**：在 `CheckRequiredAttrExistence()` 中补充 `preTokens`、`nextTokens`、`attentionMode`、`returnSoftmaxLse` 四个属性指针的空指针校验，确保在 `Parse()` 流程中完成校验后再使用，符合规范2.8"指针操作使用前必须要判空"的要求，避免因属性未定义导致的空指针解引用崩溃。

---

### 问题ID：ISSUE-002 | 严重级别：HIGH（高）

#### 🔬 假设检验过程
**代码段**：`CalcUbBmm()` 函数中的乘法运算
**假设**：H0: 该代码段是安全的

| 证据序号 | 证据类型 | 规范ID | 证据描述 | 分值增量 | 累计自信值 |
|---------|---------|--------|---------|---------|-----------|
| 1 | 红线规范违反 | 2.2 | 行299中 `gSize * (headDimAlign_ + 64U) * s1Size` 为三重 uint32_t 乘法，`gSize` 来自外部数据（最大128），`headDimAlign_` 为512，`s1Size` 来自张量shape，乘积 `128 * 576 * s1Size = 73728 * s1Size`，当 `s1Size > 58177` 时发生无符号整数回绕 | +40% | 40% |
| 2 | 上下文防御缺失 | 2.2 | 代码中未对 `s1Size` 的上界进行限制校验（仅校验了 `gSize` 的取值范围），无法防止乘法回绕 | +30% | 70% |
| 3 | 数据流追踪风险 | 2.2 | `qPreSizeMla_` 类型为 `size_t`，但中间乘法在 `uint32_t` 范围内计算后隐式转换，溢出值已丢失精度，赋值给 `size_t` 时得到错误结果 | +25% | 95% |

**结论**：自信值 **95%** > 60%，**推翻原假设H0**，该代码段存在风险。

---

**关联红线条款**：规范 2.2（确保无符号整数运算不回绕）
**代码路径**：sparse_flash_attention_tiling.cpp:296, :299
**问题类型**：无符号整数乘法回绕
**问题描述**：`CalcUbBmm()` 函数中存在两处 uint32_t 乘法回绕风险：
1. 行296：`mmResUbSize_ = sInnerSizeAlign_ * Align(cubeMSize, 16U);` —— `sInnerSizeAlign_`（uint32_t）与 `Align(cubeMSize, 16U)` 的乘积在 uint32_t 范围内计算，可能回绕后赋值给 `size_t` 类型的 `mmResUbSize_`。
2. 行299：`qPreSizeMla_ = sfaInfo_->gSize * (headDimAlign_ + 64U) * sfaInfo_->s1Size;` —— 三重乘法 `128 * 576 * s1Size`，当 `s1Size` 较大时在 uint32_t 中回绕，赋值给 `size_t` 类型的 `qPreSizeMla_` 时得到错误值。

虽然当前算子场景下 `s1Size` 通常不会超过数万，但从编码规范角度，参与乘法的外部数据应校验上界防止回绕。

#### 修改建议
**修改前代码**：
```cpp
// sparse_flash_attention_tiling.cpp:296-299
mmResUbSize_ = sInnerSizeAlign_ * Align(cubeMSize, 16U);
bmm2ResUbSize_ = headDimAlign_ * Align(cubeMSize, 16U);

qPreSizeMla_ = sfaInfo_->gSize * (headDimAlign_ + 64U) * sfaInfo_->s1Size;
```
**修改后代码**：
```cpp
// 使用 size_t 字面量或显式类型转换确保在 size_t 范围内计算
mmResUbSize_ = static_cast<size_t>(sInnerSizeAlign_) * static_cast<size_t>(Align(cubeMSize, 16U));
bmm2ResUbSize_ = static_cast<size_t>(headDimAlign_) * static_cast<size_t>(Align(cubeMSize, 16U));

qPreSizeMla_ = static_cast<size_t>(sfaInfo_->gSize) *
               static_cast<size_t>(headDimAlign_ + 64U) *
               static_cast<size_t>(sfaInfo_->s1Size);
```
**修改说明**：将参与乘法的操作数显式转换为 `size_t`（64位），确保中间乘法运算在 64 位范围内完成，避免 uint32_t 回绕导致计算结果错误。符合规范2.2"确保无符号整数运算不回绕"的要求。

---

### 问题ID：ISSUE-003 | 严重级别：HIGH（高）

#### 🔬 假设检验过程
**代码段**：`CheckDimNumInLayoutSupport()` 和 `GetAxisIdx()` 函数中对 `map::find()` 返回迭代器的解引用
**假设**：H0: 该代码段是安全的

| 证据序号 | 证据类型 | 规范ID | 证据描述 | 分值增量 | 累计自信值 |
|---------|---------|--------|---------|---------|-----------|
| 1 | 红线规范违反 | 2.8 | 行616-617：`SFA_LAYOUT_DIM_MAP.find(layout)` 返回值未与 `end()` 比较，直接访问 `dimIt->second`；行1468-1470：`SFA_LAYOUT_AXIS_MAP.find(layout)->second` 同样未校验 `find()` 结果 | +40% | 40% |
| 2 | 上下文防御缺失 | 2.8 | `CheckDimNumInLayoutSupport` 虽然总是在 `CheckLayoutSupport` 之后调用，但函数本身未做防御；`GetAxisIdx` 虽由 `GetAxisNum`（先调 `HasAxis`）保护，但 `GetAxisIdx` 本身作为独立方法缺乏防御 | +20% | 60% |
| 3 | 数据流追踪风险 | 2.8 | 若后续维护时向 `LAYOUT_SUPPORT_MAP` 新增 layout 但遗漏 `SFA_LAYOUT_DIM_MAP`，或直接调用 `GetAxisIdx` 传入不存在的 layout，将导致对 `end()` 迭代器的解引用，引发 UB | +25% | 85% |

**结论**：自信值 **85%** > 60%，**推翻原假设H0**，该代码段存在风险。

---

**关联红线条款**：规范 2.8（指针操作，使用前必须要判空）
**代码路径**：sparse_flash_attention_tiling.cpp:616-617, :1468-1470
**问题类型**：迭代器未校验即解引用
**问题描述**：两处使用 `std::map::find()` 后未校验返回值是否为 `end()` 即直接解引用：
1. 行616-617：`CheckDimNumInLayoutSupport()` 中 `const auto& dimIt = SFA_LAYOUT_DIM_MAP.find(layout); OP_CHECK_IF(shape->GetStorageShape().GetDimNum() != dimIt->second, ...);`
2. 行1468-1470：`GetAxisIdx()` 中 `const std::vector<SFAAxis>& axes = SFA_LAYOUT_AXIS_MAP.find(layout)->second;`

若 layout 值不在 map 中，`find()` 返回 `end()`，解引用 `end()` 迭代器将导致未定义行为。虽然当前调用链中 layout 已被校验，但函数本身缺乏防御性编程。

#### 修改建议
**修改前代码**：
```cpp
// sparse_flash_attention_tiling.cpp:613-622
ge::graphStatus SFATilingCheck::CheckDimNumInLayoutSupport(const SFALayout &layout,
    const gert::StorageShape *shape, const std::string &name) const
{
    const auto& dimIt = SFA_LAYOUT_DIM_MAP.find(layout);
    OP_CHECK_IF(shape->GetStorageShape().GetDimNum() != dimIt->second,
        OP_LOGE(opName_, "When layout is %s, ..."), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}
```
**修改后代码**：
```cpp
ge::graphStatus SFATilingCheck::CheckDimNumInLayoutSupport(const SFALayout &layout,
    const gert::StorageShape *shape, const std::string &name) const
{
    const auto& dimIt = SFA_LAYOUT_DIM_MAP.find(layout);
    OP_CHECK_IF(dimIt == SFA_LAYOUT_DIM_MAP.end(),
        OP_LOGE(opName_, "layout %s not found in SFA_LAYOUT_DIM_MAP", SFALayoutToSerialString(layout).c_str()),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(shape->GetStorageShape().GetDimNum() != dimIt->second,
        OP_LOGE(opName_, "When layout is %s, ..."), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}
```

```cpp
// sparse_flash_attention_tiling.cpp:1468-1473
size_t SFAInfoParser::GetAxisIdx(const SFAAxis &axis, const SFALayout &layout) const
{
    const auto& layoutIt = SFA_LAYOUT_AXIS_MAP.find(layout);
    if (layoutIt == SFA_LAYOUT_AXIS_MAP.end()) {
        return 0; // 或返回错误码
    }
    const std::vector<SFAAxis>& axes = layoutIt->second;
    const auto& axisIt = std::find(axes.begin(), axes.end(), axis);
    return std::distance(axes.begin(), axisIt);
}
```
**修改说明**：在 `CheckDimNumInLayoutSupport()` 和 `GetAxisIdx()` 中添加 `map::find()` 返回值与 `end()` 的比较校验，确保迭代器有效后再解引用。符合规范2.8"指针操作使用前必须要判空"的精神（迭代器类似于指针，使用前必须校验有效性），提升代码健壮性和可维护性。

---

### 问题ID：ISSUE-004 | 严重级别：MEDIUM（中等）

#### 🔬 假设检验过程
**代码段**：`GetWorkspaceSize()` 函数中的乘法运算
**假设**：H0: 该代码段是安全的

| 证据序号 | 证据类型 | 规范ID | 证据描述 | 分值增量 | 累计自信值 |
|---------|---------|--------|---------|---------|-----------|
| 1 | 一般规范违反 | 2.2 | 行449：`4 * 512 * (512 + 64) * 2 * actCoreNum` 整个表达式在 uint32_t 范围内计算（因 `actCoreNum` 为 uint32_t，所有 int 字面量提升为 uint32_t），乘积为 `2359296 * actCoreNum`，当 `actCoreNum > 1819` 时回绕 | +20% | 20% |
| 2 | 上下文防御缺失 | 2.2 | 代码中未对 `actCoreNum`（即 `aicNum_`）的合理上界进行限制，仅校验了 `!= 0` | +30% | 50% |
| 3 | 数据流追踪风险 | 2.2 | `workspaceSize_` 为 `uint64_t`，但中间乘法在 uint32_t 中回绕后得到错误值，再赋值给 uint64_t 时已是错误结果 | +25% | 75% |

**结论**：自信值 **75%** > 60%，**推翻原假设H0**，该代码段存在风险。

---

**关联红线条款**：规范 2.2（确保无符号整数运算不回绕）
**代码路径**：sparse_flash_attention_tiling.cpp:449-451
**问题类型**：无符号整数乘法回绕
**问题描述**：`GetWorkspaceSize()` 函数中行449-451的 workspace 计算表达式中，整型字面量与 `actCoreNum`（uint32_t）的乘法运算在 uint32_t 范围内完成。虽然 `workspaceSize_` 是 `uint64_t`，但中间计算 `4 * 512 * (512 + 64) * 2 * actCoreNum = 2359296 * actCoreNum` 在赋值前已发生 uint32_t 回绕。当前 Ascend 芯片的 AIC 核数通常不超过数百，实际触发概率较低，但从防御性编程角度仍存在隐患。

#### 修改建议
**修改前代码**：
```cpp
// sparse_flash_attention_tiling.cpp:449-451
workspaceSize_ += 4 * 512 * (512 + 64) * 2 * actCoreNum;
workspaceSize_ += 4 * 128 * 4 * (2 * actCoreNum);
```
**修改后代码**：
```cpp
// 使用 size_t/uint64_t 字面量确保在64位范围内计算
workspaceSize_ += static_cast<size_t>(4) * 512 * (512 + 64) * 2 * actCoreNum;
workspaceSize_ += static_cast<size_t>(4) * 128 * 4 * (2 * actCoreNum);
```
**修改说明**：通过 `static_cast<size_t>(4)` 将首个操作数提升为 `size_t`，使整个乘法链在 `size_t`（64位）范围内计算，避免中间结果回绕。符合规范2.2"确保无符号整数运算不回绕"的要求。

---

## 报告生成时间
2026-03-20
## 报告状态
已完成检视，待修复验证
