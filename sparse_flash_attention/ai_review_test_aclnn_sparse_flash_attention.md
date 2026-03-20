# 代码检视报告
**项目名称**：NPU算子检视报告
**检视模块**：attention/sparse_flash_attention/examples/test_aclnn_sparse_flash_attention.cpp
**检视人**：Turing Team
**检视日期**：2026-03-20


## 检视概览
| 统计项 | 数值 |
| ---- | ---- |
| 发现问题总数 | 4 个 |
| 严重级（HIGH）问题 | 1 个 |
| 中等级（MEDIUM）问题 | 2 个 |
| 轻微级（LOW）问题 | 1 个 |
| 误报数量 | 0 个 |

**核心结论**：代码整体结构清晰、资源清理逻辑较为完善（错误分支均调用了 CleanupResources）。存在1处HIGH级 executor 资源泄漏需优先修复，2处MEDIUM级问题（tensor 空指针未检查、shape 定义不一致）需修复，1处LOW级数据类型不匹配问题需改进。

---

## 问题详情及修改建议

### 问题ID：ISSUE-001 | 严重级别：HIGH（严重）

#### 假设检验过程
**代码段**：`ExecuteSparseFlashAttention()` 函数中 `aclOpExecutor` 的生命周期管理
**假设**：H0: 该代码段的资源申请与释放是配对的

| 证据序号 | 证据类型 | 规范ID | 证据描述 | 分值增量 | 累计自信值 |
|---------|---------|--------|---------|---------|-----------|
| 1 | 红线规范违反 | 2.12 | `aclOpExecutor* executor` 在 line 231 声明，line 235 由 `aclnnSparseFlashAttentionGetWorkspaceSize` 初始化，line 249 调用 `aclnnSparseFlashAttention` 执行完毕后，整个函数返回前未调用 `aclDestroyAclOpExecutor` 释放 executor | +40% | 40% |
| 2 | 上下文防御缺失 | 2.12 | 搜索 `CleanupResources` 函数（line 273-341）和 `ExecuteSparseFlashAttention` 函数（line 207-256），均未发现对 executor 的释放逻辑 | +30% | 70% |
| 3 | API 确认 | 2.12 | 经查 `/home/developer/Ascend/cann/aarch64-linux/include/aclnn/acl_meta.h:102`，存在 `aclDestroyAclOpExecutor(aclOpExecutor *executor)` API，确认为必需的释放操作 | +15% | 85% |

**结论**：自信值 **85%** > 60%，**推翻原假设H0**，该代码段存在资源泄漏风险。

---

**关联红线条款**：2.12 资源泄露（内存、句柄、锁等）— 资源申请和释放必须匹配
**代码路径**：test_aclnn_sparse_flash_attention.cpp:231, 249
**问题类型**：资源泄漏
**问题描述**：`aclOpExecutor* executor` 是由 `aclnnSparseFlashAttentionGetWorkspaceSize` 分配的显式资源，在 `aclnnSparseFlashAttention` 执行完毕后未被 `aclDestroyAclOpExecutor` 释放。每次调用 `ExecuteSparseFlashAttention` 均会泄漏一个 executor 对象，违反"资源申请和释放必须匹配"的规范要求。

#### 修改建议
**修改前代码**：
```cpp
// line 249
ret = aclnnSparseFlashAttention(*workspaceAddr, *workspaceSize, executor, stream);
if (!CHECK_RET(ret == ACL_SUCCESS)) {
    LOG_PRINT("aclnnSparseFlashAttention failed. ERROR: %d\n", ret);
    return ret;
}

return ACL_SUCCESS;
```
**修改后代码**：
```cpp
ret = aclnnSparseFlashAttention(*workspaceAddr, *workspaceSize, executor, stream);
if (!CHECK_RET(ret == ACL_SUCCESS)) {
    LOG_PRINT("aclnnSparseFlashAttention failed. ERROR: %d\n", ret);
    aclDestroyAclOpExecutor(executor);
    return ret;
}

aclDestroyAclOpExecutor(executor);
return ACL_SUCCESS;
```
**修改说明**：在 `aclnnSparseFlashAttention` 执行完毕后（无论成功或失败），均调用 `aclDestroyAclOpExecutor` 释放 executor 资源，确保资源申请与释放配对，符合规范 2.12 要求。

---

### 问题ID：ISSUE-002 | 严重级别：MEDIUM（中等）

#### 假设检验过程
**代码段**：`CreateAclTensor()` 函数中 `aclCreateTensor` 返回值处理
**假设**：H0: `aclCreateTensor` 返回的指针在使用前已做非空检查

| 证据序号 | 证据类型 | 规范ID | 证据描述 | 分值增量 | 累计自信值 |
|---------|---------|--------|---------|---------|-----------|
| 1 | 红线规范违反 | 2.8 | line 83 调用 `aclCreateTensor(...)` 返回 `aclTensor*`，未做空指针检查，直接赋值给 `*tensor` 后在 line 85 返回成功。若 `aclCreateTensor` 失败返回 nullptr，调用方 `InitializeTensors` 会认为创建成功并继续使用空指针 tensor | +40% | 40% |
| 2 | 函数调用链风险 | 2.9 | `InitializeTensors` 函数（line 110-205）中所有 `CreateAclTensor` 调用后仅检查 ret 是否为 ACL_SUCCESS，未检查 tensor 指针是否为 nullptr。若 tensor 为空，后续传入 `aclnnSparseFlashAttentionGetWorkspaceSize` 将导致空指针解引用 | +25% | 65% |

**结论**：自信值 **65%** > 60%，**推翻原假设H0**，该代码段存在空指针未保护风险。

---

**关联红线条款**：2.8 指针操作，使用前必须要判空；2.9 资源申请后必须判断是否成功
**代码路径**：test_aclnn_sparse_flash_attention.cpp:83-85
**问题类型**：空指针未保护
**问题描述**：`aclCreateTensor` 的返回值（`aclTensor*`）未做空指针检查。根据 `acl_meta.h:62-65` 声明，`aclCreateTensor` 返回指针类型，存在返回 nullptr 的可能性。当前代码直接赋值给 `*tensor` 并返回成功码，掩盖了创建失败的情况。

#### 修改建议
**修改前代码**：
```cpp
// line 83-85
*tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                          shape.data(), shape.size(), *deviceAddr);
return 0;
```
**修改后代码**：
```cpp
*tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                          shape.data(), shape.size(), *deviceAddr);
if (*tensor == nullptr) {
    LOG_PRINT("aclCreateTensor failed, returned nullptr.\n");
    return -1;
}
return 0;
```
**修改说明**：在 `aclCreateTensor` 调用后增加 nullptr 检查，创建失败时打印错误日志并返回错误码，符合规范 2.8（指针使用前必须判空）和规范 2.9（资源申请后必须判断是否成功）要求。

---

### 问题ID：ISSUE-003 | 严重级别：MEDIUM（中等）

#### 假设检验过程
**代码段**：`main()` 函数中 `attentionOutShape` 的定义与 `InitializeTensors()` 中的定义不一致
**假设**：H0: `PrintOutResult` 使用的 shape 与实际 tensor 的 shape 一致

| 证据序号 | 证据类型 | 规范ID | 证据描述 | 分值增量 | 累计自信值 |
|---------|---------|--------|---------|---------|-----------|
| 1 | 数据流追踪风险 | 2.11 | `InitializeTensors` line 115 定义 `attentionOutShape = {1, 2, 1, 512}`（size=1024），`main` line 352 定义 `attentionOutShape = {1, 2, 1, 16}`（size=32）。`PrintOutResult` 使用 main 中的 shape（line 388），只读取 32 个元素而非完整的 1024 个元素 | +30% | 30% |
| 2 | 上下文防御缺失 | 2.11 | 对比同函数中 `softmaxMaxShape` 和 `softmaxSumShape`，`InitializeTensors` (line 116-117) 和 `main` (line 353-354) 中定义完全一致（均为 `{1, 2, 1, 16}`），唯独 `attentionOutShape` 不一致，确认为复制粘贴遗漏 | +35% | 65% |

**结论**：自信值 **65%** > 60%，**推翻原假设H0**，该代码段存在数据读取不完整风险。

---

**关联红线条款**：2.11 外部输入数据需要做合法性校验（数据一致性校验）
**代码路径**：test_aclnn_sparse_flash_attention.cpp:352 vs 115
**问题类型**：数据一致性错误
**问题描述**：`main()` 中 `attentionOutShape = {1, 2, 1, 16}` 与 `InitializeTensors()` 中 `attentionOutShape = {1, 2, 1, 512}` 不一致。`PrintOutResult` 使用 main 中的 shape 计算读取大小（32个float16元素=64字节），但设备端 tensor 实际分配了 1024 个元素（2048字节）。导致测试仅验证了输出结果的 1/32，其余 992 个元素未被打印验证。

#### 修改建议
**修改前代码**：
```cpp
// main() line 352
std::vector<int64_t> attentionOutShape = {1, 2, 1, 16};
```
**修改后代码**：
```cpp
// main() line 352 - 与 InitializeTensors() 中保持一致
std::vector<int64_t> attentionOutShape = {1, 2, 1, 512};
```
**修改说明**：将 `main()` 中的 `attentionOutShape` 修改为 `{1, 2, 1, 512}`，与 `InitializeTensors()` 中创建 tensor 时使用的 shape 保持一致，确保 `PrintOutResult` 能完整读取并验证所有输出数据。

---

### 问题ID：ISSUE-004 | 严重级别：LOW（轻微）

#### 假设检验过程
**代码段**：`PrintOutResult()` 函数读取 `softmaxMaxTensor` 和 `softmaxSumTensor` 时数据类型不匹配
**假设**：H0: `PrintOutResult` 使用的数据类型与设备端 tensor 的数据类型一致

| 证据序号 | 证据类型 | 规范ID | 证据描述 | 分值增量 | 累计自信值 |
|---------|---------|--------|---------|---------|-----------|
| 1 | 数据流追踪风险 | 2.11 | `PrintOutResult` 函数（line 260）固定使用 `std::vector<aclFloat16>` 读取设备数据。但 `softmaxMaxTensor`（line 192）和 `softmaxSumTensor`（line 198）均使用 `ACL_FLOAT` 类型创建。读取时将 float32 数据（4字节/元素）按 float16（2字节/元素）解析，每个元素只读一半数据，输出值全部为乱码 | +25% | 25% |
| 2 | 上下文防御缺失 | — | `attentionOutTensor` 使用 `ACL_FLOAT16`，用 `aclFloat16` 读取是正确的。但 `softmaxMax` 和 `softmaxSum` 使用 `ACL_FLOAT`，`PrintOutResult` 无法感知数据类型差异，函数设计缺乏类型参数 | +20% | 45% |

**结论**：自信值 **45%** < 60%，**无法推翻原假设H0**，但该问题确认为测试逻辑缺陷，以建议级别提出。

---

**关联红线条款**：无直接对应的红线条款，属于测试逻辑正确性问题
**代码路径**：test_aclnn_sparse_flash_attention.cpp:260, 390-392
**问题类型**：数据类型不匹配
**问题描述**：`PrintOutResult` 函数硬编码使用 `aclFloat16` 类型读取设备数据，但 `softmaxMaxTensor`（`ACL_FLOAT`）和 `softmaxSumTensor`（`ACL_FLOAT`）实际存储的是 float32 数据。将 float32 数据按 float16 解析会导致：每个元素只读取一半字节（2字节而非4字节），输出的数值全部为乱码值，测试无法正确验证 softmaxMax 和 softmaxSum 的计算结果。

#### 修改建议
**修改前代码**：
```cpp
// line 258-271
int32_t PrintOutResult(std::vector<int64_t> &shape, void** deviceAddr) {
  auto size = GetShapeSize(shape);
  std::vector<aclFloat16> resultData(size, 0);
  auto ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]),
                         *deviceAddr, size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  ...
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("mean result[%ld] is: %f\n", i, aclFloat16ToFloat(resultData[i]));
  }
```
**修改后代码**：
```cpp
template <typename T>
int32_t PrintOutResult(std::vector<int64_t> &shape, void** deviceAddr) {
  auto size = GetShapeSize(shape);
  std::vector<T> resultData(size, 0);
  auto ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(T),
                         *deviceAddr, size * sizeof(T), ACL_MEMCPY_DEVICE_TO_HOST);
  if (!CHECK_RET(ret == ACL_SUCCESS)) {
    return ret;
  }
  for (int64_t i = 0; i < size; i++) {
    if constexpr (std::is_same_v<T, aclFloat16>) {
      LOG_PRINT("mean result[%ld] is: %f\n", i, aclFloat16ToFloat(resultData[i]));
    } else {
      LOG_PRINT("mean result[%ld] is: %f\n", i, static_cast<float>(resultData[i]));
    }
  }
  return ACL_SUCCESS;
}

// 调用处修改：
PrintOutResult<aclFloat16>(attentionOutShape, &resources.attentionOutDeviceAddr);
PrintOutResult<float>(softmaxMaxShape, &resources.softmaxMaxDeviceAddr);
PrintOutResult<float>(softmaxSumShape, &resources.softmaxSumDeviceAddr);
```
**修改说明**：将 `PrintOutResult` 改为模板函数，支持按实际 tensor 数据类型读取设备数据。`attentionOutTensor` 使用 `aclFloat16`，`softmaxMaxTensor` 和 `softmaxSumTensor` 使用 `float`，确保读取数据的类型与设备端 tensor 一致。

---

## 报告生成时间
2026-03-20 15:30:00
## 报告状态
已完成检视，待修复验证
