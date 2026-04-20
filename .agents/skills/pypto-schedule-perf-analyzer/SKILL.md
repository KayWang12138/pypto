---
name: pypto-schedule-perf-analyzer
description: PyPTO 调度性能分析技能。用于分析 AICPU 调度时间戳、定位调度性能瓶颈、提供优化建议。支持：1)开启打点运行测试；2)分析泳道图时间点时长；3)对比不同线程差异；4)多次运行波动分析。触发词：调度性能、调度分析、schedule perf、aicpu trace、PerfMtTrace、调度打点、调度延迟、任务分发、调度时间戳、泳道图分析、线程对比、波动分析。
---

# PyPTO 调度性能分析技能

## 概述

本技能用于分析 PyPTO AICPU 调度性能，通过 PerfMtTrace 打点数据定位调度瓶颈，提供优化建议。

## 核心功能

1. **开启调度性能打点** - 配置环境变量启用性能追踪
2. **分析调度时间戳** - 解析 machine_trace_perf_data.json
3. **定位调度瓶颈** - 识别关键调度路径的性能问题
4. **提供优化建议** - 基于打点数据给出针对性优化方案

---

## PerfMtTrace 机制

### 功能说明

PerfMtTrace 是 PyPTO 框架中的 AICPU 调度时间戳打点函数，用于：
- 记录调度流程关键节点的时间戳
- 分析任务分发、依赖解析、核心同步等调度环节的性能
- 定位调度性能波动和瓶颈

### 打点类型定义

**定义位置：** `framework/src/interface/machine/device/tilefwk/aicpu_perf.h`

#### 基础打点类型

```c
PERF_TRACE_BEGIN                    // 调度开始
PERF_TRACE_ALLOC_THREAD_ID          // 分配线程ID
PERF_TRACE_INIT                     // 初始化
PERF_TRACE_CORE_HAND_SHAKE          // 核心握手
PERF_TRACE_WAIT_ALL_DEV_TASK_FINISH // 等待所有设备任务完成
PERF_TRACE_WAIT_CORE_EXIT           // 等待核心退出
PERF_TRACE_EXIT                     // 调度退出
```

#### 任务打点类型（高频打点）

```c
PERF_TRACE_DEV_TASK_BUILD           // 任务构建
PERF_TRACE_DEV_TASK_RCV             // 任务接收
PERF_TRACE_DEV_TASK_RUN_CORE_TASK   // 运行核心任务
PERF_TRACE_DEV_TASK_SEND_CALLOP_TASK // 发送任务
PERF_TRACE_DEV_TASK_DISPATCH_TASK   // 分发任务
PERF_TRACE_DEV_TASK_DISPATCH_RESOL_TASK // 分发解析任务
PERF_TRACE_DEV_TASK_DISPATCH_RESOL_REG_TASK // 分发解析寄存器任务
PERF_TRACE_DEV_TASK_ENTER_DISPATCH_TASK // 进入分发任务
PERF_TRACE_DEV_TASK_ENTER_RESOLVE_DEP_DYN // 进入动态依赖解析
PERF_TRACE_DEV_TASK_OUT_RESOLVE_DEP_DYN // 退出动态依赖解析
PERF_TRACE_DEV_TASK_SCHED_EXEC      // 调度执行
PERF_TRACE_DEV_TASK_SYNC_CORE_STOP  // 核心同步停止
PERF_TRACE_DEV_TASK_RSP             // 任务响应
```

---

## 快速开始

### 一句话开启

```bash
export DUMP_DEVICE_PERF=true && pytest python/tests/st/test_swim_line.py::test_swim -v
```

### 快速分析

```bash
# 获取日志目录
python3 -c "import pypto; print(pypto.pypto_impl.LogTopFolder())"

# 综合分析
python3 scripts/analyze_schedule_perf.py <log_dir> \
    --swimlane-details --compare-threads --multi-run --run-count 5
```

---

## 使用方法（5 步分析流程）

### 步骤 1：开启调度性能打点

**方法 1：环境变量（推荐）**
```bash
export DUMP_DEVICE_PERF=true
```

**方法 2：Python 代码**
```python
import os
os.environ["DUMP_DEVICE_PERF"] = "true"
```

**方法 3：C++ 代码**
```cpp
setenv("DUMP_DEVICE_PERF", "true", 1);
```

### 步骤 2：运行测试用例

**pytest 运行（推荐）：**
```bash
pytest python/tests/st/test_swim_line.py::test_swim -v
```

**手动运行示例：** 见 [附录 A：手动运行示例](#附录-a-手动运行示例)

### 步骤 3：定位性能数据文件

**关键文件：**

| 文件名 | 说明 |
|--------|------|
| `machine_trace_perf_data_<turn>.json` | AICPU 调度时间戳（核心文件） |
| `merged_swimlane.json` | 合并泳道图（可视化分析） |
| `tilefwk_L1_prof_data.json` | L1 性能数据 |

**查找日志目录：**
```python
import pypto
log_dir = pypto.pypto_impl.LogTopFolder()
```

### 步骤 4：分析泳道图时间点时长

**分析方法：**
```bash
python3 scripts/analyze_schedule_perf.py <log_dir> --swimlane-details
```

**阈值判定标准：** 见 [附录 B：阈值判定表](#附录-b-阈值判定表)

**细化打点方法：** 见 [附录 C：添加细化打点](#附录-c-添加细化打点)

### 步骤 5：对比不同线程差异

**分析方法：**
```bash
python3 scripts/analyze_schedule_perf.py <log_dir> --compare-threads
```

**分析维度：**
- AICPU-CTRL vs AICPU-SCHED 对比
- 不同 AICPU-SCHED 线程对比（线程 1 vs 线程 2 vs 线程 3...）

**差异阈值：** 波动率 > 30% 需细化分析，方法见 [附录 C](#附录-c-添加细化打点)

### 步骤 6：运行 5 次，分析波动

**自动多次运行：**
```bash
python3 scripts/run_multi_perf_analysis.py \
    --test-case python/tests/st/test_swim_line.py::test_swim \
    --run-count 5 --output-dir ./multi_run_logs
```

**波动阈值：**

| 波动率 | 评定 |
|--------|------|
| < 5% | 稳定 |
| 5% - 15% | 轻微波动 |
| > 15% | 波动较大，需细化分析 |

**波动分析方法：** 见 [附录 D：波动分析深入](#附录-d-波动分析深入)

---

## 调度性能分析流程

### 关键指标计算

| 指标名称 | 计算公式 | 说明 |
|---------|---------|------|
| 调度延迟 | `DEV_TASK_RCV - DEV_TASK_BUILD` | 任务从构建到接收的延迟 |
| 任务分发耗时 | `DEV_TASK_DISPATCH_TASK - DEV_TASK_RUN_CORE_TASK` | 任务分发总耗时 |
| 依赖解析耗时 | `DEV_TASK_DISPATCH_RESOL_TASK - DEV_TASK_ENTER_DISPATCH_TASK` | 依赖关系解析耗时 |
| 任务发送耗时 | `DEV_TASK_SEND_CALLOP_TASK - DEV_TASK_DISPATCH_TASK` | 任务发送耗时 |

### 瓶颈识别

**瓶颈类型判定：** 见 [附录 B](#附录-b-阈值判定表)

### 多轮调度分析

对于多轮调度场景：
- 分析不同 turn 的调度时间戳差异
- 识别调度性能波动
- 定位性能异常轮次

---

## 性能优化建议

### 优化建议索引

| 症状 | 关键打点 | 阈值 | 优化方案链接 |
|------|---------|------|-------------|
| 任务构建慢 | DEV_TASK_BUILD | > 100us | [方案 A](#优化方案-a任务构建慢) |
| 任务接收慢 | DEV_TASK_RCV | > 50us | [方案 B](#优化方案-b任务接收慢) |
| 依赖解析慢 | ENTER_DISPATCH → DISPATCH_RESOL | > 200us | [方案 C](#优化方案-c依赖解析慢) |
| 任务分发慢 | DEV_TASK_DISPATCH | > 100us | [方案 D](#优化方案-d任务分发慢) |
| 核心同步慢 | SYNC_CORE_STOP | > 150us | [方案 E](#优化方案-e核心同步慢) |

### 优化方案 A：任务构建慢

**症状：** DEV_TASK_BUILD 耗时 > 100us

**优化代码：**
```python
# 1. 合并小任务
pypto.set_pass_options(cube_nbuffer_setting={0: 8})

# 2. 启用 Stitch 合并
@pypto.frontend.jit(runtime_options={"stitch_function_max_num": 128})
```

### 优化方案 B：任务接收慢

**症状：** DEV_TASK_RCV 耗时 > 50us

**优化方案：**
- 检查任务构建复杂度
- 合并小任务减少构建开销
- 使用 cube_nbuffer_setting 合并同构子图

### 优化方案 C：依赖解析慢

**症状：** 依赖解析耗时 > 200us

**优化代码：**
```python
# 1. 使用 loop_unroll 减少动态依赖
for idx in pypto.loop(count, unroll_list=[8, 4, 2, 1]):
    # ...

# 2. 优化任务调度模式
@pypto.jit(runtime_options={"device_sched_mode": 1})
```

**代码层面优化建议：**
- 减少依赖层级，使用 `loop_unroll` 展开循环
- 优化任务调度策略，降低动态依赖复杂度
- 参考 [附录 C](#附录-c添加细化打点) 添加细化打点定位瓶颈

### 优化方案 D：任务分发慢

**症状：** DEV_TASK_DISPATCH 耗时 > 100us

**优化代码：**
```python
# 增大 tile size
pypto.set_cube_tile_shapes([128, 128], [128, 512], [128, 128])
```

### 优化方案 E：核心同步慢

**症状：** DEV_TASK_SYNC_CORE_STOP 耗时 > 150us

**优化代码：**
```python
# 优化任务依赖关系
pypto.set_pass_options(sg_set_scope=1)
```

---

## 完整分析流程示例

### 场景：分析调度性能波动

**问题描述：** 算子在多次运行时性能波动较大，需要定位调度环节的性能瓶颈。

**分析流程（引用各章节）：**

1. **开启打点并运行** → 见 [步骤 1-2](#步骤-1开启调度性能打点)

2. **查看泳道图时间点时长** → 见 [步骤 4](#步骤-4分析泳道图时间点时长)
   
   输出示例：
   ```
   **耗时较大的打点路径（需细化分析）：**
   - ALLOC_THREAD_ID → DEV_TASK_BUILD: 112.50 us (阈值: 50 us)
   ```

3. **针对时长较大的打点细化** → 见 [附录 C](#附录-c-添加细化打点)

4. **对比不同线程** → 见 [步骤 5](#步骤-5对比不同线程差异)
   
   输出示例：
   ```
   ⚠️ 波动率超过 30%，需要细化分析！
   线程 2 的依赖解析耗时比线程 1 高 2.3 倍
   ```

5. **运行 5 次，分析波动** → 见 [步骤 6](#步骤-6运行-5-次分析波动)
   
   输出示例：
   ```
   波动率: 44.15% (> 15%)
   Run 2 的依赖解析耗时 280 us（最大）
   Run 4 的依赖解析耗时 90 us（最小）
   ```

6. **针对波动最大的一次深入分析** → 见 [附录 D](#附录-d-波动分析深入)

---

## 附录

### 附录 A：手动运行示例

```python
import os
import torch
import torch_npu
import pypto

os.environ["DUMP_DEVICE_PERF"] = "true"
device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
torch.npu.set_device(device_id)

@pypto.frontend.jit(debug_options=dict(runtime_debug_mode=1))
def matmul_add(a, b, c, out):
    tiling = 32
    n, k, m = tiling * 8, tiling * 8, tiling * 8
    pypto.set_vec_tile_shapes(tiling, tiling)
    pypto.set_cube_tile_shapes([tiling, tiling], [tiling, tiling], [tiling, tiling])
    for _ in pypto.loop(1, name="s0", idx_name="i"):
        a0 = pypto.view(a, [n, k], [0, 0])
        b0 = pypto.view(b, [k, m], [0, 0])
        out.move(pypto.add(pypto.matmul(a0, b0, pypto.DT_INT32), c))

a_data = torch.ones((256, 256), dtype=torch.int8, device=f'npu:{device_id}')
b_data = torch.ones((256, 256), dtype=torch.int8, device=f'npu:{device_id}')
c_data = torch.zeros((256, 256), dtype=torch.int32, device=f'npu:{device_id}')
d_data = torch.zeros((256, 256), dtype=torch.int32, device=f'npu:{device_id}')

for i in range(5):
    matmul_add(a_data, b_data, c_data, d_data)

torch_npu.npu.synchronize()
log_dir = pypto.pypto_impl.LogTopFolder()
print(f"性能数据保存在: {log_dir}")
```

### 附录 B：阈值判定表

#### 打点路径阈值

| 打点路径 | 阈值 | 超标判定 |
|---------|------|---------|
| BEGIN → ALLOC_THREAD_ID | 10us | > 10us 为慢速分配 |
| ALLOC_THREAD_ID → DEV_TASK_BUILD | 50us | > 50us 为构建延迟 |
| DEV_TASK_BUILD → DEV_TASK_RCV | 50us | > 50us 为接收延迟 |
| DEV_TASK_ENTER_DISPATCH → DEV_TASK_DISPATCH_RESOL | 200us | > 200us 为解析瓶颈 |
| DEV_TASK_DISPATCH → DEV_TASK_SEND_CALLOP | 50us | > 50us 为发送延迟 |

#### 瓶颈类型阈值

| 瓶颈类型 | 关键打点 | 阈值 |
|---------|---------|------|
| 任务构建慢 | DEV_TASK_BUILD | > 100us |
| 任务接收慢 | DEV_TASK_RCV | > 50us |
| 依赖解析慢 | ENTER_DISPATCH → DISPATCH_RESOL | > 200us |
| 任务分发慢 | DEV_TASK_DISPATCH | > 100us |
| 核心同步慢 | SYNC_CORE_STOP | > 150us |

#### 线程差异阈值

| 波动率 | 评定 |
|--------|------|
| < 10% | 线程负载均衡良好 |
| 10% - 30% | 线程负载轻微不均衡 |
| > 30% | 线程负载严重不均衡，需细化分析 |

#### 多次运行波动阈值

| 波动率 | 评定 |
|--------|------|
| < 5% | 稳定 |
| 5% - 15% | 轻微波动 |
| > 15% | 波动较大，需细化分析 |

### 附录 C：添加细化打点

当某个打点路径耗时超标时，需要在该路径内部添加更细化的打点。

#### 核心问题：当前库上代码局限性

**现状：** 当前库上代码只记录关键流程的第一次执行时间戳，无法反映：
- 多轮迭代中不同任务的耗时差异
- 循环内部单次操作的详细耗时
- 动态依赖解析、任务下发等高频操作的累计耗时

**解决思路：** 针对耗时较高的环节（如接依赖、任务下发），继续添加细化打点，记录每次操作的时间戳。

---

#### 打点细化策略总览

| 耗时环节 | 细化方向 | 参考函数 | 建议打点 |
|---------|---------|---------|---------|
| 任务分发 | 分解构建→分发→发送三步 | `DispatchAiCoreTask` | RUN_CORE_TASK, DISPATCH_TASK, SEND_CALLOP_TASK |
| 依赖解析 | 分解进入→单核解析→退出三步 | `ResolveDepForAllAiCore` | ENTER_DISPATCH, RESOL_REG, RESOL_TASK |
| 动态依赖解析 | 记录进入和退出时间 | `ResolveDepDyn` | ENTER_RESOLVE_DEP_DYN, OUT_RESOLVE_DEP_DYN |
| 循环内高频操作 | 在循环前后加打点 | 各循环函数 | 循环进入打点 + 循环退出打点 |

---

#### 步骤 1：识别耗时超标环节

**分析方法：** 使用 `analyze_schedule_perf.py` 的 `--swimlane-details` 参数：

```bash
python3 scripts/analyze_schedule_perf.py <log_dir> --swimlane-details
```

**输出示例：**
```
**耗时较大的打点路径（需细化分析）：**
- ALLOC_THREAD_ID → DEV_TASK_BUILD: 112.50 us (阈值: 50 us)
- DEV_TASK_ENTER_DISPATCH_TASK → DEV_TASK_DISPATCH_RESOL_TASK: 280.30 us (阈值: 200 us)
```

**识别策略：**
- 阈值超标：耗时 > 阈值（见附录 B）
- 高频操作：打点类型带 `(N)` 后缀（如 `DEV_TASK_BUILD(5)` 表示第 5 次执行）
- 波动明显：多次运行波动率 > 15%

---

#### 步骤 2：定位关键函数

根据超标打点路径，定位到具体函数：

| 超标路径 | 关键函数 | 文件位置 |
|---------|---------|---------|
| 任务构建慢 | `DispatchAiCoreTask` | `aicore_manager.h` |
| 依赖解析慢 | `ResolveDepForAllAiCore` | `aicore_manager.h` |
| 动态依赖慢 | `ResolveDepDyn` | `aicore_manager.h` |
| 核心同步慢 | `WaitAiCoreExit` | `aicore_manager.h` |

---

#### 步骤 3：定义新打点类型

**文件位置：** `framework/include/interface/machine/device/tilefwk/aicpu_perf.h`

**添加方式：**
```c
#define PERF_TRACES                             \
    X(BEGIN)                                    \
    X(ALLOC_THREAD_ID)                          \
    ...                                         \
    XDEVTASK(DEV_TASK_NEW_TRACE_POINT)         \
    X(MAX)
```

**命名规范：**
- 使用 `DEV_TASK_` 前缀标识任务相关打点
- 使用 `ENTER_` / `OUT_` 标识流程边界
- 使用 `_START` / `_END` 标识循环内单次操作

---

#### 步骤 4：添加打点代码

**通用模式：**
```cpp
// 函数入口打点
PerfMtTrace(PERF_TRACE_DEV_TASK_ENTER, aicpuIdx_);

// 关键逻辑步骤 1
PerfMtTrace(PERF_TRACE_DEV_TASK_STEP1, aicpuIdx_);

// 关键逻辑步骤 2
PerfMtTrace(PERF_TRACE_DEV_TASK_STEP2, aicpuIdx_);

// 循环内操作（高频）
for (auto coreIdx : activeCoreIdx_) {
    PerfMtTrace(PERF_TRACE_DEV_TASK_LOOP_ITEM_START, aicpuIdx_);
    // 单次循环操作
    PerfMtTrace(PERF_TRACE_DEV_TASK_LOOP_ITEM_END, aicpuIdx_);
}

// 函数出口打点
PerfMtTrace(PERF_TRACE_DEV_TASK_EXIT, aicpuIdx_);
```

---

#### 步骤 5：性能优化配合

**引入 activeCoreIdx_ 优化（避免遍历所有核心）：**

```cpp
// 在类成员中添加
std::vector<int> activeCoreIdx_;

// 在任务发送时记录活跃核心
pendingIds_[coreIdx] = newTask;
activeCoreIdx_.push_back(coreIdx);

// 在依赖解析时只遍历活跃核心（优化性能）
for (auto coreIdx : activeCoreIdx_) {
    ret = ResolveByRegVal(type, coreIdx);
    // 添加单核解析打点
    PerfMtTrace(PERF_TRACE_DEV_TASK_RESOL_SINGLE_END, aicpuIdx_);
}

// 在任务完成时清理非活跃核心
if (runningIds_[coreIdx] == AICORE_TASK_INIT) {
    auto it = std::find(activeCoreIdx_.begin(), activeCoreIdx_.end(), coreIdx);
    if (it != activeCoreIdx_.end()) {
        activeCoreIdx_.erase(it);
    }
}
```

---

#### 步骤 6：增大数组容量

**问题：** 高频打点可能导致数组溢出

**解决：** 在 `aicpu_perf.h` 中调整容量：

```c
// 默认值：20
// 建议：依赖解析等高频场景提升到 35-50
inline constexpr uint32_t PERF_TRACE_COUNT_DEVTASK_MAX_NUM = 50;
```

---

#### 步骤 7：验证打点效果

```bash
export DUMP_DEVICE_PERF=true
pytest python/tests/st/test_swim_line.py::test_swim -v

# 检查新打点是否出现在数据中
python3 scripts/analyze_schedule_perf.py <log_dir> --swimlane-details

# 对比优化前后性能
python3 scripts/run_multi_perf_analysis.py --run-count 5
```

---

#### 细化示例 A：任务分发耗时过大

**场景：** `DEV_TASK_RUN_CORE_TASK → DEV_TASK_DISPATCH_TASK` 耗时 > 100us

**细化策略：**
```cpp
// aicore_manager.h DispatchAiCoreTask
inline int32_t DispatchAiCoreTask(CoreType type, ReadyCoreFunctionQueue* readyQue,
                                   int coreIdxStart, int coreIdxEnd) {
    // 打点：开始运行核心任务
    PerfMtTrace(PERF_TRACE_DEV_TASK_RUN_CORE_TASK, aicpuIdx_);
    
    int32_t ret = DEVICE_MACHINE_OK;
    if (context_->waitTaskCnt_[static_cast<int>(type)] > 0) {
        ret = ResolveDepForAllAiCore(type, coreIdxStart, coreIdxEnd);
        // ...
    }
    
    // 打点：分发任务完成
    PerfMtTrace(PERF_TRACE_DEV_TASK_DISPATCH_TASK, aicpuIdx_);
    
    // 任务发送逻辑...
    
    // 打点：任务发送完成
    PerfMtTrace(PERF_TRACE_DEV_TASK_SEND_CALLOP_TASK, aicpuIdx_);
    
    return ret;
}
```

**分析收益：** 可识别耗时集中在哪个阶段（解析依赖、任务分发、任务发送）

---

#### 细化示例 B：依赖解析耗时过大

**场景：** `DEV_TASK_ENTER_DISPATCH_TASK → DEV_TASK_DISPATCH_RESOL_TASK` 耗时 > 200us

**细化策略：**
```cpp
// aicore_manager.h ResolveDepForAllAiCore
inline int32_t ResolveDepForAllAiCore(CoreType type, int coreIdxStart, int coreIdxEnd) {
    // 打点：进入依赖解析流程
    PerfMtTrace(PERF_TRACE_DEV_TASK_ENTER_DISPATCH_TASK, aicpuIdx_);
    
    int32_t ret = DEVICE_MACHINE_OK;
    PerfMtBegin(static_cast<int>(PERF_EVT_RESOLVE_DEPENDENCE), aicpuIdx_);
    
    // 使用 activeCoreIdx_ 优化遍历
    for (auto coreIdx : activeCoreIdx_) {
        if (coreIdx >= coreIdxStart && coreIdx < coreIdxEnd) {
            ret = ResolveByRegVal(type, coreIdx);
            // 打点：单核解析寄存器完成
            PerfMtTrace(PERF_TRACE_DEV_TASK_DISPATCH_RESOL_REG_TASK, aicpuIdx_);
            
            if (unlikely(ret != DEVICE_MACHINE_OK)) {
                return ret;
            }
        }
    }
    
    // 打点：所有核心依赖解析完成
    PerfMtTrace(PERF_TRACE_DEV_TASK_DISPATCH_RESOL_TASK, aicpuIdx_);
    
    ret = BatchPushReadyQueue();
    return ret;
}
```

**分析收益：**
- 可识别单核解析耗时（RESOL_REG 打点间隔）
- 可计算核心数量影响（activeCoreIdx_ 长度）
- 可定位异常核心（某个 coreIdx 解析耗时异常高）

---

#### 细化示例 C：动态依赖解析耗时过大

**场景：** 动态依赖解析环节耗时异常

**细化策略：**
```cpp
// aicore_manager.h ResolveDepDyn
inline int32_t ResolveDepDyn(uint64_t finishId, size_t resolveIndexBase = 0, int coreIdx = 0) {
    // 打点：进入动态依赖解析
    PerfMtTrace(PERF_TRACE_DEV_TASK_ENTER_RESOLVE_DEP_DYN, aicpuIdx_);
    
    int32_t ret = DEVICE_MACHINE_OK;
    auto dyntask = reinterpret_cast<DynDeviceTask *>(curDevTask_);
    auto funcId = FuncID(finishId);
    
    // 动态依赖解析逻辑...
    for (size_t i = 0; i < dyntask->outTensorNum; i++) {
        // 可在此处添加更细化的打点（如有需要）
    }
    
    // 打点：退出动态依赖解析
    PerfMtTrace(PERF_TRACE_DEV_TASK_OUT_RESOLVE_DEP_DYN, aicpuIdx_);
    
    ret = ResolveDynStitched(dyntask, funcId, opIndex, coreIdx);
    return ret;
}
```

---

#### 细化策略总结

| 优化目标 | 策略 | 打点类型 | 配合优化 |
|---------|-----|---------|---------|
| 识别整体耗时 | 函数入口+出口打点 | ENTER/EXIT | 无 |
| 识别阶段耗时 | 关键步骤打点 | STEP1/STEP2/STEP3 | 无 |
| 识别循环耗时 | 循环内单次操作打点 | LOOP_ITEM_START/END | 无 |
| 识别高频操作 | 每次执行打点 | 带 `(N)` 后缀 | 增大数组容量 |
| 优化遍历性能 | 只遍历活跃核心 | 无（性能优化） | 引入 activeCoreIdx_ |

**后续细化方向：**
- 遇到接依赖耗时过大：参考示例 B
- 遇到任务下发耗时过大：参考示例 A
- 遇到动态依赖耗时过大：参考示例 C
- 遇到核心数量波动：配合 activeCoreIdx_ 优化

### 附录 D：波动分析深入

当多次运行波动率 > 15% 时，需要深入分析。

#### 1. 对比最大最小耗时运行

```bash
# 分析 Run 2（最大耗时）
python3 scripts/analyze_schedule_perf.py ./multi_run_logs/run_2 --swimlane-details

# 对比 Run 2 和 Run 4
python3 scripts/analyze_schedule_perf.py ./multi_run_logs/run_2 --compare-threads
python3 scripts/analyze_schedule_perf.py ./multi_run_logs/run_4 --compare-threads
```

#### 2. 分析波动原因

可能原因：
- **任务数量波动**：不同运行的任务数差异
- **核心负载波动**：activeCoreIdx_ 数量变化
- **依赖解析复杂度波动**：寄存器读取频率变化
- **任务队列长度波动**：队列积压程度变化

#### 3. 针对性优化

**分析发现：** 不同运行的核心负载差异明显

**优化策略：**

1. **添加核心负载打点**
   - 在任务发送时记录活跃核心数量
   - 在依赖解析时打点记录当前活跃核心列表长度
   
   参考 [附录 C](#附录-c添加细化打点) 步骤 5，引入核心活跃度追踪机制

2. **优化核心管理**
   - 及时清理已完成任务的核心
   - 避免遍历所有核心，只处理活跃核心
   - 参考附录 C 中 `activeCoreIdx_` 优化策略

3. **优化任务调度策略**
   ```python
   @pypto.jit(runtime_options={"device_sched_mode": 1})
   ```

---

## 分析脚本汇总

### analyze_schedule_perf.py

**基础用法：**
```bash
python3 scripts/analyze_schedule_perf.py <log_dir> [--turn 0]
```

**参数说明：**

| 参数 | 说明 |
|------|------|
| `--turn` | 分析轮次编号（默认: 0） |
| `--output` | 输出报告文件路径 |
| `--swimlane-details` | 详细分析泳道图时间点 |
| `--compare-threads` | 对比不同线程性能 |
| `--multi-run` | 多次运行波动分析 |
| `--run-count` | 多次运行的次数（默认: 5） |

**综合分析示例：**
```bash
python3 scripts/analyze_schedule_perf.py /path/to/logs \
    --swimlane-details \
    --compare-threads \
    --multi-run --run-count 5 \
    --output comprehensive_report.md
```

### run_multi_perf_analysis.py

**自动多次运行：**
```bash
python3 scripts/run_multi_perf_analysis.py \
    --test-case python/tests/st/test_swim_line.py::test_swim \
    --run-count 5 \
    --output-dir ./multi_run_logs \
    --report-output波动分析_report.md
```

---

## 可视化分析

**Perfetto 可视化：** 上传 `merged_swimlane.json` 到 https://ui.perfetto.dev/

**泳道图解读：**

| 行名 | 说明 |
|------|------|
| AICPU-CTRL | 控制 CPU 调度时间线 |
| AICPU-SCHED | 调度 CPU 调度时间线 |
| AIC/AIV | AI Core 执行时间线 |

**重点关注：**
- 任务构建到分发的延迟
- 依赖解析的耗时分布
- 核心同步的等待时间

---

## 参考资料

- [aicore_manager.h](../../framework/src/machine/device/dynamic/aicore_manager.h) - 核心调度管理
- [device_perf.h](../../framework/src/machine/device/dynamic/device_perf.h) - 性能打点实现
- [aicpu_perf.h](../../framework/src/interface/machine/device/tilefwk/aicpu_perf.h) - 打点类型定义
- [dump_device_perf.cpp](../../framework/src/machine/runtime/dump_device_perf.cpp) - 性能数据导出
- [test_swim_line.py](../../python/tests/st/test_swim_line.py) - 调度性能测试示例
- [pypto-op-perf-analyzer](../pypto-op-perf-analyzer/SKILL.md) - 算子性能分析技能（可配合使用）