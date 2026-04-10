# 通信算子 shmem_wait_until 超时定位指导
本教程旨在为用户提供一套分析、定位通信算子 shmem_wait_until 超时问题，助力用户在使用 PyPTO 进行通信算子开发时能快速定位 shmem_wait_until 超时问题。

## 背景介绍

目前主流的模型训练/推理过程中都会通过集合通信库 (CCL) 来实现多卡环境下的数据交换以减少算子的训练/推理时间。下面通过通过该示例介绍超时报错的大致步骤，完整样例请参考：[glm_matmul_allreduce_add_rmsnorm.py](../../../models/glm_v4_5/glm_matmul_allreduce_add_rmsnorm.py)。

本文只关注 allreduce 算子相关代码段，即:
```python
pypto.set_vec_tile_shapes(view_row_shape, hidden_size)
for dyn_idx in range(world_size):
    put_out = pypto.distributed.shmem_put(matmul_result, [0, 0], shmem_tensor, dyn_idx,
        put_op=pypto.AtomicType.ADD, pred=[barrier_out])
    pypto.distributed.shmem_signal(shmem_tensor, dyn_idx, 1, shmem_shape,
        [0, 0], target_pe=dyn_idx, sig_op=pypto.AtomicType.ADD, pred=[put_out])
wait_until_out = pypto.distributed.shmem_wait_until(shmem_tensor, my_pe, world_size,
    shmem_shape, [0, 0], cmp=pypto.OpType.EQ, clear_signal=True, pred=[in_tensor_tile])
pypto.set_vec_tile_shapes(1, hidden_size)
all_reduce_out = pypto.experimental.shmem_load(
    shmem_tensor, my_pe, shmem_shape, [0, 0], pred=[wait_until_out], valid_shape=shmem_shape
)
```
其中：
- **shmem_put**：将当前 rank 对应的数据写入远端 rank 对应的数据缓冲区内存；
- **shmem_signal**：通知远端 rank 当前 rank 已经完成其数据缓冲区内存的写入，并在远端 rank 的信号缓冲区中写入指定值；
- **shmem_wait_until**：等待所有远端 rank 完成当前 rank 的数据缓冲区内存写入操作，即判断当前 rank 的共享信号缓冲区中是否为指定值；
- **shmem_load**：从某个 rank 上读取数据缓冲区的内存，并加载到当前 rank 的内存中。

## 问题定位

假设在NPU上板运行时，glm_matmul_allreduce_add_rmsnorm 测试用例运行失败并报错，此时需要进行调试，定位问题。

### 报错
运行通信测试用例， 出现报错 rtDeviceSynchronizeWithTimeout 字眼可以初步判断为 shmem_wait_until 超时，即当前 rank 的共享信号缓冲区值不为指定值。

### 定位步骤
1. 打开 device 侧日志开关
    ```bash
    export ASCEND_GLOBAL_LOG_LEVEL=0/(1/2/3) 分别对应【debug /info /warning /error】日志
    # 重定向日志落盘路径, 默认路径 $HOME/ascend/log
    export ASCEND_PROCESS_LOG_PATH={YOUR_PATH}
    ```
2. 确定 shmem_wait_until 超时

    在 [shmem_wait_until.cpp](../../../framework/src/machine/device/distributed/shmem_wait_until.cpp) 文件 SignalTileOp::PollCompleted() 函数第一行增加打印日志:
    ```cpp
    DEV_ERROR(DistributedErrorCode::NULLPTR, "expectedSum_=%d, addr_[0]=%d, resolve taskId=%lu", expectedSum_, addr_[0], taskId);
    ```
    出现海量日志打印 expectedSum_ 的值和 addr_ 不一致，同时日志打印 RunTask Errcode: F72001! "#sche.dtask.leave: Aicpu[x] proc finish: finishedFunctionCnt=x, coreFunctionCnt=x, taskId=x, but timeout!.", 可以确认为 shmem_wait_until 超时。其中 finishedFunctionCnt 表示已完成task数量， coreFunctionCnt表示总task数量。

## 解决方案
1. 增加定位日志，查看在 shmem_wait_until 任务完成几个

    修改 [aicore_manager.h](../../../framework/src/machine/device/dynamic/aicore_manager.h) 相关如下修改代码：
    ```cpp
    diff --git a/framework/src/machine/device/dynamic/aicore_manager.h b/framework/src/machine/device/dynamic/aicore_manager.h
    index d7687786..467b055d 100644
    --- a/framework/src/machine/device/dynamic/aicore_manager.h
    +++ b/framework/src/machine/device/dynamic/aicore_manager.h
    @@ -1018,7 +1018,7 @@ private:
                        "newTask[%lu][%lx] is not mix task, but core[%d] is not available!", newTask, newTask, coreIdx);
                }
            }
    -        DEV_VERBOSE_DEBUG("Send task %lu, at core %d ,type:%d.", newTask, coreIdx, static_cast<int>(type));
    +        DEV_ERROR(SchedErr::TASK_WAIT_TIMEOUT, "SendTaskToAicore task %lu, at core %d ,type:%d.", newTask, coreIdx, static_cast<int>(type));
        }

        inline void SetAiCpuStat(int coreIdx, uint64_t taskId)
    @@ -1214,9 +1214,10 @@ private:
            [[maybe_unused]] uint32_t aicpuCallCode = finTaskRegVal >> 32;
            uint32_t finTaskId = REG_LOW_TASK_ID(finTaskRegVal);
            uint32_t finTaskState = REG_LOW_TASK_STATE(finTaskRegVal);
    -        DEV_VERBOSE_DEBUG(
    -            "reslove task core index: %d, finishtaskid:%x, finishstate: %u.", coreIdx, finTaskId, finTaskState);
    -
    +        if (finTaskState == TASK_FIN_STATE) {
    +            DEV_VERBOSE_DEBUG(SchedErr::TASK_WAIT_TIMEOUT,
    +            "ReSolveByRegVal task core index: %d, finishtaskid:%u, finishstate: %u.", coreIdx, finTaskId, finTaskState);
    +        }
    #if SCHEDULE_USE_PENDING_AND_RUNING_SWITCH
            auto& pendingIdRef = pendingIds_[coreIdx];
            auto& pendingResolveIndexBaseRef = pendingResolveIndexList_[coreIdx];
    @@ -1326,7 +1327,10 @@ private:
            return ret;
        }

    -    inline void PushAicpuTaskQueue(uint64_t taskId) { PushReadyQue(readyAicpuFunctionQue_, &taskId, 1); }
    +    inline void PushAicpuTaskQueue(uint64_t taskId) {
    +        DEV_ERROR(SchedErr::TASK_WAIT_TIMEOUT, "PushAicpuTaskQueue task %lu,", taskId);
    +        PushReadyQue(readyAicpuFunctionQue_, &taskId, 1);
    +    }

        inline bool TrySendTaskDirectly(int coreType, uint32_t taskId)
        {
    ```
    其中：
    （1）SendTaskToAicore 表示 task 下发到 aicore, taskId及前后依赖关系见 dyn_topo.txt (dyn_topo.txt 生成请参考：[machine.md](../../trouble_shooting/machine.md) leaf function 粒度的内存重叠检测章节)。
    （2）ReSolveByRegVal 表示下发到 aicore task 已完成。
    （3）PushAicpuTaskQueue 表示aicpu task 任务解依赖完成, 此用例 aicpu task 为 shmem_wait_until task。
    通过以上日志 以及 本文前面增加的日志 可以判断当前哪些 shmem_wait_until task未完成，方便判断超时原因。

2. 检查共享信号缓冲区，即 signal 地址是否正确。
    - tileop_shmem.h 文件 ShmemSignal() 增加日志， 请参考：[machine.md](../../trouble_shooting/machine.md) **启用追踪日志章节** 。
    - 查看 signal 写入共享信号缓冲区地址是否符合预期：和 shmem_wait_until 对应的共享信号缓冲区地址对比: [shmem_wait_until.cpp](../../../framework/src/machine/device/distributed/shmem_wait_until.cpp) SignalTileOp::PollCompleted() 函数处打印的地址对比。
    ```cpp
    // 头文件增加依赖
    #include "tilefwk/aicore_print.h"
    // ShmemSignal增加打印日志，查看地址是否合理
    AiCoreLogF(param->ctx, "value = %p ", shmemSignalAddr);
    ```
    - 对比 host 侧 和 device 侧地址是否一致：device 侧在[shmem_wait_until.cpp](../../../framework/src/machine/device/distributed/shmem_wait_until.cpp) ShmemWaitUntilImpl::GetRawAddr() 函数增加日志，打印 device 侧解析的共享信号缓冲区地址；host 侧增加的日志在 [distributed_context.cpp](../../../framework/src/machine/runtime/distributed/distributed_context.cpp) AllocCommContext() 函数内，不同的芯片对应不同的特化模板函数。
    ```cpp
    // shmem_wait_until.cpp
    for (uint64_t i = 0; i < ctxHost->rankNum * 3; ++i) {
        DEV_ERROR(DevCommonErr::PARAM_CHECK_FAILED, "#ctrl.unknown: shmemAddrEndOffset = : %lu", hcclOpParam->winAddr[i]);
    }
    ```

    ```cpp
    // distributed_context.cpp
    for (uint64_t i = 0; i < ctxHost->rankNum * 3; ++i) {
        std::cout << "winAddr: "<< ctxHost->winAddr[i];
    }
    ```

3. 检查写共享信号缓冲区的信号值是否符合预期。
    - 前端 shmem_signal接口 signal 参数是否预期。
    - 检查 output/output_xxxx/kernel_aicore/ 目录中生成的 cce (cpp后缀) 的值是否符合预期，不符合预期则需要定位 shmem_expand_function.cpp 和 codegen_distributed.cpp 相关逻辑是否正确
    - tileop_shmem.h 文件 ShmemSignal() 增加日志，开启打印方式见**本章节第2点**，查看信号值和 atomicType 是否符合预期。
    ```cpp
    // ShmemSignal增加打印日志，查看是否符合预期
    AiCoreLogF(param->ctx, "value = %d atomicType = %d", value, atomicType);
    ```
