/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file machine_ws_intf.h
 * \brief
 */

#ifndef MACHINE_WS_INTF_H
#define MACHINE_WS_INTF_H

#include "tilefwk/aicpu_common.h"
#include "interface/utils/common.h"
#include "tilefwk/core_func_data.h"
#include "machine/utils/concurrent_queue/concurrent_queue.h"

namespace npu::tile_fwk {
enum class MachineStatus { START = 0, FINISH = 1, STOP = 2 };

// aic aiv 已经ready的core function id队列
struct ReadyCoreFunctionQueue {
  uint32_t head;
  uint32_t tail;
  uint32_t capacity;
  uint32_t* elem;
  size_t lock;

  uint64_t Size() { return tail - head;}
};

#define TASK_LIST_MAX_SIZE 128
#define MAX_QUEUED_TASKS 4096
#define MAX_QUEUED_CORES 4096
#define MAX_QUEUED_PAIRS 4096
typedef uint64_t aicoreFunction_t;
constexpr aicoreFunction_t aicoreNullFunction = 0xFFFFFFFFFFFFFFFFUL;

class StaticReadyCoreFunctionQueue {

  public: 
  
  // The use of past tense in these functions obeys to the fact that they are not (and cannot be) concurrency-safe
  // Therefore, the return value could have changed by the time it is returned
  inline bool wasEmpty() const { return head >= tail; }
  inline size_t wasSize() const { return tail - head; }

  inline std::pair<aicoreFunction_t*, size_t> pop(aicoreFunction_t taskList[TASK_LIST_MAX_SIZE], const size_t n = 1)
  {
    lock();
    const auto curHead = head;
    size_t count = std::min(n, (size_t)(tail - head));
    head += count;
    auto taskListPtr = taskList;
    taskListPtr = &elem[curHead];
    
    // memcpy_s(taskList, TASK_LIST_MAX_SIZE * sizeof(aicoreFunction_t), taskListPtr, count * sizeof(aicoreFunction_t));

    // while (count < n)
    // {
    //   const auto taskId = _lockFreeQueue->pop();
    //   if (taskId == aicoreNullFunction) break;
    //   taskList[count++] = taskId;
    // }
    
    unlock();
    return { taskListPtr, count };
  }

  inline void push(aicoreFunction_t* const input, const size_t count = 1)
  {
    lock();
    memcpy_s(&elem[tail], count * sizeof(aicoreFunction_t), input, count * sizeof(aicoreFunction_t));
    tail += count;
    unlock();
  }


  inline void lock() {
     while (!__sync_bool_compare_and_swap(&_lock, 0, 1)) {
    }
  }

  inline void unlock() {
      while (!__sync_bool_compare_and_swap(&_lock, 1, 0)) {
    }
  }

  inline void setBuffer(aicoreFunction_t* const buffer) { elem = buffer; }
  inline void setCapacity(const size_t capacity) { _capacity = capacity; }
  inline void setCount(const size_t count) { tail = count; head = 0; }
  inline aicoreFunction_t* getBuffer() const { return elem; }

  inline void initializeLockFree()
  {
   _lockFreeQueue = new pypto::utils::ConcurrentQueue<aicoreFunction_t, aicoreNullFunction>(MAX_QUEUED_TASKS);
  }

  inline void finalizeLockFree()
  {
    delete _lockFreeQueue;
  }

  inline void push_no_lock(aicoreFunction_t* const input, const size_t count = 1)
  {
    for (size_t i = 0; i < count; i++) _lockFreeQueue->push(input[i]);
  }

  private: 

  pypto::utils::ConcurrentQueue<aicoreFunction_t, aicoreNullFunction>* _lockFreeQueue;

  ssize_t head = 0;
  ssize_t tail = 0;
  aicoreFunction_t* elem = nullptr;
  ssize_t _capacity = 0;
  size_t _lock = 0;
};

typedef uint32_t aicoreTask_t;
typedef uint32_t aicoreCore_t;
typedef uint64_t aicorePair_t;
 
constexpr aicoreTask_t aicoreNullTask = 0xFFFFFFFFUL;
constexpr aicoreCore_t aicoreNullCore = 0xFFFFFFFFUL;
constexpr aicorePair_t aicoreNullPair = 0xFFFFFFFFFFFFFFFFUL;

// Added this structure to separate concerns between the static and dynamic schedulers.
struct StaticWrapQueue {
  uint64_t head;
  uint64_t tail;
  uint64_t* elem;
  size_t lock;
};


struct WrapInfo {
    uint32_t wrapId;
    uint32_t aicCoreIdx;
    uint32_t aivCoreIdxZero;
    uint32_t aivCoreIdxOne;
    uint32_t taskCnt {0};
    uint32_t mixResourceType;
    ReadyCoreFunctionQueue tasklist;
};

struct WrapInfoQueue {
  uint32_t head;
  uint32_t tail;
  uint32_t capacity;
  WrapInfo* elem;
  size_t lock;
  uint64_t Size() { return tail - head;}
};

inline void ReadyQueueLock(ReadyCoreFunctionQueue* rq) {
  while (!__sync_bool_compare_and_swap(&rq->lock, 0, 1)) {
  }
}

inline void ReadyQueueUnLock(ReadyCoreFunctionQueue* rq) {
  while (!__sync_bool_compare_and_swap(&rq->lock, 1, 0)) {
  }
}

enum class BinDataType {
  READY_STATUS,          // CoreFunction ready_status(no need update)
  READY_AIC_CORE_FUNC,   // ready aic CoreFunction id list(no need update)
  READY_AIV_CORE_FUNC,   // ready aiv CoreFunction id list(no need update)
  CACHE_HEADER,          // cache header
  CCE_BIN,               // all cce bin
  TOPO,                  // TOPO
  INVOKE_OFFSET_TABLE,   // invoke offset
  INVODE_TENSOR_INDEX,   // invoke tensor index
  INVODE_TENSOR_INFO,    // invoke tensor
  INVOKE_PARA_OFFSET,    // invoke para offset
  CORE_FUNC_WS_ADDR,     // corefunc args
  END
};
#pragma pack (8)
struct DeviceTaskBin {
    BaseArgs baseArgs;
    DeviceTask deviceTask;        // initial task data
    uint64_t dataSize[static_cast<size_t>(BinDataType::END)];
    uint64_t dataOffset[static_cast<size_t>(BinDataType::END)];
    uint8_t data[0];
};
#pragma pack ()

constexpr int64_t DEVICE_QUEUE_SIZE = 512;
#define DEVICE_TASK_STOP 0x7FFFFFFE

struct DeviceKernelArgs {
    int64_t *ctrlFlowCache{nullptr};
    int64_t *inputs{nullptr};
    int64_t *outputs{nullptr};
    int64_t *workspace{nullptr};
    int64_t *tilingdata{nullptr};
    int64_t *cfgdata{nullptr};
    // following 4 paras need remove to binary
    void *costmodeldata{nullptr};
    void *aicoreModel{nullptr};
    uint64_t taskWastTime{0};
    uint8_t machineConfig;
    ToSubMachineConfig toSubMachineConfig;
    DeviceKernelArgsParameter parameter;
};

struct LogHead {
    int type;
    int len;
    int64_t data[];
};

} // namespace npu::tile_fwk
#endif
