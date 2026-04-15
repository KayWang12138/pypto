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

namespace npu::tile_fwk {
enum class MachineStatus { START = 0, FINISH = 1, STOP = 2 };

#define MAX_WRAP_TASK_NUM 3 // 最多1C2V任务
#define WRAP_IDX_AIC 0
#define WRAP_IDX_AIV0 1
#define WRAP_IDX_AIV1 2

namespace dynamic {
    class DevControlFlowCache;
}

template <class T>
struct QueueGeneric {

    QueueGeneric(uint32_t capacity, T *_elem):head(0), tail(0), elem(_elem), _capacity(capacity) {}

    QueueGeneric& operator=(const QueueGeneric &rhs) {
        head = rhs.head;
        tail = rhs.tail;
        if (capacity() == 0) {
            return *this;
        }
        // const size_t datasize = sizeof(value_type) * capacity();
        // memcpy_s(elem, datasize, rhs.elem, datasize);
        std::copy(rhs.elem, rhs.elem + rhs.size(), elem);
        return *this;
    }

	uint32_t capacity() const {
	    return _capacity;
	}

	uint32_t size() const {
	    return tail - head;
	}

	typedef T value_type;
protected:
    uint32_t head;
    uint32_t tail;
    value_type* elem;

    friend class dynamic::DevControlFlowCache;
    // Only for relocation! Do not use it for elements reading/writing!
	value_type*& relocable_elem() {
	    return elem;
	}
private:
    uint32_t _capacity;
};

template <class T>
struct LockableQueueGeneric:public QueueGeneric<T> {

    LockableQueueGeneric(uint32_t capacity, T *_elem):QueueGeneric<T>(capacity, _elem), lockFlag(0) {}

    using QueueGeneric<T>::operator=;

	void lock() {
    	while (!__sync_bool_compare_and_swap(&lockFlag, 0, 1)) {
    	}
	}

	void unlock() {
    	while (!__sync_bool_compare_and_swap(&lockFlag, 1, 0)) {
    	}
	}

	uint32_t unsafe_size() const {
	    return this->size();
	}

	void unsafe_enqueue(T x) {
	    this->elem[this->tail++] = x;
	}

	void unsafe_enqueue(T *x, uint32_t count) {
	    std::copy(x, x+count, this->elem);
	    this->tail += count;
	}
	
private:
    size_t lockFlag;

	using QueueGeneric<T>::size;
};

// aic aiv 已经ready的core function id队列
typedef LockableQueueGeneric<uint32_t> ReadyCoreFunctionQueue;
typedef QueueGeneric<uint32_t> ReadyCoreFunctionQueueUnsafe;



struct StaticReadyCoreFunctionQueue {
    uint64_t head;
    uint64_t tail;
    uint64_t* elem;
    size_t lock;
};

struct WrapInfo {
    uint32_t wrapId;
    uint32_t aicoreIdxList[MAX_WRAP_TASK_NUM]; // 顺序C、V1、V2
    uint32_t tasklist[MAX_WRAP_TASK_NUM];      // 顺序C、V1、V2
    uint8_t mixResourceType;
};

struct WrapInfoQueue {
    uint32_t head;
    uint32_t tail;
    uint32_t capacity;
    WrapInfo* elem;
    size_t lock;
    uint64_t Size() { return tail - head; }
};

inline void ReadyQueueLock(ReadyCoreFunctionQueue* rq)
{
	rq->lock();
}

inline void ReadyQueueUnLock(ReadyCoreFunctionQueue* rq)
{
	rq->unlock();
}

enum class BinDataType {
    READY_STATUS,        // CoreFunction ready_status(no need update)
    READY_AIC_CORE_FUNC, // ready aic CoreFunction id list(no need update)
    READY_AIV_CORE_FUNC, // ready aiv CoreFunction id list(no need update)
    CACHE_HEADER,        // cache header
    CCE_BIN,             // all cce bin
    TOPO,                // TOPO
    INVOKE_OFFSET_TABLE, // invoke offset
    INVODE_TENSOR_INDEX, // invoke tensor index
    INVODE_TENSOR_INFO,  // invoke tensor
    INVOKE_PARA_OFFSET,  // invoke para offset
    CORE_FUNC_WS_ADDR,   // corefunc args
    END
};
#pragma pack(8)
struct DeviceTaskBin {
    BaseArgs baseArgs;
    DeviceTask deviceTask; // initial task data
    uint64_t dataSize[static_cast<size_t>(BinDataType::END)];
    uint64_t dataOffset[static_cast<size_t>(BinDataType::END)];
    uint8_t data[0];
};
#pragma pack()

constexpr int64_t DEVICE_QUEUE_SIZE = 512;
#define DEVICE_TASK_STOP 0x7FFFFFFE

struct DeviceKernelArgs {
    int64_t* ctrlFlowCache{nullptr};
    int64_t* inputs{nullptr};
    int64_t* outputs{nullptr};
    int64_t* workspace{nullptr};
    int64_t* tilingdata{nullptr};
    int64_t* cfgdata{nullptr};
    int64_t* commContexts{nullptr};
    // following 4 paras need remove to binary
    void* costmodeldata{nullptr};
    void* aicoreModel{nullptr};
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
