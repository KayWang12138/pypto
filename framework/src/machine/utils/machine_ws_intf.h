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
#include "tilefwk/error_code.h"
#include "machine/utils/device_log.h"
#include <string>
#include <sstream>
#include <mutex>

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
    QueueGeneric(uint32_t capacity, T* _elem) : head(0), tail(0), elem(_elem), _capacity(capacity) {}

    QueueGeneric& operator=(const QueueGeneric& rhs)
    {
        head = 0;
        tail = rhs.size();
        if (capacity() == 0) {
            return *this;
        }
        ASSERT(ProgEncodeErr::RANGE_VERIFY_FAILED, rhs.size() <= capacity());
        std::copy(rhs.elem + rhs.head, rhs.elem + rhs.tail, elem);
        return *this;
    }

    __attribute__((always_inline)) uint32_t capacity() const { return _capacity; }

    __attribute__((always_inline)) uint32_t size() const { return tail - head; }

    std::string str() const
    {
        std::stringstream ss;
        ss << "Queue at " << this << " head=" << head << " tail=" << tail << " capacity=" << capacity();
        return ss.str();
    }

    std::string dump() const
    {
        std::stringstream ss;
        for (value_type* it = elem + head; it != elem + tail; ++it) {
            ss << *it << " ";
        }
        return ss.str();
    }

    const T* begin() const { return elem + head; }

    const T* end() const { return elem + tail; }

    typedef T value_type;

protected:
    uint32_t head;
    uint32_t tail;
    value_type* elem;

    friend class dynamic::DevControlFlowCache;
    // Only for relocation! Do not use it for elements reading/writing!
    value_type*& relocable_elem() { return elem; }

private:
    uint32_t _capacity;
};

template <class T>
struct LockableQueueGeneric : public QueueGeneric<T> {
    using datarange = std::pair<const T*, const T*>;
    using QueueGeneric<T>::operator=;

    LockableQueueGeneric(uint32_t capacity = 0, T* _elem = nullptr) : QueueGeneric<T>(capacity, _elem), lockFlag(0) {}

    __attribute__((always_inline)) inline void lock()
    {
        while (!__sync_bool_compare_and_swap(&lockFlag, 0, 1)) {
        }
    }

    __attribute__((always_inline)) inline void unlock()
    {
        while (!__sync_bool_compare_and_swap(&lockFlag, 1, 0)) {
        }
    }

    __attribute__((always_inline)) inline uint32_t unsafe_size() const
    {
        return __atomic_load_n(&this->tail, __ATOMIC_RELAXED) - __atomic_load_n(&this->head, __ATOMIC_RELAXED);
    }

    __attribute__((always_inline)) inline void unsafe_enqueue(T x)
    {
        const uint32_t t = __atomic_fetch_add(&this->tail, 1, std::memory_order_release);
        ASSERT(ProgEncodeErr::RANGE_VERIFY_FAILED, t < this->capacity());
        this->elem[t] = x;
    }

    __attribute__((always_inline)) inline void unsafe_enqueue(T* x, uint32_t count)
    {
        const uint32_t t = __atomic_fetch_add(&this->tail, count, std::memory_order_release);
        // Faster analog of std::copy(x, x + count, this->elem + t);
        errno_t err = memcpy_s(this->elem + t, sizeof(T) * (this->capacity() - t), x, sizeof(T) * count);
        ASSERT(ProgEncodeErr::RANGE_VERIFY_FAILED, err == 0);
    }

    __attribute__((always_inline)) inline bool try_enqueue(T x)
    {
        std::scoped_lock slock(*this);
        uint32_t t = __atomic_fetch_add(&this->tail, 1, std::memory_order_release);
        if (unlikely(t >= this->capacity())) {
            __atomic_store_n(&this->tail, t, __ATOMIC_RELAXED);
            return false;
        }
        this->elem[t] = x;
        return true;
    }

    __attribute__((always_inline)) inline bool try_enqueue(const T* x, uint32_t count)
    {
        std::scoped_lock slock(*this);
        uint32_t t = __atomic_fetch_add(&this->tail, count, std::memory_order_release);
        if (unlikely(t + count > this->capacity())) {
            __atomic_store_n(&this->tail, t, __ATOMIC_RELAXED);
            return false;
        }
        // Faster analog of std::copy(x, x + count, this->elem + t);
        memcpy(this->elem + t, x, sizeof(T) * count);
        return true;
    }

    __attribute__((always_inline)) inline std::pair<const T*, const T*> dequeue_all()
    {
        std::scoped_lock slock(*this);
        uint32_t t = __atomic_load_n(&this->tail, __ATOMIC_RELAXED);
        uint32_t h = __atomic_exchange_n(&this->head, t, __ATOMIC_RELAXED);
        return std::make_pair(this->elem + h, this->elem + t);
    }

    __attribute__((always_inline)) inline datarange dequeue(uint32_t max_count)
    {
        std::scoped_lock slock(*this);
        uint32_t t = __atomic_load_n(&this->tail, __ATOMIC_RELAXED);
        uint32_t h = __atomic_load_n(&this->head, __ATOMIC_RELAXED);
        uint32_t cnt = std::min(t - h, max_count);
        if (cnt == 0) {
            return datarange(nullptr, nullptr);
        }
        __atomic_store_n(&this->head, h + cnt, __ATOMIC_RELAXED);
        return datarange(this->elem + h, this->elem + h + cnt);
    }

    __attribute__((always_inline)) inline datarange dequeue_tail(uint32_t max_count, T* out)
    {
        std::scoped_lock slock(*this);
        uint32_t t = __atomic_load_n(&this->tail, __ATOMIC_RELAXED);
        uint32_t h = __atomic_load_n(&this->head, __ATOMIC_RELAXED);
        uint32_t cnt = std::min(t - h, max_count);
        if (cnt == 0) {
            return datarange(nullptr, nullptr);
        }
        __atomic_store_n(&this->tail, t - cnt, __ATOMIC_RELAXED);
        // Faster analog og std::copy(this->elem + t - cnt, this->elem + t, out);
        memcpy(out, this->elem + t - cnt, sizeof(T) * cnt);
        return datarange(out, out + cnt);
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

inline void ReadyQueueLock(ReadyCoreFunctionQueue* rq) { rq->lock(); }

inline void ReadyQueueUnLock(ReadyCoreFunctionQueue* rq) { rq->unlock(); }

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
