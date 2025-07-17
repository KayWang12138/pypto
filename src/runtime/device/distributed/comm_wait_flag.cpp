/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file comm_wait_flag.cpp
 * \brief
 */

#include "comm_wait_flag.h"

#include <cstddef>
#include <type_traits>
#include <vector>
#include <cstring>
#include <cstdint>

#include "securec.h"

#include "tileop/a2a3/hccl_context.h"
#include "runtime/utils/device_log.h"
#include "interface/cache/core_func_data.h"
#include "neon_stub.h"


namespace npu::tile_fwk {
namespace Distributed {

constexpr uint32_t NEON_BLOCK_NUM = 16;
constexpr uint32_t NEON_BLOCK_BITS = 8;
constexpr uint32_t LONG_LONG_BITS = 64;
constexpr uint32_t TILE_INDEX_MAX_NUM = 65535;

void FlagPoller::Init(uint32_t rankId, uint32_t rankSize, uint8_t *winFlag) {
    rankId_ = rankId;
    rankSize_ = rankSize;
    winFlag_ = winFlag;
}

void FlagPoller::EnqueueOp(uint64_t taskId, uint32_t rankShape, uint32_t rankOffset, uint32_t tileIndex) {
    if ((rankShape < 1) || (tileIndex > TILE_INDEX_MAX_NUM) || (rankOffset + rankShape > TileOp::AICPU_MAX_RANK_NUM)) {
        DEV_ERROR("FlagPoller EnqueueOp failed: tileIndex=%u, rankShape=%u, rankOffset=%u\n", tileIndex, rankShape,
            rankOffset);
        return;
    }

    uint32_t startIndex = tileIndex * rankSize_ + rankOffset;
    uint32_t endIndex = startIndex + rankShape - 1;

    SetUndoFlag(startIndex, endIndex);

    if (endIndex >= opInfo_.size()) {
        opInfo_.resize(endIndex + 1, OpInfo{0, 0, 0});
    }
    opInfo_[startIndex].taskId = taskId;
    opInfo_[startIndex].flagCount = endIndex - startIndex + 1;
    for (uint32_t index = startIndex + 1; index <= endIndex; index++) {
        opInfo_[index].offset = index - startIndex;
    }
    opCount_++;

    // 如果区间内包含local，需要剔除
    auto localIndex = tileIndex * rankSize_ + rankId_;
    if ((localIndex >= startIndex) && (localIndex <= endIndex)) {
        // 将该位置flag清零
        size_t clearByte = localIndex / NEON_BLOCK_BITS;
        uint8_t clearBit = localIndex % NEON_BLOCK_BITS;
        undoFlag_[clearByte] &= ~(0x1 << clearBit);

        // 直接添加到readyQueue中
        // 考虑若该OP仅包含1个flag并且是local，则在PollCompleted的时候会直接完成
        // 若多于1个flag，则正常轮询flag，无需处理local bit
        readyQueue_.push_back(localIndex);
    }
}

void FlagPoller::PollCompleted(std::vector<uint64_t> &completed) {
    if (opCount_ == 0) {
        return;
    }

    ProcessFlag();

    // 从头部遍历并处理元素（条件不满足时重新入队到尾部）
    size_t readyCount = readyQueue_.size();
    while (readyCount-- > 0) {
        size_t readyIndex = readyQueue_.front();
        readyQueue_.pop_front();

        uint32_t startIndex = readyIndex - opInfo_[readyIndex].offset;

        // 可能这个flag对应的子图还未执行添加进来，先跳过
        if (opInfo_[startIndex].flagCount == 0) {
            // 重新入队到尾部
            readyQueue_.push_back(readyIndex);
        } else {
            opInfo_[startIndex].flagCount--;
            if (opInfo_[startIndex].flagCount == 0) {
                completed.push_back(opInfo_[startIndex].taskId);
                opCount_--;
            }
        }
    }
}

void FlagPoller::SetUndoFlag(uint32_t startIndex, uint32_t endIndex) {
    size_t startByte = startIndex / NEON_BLOCK_BITS;
    size_t endByte = endIndex / NEON_BLOCK_BITS;

    if (undoFlag_.size() < endByte + 1) {
        undoFlag_.resize(endByte + 1, 0);
    }

    if (startByte == endByte) {
        uint8_t startBit = startIndex % NEON_BLOCK_BITS;
        uint8_t endBit = endIndex % NEON_BLOCK_BITS;
        uint8_t mask = (0xFF << startBit) & (0xFF >> (NEON_BLOCK_BITS - 1 - endBit));
        undoFlag_[startByte] |= mask;
    } else {
        // 处理起始字节
        uint8_t startBit = startIndex % NEON_BLOCK_BITS;
        undoFlag_[startByte] |= (0xFF << startBit);

        // 处理结束字节
        uint8_t endBit = endIndex % NEON_BLOCK_BITS;
        undoFlag_[endByte] |= (0xFF >> (NEON_BLOCK_BITS - 1 - endBit));

        // 处理中间的完整字节
        for (size_t i = startByte + 1; i < endByte; ++i) {
            undoFlag_[i] = 0xFF;
        }
    }
}

// 处理16字节的块
void FlagPoller::ProcessBlock(uint8_t *winFlag, uint8_t *undoFlag, size_t start) {
    // 加载NEON寄存器
    uint8x16_t vecA = vld1q_u8(winFlag);
    uint8x16_t vecB = vld1q_u8(undoFlag);

    // 按位与操作
    uint8x16_t vecAnd = vandq_u8(vecA, vecB);

    // 将结果转换为64位掩码
    uint64x2_t vecAnd64 = vreinterpretq_u64_u8(vecAnd);
    uint64_t maskLow = vgetq_lane_u64(vecAnd64, 0);
    uint64_t maskHigh = vgetq_lane_u64(vecAnd64, 1);

    // 初始化清除掩码为全1
    uint8_t clearMasks[NEON_BLOCK_NUM];
    memset_s(clearMasks, sizeof(clearMasks), 0xFF, sizeof(clearMasks));

    // 处理低64位
    while (maskLow) {
        int bitPos = __builtin_ctzll(maskLow);
        size_t x = start + bitPos;
        readyQueue_.push_back(x);

        size_t byteOffset = bitPos / NEON_BLOCK_BITS;
        uint8_t bit = bitPos % NEON_BLOCK_BITS;
        clearMasks[byteOffset] &= ~(1 << bit);

        maskLow &= maskLow - 1;
    }

    // 处理高64位
    while (maskHigh) {
        int bitPos = __builtin_ctzll(maskHigh);
        size_t x = start + LONG_LONG_BITS + bitPos;
        readyQueue_.push_back(x);

        size_t globalBit = LONG_LONG_BITS + bitPos;
        size_t byteOffset = globalBit / NEON_BLOCK_BITS;
        uint8_t bit = globalBit % NEON_BLOCK_BITS;
        clearMasks[byteOffset] &= ~(1 << bit);

        maskHigh &= maskHigh - 1;
    }

    // 应用清除掩码到数组b
    uint8x16_t vecClear = vld1q_u8(clearMasks);
    vecB = vandq_u8(vecB, vecClear);
    vst1q_u8(undoFlag, vecB);
}

// 处理剩余不足16字节的部分
void FlagPoller::ProcessRemaining(uint8_t *winFlag, uint8_t *undoFlag, size_t remaining, size_t start) {
    for (size_t i = 0; i < remaining; ++i) {
        uint8_t byteA = winFlag[i];
        uint8_t byteB = undoFlag[i];
        uint8_t byteAnd = byteA & byteB;

        if (byteAnd == 0) {
            continue;
        }

        uint8_t clearMask = 0xFF;
        uint8_t temp = byteAnd;
        while (temp) {
            int bit = __builtin_ctz(temp);
            size_t x = start + i * NEON_BLOCK_BITS + bit;
            readyQueue_.push_back(x);
            clearMask &= ~(1 << bit);
            temp &= temp - 1;
        }
        undoFlag[i] &= clearMask;
    }
}

void FlagPoller::ProcessFlag() {
    // 处理完整块
    size_t blockCount = undoFlag_.size() / NEON_BLOCK_NUM;
    for (size_t i = 0; i < blockCount; ++i) {
        ProcessBlock(
            winFlag_ + i * NEON_BLOCK_NUM, undoFlag_.data() + i * NEON_BLOCK_NUM, i * NEON_BLOCK_NUM * NEON_BLOCK_BITS);
    }

    // 处理剩余字节
    size_t remaining = undoFlag_.size() % NEON_BLOCK_NUM;
    if (remaining > 0) {
        ProcessRemaining(winFlag_ + blockCount * NEON_BLOCK_NUM, undoFlag_.data() + blockCount * NEON_BLOCK_NUM,
            remaining, blockCount * NEON_BLOCK_NUM * NEON_BLOCK_BITS);
    }
}

void CommWaitFlag::Init(DeviceTask *deviceTask) {
    uint64_t *hcclContextAddr = deviceTask->coreFuncData.hcclContextAddr;
    uint32_t commGroupNum = static_cast<uint32_t>(deviceTask->coreFuncData.commGroupNum);
    if (commGroupNum > DIST_COMM_GROUP_NUM) {
        commGroupNum = 0;
        DEV_ERROR("CommWaitFlag comm group invalid, num=%u\n", commGroupNum);
        return;
    }
    hcclContextAddr_ = hcclContextAddr;
    commGroupNum_ = commGroupNum;
}

bool CommWaitFlag::Prepare(uint32_t groupIndex) {
    if (inited_[groupIndex]) {
        return true;
    }

    struct TileOp::HcclCombinOpParam *hcclOpParam = (struct TileOp::HcclCombinOpParam *)hcclContextAddr_[groupIndex];
    uint32_t rankId = hcclOpParam->rankId;
    uint32_t rankSize = hcclOpParam->rankNum;
    uint8_t *winFlag = (uint8_t *)hcclOpParam->windowsExp[rankId];
    if ((rankSize <= 1) || (rankSize > TileOp::AICPU_MAX_RANK_NUM) || (rankId >= rankSize)) {
        DEV_ERROR("CommWaitFlag Prepare failed: groupIndex=%u, rankSize=%u, rankId=%u\n", groupIndex, rankSize, rankId);
        return false;
    }
    flagPoller_[groupIndex].Init(rankId, rankSize, winFlag);
    inited_[groupIndex] = true;
    return true;
}

void CommWaitFlag::EnqueueOp(uint64_t taskId, uint64_t *paramList, uint32_t paramSize) {
    if (paramSize != 0x4) {
        DEV_ERROR("CommWaitFlag EnqueueOp param size inlvaid: %u\n", paramSize);
        return;
    }
    uint32_t tileIndex = paramList[0x0];
    uint32_t groupIndex = paramList[0x1];
    uint32_t rankShape = paramList[0x2];
    uint32_t rankOffset = paramList[0x3];
    if ((groupIndex >= commGroupNum_) || (hcclContextAddr_[groupIndex] == 0)) {
        DEV_ERROR("CommWaitFlag EnqueueOp param inlvaid: groupIndex=%u\n", groupIndex);
        return;
    }
    if (!Prepare(groupIndex)) {
        return;
    }

    flagPoller_[groupIndex].EnqueueOp(taskId, rankShape, rankOffset, tileIndex);
}

void CommWaitFlag::PollCompleted(std::vector<uint64_t> &completed) {
    for (uint32_t groupIndex = 0; groupIndex < commGroupNum_; ++groupIndex) {
        if (inited_[groupIndex]) {
            flagPoller_[groupIndex].PollCompleted(completed);
        }
    }
}

} // namespace Distributed
} // namespace npu::tile_fwk
