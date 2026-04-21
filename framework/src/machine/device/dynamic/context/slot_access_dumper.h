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
 * \file slot_access_dumper.h
 * \brief Dump per-instance slot cell access events (writer/reader) to
 *        dyn_slot_access.csv for the runtime dependency correctness
 *        verification framework. Only active on non-device (host/simulator)
 *        compilation.
 */
#pragma once

#include <cstddef>
#include <cstdint>

#ifndef __DEVICE__
#include <fstream>
#include <string>
#include "interface/configs/config_manager.h"
#endif

namespace npu::tile_fwk::dynamic {

// Append one slot access event to dyn_slot_access.csv.
// accessType: 'W' for producer write, 'R' for consumer read.
// cellIdxList / cellCount describe which cell indices of the slot's
// CellMatchTable are touched by this op instance.
// allConcrete: whether the op's offset/shape were fully concrete at dump time
// (false => cellIdxList is a conservative estimate).
inline void DumpSlotAccessEvent(
    uint32_t seqNo, int slotIdx, uint64_t rootHash, uint64_t funcKey, uint32_t funcIdx, uint32_t opIdx, char accessType,
    const int* cellIdxList, std::size_t cellCount, bool allConcrete)
{
#ifndef __DEVICE__
    if (config::GetDebugOption<int64_t>(CFG_RUNTIME_DBEUG_MODE) != CFG_DEBUG_ALL) {
        return;
    }
    const std::string path = config::LogTopFolder() + "/dyn_slot_access.csv";
    static std::string lastPath;
    static std::ofstream slotAccessOf;
    if (path != lastPath) {
        if (slotAccessOf.is_open()) {
            slotAccessOf.flush();
            slotAccessOf.close();
        }
        lastPath = path;
        slotAccessOf.open(path);
    }
    if (slotAccessOf.tellp() == 0) {
        slotAccessOf << "seqNo,slotIdx,rootHash,funcKey,funcIdx,opIdx,taskId,accessType,cellIdxList,allConcrete\n";
    }
    const uint32_t taskId = (static_cast<uint32_t>(funcIdx) << 16) | (opIdx & 0xffffU);
    slotAccessOf << seqNo << "," << slotIdx << "," << rootHash << "," << funcKey << "," << funcIdx << "," << opIdx
                 << "," << taskId << "," << accessType << ",\"[";
    for (std::size_t i = 0; i < cellCount; ++i) {
        if (i != 0) {
            slotAccessOf << ",";
        }
        slotAccessOf << cellIdxList[i];
    }
    slotAccessOf << "]\"," << (allConcrete ? 1 : 0) << "\n";
    slotAccessOf.flush();
#else
    (void)seqNo;
    (void)slotIdx;
    (void)rootHash;
    (void)funcKey;
    (void)funcIdx;
    (void)opIdx;
    (void)accessType;
    (void)cellIdxList;
    (void)cellCount;
    (void)allConcrete;
#endif
}

} // namespace npu::tile_fwk::dynamic