/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "test_codegen_utils.h"

#include <iostream>

#include "interface/configs/config_manager.h"
#include "interface/function/function.h"

namespace npu::tile_fwk {
std::shared_ptr<LogicalTensor> CreateLogicalTensor(const LogicalTensorInfo &info) {
    if (info.memType == MemoryType::MEM_DEVICE_DDR) {
        std::shared_ptr<RawTensor> ddrRawTensor = std::make_shared<RawTensor>(info.dType, info.shape, info.tensorName);
        const std::vector<int64_t> offset = {0, 0};
        auto ddrTensor = std::make_shared<LogicalTensor>(info.function, ddrRawTensor, offset, info.shape);
        ddrTensor->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR);
        ddrTensor->SetMemoryTypeToBe(MemoryType::MEM_DEVICE_DDR);
        return ddrTensor;
    }

    auto localTensor = std::make_shared<LogicalTensor>(info.function, info.dType, info.shape, info.tensorName);
    localTensor->UpdateSubgraphID(0);
    localTensor->SetMemoryTypeOriginal(info.memType);
    localTensor->SetMemoryTypeToBe(info.memType);
    localTensor->SetAttr(OpAttributeKey::needAlloc, true);
    localTensor->memoryrange.memId = 0;
    localTensor->memoryrange.start = 0;
    localTensor->memoryrange.end = 0;
    if (info.magic != -1) {
        localTensor->SetMagic(info.magic);
    }
    if (!info.dynValidShape.empty()) {
        localTensor->UpdateDynValidShape(info.dynValidShape);
    }
    return localTensor;
}

std::string GetResultFromCpp(const Function &function) {
    const auto &subFunc = function.rootFunc_->programs_[0];
    auto leafFuncAttr = subFunc->GetLeafFuncAttribute();
    ASSERT(leafFuncAttr != nullptr);
    std::string binPath = leafFuncAttr->binPath;
    std::string cppFile = binPath.substr(0, binPath.rfind('.')) + ".cpp";
    std::ifstream ifs(cppFile);
    std::string res((std::istreambuf_iterator<char>(ifs)), (std::istreambuf_iterator<char>()));
    ifs.close();
    return res;
}

} // namespace npu::tile_fwk