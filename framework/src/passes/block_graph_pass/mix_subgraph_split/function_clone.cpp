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
 * \file function_clone.cpp
 * \brief
 */

#include "function_clone.h"


namespace npu {
namespace tile_fwk {
Function* MixSubgraphSplit::CreateSplitLeafFunction(const InternalComponentInfo& component,
                                                    uint64_t newProgramID,
                                                    SubgraphToFunction& subgraphToFunction) {
    // 创建新的function名称
    std::string leafName = originalMixFunc.GetRawName() + "_leaf";
    ALOG_DEBUG_F("Add leafFunction %s", leafName.c_str());
    // 手动创建function对象
    auto funcMagicName = leafName + "_" + std::to_string(IdGen<IdType::FUNCTION>::Inst().CurId());
    auto newFunc = std::make_shared<Function>(Program::GetInstance(), funcMagicName, leafName, &rootFunc);
    // 设置function类型
    newFunc->SetFunctionType(FunctionType::STATIC); //TODO:  继承
    newFunc->SetGraphType(GraphType::BLOCK_GRAPH); //TODO:  继承

    // 获取原始Mix子图的所有op（按原始顺序）
    auto originalOps = originalMixFunc.Operations(false).DuplicatedOpList();
    // 按原始顺序筛选属于当前component的op
    for (auto *originalOp : originalOps) {
        if (originalOp->IsNOP()) {
            continue;
        }
        // 检查这个op是否属于当前component
        bool belongsToComponent = false;
        for (auto* compOp : component.operations) {
            if (compOp == originalOp) {
                belongsToComponent = true;
                break;
            }
        }

        if (belongsToComponent) {
            // 判断是否为同步op
            bool isSyncOp = IsSyncOperation(originalOp);
            if (isSyncOp) {
                // 对于同步op，直接使用原始op的shared_ptr
                std::shared_ptr<Operation> opPtr = originalOp->shared_from_this();
                programOps.push_back(opPtr);
                int originalMagic = originalOp->GetOpMagic();
                magicMap[originalMagic] = originalMagic;
                ALOG_DEBUG_F("Reuse sync op %d in leaf function %s",
                        originalMagic, leafName.c_str());
            } else {
                // 对于非同步op，进行克隆
                auto clonedOp = CloneOperation(originalOp, newFunc);
                programOps.push_back(clonedOp.shared_from_this());
                ALOG_DEBUG_F("Cloned op %d to leaf function %s (original order preserved)",
                                originalOp->GetOpMagic(), leafName.c_str());
            }
        }
    }
    // 保存映射关系
    LeafFuncMagicMap leafMap;
    leafMap.leafFunc = newFunc.get();
    leafMap.originalToClonedMagic = std::move(magicMap);
    leafFuncMagicMaps_[newFunc.get()] = std::move(leafMap);
    // 验证顺序正确性
    ALOG_DEBUG_F("Leaf function %s has %zu ops in original order",
                leafName.c_str(), programOps.size());
    newFunc->SetProgramOp(programOps);
    // 创建并设置LeafFuncAttribute
    auto leafAttr = std::make_shared<LeafFuncAttribute>();
    // 设置aivCore属性
    leafAttr->aivCore = component.aivCore;
    newFunc->SetLeafFuncAttribute(leafAttr);
    newFunc->UpdateBelongToThis();
    newFunc->SetProgramId(newProgramID);
    // 复制参数配置
    newFunc->paramConfigs_ = originalMixFunc.paramConfigs_;
    ALOG_DEBUG_F("Called UpdateBelongToThis for new function: %s", leafName.c_str());
    auto* resultFunc = newFunc.get();
    return resultFunc;
}

Operation* CloneOperation(Operation& originalOp, Function& targetFunc){
    auto iOperands = originalOp->GetIOperands();
    auto oOperands = originalOp->GetOOperands();
    Operation& clonedOp = originalOp->CloneOperation(*targetFunc, iOperands, oOperands);
    // 记录映射关系
    int originalMagic = originalOp->GetOpMagic();
    int clonedMagic = clonedOp.GetOpMagic();
    magicMap[originalMagic] = clonedMagic;
    // 复制offset信息
    for (size_t idx = 0; idx < iOperands.size(); ++idx) {
        int offset = originalOp->GetIOpAttrOffset(idx);
        if (offset != -1) { //TODO -1?
            clonedOp.SetIOpAttrOffset(idx, offset);
        }
    }
    for (size_t idx = 0; idx < oOperands.size(); ++idx) {
        int offset = originalOp->GetOOpAttrOffset(idx);
        if (offset != -1) {
            clonedOp.SetOOpAttrOffset(idx, offset);
        }
    }
    return clonedOp;
}

}
}