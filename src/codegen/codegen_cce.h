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
 * \file codegen.h
 * \brief
 */

#ifndef CODEGEN_CCE_H
#define CODEGEN_CCE_H

#include <unordered_set>
#include <utility>

#include "interface/operation/operation.h"
#include "interface/machine/host/machine_task.h"

namespace npu::tile_fwk {

class CodeGenCCE {
public:
    explicit CodeGenCCE(std::string path = "") : path_(std::move(path)) {
        if (path_.empty()) {
            PrepareDefaultOutputPath();
        }
    }
    virtual ~CodeGenCCE() = default;

    virtual void GenCode(
        Function &topFunc, const std::map<uint64_t, std::list<InvokeParaOffset>> &invokeParaOffset) = 0;
    virtual void GenCode(
        const std::string &jsonPath, const std::map<uint64_t, std::list<InvokeParaOffset>> &invokeParaOffset) = 0;

protected:
    std::string path_;

private:
    void PrepareDefaultOutputPath();
};

std::map<int, int> GenRealizeIdMap(const SubfuncParam &subFuncParam);

} // namespace npu::tile_fwk

#endif // CODEGEN_CCE_H
