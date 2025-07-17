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
 * \file merge_src_dst_buffer.h
 * \brief
 */

#ifndef PASS_MERGE_SRC_DST_BUFFER_H
#define PASS_MERGE_SRC_DST_BUFFER_H
#include "interface/function/function.h"
#include "interface/tensor/logical_tensor.h"
#include "passes/pass_interface/pass.h"
#include "interface/utils/log.h"
namespace npu::tile_fwk {

class SrcDstBufferMergePVC2 {
  public:
    SrcDstBufferMergePVC2() = default;
    ~SrcDstBufferMergePVC2() = default;
    void Run(Function &func);
  private:
    void Init(const std::vector<Operation *> &opList);
    bool CanSrcDstReuse(const std::vector<Operation *> &opList, size_t idx,
                        std::shared_ptr<LogicalTensor> ioperand, bool strict = false);
    std::map<int, std::set<int>> tensorConsumers_;
    std::map<int, int> tensorMaxSize_;
    int subGrpahID_{-1};
};

class SrcDstBufferMergePass : public Pass {
  public:
    SrcDstBufferMergePass() : Pass("SrcDstBufferMergePass") {}

private:
    Status RunOnFunction(Function &function) override {
        ALOG_INFO_F("===> Start SrcDstBufferMergePass.");
        SrcDstBufferMergePVC2 merge;
        merge.Run(function);
        ALOG_INFO_F("===> Finish SrcDstBufferMergePass.");
        return SUCCESS;
    }
};
}  // namespace npu::tile_fwk
#endif // PASS_MERGE_SRC_DST_BUFFER_H