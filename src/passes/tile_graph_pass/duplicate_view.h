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
 * \file duplicate_view.h
 * \brief
 */

#ifndef PASS_DUPLICATE_VIEW_H_
#define PASS_DUPLICATE_VIEW_H_

#include <vector>
#include "passes/pass_interface/pass.h"
namespace npu::tile_fwk {
/*
    DuplicateView: 对于一个view OP，如果存在多消费者的情况，则为每一个消费者创建一个新的view OP
*/
class DuplicateView : public Pass {
public:
    DuplicateView() : Pass("DuplicateView") {}
    ~DuplicateView() override = default;

private:
    Status RunOnFunction(Function &function) override;
    Status RunOnOperation(Function &function, Operation &operation, std::vector<std::pair<LogicalTensorPtr, LogicalTensorPtr>> &viewResults) const;
    Status DuplicateViewPass(Function &function) const;
};

}
#endif // PASS_DUPLICATE_VIEW_H_