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
 * \file remove_redundant_op_checker.cpp
 * \brief
 */

#include "remove_redundant_op_checker.h"

namespace npu{
namespace tile_fwk {
Status RemoveRedundantOpChecker::PreCheckAssemble(const Operation &op, const LogicalTensorPtr &in) {
    uint32_t assembleRemoveNum = 0;
    uint32_t otherOpNum = 0;
    for (auto &childOp : in->GetConsumers()) {
        if (childOp->GetOpcode() == Opcode::OP_ASSEMBLE) {
            auto child_in = op.iOperand.front();
            auto child_out = op.oOperand.front();
            if (child_out->GetConsumers().empty()) {
                if (child_in->shape == child_out->shape &&
                    child_in->GetMemoryTypeOriginal() == MemoryType::MEM_DEVICE_DDR &&
                    child_out->GetMemoryTypeOriginal() == MemoryType::MEM_DEVICE_DDR) {
                    ++assembleRemoveNum;
                }
            } else {
                ++otherOpNum;
            }
        } else {
            ++otherOpNum;
        }
    }
    if (assembleRemoveNum > 1 && otherOpNum > 0) {
        ALOG_ERROR_F("More than one assemble ddr op without consumer!");
        return FAILED;
    }
    return SUCCESS;
}

Status RemoveRedundantOpChecker::PreCheckView(const Operation &op, const LogicalTensorPtr &in) {
    auto out = op.oOperand.front();
    if (in->shape == out->shape && op.ConsumerOps().empty() && in->GetConsumers().size() > 1) {
        ALOG_ERROR_F("There is another op consumes the input of a view op without consumer!");
        return FAILED;
    }
    return SUCCESS;
}

Status RemoveRedundantOpChecker::ProcessPreCheck(const Operation &op) {
    if (op.GetOpcode() == Opcode::OP_ASSEMBLE) {
        auto assemble_in = op.iOperand.front();
        if (PreCheckAssemble(op, assemble_in) != SUCCESS) {
            ALOG_ERROR_F("PreCheck for assemble op[%d] failed!", op.GetOpMagic());
            return FAILED;
        }
    } else if (op.GetOpcode() == Opcode::OP_VIEW) {
        auto view_in = op.iOperand.front();
        if (PreCheckView(op, view_in) != SUCCESS) {
            ALOG_ERROR_F("PreCheck for view op[%d] failed!", op.GetOpMagic());
            return FAILED;
        }  
    }
    return SUCCESS;
}

Status RemoveRedundantOpChecker::PostCheckAssemble(const Operation &op) {
    auto assemble_in = op.iOperand.front();
    auto assemble_out = op.oOperand.front();
    auto parentOp = *assemble_in->GetProducers().begin();
    if (parentOp == nullptr) {
        ALOG_ERROR_F("The input of assemble [%d] has no producer!", op.GetOpMagic());
        return FAILED;
    }
    if (assemble_in->shape == assemble_out->shape) {
        if (assemble_in->GetMemoryTypeOriginal() == MemoryType::MEM_DEVICE_DDR &&
            assemble_out->GetMemoryTypeOriginal() == MemoryType::MEM_DEVICE_DDR) {
            ALOG_ERROR_F("PostCheck for assembleDDR op[%d] failed with the same shape!", op.GetOpMagic());
            return FAILED;
        } else if (assemble_in->GetMemoryTypeOriginal() == MemoryType::MEM_UB &&
            assemble_out->GetMemoryTypeOriginal() == MemoryType::MEM_UB) {
            ALOG_ERROR_F("PostCheck for assembleUB op[%d] failed with the same shape!", op.GetOpMagic());
            return FAILED;
        }
    }
    return SUCCESS;
}

Status RemoveRedundantOpChecker::PostCheckView(const Operation &op) {
    auto view_in = op.iOperand.front();
    auto view_out = op.oOperand.front();
    auto viewOpAttribute = dynamic_cast<ViewOpAttribute *>(op.GetOpAttribute().get());
    if (viewOpAttribute != nullptr && viewOpAttribute->GetToDynValidShape().empty() &&
        view_in->shape == view_out->shape && view_in->GetMemoryTypeOriginal() == view_out->GetMemoryTypeOriginal()) {
        ALOG_ERROR_F("PostCheck for view op[%d] failed, DynValidShape is empty with the same shape and memory type!", op.GetOpMagic());
        return FAILED;
    } else if (view_out->GetConsumers().size() == 1) {
        auto childOp = *(view_out->GetConsumers().begin());
        if (childOp == nullptr) {
            ALOG_ERROR_F("Found null childOp of op[%d]", op.GetOpMagic());
            return FAILED;
        }
        if (childOp->GetOpcode() == Opcode::OP_COMM_WAIT_FLAG) {
            ALOG_ERROR_F("View op[%d] has only one commit wait child!", op.GetOpMagic());
            return FAILED;
        }
    }
    return SUCCESS;
}

Status RemoveRedundantOpChecker::PostCheckRegCopy(const Operation &op) {
    auto regcopy_in = op.iOperand.front();
    auto regcopy_out = op.oOperand.front();
    if (regcopy_in->shape == regcopy_out->shape && regcopy_in->GetMemoryTypeOriginal() == regcopy_out->GetMemoryTypeOriginal()) {
        ALOG_ERROR_F("PostCheck for regcopy op[%d] failed, in->shape == out->shape!", op.GetOpMagic());
        return FAILED;
    }
    return SUCCESS;
}

Status RemoveRedundantOpChecker::PostCheckCopyIn(const Operation &op) {
    auto copy_in = op.iOperand.front();
    auto copy_out = op.oOperand.front();
    if (copy_in->shape == copy_out->shape && copy_out->GetMemoryTypeOriginal() == npu::tile_fwk::MEM_L1) {
        bool isRedundant = true;
        for (auto &producerOp : op.ProducerOps()) {
            if (producerOp == nullptr) {
                ALOG_ERROR_F("Found null producer of op[%d]", op.GetOpMagic());
                return FAILED;
            }
            if (producerOp->GetOpcode() != Opcode::OP_VIEW) {
                isRedundant = false;
                break;
            }
        }
        if (isRedundant) {
            ALOG_ERROR_F("PostCheck for copyin op[%d] failed!", op.GetOpMagic());
            return FAILED;
        }
    }
    return SUCCESS;
}

Status RemoveRedundantOpChecker::PostCheckExpand(const Operation &op) {
    auto expand_in = op.iOperand.front();
    auto expand_out = op.oOperand.front();
    if (expand_in->shape == expand_out->shape) {
        ALOG_ERROR_F("PostCheck for expand op[%d] failed!", op.GetOpMagic());
        return FAILED;
    }
    return SUCCESS;
}

Status RemoveRedundantOpChecker::ProcessPostCheck(const Operation &op) {
    if (op.GetOpcode() == Opcode::OP_ASSEMBLE) {
        if (PostCheckAssemble(op) != SUCCESS) {
            ALOG_ERROR_F("PostCheck for Assemble failed!");
            return FAILED;
        }
    } else if (op.GetOpcode() == Opcode::OP_VIEW) {
        if (PostCheckView(op) != SUCCESS) {
            ALOG_ERROR_F("PostCheck for View failed!");
            return FAILED;
        }
    } else if (op.GetOpcode() == Opcode::OP_REGISTER_COPY) {
        if (PostCheckRegCopy(op) != SUCCESS) {
            ALOG_ERROR_F("PostCheck for RegCopy failed!");
            return FAILED;
        }
    } else if (op.GetOpcode() == Opcode::OP_COPY_IN) {
        if (PostCheckCopyIn(op) != SUCCESS) {
            ALOG_ERROR_F("PostCheck for CopyIn failed!");
            return FAILED;
        }
    } else if (op.GetOpcode() == Opcode::OP_EXPAND) {
        if (PostCheckExpand(op) != SUCCESS) {
            ALOG_ERROR_F("PostCheck for Expand failed!");
            return FAILED;
        }
    }
    return SUCCESS;
}

Status RemoveRedundantOpChecker::DoPreCheck(Function &function) {
    ALOG_INFO_F("PreCheck for RemoveRedundantOp");
    if (CheckValidOp(function) != SUCCESS) {
        ALOG_ERROR_F("Found invalid op in the function.");
        return FAILED;
    }
    if (CheckOpIOValid(function) != SUCCESS) {
        ALOG_ERROR_F("Found invalid input/output from the function.");
        return FAILED;
    }
    for (auto &op : function.Operations()) {
        if (ProcessPreCheck(op) != SUCCESS) {
            ALOG_ERROR_F("PreCheck for RemoveRedundantOp failed!");
            return FAILED;
        }
    }
    return SUCCESS;
}

Status RemoveRedundantOpChecker::DoPostCheck(Function &function) {
    ALOG_INFO_F("PostCheck for RemoveRedundantOp");
    if (CheckOpIOValid(function) != SUCCESS) {
        ALOG_ERROR_F("Found invalid input/output in the function.");
        return FAILED;
    }
    for (auto &op : function.Operations()) {
        if (ProcessPostCheck(op) != SUCCESS) {
            ALOG_ERROR_F("PostCheck for RemoveRedundantOp failed!");
            return FAILED;
        }
    }
    return SUCCESS;
}
} // namespace tile_fwk
} // namespace npu