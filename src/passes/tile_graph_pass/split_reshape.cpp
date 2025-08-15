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
 * \file split_reshape.cpp
 * \brief
 */

#include "split_reshape.h"
#include "interface/tensor/logical_tensor.h"

namespace npu::tile_fwk {
namespace {
    std::string GetStr(const std::vector<int> &vec) {
        std::string ret;
        for (const auto &val : vec) {
            ret += val + ", ";
        }
        return "{" + ret + "}";
    }

    std::string GetStr(const ReshapeTilePara &para) {
        std::string ret;
        ret += "shape = " + GetStr(para.shape);
        ret += "newShape = " + GetStr(para.newShape);
        ret += "shape's tile = " + GetStr(para.tileShape);
        ret += "shape's offset = " + GetStr(para.tileOffset);
        return ret;
    }
}

Status SplitReshape::RunOnFunction(Function &function) {
    ALOG_INFO_F("===> start SplitReshapeOp");
    if (Init() != SUCCESS) {
        ALOG_ERROR_F("Init failed!");
        return FAILED;
    }
    if (CollectCopyOut(function) != SUCCESS) {
        ALOG_ERROR_F("CollectCopyOut failed!");
        return FAILED;
    }
    if (CheckCopyIn(function) != SUCCESS) {
        ALOG_ERROR_F("CheckCopyIn failed!");
        return FAILED;
    }
    if (AddOperation(function) != SUCCESS) {
        ALOG_ERROR_F("AddOperation failed!");
        return FAILED;
    }
    if (EraseReshape(function) != SUCCESS) {
        ALOG_ERROR_F("EraseReshape failed!");
        return FAILED;
    }
    if (EliminateDeadOperation(function) != SUCCESS) {
        ALOG_ERROR_F("EliminateDeadOperation failed!");
        return FAILED;
    }
    if (SetMemoryType(function) != SUCCESS) {
        ALOG_ERROR_F("SetMemoryType failed!");
        return FAILED;
    }
    ALOG_INFO_F("===> end SplitReshapeOp");
    return SUCCESS;
}

Status SplitReshape::Init() {
    copyOutSources.clear();
    reshapeSources.clear();
    mapOffset.clear();
    assembles.clear();
    reshapes.clear();
    redundentViewops.clear();
    reshapeRawOutputs.clear();
    return SUCCESS;
}

bool SplitReshape::CheckDynStatus(const LogicalTensorPtr &input, const LogicalTensorPtr &output) {
    if (!input->GetDynValidShape().empty()) {
        for (const auto &shape : input->GetDynValidShape()) {
            if (!shape.IsImmediate()) {
                return false;
            }
        }
    }
    if (!output->GetDynValidShape().empty()) {
        for (const auto &shape : output->GetDynValidShape()) {
            if (!shape.IsImmediate()) {
                return false;
            }
        }
    }
    return true;
}

bool SplitReshape::CheckSplit(const LogicalTensorPtr &reshapeSource) {
    auto copySources = copyOutSources[reshapeSource->tensor->rawmagic];
    auto copyoutSourceFirst = copySources.begin();
    if (copyoutSourceFirst == copySources.end()) {
        return true;
    }
    for (auto &copyOutSource : copySources) {
        if (copyOutSource->tensor->rawmagic != (*copyoutSourceFirst)->tensor->rawmagic) {
            return false;
        }
    }
    return true;
}

std::shared_ptr<ReshapeOp> SplitReshape::ReshapeOperationExist(const std::shared_ptr<ReshapeOp> &isAddReshapeop) {
    const auto hashKey = ComputeReshapeHash(isAddReshapeop->input, isAddReshapeop->output);
    auto it = reshapes.find(hashKey);
    if (it != reshapes.end()) {
        return it->second;
    }
    reshapes[hashKey] = isAddReshapeop;
    return nullptr;
}

unsigned long SplitReshape::ComputeReshapeHash(const LogicalTensorPtr &input, const LogicalTensorPtr &output) const {
    unsigned long operationHash = ComputeReshapeHashOrderless(input, output);
    return operationHash;
}

unsigned long SplitReshape::ComputeReshapeHashOrderless(
    const LogicalTensorPtr &input, const LogicalTensorPtr &output) const {
    std::stringstream ss;
    ss << "[i";
    ss << "$" << input->tensor->DumpSSA(false, false);
    ss << input->DumpType();
    ss << "(";
    for (size_t i = 0; i < input->offset.size(); ++i) {
        ss << input->offset[i];
        if (i != input->offset.size() - 1) {
            ss << ", ";
        }
    }
    ss << ")";
    ss << "]";
    ss << "[j";
    ss << "$" << output->tensor->DumpSSA(false, false);
    ss << output->DumpType();
    ss << "(";
    for (size_t i = 0; i < output->offset.size(); ++i) {
        ss << output->offset[i];
        if (i != output->offset.size() - 1) {
            ss << ", ";
        }
    }
    ss << ")";
    ss << "]";
    std::string s = ss.str();
    std::hash<std::string> hasher;
    auto result = hasher(s);
    return result;
}

Status SplitReshape::CollectCopyOut(Function &function) {
    for (auto &op : function.Operations()) {
        if (op.GetOpcode() == Opcode::OP_RESHAPE) {
            auto input = op.GetIOperands().front();
            auto output = op.GetOOperands().front();
            if (input == nullptr || output == nullptr || output->tensor == nullptr) {
                ALOG_ERROR_F("input == nullptr || output == nullptr || output->tensor == nullptr");
                return FAILED;
            }
            reshapeSources[output->tensor->rawmagic] = input;
        }
        if (op.GetOpcode() == Opcode::OP_ASSEMBLE) { // output应该是reshape的input
            auto input = op.GetIOperands().front();
            auto output = op.GetOOperands().front();
            if (input == nullptr || output == nullptr || input->tensor == nullptr || output->tensor == nullptr) {
                ALOG_ERROR_F("input == nullptr || output == nullptr || input->tensor == nullptr || output->tensor == nullptr");
                return FAILED;
            }
            copyOutSources[output->tensor->rawmagic].insert(input);
            auto offset = dynamic_cast<AssembleOpAttribute *>(op.GetOpAttribute().get())->GetToOffset();
            mapOffset[input->magic][output->magic] = offset;
        }
    }
    return SUCCESS;
}

// shape1和shape2分别为reshape前后的shape
// 移除shape1和shape2中为1的元素
// 进行的操作为细分对齐：
// example1: [2,4] + [8] -> [2,4]
// example2: [2,4] + [4,2] -> [2,2,2]
// example3: [2,3] + [5] -> FAILED
// 返回对齐后的shape
Status SplitReshape::ShapeAlign(std::vector<int32_t> shape1, std::vector<int32_t> shape2, std::vector<int32_t> &alignedShape) {
    size_t i1 = 0UL;
    size_t i2 = 0UL;
    int32_t prod1 = 1;
    int32_t prod2 = 1;
    shape1.erase(std::remove(shape1.begin(), shape1.end(), 1), shape1.end());
    shape2.erase(std::remove(shape2.begin(), shape2.end(), 1), shape2.end());
    while (i1 < shape1.size() || i2 < shape2.size()) {
        if (prod1 != 1) {
            if (prod1 % shape2[i2] != 0 && shape2[i2] % prod1 != 0) {
                ALOG_WARN_F("Non-segmentable axis");
                return WARNING;
            }
            if (shape2[i2] > prod1) {
                alignedShape.push_back(prod1);
                prod2 = shape2[i2] / prod1;
                prod1 = 1;
            } else {
                alignedShape.push_back(shape2[i2]);
                prod1 = prod1 / shape2[i2];
            }
            i2++;
        } else if (prod2 != 1) {
            std::swap(i1, i2);
            std::swap(prod1, prod2);
            std::swap(shape1, shape2);
        } else {
            if ((i1 >= shape1.size() || i2 >= shape2.size())) {
                ALOG_ERROR_F("i1 >= shape1.size() || i2 >= shape2.size()");
                return FAILED;
            }
            if (shape2[i2] % shape1[i1] != 0 && shape1[i1] % shape2[i2] != 0) {
                ALOG_WARN_F("Non-segmentable axis");
                return WARNING;
            }
            if (shape2[i2] > shape1[i1]) {
                alignedShape.push_back(shape1[i1]);
                prod2 = shape2[i2] / shape1[i1];
            } else {
                alignedShape.push_back(shape2[i2]);
                prod1 = shape1[i1] / shape2[i2];
            }
            i1++;
            i2++;
        }
    }
    return SUCCESS;
}

Status SplitReshape::UpdateShapeOffset(UpdatePara &para, bool &flag, int &currentShape, int &currentOffset) {
    auto stride = para.stride;
    if (flag && currentShape <= stride) {
        para.OffsetVal = currentOffset / stride;
        para.ShapeVal = 1;
        currentOffset = currentOffset - para.OffsetVal * stride;
        if (currentOffset + currentShape > stride) {
            ALOG_WARN_F("Found tail cond, currentOffset = %d, currentShape = %d, stride = %d.", currentOffset, currentShape, stride);
            return WARNING;
        }
    } else {
        if (currentOffset % stride != 0) {
            ALOG_WARN_F("Found tail cond, currentOffset = %d, stride = %d.", currentOffset, stride);
            return WARNING;
        }
        if (currentShape % stride != 0) {
            ALOG_WARN_F("Found tail cond, currentShape = %d, stride = %d.", currentShape, stride);
            return WARNING;
        }
        para.OffsetVal = currentOffset / stride;
        para.ShapeVal = currentShape / stride;
        currentOffset = 0;
        currentShape = stride;
        flag = false;
    }
    return SUCCESS;
}

Status SplitReshape::ConstructShapeOffset(const ReshapeTilePara &shapePara, size_t &i, size_t j, std::vector<int32_t> &newOffset, std::vector<int32_t> &newShape) {
    UpdatePara updatePara;
    auto shape = shapePara.shape;
    auto alignedShape = shapePara.newShape;
    auto tileOffset = shapePara.tileOffset;
    auto tileShape = shapePara.tileShape;
    int stride = shape[j];
    int currentOffset = tileOffset[j];
    int currentShape = tileShape[j];
    bool flag = true; // flag为true代表之前拆分的轴均为1，目前的tile依然占据维度的连续一段，可以不完整。
    while (stride > 1) {
        if (stride % alignedShape[i] != 0) {
            ALOG_WARN_F("Cannot cut shape, Stride = %d, alignedShape[%zu] = %d.", stride, i, alignedShape[i]);
            return WARNING;
        }
        stride /= alignedShape[i];
        updatePara.stride = stride;
        if (UpdateShapeOffset(updatePara, flag, currentShape, currentOffset) == WARNING) {
            ALOG_WARN_F("Found tail condition.");
            return WARNING;
        }
        newOffset[i] = updatePara.OffsetVal;
        newShape[i] = updatePara.ShapeVal;
        i++;
    }
    return SUCCESS;
}

// 将原始shape的分片信息转换到alignedShape维度上
// 若原始维度shape的某轴为1：需保证对应分片偏移tileOffset为0且分片大小tileShape为1，直接跳过该维度
// 若原始维度shape的某轴非1：
// 用stride记录当前维度的"剩余未拆分长度"，currentOffset和currentShape记录当前维度的分片偏移和大小
// 用flag标记 "当前分片是否仍为连续段"（初始为true，表示可拆分）
//     根据flag和currentShape与stride的大小关系，计算newOffset[i]和newShape[i]，并更新currentOffset、currentShape和flag
// shape = [8], alignedShape = [2, 2, 2], tileOffset = [2], tileShape = [2]
// offset: [0, 1, 0], shape = [1, 1, 2]
// shape = [64, 64], alignedShape = [32, 2, 64], tileOffset = [32, 32], tileShape = [32, 32]
// offset: [16, 0, 32], shape = [16, 2, 32]
Status SplitReshape::RawToAlign(const ReshapeTilePara &shapePara,
                                       std::vector<int32_t> &newOffset, std::vector<int32_t> &newShape) {
    auto shape = shapePara.shape;
    auto alignedShape = shapePara.newShape;
    auto tileOffset = shapePara.tileOffset;
    auto tileShape = shapePara.tileShape;
    newOffset.clear();
    newShape.clear();
    for (size_t j = 0UL; j < alignedShape.size(); j++) {
        newOffset.emplace_back(0);
        newShape.emplace_back(0);
    }
    size_t i = 0UL;
    for (size_t j = 0UL; j < shape.size(); j++) {
        if (shape[j] == 1) {
            if (tileOffset[j] != 0 || tileShape[j] != 1) {
                ALOG_ERROR_F("shape[%zu] = 1, but tileOffset[%zu] = %d, tileShape[%zu] = %d, incorrect shape.", j, j, tileOffset[j], j, tileShape[j]);
                return FAILED;
            }
            continue;
        }
        if (ConstructShapeOffset(shapePara, i, j, newOffset, newShape) != SUCCESS) {
            ALOG_WARN_F("ConstructShapeOffset failed.");
            return WARNING;
        }
    }
    if (i != alignedShape.size()) {
        ALOG_WARN_F("i = %zu, alignedShape.size() = %zu.", i, alignedShape.size());
        return WARNING;
    }
    return SUCCESS;
}

// 用于将rawShape的分片信息映射到新的目标形状newRawshape上(本质上是加细的还原)
// 生成对应的newOffset和新形状newShape
// 核心逻辑将原始分片信息 “重组” 到新维度上，特定条件下会清空结果并返回成功。
// example1 : rawShape = [2, 3], newRawshape = [6], tileOffset = [1, 0], tileShape = [1, 3]
//          newOffset = [3], newShape = [3]
// example2 : rawShape = [1], newRawshape = [1], tileOffset = [0], tileShape = [2]
//          newOffset = [0], newShape = [1]
// example3 : rawShape = [32, 2, 64], newRawshape = [64, 64], tileOffset = [16, 0, 32], tileShape = [16, 2, 32]
//          newOffset = [32, 32], newShape = [32, 32]
Status SplitReshape::AlignToRaw(const ReshapeTilePara &shapePara,
                                std::vector<int32_t> &newOffset, std::vector<int32_t> &newShape) {
    auto rawShape = shapePara.shape;
    auto newRawshape = shapePara.newShape;
    auto tileOffset = shapePara.tileOffset;
    auto tileShape = shapePara.tileShape;
    newOffset.clear();
    newShape.clear();
    for (size_t j = 0UL; j < newRawshape.size(); j++) {
        newOffset.emplace_back(0);
        newShape.emplace_back(0);
    }
    size_t i = 0UL;
    for (size_t j = 0UL; j < newRawshape.size(); j++) {
        if (newRawshape[j] == 1) {
            newOffset[j] = 0;
            newShape[j] = 1;
            continue;
        }
        int stride = newRawshape[j];
        bool flag = true; // flag为true表示尚未遇到不为1的tile_shape
        while (stride > 1) {
            if (stride % rawShape[i] != 0) {
                ALOG_ERROR_F("Incorrect alignment.");
                return FAILED;
            }
            stride /= rawShape[i];
            if (flag && tileShape[i] != 1) {
                newOffset[j] += tileOffset[i] * stride;
                newShape[j] = tileShape[i] * stride;
                flag = false;
            } else if (flag && tileShape[i] == 1) {
                newOffset[j] += tileOffset[i] * stride;
            } else if ((!flag) && (!(tileOffset[i] == 0 && tileShape[i] == rawShape[i]))) {
                newOffset.clear();
                newShape.clear();
                return SUCCESS;
            }
            i++;
        }
        if (flag) {
            newShape[j] = 1;
        }
    }
    return SUCCESS;
}

Status SplitReshape::DynRawToAlign(const DynReshapeTilePara &shapePara,
                                    std::vector<SymbolicScalar> &newOffset,
                                    std::vector<SymbolicScalar> &newShape) {
    auto shape = shapePara.shape;
    auto alignedShape = shapePara.newShape;
    auto dynOffset = shapePara.dynOffset;
    auto dynShape = shapePara.dynShape;
    newOffset.clear();
    newShape.clear();
    for (size_t j = 0UL; j < alignedShape.size(); j++) {
        newOffset.emplace_back(SymbolicScalar(0));
        newShape.emplace_back(SymbolicScalar(0));
    }
    size_t i = 0UL;
    for (size_t j = 0UL; j < shape.size(); j++) {
        int stride = shape[j];
        if (stride == 1) {
            if (dynOffset[j].Concrete() != 0 || dynShape[j].Concrete() != 1) {
                ALOG_WARN_F("shape[%zu] = 1, but dynOffset[%zu] = %s, dynShape[%zu] = %s.", j, j, dynOffset[j].Dump().c_str(), j, dynShape[j].Dump().c_str());
                return WARNING;
            }
            continue;
        }
        bool flag = true;
        if (stride != alignedShape[i]) {
            if (!dynShape[j].IsImmediate() || !dynOffset[j].IsImmediate()) {
                ALOG_WARN_F("dynShape[%zu] = %s, dynOffset[%zu] = %s, shape[%zu] has split info.", j, dynShape[j].Dump().c_str(), j);
                return WARNING;
            }
            int currentShape = dynShape[j].Concrete();
            int currentOffset = dynOffset[j].Concrete();
            while (stride > 1) {
                stride /= alignedShape[i];
                UpdatePara updatePara = {0, 0, stride};
                if (UpdateShapeOffset(updatePara, flag, currentShape, currentOffset) == WARNING) {
                    return WARNING;
                }
                newOffset[i] = SymbolicScalar(updatePara.OffsetVal);
                newShape[i] = SymbolicScalar(updatePara.ShapeVal);
                i++;
            }
        } else {
            newShape[i] = dynShape[j];
            newOffset[i] = dynOffset[j];
            i++;
        }
    }
    return SUCCESS;
}

/*
DynAlignToRaw不需要考虑是否为空的场景
*/
Status SplitReshape::DynAlignToRaw(const DynReshapeTilePara &shapePara, std::vector<SymbolicScalar> &newOffset, std::vector<SymbolicScalar> &newShape) {
    auto rawShape = shapePara.shape;
    auto newRawshape = shapePara.newShape;
    auto dynOffset = shapePara.dynOffset;
    auto dynShape = shapePara.dynShape;
    newOffset.clear();
    newShape.clear();
    for (size_t j = 0UL; j < newRawshape.size(); j++) {
        newOffset.emplace_back(0);
        newShape.emplace_back(0);
    }
    size_t i = 0UL;
    for (size_t j = 0UL; j < newRawshape.size(); j++) {
        if (newRawshape[j] == 1) {
            newOffset[j] = SymbolicScalar(0);
            newShape[j] = SymbolicScalar(1);
            continue;
        }
        int stride = newRawshape[j];
        bool flag = true; // flag为true表示尚未遇到不为1的tile_shape
        while (stride > 1) {
            stride /= rawShape[i];
            if (flag && (!dynShape[i].IsImmediate() || dynShape[i].Concrete() != 1)) {
                newOffset[j] = newOffset[j] + dynOffset[i] * stride;
                newShape[j] = dynShape[i] * stride;
                flag = false;
            } else if (flag && dynShape[i].IsImmediate() && dynShape[i].Concrete() == 1) {
                newOffset[j] = newOffset[j] + dynOffset[i] * stride;
            } else if (!flag && (!dynOffset[i].IsImmediate() || dynOffset[i].Concrete() != 0)) {
                ALOG_ERROR_F("Incorrect alignment, found non-zero after a non-one shape.");
                return FAILED;
            } else if (!flag && (!dynShape[i].IsImmediate() || dynShape[i].Concrete() != rawShape[i])) {
                ALOG_ERROR_F("Incorrect alignment, the undetermined value is a constant value after a non-one shape.");
                return FAILED;
            }
            i++;
        }
        if (flag) {
            newShape[j] = SymbolicScalar(1);
        }
    }
    return SUCCESS;
}

Status SplitReshape::AddReshapeRemoveView(Operation &op, const OpPara &para) {
    auto overlap = para.newInput;
    auto output = para.oldOutput;
    auto reshapeOutput = para.newOutput;
    auto isAddReshapeOp = std::make_shared<ReshapeOp>(overlap, reshapeOutput);
    if (isAddReshapeOp == nullptr) {
        return FAILED;
    }
    auto consumers = output->GetConsumers();
    auto existOp = ReshapeOperationExist(isAddReshapeOp);
    if (existOp != nullptr) {
        for (auto &consumerOp : consumers) {
            if (consumerOp == nullptr) {
                return FAILED;
            }
            consumerOp->ReplaceInput(existOp->output, output);
        }
    } else {
        for (auto &consumerOp : consumers) {
            if (consumerOp == nullptr) {
                return FAILED;
            }
            consumerOp->ReplaceInput(reshapeOutput, output);
        }
    }
    redundentViewops.insert(&op);
    return SUCCESS;
}

Status SplitReshape::AddReshape(Operation &op, const OpPara &para) {
    auto input = para.oldInput;
    auto overlap = para.newInput;
    auto reshapeOutput = para.newOutput;
    auto viewOpAttribute = dynamic_cast<ViewOpAttribute *>(op.GetOpAttribute().get());
    if (viewOpAttribute == nullptr) {
        return FAILED;
    }
    auto isAddReshapeOp = std::make_shared<ReshapeOp>(overlap, reshapeOutput);
    if (isAddReshapeOp == nullptr) {
        return FAILED;
    }
    auto existOp = ReshapeOperationExist(isAddReshapeOp);
    if (existOp != nullptr) {
        op.ReplaceInput(existOp->output, input);
        viewOpAttribute->SetFromOffset(existOp->output->offset);
    } else {
        op.ReplaceInput(reshapeOutput, input);
        viewOpAttribute->SetFromOffset(reshapeOutput->offset);
    }
    return SUCCESS;
}

Status SplitReshape::ObtainCopyOutTile(Function &function, const copyOutTilePara &copyOutTile, LogicalTensors &overlaps, LogicalTensors &newOverlaps) {
    auto reshapeSource = copyOutTile.reshapeSource;
    auto inputView = copyOutTile.inputView;
    auto alignedShape = copyOutTile.alignedShape;
    auto newInputView = copyOutTile.newInputView;
    auto validShape = copyOutTile.validShape;
    for (auto &copyOutSource : copyOutSources[reshapeSource->tensor->rawmagic]) {
        // 存在多个tensor assemble成一个tensor再reshape的场景，需要使用assemble op的offset计算
        std::vector<int> copyOutOffset = mapOffset[copyOutSource->magic][reshapeSource->magic];
        std::vector<int32_t> newCopyOutTileShape;
        std::vector<int32_t> newCopyOutTileOffset;
        ReshapeTilePara CopyOutInfo = {reshapeSource->tensor->rawshape, alignedShape, copyOutOffset, copyOutSource->shape};
        Status ret = RawToAlign(CopyOutInfo, newCopyOutTileOffset, newCopyOutTileShape);
        if (ret == WARNING) {
            ALOG_WARN_F("Found RawToAlign warning case. %s", GetStr(CopyOutInfo).c_str());
            return WARNING;
        } else if (ret == FAILED) {
            ALOG_ERROR_F("Run RawToAlign failed. %s", GetStr(CopyOutInfo).c_str());
            return FAILED;
        }
        auto newCopyOutSource = std::make_shared<LogicalTensor>(function, reshapeSource->tensor, newCopyOutTileOffset, newCopyOutTileShape, validShape);
        auto status = CalcOverlap(newInputView, newCopyOutSource, true);
        if (status == OverlapStatus::PERFECTLY_MATCH || status == OverlapStatus::BE_COVERED) {
            overlaps.push_back(copyOutSource);
            newOverlaps.push_back(newCopyOutSource);
            break;
        }
        if (status == OverlapStatus::COVERED) {
            overlaps.push_back(copyOutSource);
            newOverlaps.push_back(newCopyOutSource);
        }
    }
    return SUCCESS;
}

Status SplitReshape::UpdateForPerfectlyMatchWithUB(Operation &op, const PerfectlyMatchPara &para) {
    auto overlap = para.overlap;
    auto output = para.output;
    auto reshapeOutput = para.reshapeOutput;
    OpPara reshapePara = {nullptr, output, overlap, reshapeOutput};
    if (AddReshapeRemoveView(op, reshapePara) != SUCCESS) {
        ALOG_ERROR_F("AddReshapeRemoveView failed.");
        return FAILED;
    }
    return SUCCESS;
}

Status SplitReshape::UpdateForPerfectlyMatchWithDDR(Operation &op, const PerfectlyMatchPara &para) {
    auto overlap = para.overlap;
    auto reshapeOutput = para.reshapeOutput;
    auto input = para.input;
    OpPara reshapePara = {input, nullptr, overlap, reshapeOutput};
    if (AddReshape(op, reshapePara) != SUCCESS) {
        ALOG_ERROR_F("AddReshape failed.");
        return FAILED;
    }
    return SUCCESS;
}

Status SplitReshape::ObtainReshapeSource(Function &function, const CheckParam &para, LogicalTensorPtr &newReshapeSource) {
    auto overlap = para.input;
    auto reshapeSource = para.output;
    std::vector<int> assembleOffset = mapOffset[overlap->magic][reshapeSource->magic];
    if (reshapeRawInputs.find(overlap->tensor->rawmagic) == reshapeRawInputs.end()) {
        auto reshapeRawInput = std::make_shared<RawTensor>(overlap->Datatype(), overlap->tensor->rawshape);
        if (reshapeRawInput == nullptr) {
            return FAILED;
        }
        reshapeRawInputs[overlap->tensor->rawmagic] = reshapeRawInput;
    }
    newReshapeSource = std::make_shared<LogicalTensor>(
        function, reshapeRawInputs[overlap->tensor->rawmagic], assembleOffset, overlap->shape);
    if (newReshapeSource == nullptr) {
        return FAILED;
    }
    newReshapeSource->SetMemoryTypeBoth(reshapeSource->GetMemoryTypeOriginal());
    return SUCCESS;
}

Status SplitReshape::UpdateForPerfectlyMatchOtherCase(Function &function, Operation &op, const PerfectlyMatchPara &para) {
    auto overlap = para.overlap;
    auto reshapeSource = para.reshapeSource;
    auto input = para.input;
    auto reshapeOutput = para.reshapeOutput;
    LogicalTensorPtr newReshapeSource;
    CheckParam checkPara = {overlap, reshapeSource, nullptr};
    std::vector<int> assembleOffset = mapOffset[overlap->magic][reshapeSource->magic];
    if (ObtainReshapeSource(function, checkPara, newReshapeSource) != SUCCESS) {
        ALOG_ERROR_F("ObtainReshapeSource failed.");
        return FAILED;
    }
    auto viewOpAttribute = dynamic_cast<ViewOpAttribute *>(op.GetOpAttribute().get());
    if (viewOpAttribute == nullptr) {
        return FAILED;
    }
    auto isAddReshapeOp = std::make_shared<ReshapeOp>(newReshapeSource, reshapeOutput);
    if (isAddReshapeOp == nullptr) {
        return FAILED;
    }
    auto existOp = ReshapeOperationExist(isAddReshapeOp);
    if (existOp != nullptr) {
        op.ReplaceInput(existOp->output, input);
        viewOpAttribute->SetFromOffset(existOp->output->offset);
    } else {
        assembles.emplace_back(
            AssembleOp{overlap->GetMemoryTypeOriginal(), assembleOffset, overlap, newReshapeSource});
        op.ReplaceInput(reshapeOutput, input);
        viewOpAttribute->SetFromOffset(reshapeOutput->offset);
    }
    return SUCCESS;
}

Status SplitReshape::UpdateForPerfectlyMatch(Function &function, Operation &op, const CalcOverlapPara &para) {
    std::vector<int32_t> alignedShape = para.alignedShape;
    LogicalTensors newOverlaps = para.newOverlaps;
    LogicalTensors overlaps = para.overlaps;
    LogicalTensorPtr reshapeSource = para.reshapeSource;
    LogicalTensorPtr input = para.input;
    LogicalTensorPtr output = para.output;
    LogicalTensorPtr inputView = para.inputView;
    auto overlap = overlaps.front();
    auto newoverlap = newOverlaps.front();
    ReshapeTilePara reshapeInfo2 = {alignedShape, inputView->tensor->rawshape, newoverlap->offset, newoverlap->shape};
    std::vector<int32_t> reshapeTileShape;
    std::vector<int32_t> reshapeTileOffset;
    if (AlignToRaw(reshapeInfo2, reshapeTileOffset, reshapeTileShape) != SUCCESS) {
        ALOG_ERROR_F("AlignToRaw failed. %s", GetStr(reshapeInfo2).c_str());
        return FAILED;
    }
    if (reshapeTileShape != output->shape) {
        ALOG_ERROR_F("reshapeTileShape != output->shape");
        return FAILED;
    }
    if (reshapeRawOutputs.find(overlap->tensor->rawmagic) == reshapeRawOutputs.end()) {
        auto reshaperawOutput = std::make_shared<RawTensor>(input->Datatype(), inputView->tensor->rawshape);
        if (reshaperawOutput == nullptr) {
            return FAILED;
        }
        reshapeRawOutputs[overlap->tensor->rawmagic] = reshaperawOutput;
    }
    auto reshapeOutput = std::make_shared<LogicalTensor>(function,
        reshapeRawOutputs[overlap->tensor->rawmagic], reshapeTileOffset, reshapeTileShape);
    reshapeOutput->SetMemoryTypeBoth(input->GetMemoryTypeOriginal());
    PerfectlyMatchPara perfectlyMatchPara = {input, output, overlap, reshapeSource, reshapeOutput};
    if (overlap->GetMemoryTypeOriginal() == output->GetMemoryTypeOriginal() && overlap->GetMemoryTypeOriginal() == MemoryType::MEM_UB) {
        reshapeOutput->SetMemoryTypeBoth(output->GetMemoryTypeOriginal(), true);
        if (UpdateForPerfectlyMatchWithUB(op, perfectlyMatchPara) != SUCCESS) {
            ALOG_ERROR_F("UpdateForPerfectlyMatchWithUB failed");
            return FAILED;
        }
    } else if (overlap->GetMemoryTypeOriginal() != output->GetMemoryTypeOriginal() && overlap->GetMemoryTypeOriginal() == MemoryType::MEM_DEVICE_DDR) {
        if (UpdateForPerfectlyMatchWithDDR(op, perfectlyMatchPara) != SUCCESS) {
            ALOG_ERROR_F("UpdateForPerfectlyMatchWithDDR failed");
            return FAILED;
        }
    } else {
        if (UpdateForPerfectlyMatchOtherCase(function, op, perfectlyMatchPara) != SUCCESS) {
            ALOG_ERROR_F("UpdateForPerfectlyMatchOtherCase failed");
            return FAILED;
        }
    }
    return SUCCESS;
}

Status SplitReshape::UpdateForBeCoveredUBDDR(Operation &op, const BeCoveredPara &para) {
    auto input = para.input;
    auto overlap = para.overlap;
    auto reshapeOutput = para.reshapeOutput;
    auto newOffset = para.newOffset;
    auto isAddReshapeOp = std::make_shared<ReshapeOp>(overlap, reshapeOutput);
    if (isAddReshapeOp == nullptr) {
        return FAILED;
    }
    auto viewOpAttribute = dynamic_cast<ViewOpAttribute *>(op.GetOpAttribute().get());
    if (viewOpAttribute == nullptr) {
        return FAILED;
    }
    auto existOp = ReshapeOperationExist(isAddReshapeOp);
    if (existOp != nullptr) {
        viewOpAttribute->SetFromOffset(newOffset);
        op.ReplaceInput(existOp->output, input);
    } else {
        viewOpAttribute->SetFromOffset(newOffset);
        op.ReplaceInput(reshapeOutput, input);
    }
    return SUCCESS;
}

Status SplitReshape::UpdateForBeCoveredOtherCase(Function &function, Operation &op, const BeCoveredPara &para) {
    auto input = para.input;
    auto overlap = para.overlap;
    auto reshapeOutput = para.reshapeOutput;
    auto reshapeSource = para.reshapeSource;
    auto newOffset = para.newOffset;
    
    auto viewOpAttribute = dynamic_cast<ViewOpAttribute *>(op.GetOpAttribute().get());
    if (viewOpAttribute == nullptr) {
        return FAILED;
    }
    LogicalTensorPtr newReshapeSource;
    CheckParam checkPara = {overlap, reshapeSource, nullptr};
    if (ObtainReshapeSource(function, checkPara, newReshapeSource) != SUCCESS) {
        ALOG_ERROR_F("ObtainReshapeSource failed.");
        return FAILED;
    }
    auto isAddReshapeOp = std::make_shared<ReshapeOp>(newReshapeSource, reshapeOutput);
    auto existOp = ReshapeOperationExist(isAddReshapeOp);
    if (existOp != nullptr) {
        viewOpAttribute->SetFromOffset(newOffset);
        op.ReplaceInput(existOp->output, input);
    } else {
        assembles.emplace_back(AssembleOp{
            overlap->GetMemoryTypeOriginal(), newReshapeSource->offset, overlap, newReshapeSource});
        viewOpAttribute->SetFromOffset(newOffset);
        op.ReplaceInput(reshapeOutput, input);
    }
    return SUCCESS;
}

Status SplitReshape::ProcessBeCovered(Function &function, Operation &op, const CalcOverlapPara &para, const BeCoveredPara &beCoveredPara, LogicalTensorPtr &reshapeOutput) {
    auto overlap = para.overlaps.front();
    auto output = para.output;
    if ((overlap->GetMemoryTypeOriginal() == output->GetMemoryTypeOriginal() && overlap->GetMemoryTypeOriginal() == MemoryType::MEM_UB) || (overlap->GetMemoryTypeOriginal() != output->GetMemoryTypeOriginal() && overlap->GetMemoryTypeOriginal() == MemoryType::MEM_DEVICE_DDR)) {
        if (overlap->GetMemoryTypeOriginal() == output->GetMemoryTypeOriginal() && overlap->GetMemoryTypeOriginal() == MemoryType::MEM_UB) {
            reshapeOutput->SetMemoryTypeBoth(output->GetMemoryTypeOriginal(), true);
        }
        if (UpdateForBeCoveredUBDDR(op, beCoveredPara) != SUCCESS) {
            ALOG_ERROR_F("Process UpdateForBeCoveredUBDDR failed.");
            return FAILED;
        }
    } else {
        if (UpdateForBeCoveredOtherCase(function, op, beCoveredPara) != SUCCESS) {
            ALOG_ERROR_F("Process UpdateForBeCoveredOtherCase failed.");
            return FAILED;
        }
    }
    return SUCCESS;
}

Status SplitReshape::UpdateForBeCovered(Function &function, Operation &op, const CalcOverlapPara &para) {
    auto alignedShape = para.alignedShape;
    auto reshapeSource = para.reshapeSource;
    auto input = para.input;
    auto output = para.output;
    auto inputView = para.inputView;
    auto overlap = para.overlaps.front();
    auto newoverlap = para.newOverlaps.front();
    std::vector<int32_t> newShape;
    std::vector<int32_t> newOffset;
    std::vector<int32_t> reshapeTileShape;
    std::vector<int32_t> reshapeTileOffset;
    ReshapeTilePara newInfo = {alignedShape, inputView->tensor->rawshape, para.newInputViewTileOffset, para.newInputViewTileShape};
    if (AlignToRaw(newInfo, newOffset, newShape) != SUCCESS) {
        ALOG_ERROR_F("Failed to compute the tile of the raw input. %s", GetStr(newInfo).c_str());
        return FAILED;
    }
    if (newShape != output->shape) {
        ALOG_ERROR_F("The new input shape of view[%d] does not equal to output.", op.GetOpMagic());
        return FAILED;
    }
    ReshapeTilePara reshapeTileInfo = {alignedShape, inputView->tensor->rawshape, newoverlap->offset, newoverlap->shape};
    if (AlignToRaw(reshapeTileInfo, reshapeTileOffset, reshapeTileShape) != SUCCESS) {
        ALOG_ERROR_F("Failed to compute the tile of the raw input of inputView. %s", GetStr(reshapeTileInfo).c_str());
        return FAILED;
    }
    if (reshapeTileOffset.size() == 0 && reshapeTileShape.size() == 0) {
        ALOG_WARN_F("One new assemble overlap tile block cannot cover the input of view op[%d].", op.GetOpMagic());
        return SUCCESS; // 这种情况不会对reshape做切分
    }
    if (reshapeRawOutputs.find(overlap->tensor->rawmagic) == reshapeRawOutputs.end()) {
        auto reshaperawOutput = std::make_shared<RawTensor>(input->Datatype(), inputView->tensor->rawshape);
        if (reshaperawOutput == nullptr) {
            return FAILED;
        }
        reshapeRawOutputs[overlap->tensor->rawmagic] = reshaperawOutput;
    }
    auto reshapeOutput = std::make_shared<LogicalTensor>(function, reshapeRawOutputs[overlap->tensor->rawmagic], reshapeTileOffset, reshapeTileShape);
    reshapeOutput->SetMemoryTypeBoth(input->GetMemoryTypeOriginal());
    BeCoveredPara beCoveredPara = {overlap, input, reshapeOutput, reshapeSource, newOffset};
    if (ProcessBeCovered(function, op, para, beCoveredPara, reshapeOutput)) {
        ALOG_ERROR_F("Process ProcessBeCovered failed.");
        return FAILED;
    }
    return SUCCESS;
}

Status SplitReshape::UpdateForAssembleAfterReshapeWithUB(Operation &op, const AssemblePara &para) {
    auto input = para.input;
    auto output = para.output;
    auto overlap = para.overlap;
    auto newReshapeOutput = para.newReshapeOutput;
    auto inputView = para.inputView;
    auto newReshapeOutputTileOffset = para.newReshapeOutputTileOffset;
    newReshapeOutput->SetMemoryTypeBoth(output->GetMemoryTypeOriginal(), true);
    std::vector<int> newOffset(inputView->offset.size(), 0);
    std::transform(newReshapeOutputTileOffset.begin(), newReshapeOutputTileOffset.end(),
        inputView->offset.begin(), newOffset.begin(), [](int a, int b) { return a - b; });
    auto isAddReshapeOp = std::make_shared<ReshapeOp>(overlap, newReshapeOutput);
    if (isAddReshapeOp == nullptr) {
        return FAILED;
    }
    auto existOp = ReshapeOperationExist(isAddReshapeOp);
    if (existOp != nullptr) {
        assembles.emplace_back(
            AssembleOp{newReshapeOutput->GetMemoryTypeOriginal(), newOffset, existOp->output, output});
    } else {
        assembles.emplace_back(
            AssembleOp{newReshapeOutput->GetMemoryTypeOriginal(), newOffset, newReshapeOutput, output});
    }
    redundentViewops.insert(&op);
    return SUCCESS;
}

Status SplitReshape::UpdateForAssembleAfterReshapeWithDDR(Operation &op, const AssemblePara &para) {
    auto input = para.input;
    auto newInput = para.newInput;
    auto output = para.output;
    auto overlap = para.overlap;
    auto newReshapeOutput = para.newReshapeOutput;
    auto inputView = para.inputView;
    auto newReshapeOutputTileOffset = para.newReshapeOutputTileOffset;
    auto viewOpAttribute = dynamic_cast<ViewOpAttribute *>(op.GetOpAttribute().get());
    if (viewOpAttribute == nullptr) {
        return FAILED;
    }
    std::vector<int> newOffset(inputView->offset.size(), 0);
    std::transform(newReshapeOutputTileOffset.begin(), newReshapeOutputTileOffset.end(),
        inputView->offset.begin(), newOffset.begin(), [](int a, int b) { return a - b; });
    auto isAddReshapeOp = std::make_shared<ReshapeOp>(overlap, newReshapeOutput);
    
    auto existOp = ReshapeOperationExist(isAddReshapeOp);
    if (existOp != nullptr) {
        assembles.emplace_back(
            AssembleOp{newReshapeOutput->GetMemoryTypeOriginal(), newOffset, existOp->output, newInput});
        viewOpAttribute->SetFromOffset(newInput->offset);
        op.ReplaceInput(newInput, input);
    } else {
        assembles.emplace_back(
            AssembleOp{newReshapeOutput->GetMemoryTypeOriginal(), newOffset, newReshapeOutput, newInput});
        viewOpAttribute->SetFromOffset(newInput->offset);
        op.ReplaceInput(newInput, input);
    }
    return SUCCESS;
}

Status SplitReshape::UpdateForAssembleAfterReshapeOtherCase(Function &function, Operation &op, const AssemblePara &para) {
    LogicalTensorPtr newReshapeSource;
    auto input = para.input;
    auto newInput = para.newInput;
    auto output = para.output;
    auto overlap = para.overlap;
    auto reshapeSource = para.reshapeSource;
    auto newReshapeOutput = para.newReshapeOutput;
    auto inputView = para.inputView;
    auto newReshapeOutputTileOffset = para.newReshapeOutputTileOffset;
    auto viewOpAttribute = dynamic_cast<ViewOpAttribute *>(op.GetOpAttribute().get());
    if (viewOpAttribute == nullptr) {
        return FAILED;
    }
    std::vector<int> assembleOffset = mapOffset[overlap->magic][reshapeSource->magic];
    CheckParam checkPara = {overlap, reshapeSource, nullptr};
    if (ObtainReshapeSource(function, checkPara, newReshapeSource) != SUCCESS) {
        ALOG_ERROR_F("ObtainReshapeSource failed.");
        return FAILED;
    }
    std::vector<int> newOffset(inputView->offset.size(), 0);
    std::transform(newReshapeOutputTileOffset.begin(), newReshapeOutputTileOffset.end(), inputView->offset.begin(), newOffset.begin(), [](int a, int b) { return a - b; });
    auto isAddReshapeOp = std::make_shared<ReshapeOp>(newReshapeSource, newReshapeOutput);
    auto existOp = ReshapeOperationExist(isAddReshapeOp);
    if (existOp != nullptr) {
        assembles.emplace_back(AssembleOp{newReshapeOutput->GetMemoryTypeOriginal(), newOffset, existOp->output, newInput});
        viewOpAttribute->SetFromOffset(newInput->offset);
        op.ReplaceInput(newInput, input);
    } else {
        assembles.emplace_back(AssembleOp{overlap->GetMemoryTypeOriginal(), assembleOffset, overlap, newReshapeSource});
        assembles.emplace_back(AssembleOp{newReshapeOutput->GetMemoryTypeOriginal(), newOffset, newReshapeOutput, newInput});
        viewOpAttribute->SetFromOffset(newInput->offset);
        op.ReplaceInput(newInput, input);
    }
    return SUCCESS;
}

Status SplitReshape::UpdateForAssembleAfterReshape(Function &function, Operation &op, const CalcOverlapPara &para) {
    std::vector<int32_t> alignedShape = para.alignedShape;
    LogicalTensorPtr reshapeSource = para.reshapeSource;
    LogicalTensors overlaps = para.overlaps;
    LogicalTensors newOverlaps = para.newOverlaps;
    LogicalTensorPtr input = para.input;
    LogicalTensorPtr output = para.output;
    LogicalTensorPtr inputView = para.inputView;
    auto newInput = std::make_shared<LogicalTensor>(function, input->Datatype(), output->shape);
    if (newInput == nullptr) {
        return FAILED;
    }
    newInput->SetMemoryTypeBoth(input->GetMemoryTypeOriginal());
    for (size_t i = 0; i < overlaps.size(); i++) {
        std::vector<int32_t> newReshapeOutputTileShape;
        std::vector<int32_t> newReshapeOutputTileOffset;
        ReshapeTilePara newInfo = {alignedShape, inputView->tensor->rawshape, newOverlaps[i]->offset, newOverlaps[i]->shape};
        if (AlignToRaw(newInfo, newReshapeOutputTileOffset, newReshapeOutputTileShape) != SUCCESS) {
            ALOG_ERROR_F("Failed to compute the tile of the raw input of inputView. %s", GetStr(newInfo).c_str());
            return FAILED;
        }
        if (reshapeRawOutputs.find(overlaps[i]->tensor->rawmagic) == reshapeRawOutputs.end()) {
            auto reshaperawOutput = std::make_shared<RawTensor>(input->Datatype(), inputView->tensor->rawshape);
            if (reshaperawOutput == nullptr) {
                return FAILED;
            }
            reshapeRawOutputs[overlaps[i]->tensor->rawmagic] = reshaperawOutput;
        }
        auto newReshapeOutput = std::make_shared<LogicalTensor>(function, reshapeRawOutputs[overlaps[i]->tensor->rawmagic], newReshapeOutputTileOffset, newReshapeOutputTileShape);
        newReshapeOutput->SetMemoryTypeBoth(input->GetMemoryTypeOriginal());
        AssemblePara assemblePara = {input, output, reshapeSource, newInput, newReshapeOutput, inputView, overlaps[i], newReshapeOutputTileOffset};
        if ((overlaps.front()->GetMemoryTypeOriginal() == output->GetMemoryTypeOriginal() && overlaps.front()->GetMemoryTypeOriginal() == MemoryType::MEM_UB)) {
            if (UpdateForAssembleAfterReshapeWithUB(op, assemblePara) != SUCCESS) {
                ALOG_ERROR_F("Process UpdateForAssembleAfterReshapeWithUB failed.");
                return FAILED;
            }
        } else if ((overlaps.front()->GetMemoryTypeOriginal() != output->GetMemoryTypeOriginal() && overlaps.front()->GetMemoryTypeOriginal() == MemoryType::MEM_DEVICE_DDR)) {
            if (UpdateForAssembleAfterReshapeWithDDR(op, assemblePara) != SUCCESS) {
                ALOG_ERROR_F("Process UpdateForAssembleAfterReshapeWithDDR failed.");
                return FAILED;
            }
        } else {
            if (UpdateForAssembleAfterReshapeOtherCase(function, op, assemblePara) != SUCCESS) {
                ALOG_ERROR_F("Process UpdateForAssembleAfterReshapeOtherCase failed.");
                return FAILED;
            }
        }
    }
    return SUCCESS;
}

Status SplitReshape::UpdateForPerfectlyMatchWithAllWithUB(Operation &op, const PerfectlyMatchWithAllPara &para) {
    auto output = para.output;
    auto overlap = para.overlap;
    auto reshapeOutput = para.reshapeOutput;
    auto newReshapeSource = para.newReshapeSource;
    newReshapeSource->SetMemoryTypeBoth(overlap->GetMemoryTypeOriginal(), true);
    OpPara reshapePara = {nullptr, output, newReshapeSource, reshapeOutput};
    if (AddReshapeRemoveView(op, reshapePara) != SUCCESS) {
        ALOG_ERROR_F("Process AddReshapeRemoveView failed.");
        return FAILED;
    }
    return SUCCESS;
}

Status SplitReshape::UpdateForPerfectlyMatchWithAllOtherCase(Operation &op, const PerfectlyMatchWithAllPara &para) {
    auto input = para.input;
    auto reshapeOutput = para.reshapeOutput;
    auto newReshapeSource = para.newReshapeSource;
    reshapeOutput->SetMemoryTypeBoth(input->GetMemoryTypeOriginal(), true);
    OpPara reshapePara = {input, nullptr, newReshapeSource, reshapeOutput};
    if (AddReshape(op, reshapePara) != SUCCESS) {
        ALOG_ERROR_F("Process AddReshape failed.");
        return FAILED;
    }
    return SUCCESS;
}

Status SplitReshape::ProcessMultitoOne(Function &function, Operation &op, const CalcOverlapPara &para, const std::vector<int32_t> &newReshapeSourceTileShape, 
                                        const std::vector<int32_t> &newReshapeSourceTileOffset) {
    LogicalTensors overlaps = para.overlaps;
    LogicalTensorPtr reshapeSource = para.reshapeSource;
    LogicalTensorPtr input = para.input;
    LogicalTensorPtr output = para.output;
    LogicalTensorPtr inputView = para.inputView;
    if (reshapeRawInputs.find(overlaps.front()->tensor->rawmagic) == reshapeRawInputs.end()) {
        auto reshapeRawInput = std::make_shared<RawTensor>(overlaps.front()->Datatype(), overlaps.front()->tensor->rawshape);
        if (reshapeRawInput == nullptr) {
            return FAILED;
        }
        reshapeRawInputs[overlaps.front()->tensor->rawmagic] = reshapeRawInput;
    }
    auto newReshapeSource = std::make_shared<LogicalTensor>(function, reshapeRawInputs[overlaps.front()->tensor->rawmagic], newReshapeSourceTileOffset, newReshapeSourceTileShape);
    if (newReshapeSource == nullptr) {
        return FAILED;
    }
    newReshapeSource->SetMemoryTypeBoth(reshapeSource->GetMemoryTypeOriginal());
    if (reshapeRawOutputs.find(overlaps.front()->tensor->rawmagic) == reshapeRawOutputs.end()) {
        auto reshaperawOutput = std::make_shared<RawTensor>(input->Datatype(), inputView->tensor->rawshape);
        if (reshaperawOutput == nullptr) {
            return FAILED;
        }
        reshapeRawOutputs[overlaps.front()->tensor->rawmagic] = reshaperawOutput;
    }
    auto reshapeOutput = std::make_shared<LogicalTensor>(function, reshapeRawOutputs[overlaps.front()->tensor->rawmagic], inputView->offset, inputView->shape);
    if (reshapeOutput == nullptr) {
        return FAILED;
    }
    reshapeOutput->SetMemoryTypeBoth(output->GetMemoryTypeOriginal());
    for (auto &overlap : overlaps) {
        std::vector<int> overlapOffset = mapOffset[overlap->magic][reshapeSource->magic];
        assembles.emplace_back(AssembleOp{overlap->GetMemoryTypeOriginal(), overlapOffset, overlap, newReshapeSource});
    }
    PerfectlyMatchWithAllPara perfectlyMatchwithAllPara = {input, output, overlaps.front(), reshapeOutput, newReshapeSource};
    if (overlaps.front()->GetMemoryTypeOriginal() == output->GetMemoryTypeOriginal() &&
        overlaps.front()->GetMemoryTypeOriginal() == MemoryType::MEM_UB) {
        if (UpdateForPerfectlyMatchWithAllWithUB(op, perfectlyMatchwithAllPara) != SUCCESS) {
            ALOG_ERROR_F("Process UpdateForPerfectlyMatchWithAllWithUB failed.");
            return FAILED;
        }
    } else {
        if (UpdateForPerfectlyMatchWithAllOtherCase(op, perfectlyMatchwithAllPara) != SUCCESS) {
            ALOG_ERROR_F("Process UpdateForPerfectlyMatchWithAllOtherCase failed.");
            return FAILED;
        }
    }
    return SUCCESS;
}

Status SplitReshape::UpdateForPerfectlyMatchWithAll(Function &function, Operation &op, const CalcOverlapPara &para) {
    std::vector<int32_t> alignedShape = para.alignedShape;
    std::vector<int32_t> newInputViewTileOffset = para.newInputViewTileOffset;
    std::vector<int32_t> newInputViewTileShape = para.newInputViewTileShape;
    LogicalTensorPtr reshapeSource = para.reshapeSource;
    std::vector<int32_t> newReshapeSourceTileShape;
    std::vector<int32_t> newReshapeSourceTileOffset;
    ReshapeTilePara newInfo = {alignedShape, reshapeSource->tensor->rawshape, newInputViewTileOffset, newInputViewTileShape};
    if (AlignToRaw(newInfo, newReshapeSourceTileOffset, newReshapeSourceTileShape) != SUCCESS) {
        ALOG_ERROR_F("Failed to compute the tile of the raw input of reshapeSource. %s", GetStr(newInfo).c_str());
        return FAILED;
    }
    // reshape前的tile无法assemble表达成一个tile时需要先reshape成alignshape然后再assemble成dstshape
    if (newReshapeSourceTileOffset.size() == 0 && newReshapeSourceTileShape.size() == 0) {
        if (UpdateForAssembleAfterReshape(function, op, para) != SUCCESS) {
            ALOG_ERROR_F("Process UpdateForAssembleAfterReshape failed.");
            return FAILED;
        }
    } else {
        if (ProcessMultitoOne(function, op, para, newReshapeSourceTileShape, newReshapeSourceTileOffset) != SUCCESS) {
            ALOG_ERROR_F("Process ProcessMultitoOne failed.");
            return FAILED;
        }
    }
    return SUCCESS;
}

Status SplitReshape::UpdateReshapeOp(Function &function, Operation &op, const OverlapStatus &status, const CalcOverlapPara &calcpara) {
    if (status == OverlapStatus::PERFECTLY_MATCH) {
        if (UpdateForPerfectlyMatch(function, op, calcpara) != SUCCESS) {
            ALOG_ERROR_F("Process UpdateForPerfectlyMatch of view[%d] failed.", op.GetOpMagic());
            return FAILED;
        }
    } else if (status == OverlapStatus::BE_COVERED) {
        if (UpdateForBeCovered(function, op, calcpara) != SUCCESS) {
            ALOG_ERROR_F("Process UpdateForBeCovered of view[%d] failed.", op.GetOpMagic());
            return FAILED;
        }
    } else if (status == OverlapStatus::PERFECTLY_MATCH_WITH_ALL) {
        if (UpdateForPerfectlyMatchWithAll(function, op, calcpara) != SUCCESS) {
            ALOG_ERROR_F("Process UpdateForPerfectlyMatchWithAll of view[%d] failed.", op.GetOpMagic());
            return FAILED;
        }
    } else {
        ALOG_WARN_F("The new input of view[%d] intersects the input of assemble, skip splitreshape.", op.GetOpMagic());
    }
    return SUCCESS;
}

Status SplitReshape::CheckValidOp(const CheckParam &para, LogicalTensorPtr &reshapeSource,
                                std::vector<int32_t> &alignedShape, std::vector<int32_t> &newInputViewTileOffset, std::vector<int32_t> &newInputViewTileShape) {
    auto input = para.input;
    auto output = para.output;
    auto inputView = para.inputView;
    if (input->shape == output->shape) {
        ALOG_WARN_F("The input and output have the same shape");
        return WARNING;
    }
    if (reshapeSources.find(input->tensor->rawmagic) == reshapeSources.end()) {
        ALOG_WARN_F("ReshapeOp has no preceding reshapeop.");
        return WARNING;
    }
    reshapeSource = reshapeSources[input->tensor->rawmagic];
    if (!CheckSplit(reshapeSource)) {
        ALOG_WARN_F("Tiling block from different raw tensor.");
        return WARNING;
    }
    if (!CheckDynStatus(reshapeSource, input)) {
        ALOG_WARN_F("Dynamic shape found in reshape source.");
        return WARNING;
    } 
    if (ShapeAlign(reshapeSource->tensor->rawshape, input->tensor->rawshape, alignedShape) == WARNING) {
        ALOG_WARN_F("Can not construct align for rawshape.");
        return WARNING;
    }
    ReshapeTilePara reshapeInfo = {inputView->tensor->rawshape, alignedShape, inputView->offset, inputView->shape};
    Status alignRet = RawToAlign(reshapeInfo, newInputViewTileOffset, newInputViewTileShape);
    if (alignRet == WARNING) {
        ALOG_WARN_F("Cannot process the cond. %s", GetStr(reshapeInfo).c_str());
        return WARNING;
    } else if (alignRet == FAILED) {
        ALOG_WARN_F("Process RawToAlign failed. %s", GetStr(reshapeInfo).c_str());
        return FAILED;
    }
    return SUCCESS;
}

Status SplitReshape::CheckOp(Function &function, Operation &op) {
    LogicalTensors overlaps;
    LogicalTensors newOverlaps;
    LogicalTensorPtr reshapeSource;
    std::vector<SymbolicScalar> validShape;
    std::vector<int32_t> alignedShape;
    std::vector<int32_t> newInputViewTileShape;
    std::vector<int32_t> newInputViewTileOffset;
    auto input = op.GetIOperands().front();
    auto output = op.GetOOperands().front();
    if (input == nullptr || output == nullptr || input->tensor == nullptr) {
        ALOG_ERROR_F("input == nullptr || output == nullptr || input->tensor == nullptr");
        return FAILED;
    }
    auto viewOpAttribute = dynamic_cast<ViewOpAttribute *>(op.GetOpAttribute().get());
    if (viewOpAttribute == nullptr) {
        return FAILED;
    }
    auto inputView = std::make_shared<LogicalTensor>(function, input->tensor, viewOpAttribute->GetFrom(), output->shape);
    CheckParam checkParam = {input, output, inputView};
    auto checkRet = CheckValidOp(checkParam, reshapeSource, alignedShape, newInputViewTileOffset, newInputViewTileShape);
    if (checkRet == WARNING) {
        ALOG_WARN_F("Skip splitreshape for op[%d].", op.GetOpMagic());
        return SUCCESS;
    } else if (checkRet == FAILED) {
        ALOG_ERROR_F("Failed to CheckValidOp for op[%d].", op.GetOpMagic());
        return FAILED;
    }
    auto newInputView = std::make_shared<LogicalTensor>(function, inputView->tensor, newInputViewTileOffset, newInputViewTileShape, validShape);
    copyOutTilePara copyOutTile = {reshapeSource, inputView, newInputView, alignedShape, validShape};
    Status ret = ObtainCopyOutTile(function, copyOutTile, overlaps, newOverlaps);
    if (ret == WARNING) {
        ALOG_WARN_F("Obtain CopyOutTile failed, skip splitreshape for [%d].", op.GetOpMagic());
        return SUCCESS;
    } else if (ret == FAILED) {
        ALOG_ERROR_F("Process ObtainCopyOutTile failed.");
        return FAILED;
    }
    auto status = CalcOverlap(newInputView, newOverlaps, true);
    CalcOverlapPara calcpara = {alignedShape, reshapeSource, newInputViewTileOffset, newInputViewTileShape, overlaps, newOverlaps, input, inputView, output};
    if (UpdateReshapeOp(function, op, status, calcpara) != SUCCESS) {
        ALOG_ERROR_F("Process UpdateReshapeOp failed.");
        return FAILED;
    }
    return SUCCESS;
}

Status SplitReshape::CheckCopyIn(Function &function) {
    for (auto &op : function.Operations()) {
        if (op.GetOpcode() != Opcode::OP_VIEW) {
            continue;
        }
        if (CheckOp(function, op) == FAILED) {
            ALOG_WARN_F("Run CheckOp failed.");
            return FAILED;
        }
    }
    return SUCCESS;
}

Status SplitReshape::AddOperation(Function &function) {
    for (auto &a : assembles) {
        auto &newCopyOut = function.AddOperation(Opcode::OP_ASSEMBLE, {a.input}, {a.output});
        newCopyOut.SetOpAttribute(std::make_shared<AssembleOpAttribute>(a.from, a.toOffset));
        ALOG_INFO_F("ADD OP_ASSEMBLE, magic %d ,IOperand tensor magic %d OOperand tensor magic %d", newCopyOut.opmagic,
            a.input->magic, a.output->magic);
    }
    for (auto &b : reshapes) {
        auto &newReshape = function.AddOperation(Opcode::OP_RESHAPE, {b.second->input}, {b.second->output});
        // 后续可能会用到
        ALOG_INFO_F("ADD OP_RESHAPE, magic %d ,IOperand tensor magic %d OOperand tensor magic %d", newReshape.opmagic,
            b.second->input->magic, b.second->output->magic);
    }
    return SUCCESS;
}

Status SplitReshape::EraseReshape(Function &function) {
    // 先删除view使reshape的Consumers为空
    for (auto &opView : redundentViewops) {
        if (opView == nullptr) {
            return FAILED;
        }
        opView->SetAsDeleted();
    }
    function.EraseOperations(true, false);

    for (auto &op : function.Operations(false)) {
        if (op.GetOpcode() != Opcode::OP_RESHAPE) {
            continue;
        }
        if (op.oOperand.empty()) {
            op.SetAsDeleted();
            continue;
        }
        auto output=op.oOperand.front();
        if (output == nullptr) {
            return FAILED;
        }
        if (output->nodetype == NodeType::LOCAL && output->GetConsumers().empty()) {
            op.SetAsDeleted();
        }
    }
    function.EraseOperations(true, false);

    for (auto &op : function.Operations(false)) {
        if (op.GetOpcode() != Opcode::OP_ASSEMBLE) {
            continue;
        }
        auto output = op.oOperand.front();
        if (output == nullptr) {
            return FAILED;
        }
        if (output->nodetype == NodeType::LOCAL && output->GetConsumers().empty()) {
            op.SetAsDeleted();
        }
    }
    function.EraseOperations(true, true);
    return SUCCESS;
}

Status SplitReshape::SetMemoryType(Function &function) {
    for (auto& op: function.Operations()) {
        if (op.GetOpcode() != Opcode::OP_VIEW) {
            continue;
        }
        for (auto consumerOp : op.ConsumerOps()) {
            if (consumerOp->GetOpcode() == Opcode::OP_RESHAPE &&
                op.GetOOperands()[0]->GetMemoryTypeOriginal() == MemoryType::MEM_UNKNOWN) {
                op.GetOOperands()[0]->SetMemoryTypeBoth(consumerOp->GetOOperands()[0]->GetMemoryTypeOriginal());
            }
        }
    }
    return SUCCESS;
}
} // namespace npu::tile_fwk