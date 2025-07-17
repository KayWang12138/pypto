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
 * \file split_reshape_pvc2.cpp
 * \brief
 */

#include "split_reshape_pvc2.h"
#include "interface/tensor/logical_tensor.h"

namespace npu::tile_fwk {
Status SplitReshapeOpPVC2::RunOnFunction(Function &function) {
    ASLOGI("===> start SplitReshapeOpPVC2");
    copyOutSources.clear();
    reshapeSources.clear();
    mappingOffset.clear();
    assembles.clear();
    reshapes.clear();
    redundentViewops.clear();
    reshapeRawOutputs.clear();
    CollectCopyOut(function);
    CheckCopyIn(function);
    for (auto &a : assembles) {
        auto &newCopyOut = function.AddOperation(Opcode::OP_ASSEMBLE, {a.input}, {a.output});
        newCopyOut.SetOpAttribute(std::make_shared<AssembleOpAttribute>(a.from, a.toOffset));
        ASLOGI("ADD OP_ASSEMBLE, magic %d ,IOperand tensor magic %d OOperand tensor magic %d", newCopyOut.opmagic,
            a.input->magic, a.output->magic);
    }
    for (auto &b : reshapes) {
        auto &newReshape = function.AddOperation(Opcode::OP_RESHAPE, {b.second->input}, {b.second->output});
        // 后续可能会用到
        ASLOGI("ADD OP_RESHAPE, magic %d ,IOperand tensor magic %d OOperand tensor magic %d", newReshape.opmagic,
            b.second->input->magic, b.second->output->magic);
    }
    EraseReshape(function);
    EliminateDeadOperationBackward(function);
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
    ASLOGI("===> end SplitReshapeOpPVC2");
    return SUCCESS;
}

std::shared_ptr<ReshapeOp> SplitReshapeOpPVC2::ReshapeOperationExist(const std::shared_ptr<ReshapeOp> &isAddReshapeop) {
    if (reshapes.count(ComputeReshapeHash(isAddReshapeop->input, isAddReshapeop->output)) != 0) {
        return reshapes[ComputeReshapeHash(isAddReshapeop->input, isAddReshapeop->output)];
    } else {
        reshapes.insert({ComputeReshapeHash(isAddReshapeop->input, isAddReshapeop->output), isAddReshapeop});
        return nullptr;
    }
}

unsigned long SplitReshapeOpPVC2::ComputeReshapeHash(
    const std::shared_ptr<LogicalTensor> &input, const std::shared_ptr<LogicalTensor> &output) const {
    unsigned long operationHash = ComputeReshapeHashOrderless(input, output);
    return operationHash;
}

unsigned long SplitReshapeOpPVC2::ComputeReshapeHashOrderless(
    const std::shared_ptr<LogicalTensor> &input, const std::shared_ptr<LogicalTensor> &output) const {
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

void SplitReshapeOpPVC2::CollectCopyOut(Function &function) {
    for (auto &op : function.Operations()) {
        if (op.GetOpcode() == Opcode::OP_RESHAPE) {
            auto input = op.GetIOperands().front();
            auto output = op.GetOOperands().front();
            if (reshapeSources.count(output->tensor->rawmagic) == 0) {
                reshapeSources.insert({output->tensor->rawmagic, {}});
            }
            reshapeSources[output->tensor->rawmagic] = input;
        }
        if (op.GetOpcode() == Opcode::OP_ASSEMBLE) { // output应该是reshape的input
            auto input = op.GetIOperands().front();
            auto output = op.GetOOperands().front();
            if (copyOutSources.count(output->tensor->rawmagic) == 0) {
                copyOutSources.insert({output->tensor->rawmagic, {}});
            }
            copyOutSources[output->tensor->rawmagic].insert(input);
            auto offset = dynamic_cast<AssembleOpAttribute *>(op.GetOpAttribute().get())->GetToOffset();
            if (mappingOffset.count(input->magic) == 0) {
                mappingOffset.insert({input->magic, {}});
            }
            mappingOffset[input->magic].insert({output->magic, offset});
        }
    }
}

// shape1 和 shape2 分别为reshape前后的shape
// 返回对齐后的shape
std::vector<int> SplitReshapeOpPVC2::ShapeAlign(std::vector<int> shape1, std::vector<int> shape2) {
    size_t i1 = 0, i2 = 0;
    int prod1 = 1, prod2 = 1;
    std::vector<int> alignedShape;
    shape1.erase(std::remove(shape1.begin(), shape1.end(), 1), shape1.end());
    shape2.erase(std::remove(shape2.begin(), shape2.end(), 1), shape2.end());
    while (i1 < shape1.size() || i2 < shape2.size()) {
        if (prod1 != 1) {
            if (shape2[i2] > prod1) {
                assert(shape2[i2] % prod1 == 0 && "cannot align shape");
                alignedShape.push_back(prod1);
                prod2 = shape2[i2] / prod1;
                prod1 = 1;
            } else {
                assert(prod1 % shape2[i2] == 0 && "cannot align shape");
                alignedShape.push_back(shape2[i2]);
                prod1 = prod1 / shape2[i2];
            }
            i2++;
        } else if (prod2 != 1) {
            std::swap(i1, i2);
            std::swap(prod1, prod2);
            std::swap(shape1, shape2);
        } else {
            assert((i1 < shape1.size() && i2 < shape2.size()) && "cannot align shape");
            if (shape2[i2] > shape1[i1]) {
                assert((shape2[i2] % shape1[i1] == 0) && "cannot align shape");
                alignedShape.push_back(shape1[i1]);
                prod2 = shape2[i2] / shape1[i1];
            } else {
                assert((shape1[i1] % shape2[i2] == 0) && "cannot align shape");
                alignedShape.push_back(shape2[i2]);
                prod1 = shape1[i1] / shape2[i2];
            }
            i1++;
            i2++;
        }
    }
    return alignedShape;
}

// shape为reshape前或reshape后的shape
// aligned_shape为上述对齐后的shape
// tile_offset, tile_shape为当前tile块的offset和shape
// 返回对齐后的tile块的offset和shape
// assert失败为不支持的情况
std::pair<std::vector<int>, std::vector<int>> SplitReshapeOpPVC2::ReshapeTile(const std::vector<int> &shape,
    const std::vector<int> &alignedShape, const std::vector<int> &tileOffset, const std::vector<int> &tileShape) {
    std::vector<int> newOffset(alignedShape.size()), newShape(alignedShape.size());
    size_t i;
    size_t j;
    for (i = j = 0; j < shape.size(); j++) {
        if (shape[j] == 1) {
            assert(tileOffset[j] == 0 && tileShape[j] == 1);
            continue;
        }
        int stride = shape[j];
        int currentOffset = tileOffset[j];
        int currentShape = tileShape[j];
        bool flag = true; // flag为true代表之前拆分的轴均为1，目前的tile依然占据维度的连续一段，可以不完整。
        while (stride > 1) {
            assert(stride % alignedShape[i] == 0);
            stride /= alignedShape[i];
            if (flag && currentShape <= stride) {
                newOffset[i] = currentOffset / stride;
                newShape[i] = 1;
                currentOffset = currentOffset - newOffset[i] * stride;
                assert(currentOffset + currentShape <= stride && "tile is scattered in alignedShape");
            } else {
                assert(currentOffset % stride == 0 && "cannot reshape to alignedShape");
                assert(currentShape % stride == 0 && "cannot reshape to alignedShape");
                newOffset[i] = currentOffset / stride;
                newShape[i] = currentShape / stride;
                currentOffset = 0;
                currentShape = stride;
                flag = false;
            }
            i++;
        }
    }
    assert(i == alignedShape.size() && "cannot reshape to alignedShape");
    return std::make_pair(newOffset, newShape);
}

// rawshape为变换前的rawshape
// new_rawshape为变换后的rawshape
// tile_offset, tile_shape为当前tile块变换前的offset和shape
// 返回变换后的tile块的offset和shape
// 要求rawshape为new_rawshape的加细
// assert失败为不支持的情况
std::pair<std::vector<int>, std::vector<int>> SplitReshapeOpPVC2::ReshapeTile2(const std::vector<int> &rawshape,
    const std::vector<int> &newRawshape, const std::vector<int> &tileOffset, const std::vector<int> &tileShape) {
    std::vector<int> newOffset(newRawshape.size()), newShape(newRawshape.size());
    size_t i, j;
    for (i = j = 0; j < newRawshape.size(); j++) {
        if (newRawshape[j] == 1) {
            newOffset[j] = 0;
            newShape[j] = 1;
            continue;
        }
        int stride = newRawshape[j];
        bool flag = true; // flag为true表示尚未遇到不为1的tile_shape
        while (stride > 1) {
            assert(stride % rawshape[i] == 0);
            stride /= rawshape[i];
            if (flag && tileShape[i] != 1) {
                newOffset[j] += tileOffset[i] * stride;
                newShape[j] = tileShape[i] * stride;
                flag = false;
            } else if (flag && tileShape[i] == 1) {
                newOffset[j] += tileOffset[i] * stride;
            } else if ((!flag) && (!(tileOffset[i] == 0 && tileShape[i] == rawshape[i]))) {
                std::vector<int> needAlagned(0);
                return std::make_pair(needAlagned, needAlagned);
            }
            i++;
        }
        if (flag) {
            newShape[j] = 1;
        }
    }
    return std::make_pair(newOffset, newShape);
}

void SplitReshapeOpPVC2::CheckCopyIn(Function &function) {
    for (auto &op : function.Operations()) {
        if (op.GetOpcode() != Opcode::OP_VIEW) {
            continue;
        }

        auto input = op.GetIOperands().front();
        auto output = op.GetOOperands().front();
        if (input->shape == output->shape) {
            continue;
        }

        auto viewOpAttribute = dynamic_cast<ViewOpAttribute *>(op.GetOpAttribute().get());
        ASSERT(viewOpAttribute != nullptr);
        auto &fromOffset = viewOpAttribute->GetFrom();
        auto inputView = std::make_shared<LogicalTensor>(function, input->tensor, fromOffset, output->shape);
        if (reshapeSources.find(input->tensor->rawmagic) == reshapeSources.end()) {
            continue;
        }
        auto reshapeSource = reshapeSources[input->tensor->rawmagic];
        bool isSplit = true;
        auto copyoutSourceFirst = copyOutSources[reshapeSource->tensor->rawmagic].begin();
        for (auto &copyOutSource : copyOutSources[reshapeSource->tensor->rawmagic]) {
            if (copyOutSource->tensor->rawmagic != (*copyoutSourceFirst)->tensor->rawmagic) {
                isSplit = false;
                break;
            }
        }
        if (isSplit == false) {
            continue;
        }
        std::vector<int> alignedShape = ShapeAlign(reshapeSource->tensor->rawshape, input->tensor->rawshape);
        auto newinputviewTileinfo =
            ReshapeTile(inputView->tensor->rawshape, alignedShape, inputView->offset, inputView->shape);
        // give a dummy valid shape also, to avoid rank of rawTensor and shape/offset are mismatch
        std::vector<SymbolicScalar> validShape;
        auto newinputview = std::make_shared<LogicalTensor>(
            function, inputView->tensor, newinputviewTileinfo.first, newinputviewTileinfo.second, validShape);
        std::vector<std::shared_ptr<LogicalTensor>> overlaps;
        std::vector<std::shared_ptr<LogicalTensor>> newOverlaps;
        for (auto &copyOutSource : copyOutSources[reshapeSource->tensor->rawmagic]) {
            // 存在多个tensor assemble成一个tensor再reshape的场景，需要使用assemble op的offset计算
            std::vector<int> copyOutOffset = mappingOffset[copyOutSource->magic][reshapeSource->magic];
            auto newcopyOutSourceTileinfo =
                ReshapeTile(reshapeSource->tensor->rawshape, alignedShape, copyOutOffset, copyOutSource->shape);
            auto newCopyOutSource = std::make_shared<LogicalTensor>(function, reshapeSource->tensor,
                newcopyOutSourceTileinfo.first, newcopyOutSourceTileinfo.second, validShape);
            auto status = CalcOverlap(newinputview, newCopyOutSource, true);
            if (status == OverlapStatus::PERFECTLY_MATCH || status == OverlapStatus::BE_COVERED) {
                overlaps.push_back(copyOutSource);
                newOverlaps.push_back(newCopyOutSource);
                break;
            }
            if (status == OverlapStatus::COVERED) {
                overlaps.push_back(copyOutSource);
                newOverlaps.push_back(newCopyOutSource);
                continue;
            }
        }
        auto status = CalcOverlap(newinputview, newOverlaps, true);
        switch (status) {
            case OverlapStatus::PERFECTLY_MATCH: {
                auto overlap = overlaps.front();
                auto newoverlap = newOverlaps.front();
                //  计算reshape后的tileshape和offset
                auto reshapeTileInfo =
                    ReshapeTile2(alignedShape, inputView->tensor->rawshape, newoverlap->offset, newoverlap->shape);
                if (reshapeTileInfo.second != output->shape) {
                    assert(reshapeTileInfo.second == output->shape);
                }
                if (reshapeRawOutputs.count(overlap->tensor->rawmagic) == 0) {
                    auto reshaperawOutput =
                        std::make_shared<RawTensor>(input->Datatype(), inputView->tensor->rawshape);
                    reshapeRawOutputs.insert({overlap->tensor->rawmagic, reshaperawOutput});
                }
                auto reshapeOutput = std::make_shared<LogicalTensor>(function,
                    reshapeRawOutputs[overlap->tensor->rawmagic], reshapeTileInfo.first, reshapeTileInfo.second);
                reshapeOutput->SetMemoryTypeBoth(input->GetMemoryTypeOriginal());
                if (overlap->GetMemoryTypeOriginal() == output->GetMemoryTypeOriginal() &&
                    overlap->GetMemoryTypeOriginal() == MemoryType::MEM_UB) {
                    reshapeOutput->SetMemoryTypeBoth(output->GetMemoryTypeOriginal(), true);
                    auto consumers = output->GetConsumers();
                    auto isAddReshapeOp = std::make_shared<ReshapeOp>(overlap, reshapeOutput);
                    auto existOp = ReshapeOperationExist(isAddReshapeOp);
                    if (existOp != nullptr) {
                        for (auto &consumerOp : consumers) {
                            consumerOp->ReplaceInput(existOp->output, output);
                        }
                    } else {
                        for (auto &consumerOp : consumers) {
                            consumerOp->ReplaceInput(reshapeOutput, output);
                        }
                    }
                    redundentViewops.insert(&op);
                } else if (overlap->GetMemoryTypeOriginal() != output->GetMemoryTypeOriginal() &&
                           overlap->GetMemoryTypeOriginal() == MemoryType::MEM_DEVICE_DDR) {
                    auto isAddReshapeOp = std::make_shared<ReshapeOp>(overlap, reshapeOutput);
                    auto existOp = ReshapeOperationExist(isAddReshapeOp);
                    if (existOp != nullptr) {
                        op.ReplaceInput(existOp->output, input);
                        viewOpAttribute->SetFromOffset(existOp->output->offset);
                    }else {
                        op.ReplaceInput(reshapeOutput, input);
                        viewOpAttribute->SetFromOffset(reshapeOutput->offset);
                    }
                } else {
                    std::vector<int> assembleOffset = mappingOffset[overlap->magic][reshapeSource->magic];
                    if (reshapeRawInputs.count(overlap->tensor->rawmagic) == 0) {
                        auto reshapeRawInput =
                            std::make_shared<RawTensor>(overlap->Datatype(), overlap->tensor->rawshape);
                        reshapeRawInputs.insert({overlap->tensor->rawmagic, reshapeRawInput});
                    }
                    auto newReshapeSource = std::make_shared<LogicalTensor>(
                        function, reshapeRawInputs[overlap->tensor->rawmagic], assembleOffset, overlap->shape);
                    newReshapeSource->SetMemoryTypeBoth(reshapeSource->GetMemoryTypeOriginal());
                    auto isAddReshapeOp = std::make_shared<ReshapeOp>(newReshapeSource, reshapeOutput);
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
                }
                break;
            }
            case OverlapStatus::BE_COVERED: {
                auto overlap = overlaps.front();
                auto newoverlap = newOverlaps.front();
                // 变换为reshape后的offset和shape
                auto newOffsetShape = ReshapeTile2(alignedShape, inputView->tensor->rawshape,
                    newinputviewTileinfo.first, newinputviewTileinfo.second);
                if (newOffsetShape.second != output->shape) {
                    assert(newOffsetShape.second == output->shape);
                }
                //  计算reshape后view的tileshape和offset
                auto reshapeTileInfo =
                    ReshapeTile2(alignedShape, inputView->tensor->rawshape, newoverlap->offset, newoverlap->shape);
                if (reshapeTileInfo.first.size() == 0 && reshapeTileInfo.second.size() == 0) {
                    break; // 这种场景需要先view再reshape
                }
                if (reshapeRawOutputs.count(overlap->tensor->rawmagic) == 0) {
                    auto reshaperawOutput =
                        std::make_shared<RawTensor>(input->Datatype(), inputView->tensor->rawshape);
                    reshapeRawOutputs.insert({overlap->tensor->rawmagic, reshaperawOutput});
                }
                auto reshapeOutput = std::make_shared<LogicalTensor>(function,
                    reshapeRawOutputs[overlap->tensor->rawmagic], reshapeTileInfo.first, reshapeTileInfo.second);
                reshapeOutput->SetMemoryTypeBoth(input->GetMemoryTypeOriginal());
                if ((overlap->GetMemoryTypeOriginal() == output->GetMemoryTypeOriginal() &&
                        overlap->GetMemoryTypeOriginal() == MemoryType::MEM_UB) ||
                    (overlap->GetMemoryTypeOriginal() != output->GetMemoryTypeOriginal() &&
                        overlap->GetMemoryTypeOriginal() == MemoryType::MEM_DEVICE_DDR)) {
                    if (overlap->GetMemoryTypeOriginal() == output->GetMemoryTypeOriginal() &&
                        overlap->GetMemoryTypeOriginal() == MemoryType::MEM_UB) {
                        reshapeOutput->SetMemoryTypeBoth(output->GetMemoryTypeOriginal(), true);
                    }
                    auto isAddReshapeOp = std::make_shared<ReshapeOp>(overlap, reshapeOutput);
                    auto existOp = ReshapeOperationExist(isAddReshapeOp);
                    if (existOp != nullptr) {
                        viewOpAttribute->SetFromOffset(newOffsetShape.first);
                        op.ReplaceInput(existOp->output, input);
                    } else {
                        viewOpAttribute->SetFromOffset(newOffsetShape.first);
                        op.ReplaceInput(reshapeOutput, input);
                    }
                } else {
                    std::vector<int> assembleOffset = mappingOffset[overlap->magic][reshapeSource->magic];
                    if (reshapeRawInputs.count(overlap->tensor->rawmagic) == 0) {
                        auto reshapeRawInput =
                            std::make_shared<RawTensor>(overlap->Datatype(), overlap->tensor->rawshape);
                        reshapeRawInputs.insert({overlap->tensor->rawmagic, reshapeRawInput});
                    }
                    auto newReshapeSource = std::make_shared<LogicalTensor>(
                        function, reshapeRawInputs[overlap->tensor->rawmagic], assembleOffset, overlap->shape);
                    newReshapeSource->SetMemoryTypeBoth(reshapeSource->GetMemoryTypeOriginal());

                    auto isAddReshapeOp = std::make_shared<ReshapeOp>(newReshapeSource, reshapeOutput);
                    auto existOp = ReshapeOperationExist(isAddReshapeOp);
                    if (existOp != nullptr) {
                        viewOpAttribute->SetFromOffset(newOffsetShape.first);
                        op.ReplaceInput(existOp->output, input);
                    } else {
                        assembles.emplace_back(AssembleOp{
                            overlap->GetMemoryTypeOriginal(), newReshapeSource->offset, overlap, newReshapeSource});
                        viewOpAttribute->SetFromOffset(newOffsetShape.first);
                        op.ReplaceInput(reshapeOutput, input);
                    }
                }
                break;
            }
            case OverlapStatus::PERFECTLY_MATCH_WITH_ALL: {
                UpdateForPerfectlyMatchWithAll(function, op,
                    {alignedShape, reshapeSource, newinputviewTileinfo, overlaps, newOverlaps, input, inputView, output});
                break;
            }
            default: break;
        }
    }
}

void SplitReshapeOpPVC2::UpdateForPerfectlyMatchWithAll(Function &function, Operation &op, const CalcOverlapPara &para) {
    std::vector<int> &alignedShape = para.alignedShape;
    std::shared_ptr<LogicalTensor> &reshapeSource = para.reshapeSource;
    std::pair<std::vector<int>, std::vector<int>> &newinputviewTileinfo = para.newinputviewTileinfo;
    std::vector<std::shared_ptr<LogicalTensor>> &overlaps = para.overlaps;
    std::shared_ptr<LogicalTensor> &input = para.input;
    std::shared_ptr<LogicalTensor> &output = para.output;
    std::shared_ptr<LogicalTensor> &inputView = para.inputView;

    auto viewOpAttribute = dynamic_cast<ViewOpAttribute *>(op.GetOpAttribute().get());
    ASSERT(viewOpAttribute != nullptr);
    // newReshapeSource的tileinfo要变换为reshape前的
    auto newReshapeSourceTileInfo = ReshapeTile2(
        alignedShape, reshapeSource->tensor->rawshape, newinputviewTileinfo.first, newinputviewTileinfo.second);
    // reshape前的tile无法assemble表达成一个tile时需要先reshape成alignshape然后再assemble成dstshape
    if (newReshapeSourceTileInfo.first.size() == 0 && newReshapeSourceTileInfo.second.size() == 0) {
        UpdateForAssembleAfterReshape(function, op, para);
    } else {
        if (reshapeRawInputs.count(overlaps.front()->tensor->rawmagic) == 0) {
            auto reshapeRawInput =
                std::make_shared<RawTensor>(overlaps.front()->Datatype(), overlaps.front()->tensor->rawshape);
            reshapeRawInputs.insert({overlaps.front()->tensor->rawmagic, reshapeRawInput});
        }
        auto newReshapeSource =
            std::make_shared<LogicalTensor>(function, reshapeRawInputs[overlaps.front()->tensor->rawmagic],
                newReshapeSourceTileInfo.first, newReshapeSourceTileInfo.second);
        newReshapeSource->SetMemoryTypeBoth(reshapeSource->GetMemoryTypeOriginal());
        if (reshapeRawOutputs.count(overlaps.front()->tensor->rawmagic) == 0) {
            auto reshaperawOutput = std::make_shared<RawTensor>(input->Datatype(), inputView->tensor->rawshape);
            reshapeRawOutputs.insert({overlaps.front()->tensor->rawmagic, reshaperawOutput});
        }
        auto reshapeOutput = std::make_shared<LogicalTensor>(
            function, reshapeRawOutputs[overlaps.front()->tensor->rawmagic], inputView->offset, inputView->shape);
        reshapeOutput->SetMemoryTypeBoth(output->GetMemoryTypeOriginal());
        for (auto &overlap : overlaps) {
            std::vector<int> overlapOffset = mappingOffset[overlap->magic][reshapeSource->magic];
            assembles.emplace_back(AssembleOp{overlap->GetMemoryTypeOriginal(), overlapOffset, overlap, newReshapeSource});
        }
        if (overlaps.front()->GetMemoryTypeOriginal() == output->GetMemoryTypeOriginal() &&
            overlaps.front()->GetMemoryTypeOriginal() == MemoryType::MEM_UB) {
            newReshapeSource->SetMemoryTypeBoth(overlaps.front()->GetMemoryTypeOriginal(), true);
            auto isAddReshapeOp = std::make_shared<ReshapeOp>(newReshapeSource, reshapeOutput);
            auto existOp = ReshapeOperationExist(isAddReshapeOp);
            auto consumers = output->GetConsumers();
            if (existOp != nullptr) {
                for (auto &consumerOp : consumers) {
                    consumerOp->ReplaceInput(existOp->output, output);
                }
            } else {
                for (auto &consumerOp : consumers) {
                    consumerOp->ReplaceInput(reshapeOutput, output);
                }
            }
            redundentViewops.insert(&op);
        } else {
            reshapeOutput->SetMemoryTypeBoth(input->GetMemoryTypeOriginal(), true);
            auto isAddReshapeOp = std::make_shared<ReshapeOp>(newReshapeSource, reshapeOutput);
            auto existOp = ReshapeOperationExist(isAddReshapeOp);
            if (existOp != nullptr) {
                op.ReplaceInput(existOp->output, input);
                viewOpAttribute->SetFromOffset(existOp->output->offset);
            } else {
                op.ReplaceInput(reshapeOutput, input);
                viewOpAttribute->SetFromOffset(reshapeOutput->offset);
            }
        }
    }
}

void SplitReshapeOpPVC2::UpdateForAssembleAfterReshape(Function &function, Operation &op, const CalcOverlapPara &para) {
    std::vector<int> &alignedShape = para.alignedShape;
    std::shared_ptr<LogicalTensor> &reshapeSource = para.reshapeSource;
    std::vector<std::shared_ptr<LogicalTensor>> &overlaps = para.overlaps;
    std::vector<std::shared_ptr<LogicalTensor>> &newOverlaps = para.newOverlaps;
    std::shared_ptr<LogicalTensor> &input = para.input;
    std::shared_ptr<LogicalTensor> &output = para.output;
    std::shared_ptr<LogicalTensor> &inputView = para.inputView;

    auto viewOpAttribute = dynamic_cast<ViewOpAttribute *>(op.GetOpAttribute().get());
    ASSERT(viewOpAttribute != nullptr);

    auto newInput = std::make_shared<LogicalTensor>(function, input->Datatype(), output->shape);
    newInput->SetMemoryTypeBoth(input->GetMemoryTypeOriginal());
    for (size_t i = 0; i < overlaps.size(); i++) {
        auto newReshapeOutputTileInfo =
            ReshapeTile2(alignedShape, inputView->tensor->rawshape, newOverlaps[i]->offset, newOverlaps[i]->shape);
        if (reshapeRawOutputs.count(overlaps[i]->tensor->rawmagic) == 0) {
            auto reshaperawOutput = std::make_shared<RawTensor>(input->Datatype(), inputView->tensor->rawshape);
            reshapeRawOutputs.insert({overlaps[i]->tensor->rawmagic, reshaperawOutput});
        }
        auto newReshapeOutput =
            std::make_shared<LogicalTensor>(function, reshapeRawOutputs[overlaps[i]->tensor->rawmagic],
                newReshapeOutputTileInfo.first, newReshapeOutputTileInfo.second);
        newReshapeOutput->SetMemoryTypeBoth(input->GetMemoryTypeOriginal());
        if ((overlaps.front()->GetMemoryTypeOriginal() == output->GetMemoryTypeOriginal() &&
                overlaps.front()->GetMemoryTypeOriginal() == MemoryType::MEM_UB)) {
            newReshapeOutput->SetMemoryTypeBoth(output->GetMemoryTypeOriginal(), true);
            std::vector<int> newOffset(inputView->offset.size(), 0);
            std::transform(newReshapeOutputTileInfo.first.begin(), newReshapeOutputTileInfo.first.end(),
                inputView->offset.begin(), newOffset.begin(), [](int a, int b) { return a - b; });
            auto isAddReshapeOp = std::make_shared<ReshapeOp>(overlaps[i], newReshapeOutput);
            auto existOp = ReshapeOperationExist(isAddReshapeOp);
            if (existOp != nullptr) {
                assembles.emplace_back(
                    AssembleOp{newReshapeOutput->GetMemoryTypeOriginal(), newOffset, existOp->output, output});
                redundentViewops.insert(&op);
            } else {
                assembles.emplace_back(
                    AssembleOp{newReshapeOutput->GetMemoryTypeOriginal(), newOffset, newReshapeOutput, output});
                redundentViewops.insert(&op);
            }
        } else if ((overlaps.front()->GetMemoryTypeOriginal() != output->GetMemoryTypeOriginal() &&
                       overlaps.front()->GetMemoryTypeOriginal() == MemoryType::MEM_DEVICE_DDR)) {
            std::vector<int> newOffset(inputView->offset.size(), 0);
            std::transform(newReshapeOutputTileInfo.first.begin(), newReshapeOutputTileInfo.first.end(),
                inputView->offset.begin(), newOffset.begin(), [](int a, int b) { return a - b; });
            auto isAddReshapeOp = std::make_shared<ReshapeOp>(overlaps[i], newReshapeOutput);
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
        } else {
            std::vector<int> assembleOffset = mappingOffset[overlaps[i]->magic][reshapeSource->magic];
            if (reshapeRawInputs.count(overlaps[i]->tensor->rawmagic) == 0) {
                auto reshapeRawInput =
                    std::make_shared<RawTensor>(overlaps[i]->Datatype(), overlaps[i]->tensor->rawshape);
                reshapeRawInputs.insert({overlaps[i]->tensor->rawmagic, reshapeRawInput});
            }
            auto newReshapeSource = std::make_shared<LogicalTensor>(
                function, reshapeRawInputs[overlaps[i]->tensor->rawmagic], assembleOffset, overlaps[i]->shape);
            newReshapeSource->SetMemoryTypeBoth(reshapeSource->GetMemoryTypeOriginal());
            std::vector<int> newOffset(inputView->offset.size(), 0);
            std::transform(newReshapeOutputTileInfo.first.begin(), newReshapeOutputTileInfo.first.end(),
                inputView->offset.begin(), newOffset.begin(), [](int a, int b) { return a - b; });
            auto isAddReshapeOp = std::make_shared<ReshapeOp>(newReshapeSource, newReshapeOutput);
            auto existOp = ReshapeOperationExist(isAddReshapeOp);
            if (existOp != nullptr) {
                assembles.emplace_back(
                    AssembleOp{newReshapeOutput->GetMemoryTypeOriginal(), newOffset, existOp->output, newInput});
                viewOpAttribute->SetFromOffset(newInput->offset);
                op.ReplaceInput(newInput, input);
            } else {
                assembles.emplace_back(
                    AssembleOp{overlaps[i]->GetMemoryTypeOriginal(), assembleOffset, overlaps[i], newReshapeSource});
                assembles.emplace_back(
                    AssembleOp{newReshapeOutput->GetMemoryTypeOriginal(), newOffset, newReshapeOutput, newInput});
                viewOpAttribute->SetFromOffset(newInput->offset);
                op.ReplaceInput(newInput, input);
            }
        }
    }
}

void SplitReshapeOpPVC2::EraseReshape(Function &function) {
    // 先删除view使reshape的Consumers为空
    for (auto &opView : redundentViewops) {
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
        if (output->nodetype == NodeType::LOCAL && output->GetConsumers().empty()) {
            op.SetAsDeleted();
        }
    }

    function.EraseOperations(true, true);
}

} // namespace npu::tile_fwk