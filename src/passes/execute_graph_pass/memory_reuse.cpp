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
 * \file memory_reuse.cpp
 * \brief
 */
#include <deque>
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/program/program.h"
#include "interface/function/function.h"
#include "interface/tensor/logical_tensor.h"
#include "passes/execute_graph_pass/memory_reuse.h"

namespace npu::tile_fwk {
constexpr int64_t MEM_PROPORTION_COEFF = 8;
constexpr uint64_t EXTRA_SIZE_IN_BYTE = 32;
constexpr uint64_t ALIGN_SIZE_IN_BYTE = 512;
constexpr int64_t INVALID_INDEX = -1;
constexpr size_t OFFSET_INDEX = 1;
constexpr size_t RAW_SHAPE_POS = 2;
inline uint64_t Align(const uint64_t n)
{
    return (n + ALIGN_SIZE_IN_BYTE - 1) & (~(ALIGN_SIZE_IN_BYTE - 1));
}

void TensorBucket::UpdateOffset(const uint64_t offset) {
    offset_ = offset;
    for (auto &ref : refs_) {
        auto &first = (*ref.begin());
        first->storage_->start_ = offset;
        first->storage_->length_ = size_;
    }
    if (refs_.size() > 1) {
        ALOG_INFO_F("Memory Offset %lu, contains :", offset);
        for (auto &ref : refs_) {
            ALOG_INFO_F("Tensor magic %d rawmagic %d", (*ref.begin())->magic, (*ref.begin())->tensor->rawmagic);
        }
    }
}

void TensorBucket::AddRef(const TensorsDesc &tensorsDesc)
{
    if (tensorsDesc.tensors.empty()) {
        ALOG_INFO_F("Ref is empty");
        ASSERT(false);
        return;
    }
    auto &tensor = *(tensorsDesc.tensors.begin());
    uint64_t tensorSize = static_cast<uint64_t>(tensor->storage_->length_);
    size_ = std::max(tensorSize, size_);
    ALOG_INFO_F("%d %d, size is %lu", tensor->magic, tensor->tensor->rawmagic, size_);
    refs_.emplace_back(tensorsDesc.tensors);
    consumerOpIdxs_ = tensorsDesc.consumerOpIdxs;
}

bool Allocator::IsRawQualified(const WorkspaceInfo &outWspInfo, const WorkspaceInfo &inWspInfo) {
    auto &outRaw = outWspInfo.tensor->tensor;
    auto &inRaw = inWspInfo.tensor->tensor;
    if (outWspInfo.size == inWspInfo.size) {
        return true;
    }
    if (outRaw->GetDataType() != inRaw->GetDataType()) {
        return false;
    }
    // size不相等场景 校验除了最高轴其余轴都相等
    auto dimOut = outRaw->rawshape.size();
    auto dimIn = inRaw->rawshape.size();
    if (dimOut != dimIn) {
        return false;
    }
    for (size_t i = 1; i < dimOut; i++) {
        if (outRaw->rawshape[i] != inRaw->rawshape[i]) {
            return false;
        }
    }
    return true;
}

// 在不引入额外同步的情况下，完成内存的复用，找到某一个CopyOut的前驱的CopyIn，依赖关系天然存在
// 极限的复用，可以不考虑依赖关系，只看节点之间的顺序，在后续insert sync时可以插入mte3 wait mte2的同步，但是可能会有性能劣化。
void FindFirstQualifiedCopyIn(Function *leafFunc, Operation *op,
    const WorkspaceInfo &outWspInfo,
    std::unordered_map<LogicalTensorPtr, WorkspaceInfo> &inWspCnt,
    std::vector<WorkspaceInfo> &outReuseInCasts) {
    std::deque<Operation *> parents;
    
    std::unordered_set<Operation*> visited; // 已访问标记集合
    
    parents.push_back(op);
    visited.insert(op); // 标记初始操作已访问

    auto &out = outWspInfo.tensor;
    while (!parents.empty()) {
        auto parent = parents.front();
        parents.pop_front();

        for (auto &in : parent->GetIOperands()) {
            // 在leafFunc边界停止遍历
            if (std::find(leafFunc->inCasts_.begin(), leafFunc->inCasts_.end(), in) != leafFunc->inCasts_.end()) {
                continue;
            }
            for (auto &producerOfParent : in->GetProducers()) {
                // 跳过已访问的操作
                if (visited.find(producerOfParent) != visited.end()) {
                    continue;
                }
                visited.insert(producerOfParent); // 标记当前操作为已访问

                if (OpcodeManager::Inst().IsCopyIn(producerOfParent->GetOpcode())) {
                    auto &copyInInput = producerOfParent->GetIOperands()[0];
                    auto iter = inWspCnt.find(copyInInput);
                    if (iter != inWspCnt.end()) {
                        if (iter->second.count == 1 && iter->second.size >= outWspInfo.size &&
                            iter->second.size / outWspInfo.size < MEM_PROPORTION_COEFF) {
                            if (iter->second.used == false && Allocator::IsRawQualified(outWspInfo, iter->second)) {
                                ALOG_DEBUG_F("$$$$$$$$$$$$ Outcast %d raw %d can reuse incast %d raw %d size [%zu : %zu], leaf hash %lu", out->magic, out->tensor->rawmagic,
                                    copyInInput->magic, copyInInput->tensor->rawmagic,
                                    iter->second.size, outWspInfo.size, leafFunc->GetFunctionHash().GetHash());
                                iter->second.used = true;
                                outReuseInCasts[outWspInfo.position] = iter->second;
                                return;
                            }
                        }
                    } else {
                        parents.push_back(producerOfParent);
                    }
                } else {
                    parents.push_back(producerOfParent);
                }
            }
        }
    }
}

void HandleOneOutCast(Function *leafFunc, WorkspaceInfo &wspInfo,
    std::unordered_map<LogicalTensorPtr, WorkspaceInfo> &inWspCnt, std::vector<WorkspaceInfo> &outReuseInCasts) {
    auto &out = wspInfo.tensor;
    if (wspInfo.count != 1) {
        ALOG_DEBUG_F("magic %d raw %d not 1", out->magic, out->tensor->rawmagic);
        return;
    }

    auto &producers = out->GetProducers();
    if (producers.size() > 1) {
        return;
    }
    if (producers.empty()) {
        ALOG_ERROR_F("Tensor %d producer is empty function hash %lu", out->magic, leafFunc->GetFunctionHash().GetHash());
        return;
    }

    auto producer = *(producers.begin());
    if (OpcodeManager::Inst().IsCopyOut(producer->GetOpcode())) {
        return;
    }
    auto &producerIn = producer->GetIOperands()[0];
    // 通过copyOut的输入来检查输入和输出的shape是否相等, 不相等的话，意味着多写入，判断复用难度较大。
    if (producerIn->oriShape != out->shape || producerIn->oriShape != out->tensor->rawshape) {
        return;
    }
    FindFirstQualifiedCopyIn(leafFunc, producer, wspInfo, inWspCnt, outReuseInCasts);
}

bool GetCopyInSize(LogicalTensorPtr &in, Operation *copyIn, uint64_t &size) {
    if (copyIn == nullptr) {
        return false;
    }
    if (OpcodeManager::Inst().IsCopyIn(copyIn->GetOpcode())) {
        return false;
    }
    auto attr = dynamic_cast<CopyOpAttribute *>(copyIn->GetOpAttribute().get());
    if (attr == nullptr) {
        return false;
    }
    if (attr->IsDynFromOffset()) {
        return false;
    }
    size_t bytesPerEle = BytesOf(in->tensor->datatype);
    size = bytesPerEle;
    for (auto &ele : copyIn->GetOOperands()[0]->oriShape) {
        size *= ele;
    }
    return true;
}

/* Reshape的复用之后的Offset计算比较复杂，暂时不复用 */
bool HasReshapeConsumer(const LogicalTensorPtr &out)
{
    for (Operation *consumer : out->GetConsumers()) {
        if (consumer->GetOpcode() == Opcode::OP_RESHAPE) {
            return true;
        }
    }
    return false;
}

// leaf内复用注意：
// 1. leaf的Incast、Outcast如果有重复，那么是不能被复用，也不需要复用的。
// 2. reshape的相关处理
// 3. 对outcast中没有actual raw magic且数量为1，且rawshape = shape的做广度优先遍历, 找到距离最近的一个incast，并将他们标记为
// 一对可以复用的incast outcast pair。
// 4. 如果incast的size大于outcast，那么也可以复用，但是要给outcast的tensor上打上偏移量。
// 5. 如果outcast的shape不等于rawshape，那么不能复用，这种场景较为复杂，有优化空间
// 6. 如果outcast在leafFunction中存在后继的reshape，那么不需要复用
void Allocator::ProcessLeafMemoryReuse(Function *leafFunc) {
    std::unordered_map<LogicalTensorPtr, size_t> tensorToInfo;
    std::vector<WorkspaceInfo> outWspInfo;

    // 获取或创建该 leaf function 的 outReuseInCasts_ 数据
    std::vector<WorkspaceInfo>& leafOutReuseInCasts = GetOutReuseInCasts(leafFunc);

    for (size_t i = 0; i < leafFunc->outCasts_.size(); ++i) {
        auto &out = leafFunc->outCasts_[i];
        leafOutReuseInCasts.emplace_back(WorkspaceInfo());
        if (rootOutCasts_.count(out->GetRawMagic())) {
            continue;
        }
        if (HasReshapeConsumer(out)) {
            continue;
        }
        if (out->tensor->actualRawmagic != -1) {
            continue;
        }
        auto iter = tensorToInfo.find(out);
        if (iter == tensorToInfo.end()) {
            outWspInfo.emplace_back(WorkspaceInfo(1, i, static_cast<uint64_t>(out->tensor->GetRawDataSize()), out));
            tensorToInfo[out] = outWspInfo.size() - 1;
        } else {
            auto index = iter->second;
            outWspInfo[index].count++;
        }
    }

    std::unordered_map<LogicalTensorPtr, WorkspaceInfo> inWspCnt;
    for (size_t i = 0; i < leafFunc->inCasts_.size(); ++i) {
        auto &in = leafFunc->inCasts_[i];
        if (rootInCasts_.count(in->GetRawMagic())) {
            continue;
        }

        if (in->tensor->actualRawmagic != -1) {
            continue;
        }

        auto iter = inWspCnt.find(in);
        if (iter == inWspCnt.end()) {
            auto &consumers = in->GetConsumers();
            auto consumer = *(consumers.begin());
            uint64_t size = 0;
            if (GetCopyInSize(in, consumer, size)) {
                inWspCnt[in] = WorkspaceInfo(1, i, size, in);
            }
        } else {
            iter->second.count++;
        }
    }
    for (auto &wspInfo : outWspInfo) {
        HandleOneOutCast(leafFunc, wspInfo, inWspCnt, leafOutReuseInCasts);
    }
}

bool CheckAllConsumerAccessNoOverlap(const std::vector<std::vector<int>> &allOffsets,
    const std::vector<std::vector<int>> &allShapes) {
    if (allOffsets.size() == 1) {
        return true;
    }
    auto isNoOverlap = [&allOffsets, &allShapes](size_t p, size_t q) {
        auto &offsetP = allOffsets[p];
        auto &offsetQ = allOffsets[q];
        auto &shapeP = allShapes[p];
        auto &shapeQ = allShapes[q];

        for (size_t dim = 0; dim < offsetP.size(); dim++) {
            size_t pStart = offsetP[dim];
            size_t pEnd = pStart + shapeP[dim] - 1;
            size_t qStart = offsetQ[dim];
            size_t qEnd = qStart + shapeQ[dim] - 1;

            if (pEnd < qStart || qEnd < pStart) {
                return true;
            }
        }
        return false;
    };

    for (size_t pp = 0; pp < allOffsets.size(); ++pp) {
        for (size_t qq = pp + 1; qq < allOffsets.size(); ++qq) {
            if (!isNoOverlap(pp, qq)) {
                return false;
            }
        }
    }
    return true;
}

void RecordAllConsumerShapeAndOffset(LogicalTensorPtr &out, std::vector<std::vector<int>> &allOffsets,
    std::vector<std::vector<int>> &allShapes, bool &canReuse) {
    size_t outShapeSize = out->shape.size();
    size_t shapeIdx =  OFFSET_INDEX + outShapeSize;
    for (auto &consumer : out->GetConsumers()) {
        if (consumer->GetOpcode() != Opcode::OP_CALL) {
            continue;
        }
        for (size_t i = 0; i < consumer->GetIOperands().size(); i++) {
            if (consumer->GetIOperands()[i] != out) {
                continue;
            }
            allOffsets.emplace_back();
            allShapes.emplace_back();
            auto &offset = allOffsets.back();
            auto &shape = allShapes.back();

            CallOpAttribute *attr = dynamic_cast<CallOpAttribute *>(consumer->GetOpAttribute().get());
            if (attr == nullptr) {
                continue;
            }
            auto &arglist = attr->GetArgList()[i];
            for (size_t j = OFFSET_INDEX; j < OFFSET_INDEX + outShapeSize; j++) {
                if (arglist[j].IsImmediate()) {
                    offset.emplace_back(arglist[j].Concrete());
                } else {
                    canReuse = false;
                    return;
                }
            }

            for (size_t j = shapeIdx; j < shapeIdx + outShapeSize; j++) {
                if (arglist[j].IsImmediate()) {
                    shape.emplace_back(arglist[j].Concrete());
                } else {
                    canReuse = false;
                    return;
                }
            }
            if (offset.size() != outShapeSize || shape.size() != outShapeSize) {
                canReuse = false;
                return;
            }
        }
    }
}

void Allocator::MarkNonOverlappingConsumerTensors() {
    for (auto &op : function_->Operations()) {
        for (auto &out : op.GetOOperands()) {
            bool canReuse = true;
            if (rootOutCasts_.count(out->GetRawMagic()) != 0) {
                continue;
            }
            if (out->GetConsumers().size() == 1) {
                out->SetAttr("ConsumerAccessNoOverlap", true);
                continue;
            }

            std::vector<std::vector<int>> allOffsets;
            std::vector<std::vector<int>> allShapes;
            RecordAllConsumerShapeAndOffset(out, allOffsets, allShapes, canReuse);
            if (canReuse == false) {
                continue;
            }
            bool ret= CheckAllConsumerAccessNoOverlap(allOffsets, allShapes);
            if (ret) {
                out->SetAttr("ConsumerAccessNoOverlap", true);
            }
        }
    }
}

void Allocator::InitializeLeafMemoryReuse() {
    if (function_->GetFunctionType() != FunctionType::DYNAMIC_LOOP_PATH) {
        return;
    }
    auto &programs = function_->programs_;
    for (auto &program : programs) {
        ProcessLeafMemoryReuse(program.second);
    }
    MarkNonOverlappingConsumerTensors();
}

/* 如果previous tensor的consumer中包含了tensor的producer，那么不能复用。
   其余场景，如果previous tensor的所有consumer到tensor的一个producer之间有连接，那么意味着，tensor的producer的执行，一定要
   等到preivous的所有consumer都执行完。 */
bool TensorBucket::HasTopoDependency(const LargeBitmap &PreducersOp) const {
    for (auto &cons : consumerOpIdxs_) {
        if (!PreducersOp.GetBit(cons)) {
            return false;
        }
    }
    return true;
}

void Allocator::UpdateTensorMagicToBucketIdx(const std::set<LogicalTensorPtr> &tensors, int idx)
{
    for (auto tensor : tensors) {
        tensorMagicToBucketIdx_[tensor->GetMagic()] = idx;
    }
}

TensorBucket &Allocator::GetBestFitBucket(const TensorsDesc &tensorsDesc)
{
    if (tensorsDesc.isDummy) {
        return dummyPackets_;
    }
    auto &first = *(tensorsDesc.tensors.begin());
    int64_t rawDataSize = first->tensor->GetRawDataSize();
    int64_t rawDataSizeKey = rawDataSize / MEM_PROPORTION_COEFF;

    std::deque<LogicalTensorPtr> predecessorTensors(tensorsDesc.tensors.begin(), tensorsDesc.tensors.end());
    std::unordered_set<int> visitedTensor;
    std::unordered_set<int> visitedBucket;
    while (predecessorTensors.size() > 0) {
        LogicalTensorPtr ptr = predecessorTensors.front();
        predecessorTensors.pop_front();
        if (visitedTensor.count(ptr->GetMagic()) > 0) {
            continue;
        }
        visitedTensor.insert(ptr->GetMagic());

        if (tensorMagicToBucketIdx_.count(ptr->GetMagic()) > 0) {
            int bucketIdx = tensorMagicToBucketIdx_[ptr->GetMagic()];
            if (visitedBucket.count(bucketIdx) > 0) {
                continue;
            }
            visitedBucket.insert(bucketIdx);
            if (bucketsIdxToSize_[bucketIdx] >= rawDataSizeKey &&
                buckets_[bucketIdx].HasTopoDependency(tensorsDesc.connectionOpsBitmap)) {
                bucketsIdxToSize_[bucketIdx] = rawDataSize;
                UpdateTensorMagicToBucketIdx(tensorsDesc.tensors, bucketIdx);
                return buckets_[bucketIdx];
            }
        }

        for (auto &op : ptr->GetProducers()) {
            for (auto &tensor : op->GetIOperands()) {
                predecessorTensors.push_back(tensor);
            }
        }
    }

    buckets_.emplace_back();
    UpdateTensorMagicToBucketIdx(tensorsDesc.tensors, buckets_.size() - 1);
    bucketsIdxToSize_[buckets_.size() - 1] = rawDataSize;
    return buckets_.back();
}

bool GetStorageOffsetByCall(Operation& callOp, size_t incastIdx, uint64_t &storageOffset) {
    auto &input = callOp.GetIOperands()[incastIdx];
    if (input->storageOffset_ != 0) {
        storageOffset = input->storageOffset_;
        return true;
    }
    size_t rawShapeIdx =  OFFSET_INDEX + 2 * input->shape.size();

    CallOpAttribute *attr = dynamic_cast<CallOpAttribute *>(callOp.GetOpAttribute().get());
    auto &argList = attr->GetArgList()[incastIdx];
    std::vector<int> offset;
    std::vector<int> rawshape;
    for (size_t i = OFFSET_INDEX; i < OFFSET_INDEX + input->shape.size(); i++) {
        if (argList[i].IsImmediate()) {
            offset.emplace_back(argList[i].Concrete());
        } else {
            return false;
        }
    }

    for (size_t i = rawShapeIdx; i < rawShapeIdx + input->shape.size(); i++) {
        if (argList[i].IsImmediate()) {
            rawshape.emplace_back(argList[i].Concrete());
        } else {
            return false;
        }
    }

    std::vector<int> stride(rawshape.size(), 1);
    for (int i = static_cast<int>(rawshape.size() - 2); i >= 0; i--) {
        stride[i] = rawshape[i + 1] * stride[i + 1];
    }
    for (size_t i = 0; i < offset.size(); ++i) {
        auto offsetThisDim = offset[i];
        storageOffset += (offsetThisDim * stride[i]);
    }
    size_t bytesPerEle = BytesOf(input->tensor->datatype);
    storageOffset *= bytesPerEle;
    return true;
}

bool Allocator::CheckTopoDependancy(const LogicalTensorPtr &tensor, Operation &op) const {
    for (auto cons : tensor->GetConsumers()) {
        if (cons == &op) {
            continue;
        }

        if (connectionMatrix_.IsConnected(*cons, op)) {
            continue;
        } else {
            return false;
        }
    }
    return true;
}

bool Allocator::CheckReuseInnerCall(
    Operation &callOp, size_t outputIdx, LogicalTensorPtr &previous, uint64_t &storageOffset) const {
    // CallOp需要满足Topo序
    auto cacheValue = Program::GetInstance().GetHostMachine().TryHitCahce(callOp.GetCalleeHash());
    Function *program = nullptr;
    if (cacheValue == std::nullopt) {
        ALOG_ERROR_F("Cannot find program hash %lu by op %d", callOp.GetCalleeHash().GetHash(), callOp.opmagic);
        return false;
    } else {
        program = cacheValue->cacheFunction;
    }

    // 从 map 中获取该 program 的 outReuseInCasts_ 数据
    auto it = outReuseInCasts_.find(program);
    if (it == outReuseInCasts_.end()) {
        return false;
    }
    const auto& programOutReuseInCasts = it->second;

    if (outputIdx >= programOutReuseInCasts.size()) {
        return false;
    }

    auto &incastInfo = programOutReuseInCasts[outputIdx];
    auto incastIdx = incastInfo.position;
    if (incastInfo.count == -1 || incastIdx > callOp.GetIOperands().size()) {
        return false;
    }
    auto &inputs = callOp.GetIOperands();
    auto &input = inputs[incastIdx];
    bool consumerNoOverLap = false;
    (void)input->GetAttr("ConsumerAccessNoOverlap", consumerNoOverLap);
    if (consumerNoOverLap == false) {
        if (!CheckTopoDependancy(input, callOp)) {
            ALOG_DEBUG_F("input %d contains more than one consumer and does not directly linked to %d",
                input->magic, callOp.opmagic);
            return false;
        }
    }
    previous = input;
    if (!GetStorageOffsetByCall(callOp, incastIdx, storageOffset)) {
        ALOG_ERROR_F("Offset is not valid.");
        return false;
    }
    ALOG_DEBUG_F("Callop %d leaf %s %lu output %zu can reuse input %d", callOp.opmagic,
        program->GetMagicName().c_str(), program->GetFunctionHash().GetHash(), outputIdx, incastIdx);
    return true;
}

void Allocator::UpdateActualRaw(LogicalTensorPtr &input) const {
    if (input->tensor->actualRawmagic != -1) {
        auto iter = function_->GetTensorMap().tensorMap_.find(input->tensor->actualRawmagic);
        if (iter != function_->GetTensorMap().tensorMap_.end() && !iter->second.empty()) {
            for (auto &t : iter->second) {
                t->storage_ = input->storage_;
            }
        }
    }
}

void UpdateOneCall(Operation &consumer, const LogicalTensorPtr &output) {
    size_t inputIdx = 0;
    auto &consumerInputs = consumer.GetIOperands();
    size_t shapeSize = output->shape.size();
    size_t rawShapeIdx = OFFSET_INDEX + RAW_SHAPE_POS * shapeSize;
    auto attr = dynamic_cast<CallOpAttribute *>(consumer.GetOpAttribute().get());

    for (inputIdx = 0; inputIdx < consumerInputs.size(); inputIdx++) {
        if (consumerInputs[inputIdx] != output) {
            continue;
        }
        auto &arglist = attr->GetArgList()[inputIdx];
        for (size_t i = 0; i < shapeSize; i++) {
            arglist[i + rawShapeIdx] = SymbolicScalar(output->tensor->rawshape[i]);
        }
    }
}

void RefreshCallRawShape(Operation &callOp, size_t j, const LogicalTensorPtr &output, const LogicalTensorPtr &previous) {
    if (output->tensor->rawshape == previous->tensor->rawshape) {
        return;
    }

    if (output->tensor->GetRawDataSize() == previous->tensor->GetRawDataSize()) {
        return;
    }
    // 1. 刷新producer CallOp的Attr中的output rawshape数据
    output->tensor->UpdateRawShape(previous->tensor->rawshape);
    auto attr = dynamic_cast<CallOpAttribute *>(callOp.GetOpAttribute().get());
    auto &arglist = attr->GetArgList()[callOp.GetIOperands().size() + j];
    size_t shapeSize = previous->shape.size();
    size_t rawShapeIdx = OFFSET_INDEX + RAW_SHAPE_POS * shapeSize;
    for (size_t i = 0; i < shapeSize; i++) {
        arglist[i + rawShapeIdx] = SymbolicScalar(output->tensor->rawshape[i]);
    }

    // 2. 刷新对应的consumer CallOp的Attr中的input rawshape数据
    for (auto consumer : output->GetConsumers()) {
        if (consumer->GetOpcode() != Opcode::OP_CALL) {
            ALOG_WARN_F("output magic %d consumer is %d %s", output->magic, consumer->opmagic, consumer->GetOpcodeStr().c_str());
            continue;
        }
        UpdateOneCall(*consumer, output);
    }
}

bool Allocator::SetupReusedTensor(Operation& callOp, size_t outputIdx, 
                                 LogicalTensorPtr& output, LogicalTensorPtr& previous) {
    output->storage_ = previous->storage_;
    auto iterPrevious = storageMap_.find(previous->GetRawMagic());
    if (iterPrevious == storageMap_.end()) {
        ALOG_ERROR_F("Cannot find previous %d rawmagic %d", previous->magic, previous->GetRawMagic());
        return false;
    }
    RefreshCallRawShape(callOp, outputIdx, output, previous);
    auto &tensorsDesc = storageNeedToAllocate_.at(iterPrevious->second);
    tensorsDesc.tensors.emplace(output);
    storageMap_.emplace(output->GetRawMagic(), iterPrevious->second);
    ALOG_DEBUG_F("Tensor %d can inner reuse %d", output->magic, previous->magic);
    return true;
}

void Allocator::CreateNewTensorStorage(LogicalTensorPtr& output) {
    output->storage_ = std::make_shared<Storage>(MemoryType::MEM_WORKSPACE, output->GetRawMagic(),
        Align(static_cast<uint64_t>(output->tensor->GetRawDataSize())));
    TensorsDesc tensorsDesc(function_);
    tensorsDesc.tensors.emplace(output);
    tensorsDesc.isDummy = output->GetProducers().empty();
    storageNeedToAllocate_.emplace_back(tensorsDesc);
    storageMap_.emplace(output->GetRawMagic(), (storageNeedToAllocate_.size() - 1));
    UpdateActualRaw(output);
}

bool Allocator::HandleNewTensor(Operation& callOp, size_t outputIdx, LogicalTensorPtr& output) {
    // 判断是否可以复用输入
    LogicalTensorPtr previous = nullptr;
    if (CheckReuseInnerCall(callOp, outputIdx, previous, output->storageOffset_) && previous != nullptr) {
        return SetupReusedTensor(callOp, outputIdx, output, previous);
    } else {
        CreateNewTensorStorage(output);
        return false;
    }
}

void Allocator::HandleExistingTensor(size_t storageIndex, LogicalTensorPtr& output) {
    auto& tensorsDesc = storageNeedToAllocate_.at(storageIndex);
    auto& firstTensor = *(tensorsDesc.tensors.begin());
    output->storage_ = firstTensor->storage_;
    tensorsDesc.tensors.emplace(output);
}

bool Allocator::TryProcessTensor(Operation& callOp, size_t outputIdx) {
    auto &output = callOp.GetOOperands()[outputIdx];
    if (function_->IsFromInCast(output) || function_->IsFromOutCast(output)) {
        return false;
    }
    if (output->storage_ != nullptr) {
        return false;
    }
    auto iter = storageMap_.find(output->GetRawMagic());
    if (iter == storageMap_.end()) {
        return HandleNewTensor(callOp, outputIdx, output);
    } else {
        HandleExistingTensor(iter->second, output);
        return false;
    }
}

void Allocator::ProcessSingleOperation(Operation& callOp) {
    int reuseCount = 0;
    for (size_t j = 0; j < callOp.GetOOperands().size(); ++j) {
        if (TryProcessTensor(callOp, j)) {
            reuseCount++;
        }
    }
    ALOG_DEBUG_F("Call %d reuse count is %d", callOp.opmagic, reuseCount);
}

void Allocator::ProcessOperations() {
    auto callOps = function_->Operations();
    // StorageId(initialized as rawmagic) to position in storageNeedToAllocate_
    for (size_t i = 0; i < callOps.size(); ++i) {
        ProcessSingleOperation(callOps[i]);
    }
}

void Allocator::InitializeRootCasts() {
    for (auto &in : function_->inCasts_) {
        rootInCasts_.emplace(in->GetRawMagic());
    }
    for (auto &out : function_->outCasts_) {
        rootOutCasts_.emplace(out->GetRawMagic());
    }
}

void Allocator::Init() {
    connectionMatrix_.Generate(function_);
    storageMap_.clear();
    InitializeRootCasts();
    InitializeLeafMemoryReuse();
    ProcessOperations();
}

void Allocator::StorageNeedToAllocatePreProcess(TensorsDesc &tensorsDesc) {
    for (auto &tensor : tensorsDesc.tensors) {
        for (auto &cons : tensor->GetConsumers()) {
            if (cons->GetOpcode() != Opcode::OP_CALL) {
                continue;
            }
            tensorsDesc.consumerOpIdxs.emplace(connectionMatrix_.GetIndex(*cons));
        }
    }
    bool isDeleteConsumerOp = false;
    for (auto cons = tensorsDesc.consumerOpIdxs.begin(); cons !=tensorsDesc.consumerOpIdxs.end();) {
        for (auto &operation : tensorsDesc.consumerOpIdxs) {
            if (*cons == operation) {
                continue;
            }
            if (connectionMatrix_.IsConnected(*cons, operation)) {
                cons = tensorsDesc.consumerOpIdxs.erase(cons);
                isDeleteConsumerOp = true;
                break;
            }
        }
        if (isDeleteConsumerOp) {
            isDeleteConsumerOp = false;
        } else {
            cons++;
        }
    }
    tensorsDesc.connectionOpsBitmap.SetValues(0xFFFFFFFFFFFFFFFF); // And 操作前，需要将connectionOpsBitmap初始化为全1
    for (auto &tensor : tensorsDesc.tensors) {
        for (auto &prod : tensor->GetProducers()) {
            if (prod->GetOpcode() != Opcode::OP_CALL) {
                continue;
            }
            tensorsDesc.connectionOpsBitmap.And(connectionMatrix_.GetBitMap(*prod));
            // 由于customerOp和producerOp相同时，不能进行内存服用，所以这里需要将bitmap中的指向本操作的位清零
            tensorsDesc.connectionOpsBitmap.ClearBit(static_cast<size_t>(connectionMatrix_.GetIndex(*prod)));
        }
    }
}

Status Allocator::UpdateStorageId(TensorsDesc &tensorsDesc, std::unordered_map<int64_t, int> &idMap, int &storageId) {
    if (tensorsDesc.tensors.empty()) {
        ALOG_DEBUG_F("Storage tensors is empty");
        return SUCCESS;
    }
    auto &tensor = *(tensorsDesc.tensors.begin());
    if (tensor->storage_ == nullptr) {
        ALOG_ERROR_F("tensor rawMagic:%d, storage is nullptr", tensor->GetRawMagic());
        return FAILED;
    }

    auto iter = idMap.find(tensor->storage_->start_);
    if (iter == idMap.end()) {
        idMap.emplace(std::make_pair(tensor->storage_->start_, storageId));
        tensor->storage_->id_ = storageId;
        storageId++;
    } else {
        tensor->storage_->id_ = iter->second;
    }
    return SUCCESS;
}

Status Allocator::UpdateIncastOutCast() {
    auto callOps = function_->Operations();
    for (auto &callOp : callOps) {
        auto callAttr = dynamic_cast<CallOpAttribute *>(callOp.GetOpAttribute().get());
        auto &incasts = callAttr->invokeInfo_->incastTensorParamList_;
        auto &outcasts = callAttr->invokeInfo_->outcastTensorParamList_;
        if (incasts.size() > callOp.iOperand.size()) {
            ALOG_ERROR_F("incasts.size:%ld, is larger than iOperand.size:%ld, opCode:%d",
                incasts.size(), callOp.iOperand.size(), callOp.GetOpcode());
            return FAILED;
        }
        if (outcasts.size() > callOp.oOperand.size()) {
            ALOG_ERROR_F("incasts.size:%ld, is larger than iOperand.size:%ld, opCode:%d",
                incasts.size(), callOp.iOperand.size(), callOp.GetOpcode());
            return FAILED;
        }
        for (size_t i = 0; i < incasts.size(); ++i) {
            auto &incast = incasts[i];
            incast.tensor = callOp.iOperand[i];
        }
        for (size_t i = 0; i < outcasts.size(); ++i) {
            auto &outcast = outcasts[i];
            outcast.tensor = callOp.oOperand[i];
        }
    }
    return SUCCESS;
}

Status Allocator::Allocate() {
    uint64_t sizeBeforeReuse = 0;
    for (auto &tensorsDesc : storageNeedToAllocate_) {
        StorageNeedToAllocatePreProcess(tensorsDesc);
        TensorBucket &bucket = GetBestFitBucket(tensorsDesc);
        bucket.AddRef(tensorsDesc);
        auto &tensor = *(tensorsDesc.tensors.begin());
        ALOG_DEBUG_F("Start to allocate tensor rawmagic %d",tensor->GetRawMagic());
        if (tensor->storage_ == nullptr) {
            ALOG_ERROR_F("tensor rawMagic:%d, storage is nullptr", tensor->GetRawMagic());
            return FAILED;
        }
        sizeBeforeReuse += tensor->storage_->length_;
    }
    for (auto &bucket : buckets_) {
        bucket.UpdateOffset(size_);
        size_ += bucket.GetSize();
    }
    dummyPackets_.UpdateOffset(size_);
    size_ += dummyPackets_.GetSize();

    ALOG_EVENT_F("Total memory size is [%lu bytes]", size_);
    // 根据storage id刷新 DDRId, start相同认为是一个storage
    std::unordered_map<int64_t, int> idMap;
    int storageId = 0;
    for (auto &tensorsDesc : storageNeedToAllocate_) {
        if (UpdateStorageId(tensorsDesc, idMap, storageId) == FAILED) {
            return FAILED;
        }
    }

    return UpdateIncastOutCast();
}

Status MemoryReuse::RunOnFunction(Function &function) {
    /* 为incast、outcast类型申请storage，需要正确处理actual rawmagic */
    /* 标注每个CallOp输出Tensor生命周期，生命周期 */
    Allocator allocator(function.rootFunc_);
    allocator.Init();
    return allocator.Allocate();
}
} // namespace npu::tile_fwk
