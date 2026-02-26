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
 * \file distributed_dep_schedule.cpp
 * \brief Build distributed dependency scheduling metadata for shmem-first execution.
 */

#include "passes/block_graph_pass/distributed_dep_schedule.h"

#include <algorithm>
#include <cstdarg>
#include <cstdlib>
#include <cstdio>
#include <string>
#include <vector>
#include <unistd.h>
#include <limits>

#include "interface/operation/distributed/distributed_common.h"
#include "interface/operation/operation.h"
#include "passes/pass_log/pass_log.h"

#define MODULE_NAME "DistributedDepSchedule"

namespace npu::tile_fwk {
namespace {
using DistDepAxis = Distributed::DistDepAxis;
using DistOpAttr = Distributed::DistOpAttr;
using DistScheduleMode = Distributed::DistScheduleMode;
using DistSignalResetPolicy = Distributed::DistSignalResetPolicy;
using DistStagePolicy = Distributed::DistStagePolicy;
using DistTileSchedule = Distributed::DistTileSchedule;

bool DistScheduleTraceEnabled()
{
    static bool enabled = []() {
        const char *envValue = std::getenv("PYTO_DIST_SCHED_TRACE");
        return envValue != nullptr && std::string(envValue) == "1";
    }();
    return enabled;
}

const char *GetDistScheduleTraceFile()
{
    const char *envValue = std::getenv("PYTO_DIST_SCHED_TRACE_FILE");
    if (envValue == nullptr || std::string(envValue).empty()) {
        return "/tmp/pypto_dist_schedule_trace.log";
    }
    return envValue;
}

__attribute__((format(printf, 1, 2))) void EmitDistScheduleTrace(const char *fmt, ...)
{
    if (!DistScheduleTraceEnabled()) {
        return;
    }
    char msg[1024] = {0};
    va_list ap;
    va_start(ap, fmt);
    (void)vsnprintf(msg, sizeof(msg), fmt, ap);
    va_end(ap);

    FILE *fp = fopen(GetDistScheduleTraceFile(), "a");
    if (fp != nullptr) {
        (void)fprintf(fp, "[DistributedDepSchedule][pid=%d] %s\n", getpid(), msg);
        (void)fclose(fp);
    }
}

bool DistSchedulePassEnabled()
{
    static bool enabled = []() {
        const char *envValue = std::getenv("PYTO_ENABLE_DIST_SCHEDULE");
        return envValue != nullptr && std::string(envValue) == "1";
    }();
    return enabled;
}

bool IsDistScheduleTarget(Opcode opcode)
{
    switch (opcode) {
        case Opcode::OP_SHMEM_PUT:
        case Opcode::OP_SHMEM_SIGNAL:
        case Opcode::OP_SHMEM_WAIT_UNTIL:
        case Opcode::OP_SHMEM_GET:
        case Opcode::OP_SHMEM_GET_GM2UB:
        case Opcode::OP_MOE_DISTRIBUTED_COMBINE_SEND:
        case Opcode::OP_MOE_DISTRIBUTED_COMBINE_RECEIVE:
        case Opcode::OP_MOE_FFN_FUSED:
            return true;
        default:
            return false;
    }
}

DistDepAxis GetDepAxis(Opcode opcode)
{
    switch (opcode) {
        case Opcode::OP_MOE_FFN_FUSED:
            return DistDepAxis::N;
        case Opcode::OP_SHMEM_PUT:
        case Opcode::OP_SHMEM_SIGNAL:
        case Opcode::OP_SHMEM_WAIT_UNTIL:
        case Opcode::OP_SHMEM_GET:
        case Opcode::OP_SHMEM_GET_GM2UB:
        case Opcode::OP_MOE_DISTRIBUTED_COMBINE_SEND:
        case Opcode::OP_MOE_DISTRIBUTED_COMBINE_RECEIVE:
            return DistDepAxis::M;
        default:
            return DistDepAxis::NONE;
    }
}

bool IsProducerOp(Opcode opcode)
{
    switch (opcode) {
        case Opcode::OP_SHMEM_PUT:
        case Opcode::OP_SHMEM_SIGNAL:
        case Opcode::OP_MOE_DISTRIBUTED_COMBINE_SEND:
            return true;
        default:
            return false;
    }
}

bool IsConsumerOp(Opcode opcode)
{
    switch (opcode) {
        case Opcode::OP_SHMEM_WAIT_UNTIL:
        case Opcode::OP_SHMEM_GET:
        case Opcode::OP_SHMEM_GET_GM2UB:
        case Opcode::OP_MOE_DISTRIBUTED_COMBINE_RECEIVE:
        case Opcode::OP_MOE_FFN_FUSED:
            return true;
        default:
            return false;
    }
}

DistSignalResetPolicy GetSignalResetPolicy(Opcode opcode, const DistOpAttr &attr)
{
    if (opcode == Opcode::OP_SHMEM_WAIT_UNTIL && attr.aicpuOpParams.size() >= 3 && attr.aicpuOpParams[2] != 0) {
        return DistSignalResetPolicy::CONSUME_CLEAR;
    }
    return DistSignalResetPolicy::NONE;
}

int64_t InferTileRowShape(const Operation &op, const DistOpAttr &attr)
{
    if (attr.rowShape > 0) {
        return attr.rowShape;
    }
    for (const auto &tensor : op.GetOOperands()) {
        if (tensor != nullptr) {
            const auto &shape = tensor->GetShape();
            if (shape.size() >= 2 && shape[0] > 0) {
                return shape[0];
            }
        }
    }
    for (const auto &tensor : op.GetIOperands()) {
        if (tensor != nullptr) {
            const auto &shape = tensor->GetShape();
            if (shape.size() >= 2 && shape[0] > 0) {
                return shape[0];
            }
        }
    }
    return 1;
}

int64_t InferTileColShape(const Operation &op)
{
    for (const auto &tensor : op.GetOOperands()) {
        if (tensor != nullptr) {
            const auto &shape = tensor->GetShape();
            if (shape.size() >= 2 && shape[1] > 0) {
                return shape[1];
            }
        }
    }
    for (const auto &tensor : op.GetIOperands()) {
        if (tensor != nullptr) {
            const auto &shape = tensor->GetShape();
            if (shape.size() >= 2 && shape[1] > 0) {
                return shape[1];
            }
        }
    }
    return 1;
}

bool HasExplicitSchedule(const DistOpAttr &attr)
{
    const auto &sched = attr.tileSchedule;
    return attr.scheduleMode != DistScheduleMode::NONE || attr.stagePolicy != DistStagePolicy::NONE ||
        attr.signalResetPolicy != DistSignalResetPolicy::NONE || attr.enableLocalFirstSchedule ||
        attr.enableNDimColumnMajor || sched.depAxis != DistDepAxis::NONE ||
        sched.stageId >= 0 || sched.stageCount > 0 || sched.splitN > 1 || sched.barrierSlot >= 0 ||
        sched.producerRole != 0 || sched.consumerRole != 0 || sched.signalEpoch != 0;
}

int64_t ParsePositiveInt(const std::string &s, int64_t fallback)
{
    if (s.empty()) {
        return fallback;
    }
    try {
        auto value = std::stoll(s);
        return value > 0 ? value : fallback;
    } catch (...) {
        return fallback;
    }
}

int64_t InferSplitN(const Operation &op, const DistOpAttr &attr)
{
    if (attr.tileSchedule.splitN > 1) {
        return attr.tileSchedule.splitN;
    }
    if (op.GetOpcode() == Opcode::OP_MOE_FFN_FUSED) {
        // For fused FFN, extraTemplateParam stores intermediate size and is the natural N split hint.
        return ParsePositiveInt(attr.extraTemplateParam, 1);
    }
    if (attr.topK > 1) {
        return attr.topK;
    }
    return 1;
}
} // namespace

Status DistributedDepSchedule::RunOnFunction(Function &function)
{
    if (!DistSchedulePassEnabled()) {
        return SUCCESS;
    }
    EmitDistScheduleTrace("Start on function=%s", function.GetMagicName().c_str());
    APASS_LOG_INFO_F(Elements::Operation,
        "===============================================================> Start DistributedDepSchedule.");
    Function *rootFunc = function.rootFunc_;
    if (rootFunc == nullptr) {
        APASS_LOG_INFO_F(Elements::Operation, "Function[%s] has no rootFunc, skip.", function.GetMagicName().c_str());
        return SUCCESS;
    }
    for (auto &program : rootFunc->programs_) {
        if (program.second == nullptr) {
            continue;
        }
        if (AnnotateProgram(*program.second, program.first) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Operation, "AnnotateProgram failed for program[%lu].", program.first);
            return FAILED;
        }
    }
    APASS_LOG_INFO_F(Elements::Operation,
        "===============================================================> Finish DistributedDepSchedule.");
    EmitDistScheduleTrace("Finish on function=%s", function.GetMagicName().c_str());
    return SUCCESS;
}

Status DistributedDepSchedule::AnnotateProgram(Function &program, uint64_t programId) const
{
    std::vector<Operation *> targets;
    targets.reserve(program.Operations(false).size());
    for (auto &op : program.Operations(false)) {
        if (op.IsDeleted()) {
            continue;
        }
        if (IsDistScheduleTarget(op.GetOpcode())) {
            targets.emplace_back(&op);
        }
    }
    if (targets.empty()) {
        return SUCCESS;
    }

    int64_t stageCountM = 0;
    int64_t stageCountN = 0;
    for (const auto *op : targets) {
        DistDepAxis axis = GetDepAxis(op->GetOpcode());
        if (axis == DistDepAxis::M) {
            stageCountM++;
        } else if (axis == DistDepAxis::N) {
            stageCountN++;
        }
    }

    int64_t stageIdM = 0;
    int64_t stageIdN = 0;
    bool traceEnabled = DistScheduleTraceEnabled();
    EmitDistScheduleTrace("Program[%lu] targetOps=%zu stageCountM=%ld stageCountN=%ld",
        programId, targets.size(), stageCountM, stageCountN);
    for (auto *op : targets) {
        DistOpAttr distOpAttr;
        bool hasExistingAttr = op->HasAttr(OpAttributeKey::distOpAttr);
        if (hasExistingAttr) {
            op->GetAttr(OpAttributeKey::distOpAttr, distOpAttr);
        }
        bool hasExplicitSchedule = HasExplicitSchedule(distOpAttr);

        DistTileSchedule schedule = distOpAttr.tileSchedule;
        if (schedule.depAxis == DistDepAxis::NONE) {
            schedule.depAxis = GetDepAxis(op->GetOpcode());
        }

        auto allocStage = [&](DistDepAxis axis, int64_t &cursor, int64_t total) {
            if (axis == DistDepAxis::NONE) {
                schedule.stageId = -1;
                schedule.stageCount = 0;
                return;
            }
            if (schedule.stageCount <= 0) {
                schedule.stageCount = total;
            }
            if (schedule.stageId < 0) {
                schedule.stageId = cursor++;
            }
        };

        if (schedule.depAxis == DistDepAxis::M) {
            allocStage(DistDepAxis::M, stageIdM, stageCountM);
        } else if (schedule.depAxis == DistDepAxis::N) {
            allocStage(DistDepAxis::N, stageIdN, stageCountN);
        } else {
            schedule.stageId = -1;
            schedule.stageCount = 0;
        }

        int64_t rowOffset = std::max<int64_t>(0, distOpAttr.rowOffset);
        int64_t rowShape = std::max<int64_t>(1, InferTileRowShape(*op, distOpAttr));
        int64_t colShape = std::max<int64_t>(1, InferTileColShape(*op));
        if (schedule.tileMBegin == 0 && schedule.tileMEnd == 0) {
            schedule.tileMBegin = rowOffset;
            schedule.tileMEnd = rowOffset + rowShape - 1;
        }
        if (schedule.tileNBegin == 0 && schedule.tileNEnd == 0) {
            schedule.tileNBegin = 0;
            schedule.tileNEnd = colShape - 1;
        }
        if (schedule.depAxis == DistDepAxis::N && schedule.splitN <= 1) {
            schedule.splitN = InferSplitN(*op, distOpAttr);
        } else if (schedule.depAxis != DistDepAxis::N && schedule.splitN <= 0) {
            schedule.splitN = 1;
        }
        if (schedule.barrierSlot < 0) {
            schedule.barrierSlot = schedule.stageId;
        }
        if (schedule.producerRole == 0 && schedule.consumerRole == 0) {
            schedule.producerRole = IsProducerOp(op->GetOpcode()) ? 1 : 0;
            schedule.consumerRole = IsConsumerOp(op->GetOpcode()) ? 1 : 0;
        }

        if (distOpAttr.scheduleMode == DistScheduleMode::NONE) {
            distOpAttr.scheduleMode = DistScheduleMode::SHMEM_STANDARD;
        }
        if (distOpAttr.stagePolicy == DistStagePolicy::NONE) {
            distOpAttr.stagePolicy = DistStagePolicy::PIPELINE;
        }
        if (distOpAttr.signalResetPolicy == DistSignalResetPolicy::NONE) {
            distOpAttr.signalResetPolicy = GetSignalResetPolicy(op->GetOpcode(), distOpAttr);
        }
        if (!distOpAttr.enableLocalFirstSchedule && schedule.depAxis == DistDepAxis::M) {
            distOpAttr.enableLocalFirstSchedule = DistSchedulePassEnabled();
        }
        if (!distOpAttr.enableNDimColumnMajor && schedule.depAxis == DistDepAxis::N) {
            distOpAttr.enableNDimColumnMajor = DistSchedulePassEnabled();
        }
        distOpAttr.enableDistScheduleTrace = traceEnabled;
        distOpAttr.tileSchedule = schedule;
        op->SetAttr(OpAttributeKey::distOpAttr, distOpAttr);

        if (traceEnabled) {
            EmitDistScheduleTrace(
                "Program[%lu] Op[%d:%s] axis=%s stage=%ld/%ld m=[%ld,%ld] n=[%ld,%ld] splitN=%ld barrier=%ld reset=%d role(p=%ld,c=%ld) explicit=%d hasAttr=%d",
                programId, op->GetOpMagic(), op->GetOpcodeStr().c_str(),
                Distributed::DistDepAxisToString(schedule.depAxis), schedule.stageId, schedule.stageCount,
                schedule.tileMBegin, schedule.tileMEnd, schedule.tileNBegin, schedule.tileNEnd, schedule.splitN,
                schedule.barrierSlot, static_cast<int>(distOpAttr.signalResetPolicy),
                schedule.producerRole, schedule.consumerRole, hasExplicitSchedule ? 1 : 0, hasExistingAttr ? 1 : 0);
            APASS_LOG_INFO_F(Elements::Operation,
                "Program[%lu] Op[%d:%s] dist schedule: axis=%s stage=%ld/%ld m=[%ld,%ld] n=[%ld,%ld] reset=%d role(p=%ld,c=%ld).",
                programId, op->GetOpMagic(), op->GetOpcodeStr().c_str(),
                Distributed::DistDepAxisToString(schedule.depAxis), schedule.stageId, schedule.stageCount,
                schedule.tileMBegin, schedule.tileMEnd, schedule.tileNBegin, schedule.tileNEnd,
                static_cast<int>(distOpAttr.signalResetPolicy), schedule.producerRole, schedule.consumerRole);
        }
    }
    return SUCCESS;
}
} // namespace npu::tile_fwk
