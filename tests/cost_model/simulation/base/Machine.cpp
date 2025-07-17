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
 * \file Machine.cpp
 * \brief
 */

#include "simulation/base/Machine.h"
#include "simulation/base/ModelTop.h"

namespace CostModel {

void Machine::LoggerRecordTaskStart(std::string name, std::string hint)
{
    GetSim()->LoggerRecordCoreStart(name, machineId, hint);
    GetSim()->GetLogger()->AddEventBegin(name, machineId, coreTid, GetSim()->GetCycles(), hint);
}

void Machine::LoggerRecordTaskEnd()
{
    if (!sim) {
        return;
    }

    sim->LoggerRecordCoreCompleted(machineId);

    auto logger = sim->GetLogger();
    if (!logger) {
        return;
    }

    logger->AddEventEnd(machineId, coreTid, sim->GetCycles());
}

void Machine::LoggerRecordPipe(std::string name, size_t pipeId)
{
    GetSim()->GetLogger()->SetThreadName(name, machineId, pipeId + reversedTidNum);
}

void Machine::LoggerRecordTileOp(std::string name, size_t pipeId, size_t sTime, size_t eTime)
{
    LogData data;
    data.name = name;
    data.pid = machineId;
    data.tid = pipeId + reversedTidNum;
    data.sTime = sTime;
    data.eTime = eTime;
    GetSim()->GetLogger()->AddDuration(data);
}

void Machine::SetQueueCounter()
{
    GetSim()->GetLogger()->SetThreadName("SubmissionQ", machineId, (queueSeq + coreTid));
    submissionQueue.SetCounterInfo(sim->GetLogger(), machineId, (queueSeq++) + coreTid);

    GetSim()->GetLogger()->SetThreadName("CompletionQ", machineId, (queueSeq + coreTid));
    completionQueue.SetCounterInfo(sim->GetLogger(), machineId, (queueSeq++) + coreTid);

    GetSim()->GetLogger()->SetThreadName("OutcastQ", machineId, (queueSeq + coreTid));
    outcastReferenceQueue.SetCounterInfo(sim->GetLogger(), machineId, (queueSeq++) + coreTid);

    GetSim()->GetLogger()->SetThreadName("IncastQ", machineId, (queueSeq + coreTid));
    incastReferenceQueue.SetCounterInfo(sim->GetLogger(), machineId, (queueSeq++) + coreTid);

    GetSim()->GetLogger()->SetThreadName("ReleaseQ", machineId, (queueSeq + coreTid));
    releaseQueue.SetCounterInfo(sim->GetLogger(), machineId, (queueSeq++) + coreTid);

    GetSim()->GetLogger()->SetThreadName("FunctionCache", machineId, (queueSeq + coreTid));
    functionCacheTid = (queueSeq++) + coreTid;

    if (queueSeq > reversedTidNum) {
        ASSERT(0 && "Queue Counter thread id is conflict with reversedTidNum");
    }
}
void Machine::SubmitTask(TaskPack task, uint64_t extraDelay)
{
    lastCycles = GetSim()->GetCycles();
    submissionQueue.Enqueue(task, extraDelay);
}

void Machine::ResponseData(CachePacket pkt, uint64_t extraDelay)
{
    lastCycles = GetSim()->GetCycles();
    cacheRespQueue.Enqueue(pkt, extraDelay);
}
}