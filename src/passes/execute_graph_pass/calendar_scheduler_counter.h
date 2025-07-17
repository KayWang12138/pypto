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
 * \file calendar_schedule_counter.h
 * \brief
 */
#ifndef PASSES_CALENDAR_SCHEDULE_COUNTER_H
#define PASSES_CALENDAR_SCHEDULE_COUNTER_H

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <set>
#include <map>
#include <memory>
#include <iostream>
#include <iomanip>
#include <string>
#include <algorithm>

#include "interface/function/function.h"
#include "interface/tensor/logical_tensor.h"
#include "passes/pass_interface/pass.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/program/program.h"
#include "passes/pass_utils/pass_utils.h"
#include "interface/utils/log.h"

const int indexTwo = 2;
const int indexThree = 3;
const int indexFour = 4;

namespace npu::tile_fwk {
struct Task {
    std::string operation;
    int eventId;
    int timeStamp;
};

struct Core {
    int coreId;
    std::vector<Task> tasks;
};

struct JsonData {
    int coreNum;
    std::vector<Core> cores;
};

enum class OperationType
{
    SET,
    WAIT
};

struct Event
{
    int eventId;
    int counterId = -1;
    int expectedVal = -1;

    Event(int id) : eventId(id) {}
};

struct TaskOperation
{
    OperationType type;
    std::shared_ptr<Event> event;
    int timeStamp = 0;
    size_t id = -1;

    bool operator==(const TaskOperation &other) const
    {
        return type == other.type && event->eventId == other.event->eventId;
    }
};

class Sequence
{
public:
    std::vector<TaskOperation> operations;
    std::set<int> counterCandidates; // candidates counters of the sequence
    size_t processIndex = 0;         // current processing operation index of the sequence
    int lastUsedCounter = -1;        // last used counter of the sequence
    int sequenceId;                  // this sequence's id
    int lastTime = 0;                // for find candidate, to choose counter
    int durationTime = 0;            // if add #wait, add time

    Sequence(int id) : sequenceId(id) {}
};

struct ExpectedValue {
    int value;
    std::vector<TaskOperation> operations;
};

struct Counter {
    int id;
    std::vector<ExpectedValue> expectedValues;
};

class CounterAllocator
{
private:
    std::vector<std::shared_ptr<Sequence>> sequences;
    int totalCounters;
    std::map<int, int> counterCurrentVals;
    std::map<int, std::shared_ptr<Event>> counterCurrentEvent;
    std::map<int, Counter> counterEventValue;
    std::map<int, std::vector<int>> addWaitMap;
    std::map<int, int> setEventIdTimeMap;
    int addWaitFlag = 0;

public:
    CounterAllocator(int numCounters) : totalCounters(numCounters)
    {
        for (int i = 0; i < totalCounters; i++)
        {
            // init counter's expected value
            counterCurrentVals[i] = 0;
        }
    }

    void AddSequence(const std::vector<TaskOperation> &ops, int seqId);

    // event_id -> count_id & event_id -> count_id, expected_val map
    std::pair<std::map<int,int>, std::map<int, std::pair<int,int>>> GetEventCounterMap() const;

    std::map<int, std::vector<int>> GetAddWaitMap(){
        return addWaitMap;
    }
    std::map<int, int> GetSetEventIdTimeMap(){
        return setEventIdTimeMap;
    }

    bool CheckExpectedValues() const;

    // Dump original input sequence
    void DumpInputSequences() const;

    // Print input sequence with counter allocated
    void PrintAllocationResults() const;

    void AllocateCounters();

    bool ProcessSequence(std::shared_ptr<Sequence> seq);

    int FindCounterFromOtherSeq();

    void RemoveCounterFromOtherSequences(std::shared_ptr<Sequence> currentSeq, int counter);

    void AddWaitOperation(std::shared_ptr<Sequence> seq, int counter, TaskOperation setOp);
};

void createJson(int counterNum, std::map<int, int> myMap, std::string filename);

std::pair<JsonData, int> readJson(const std::string& filename);

void addParameters(const std::string& inputFilename,
                   const std::string& outputFilename,
                   std::map<int, std::pair<int, int>> eventPairMap,
                   int counterNum,
                   std::map<int, std::vector<int>> eventVecMap,
                   bool removeFlag);
} // namespace npu::tile_fwk
#endif // PASSES_CALENDAR_SCHEDULE_COUNTER_H