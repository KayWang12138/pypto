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
 * \file calendar_schedule_counter.cpp
 * \brief
 */
#include <climits>
#include "calendar_scheduler_counter.h"
#include "nlohmann/json.hpp"

using json = nlohmann::json;

namespace npu::tile_fwk {
std::string OpTypeToString(OperationType type)
{
    if (type == OperationType::SET) {
        return "SET";
    }
    return "UNKNOWN";
}

void CounterAllocator::AddSequence(const std::vector<TaskOperation> &ops, int seqId)
{
    auto seq = std::make_shared<Sequence>(seqId);
    seq->operations = ops;

    for (int i = 0; i < totalCounters; i++)
    {
        seq->counterCandidates.insert(i);
    }
    for (size_t i = 0; i < seq->operations.size(); i++)
    {
        seq->operations[i].id = i;
    }

    sequences.push_back(seq);
}

std::pair<std::map<int, int>, std::map<int, std::pair<int, int>>> CounterAllocator::GetEventCounterMap() const
{
    std::map<int, int> eventCounterMap;
    std::map<int, std::pair<int,int>> eventCounterExpectedValMap;
    const int gapTen = 10;
    const int gapTwelve = 12;
    for (const auto &seq : sequences)
    {
        for (size_t i = 0; i < seq->operations.size(); i++)
        {
            const auto &op = seq->operations[i];
            if (eventCounterMap.find(op.event->eventId) == eventCounterMap.end()) {
                std::cout << std::setw(gapTen) << op.event->eventId
                          << std::setw(gapTwelve) << (op.event->counterId == -1 ? "N/A" : std::to_string(op.event->counterId))
                          << "\n";
                eventCounterMap[op.event->eventId] = op.event->counterId;
                eventCounterExpectedValMap[op.event->eventId] = {op.event->counterId, op.event->expectedVal};
            }
        }
    }
    return {eventCounterMap, eventCounterExpectedValMap};
}

void CounterAllocator::DumpInputSequences() const
{
    std::cout << "\n=== Input Sequences ===\n";
    std::cout << "Total sequences: " << sequences.size() << "\n";
    const int gapFive = 5;
    const int gapTen = 10;
    for (const auto &seq : sequences)
    {
        std::cout << "\nSequence " << seq->sequenceId << ":\n";
        std::cout << std::setw(gapFive) << "Idx" << std::setw(gapTen) << "Type"
                  << std::setw(gapTen) << "EventID"
                  << std::setw(gapTen) << "TimtStamp" << "\n";
        std::cout << std::string(gapFive * gapTen, '-') << "\n";

        for (size_t i = 0; i < seq->operations.size(); i++)
        {
            const auto &op = seq->operations[i];
            std::cout << std::setw(gapFive) << (op.id == static_cast<size_t>(-1) ? "#" : std::to_string(op.id))
                      << std::setw(gapTen) << OpTypeToString(op.type)
                      << std::setw(gapTen) << op.event->eventId
                      << std::setw(gapTen) << op.timeStamp << "\n";
        }
    }
}

void CounterAllocator::PrintAllocationResults() const
{
    std::cout << "\n=== Counter Allocation Results ===\n";
    std::cout << "\nTotal available counters: " << totalCounters << "\n";
    const int gapFive = 5;
    const int gapTen = 10;
    const int gapTwelve = 12;
    const int gapFifteen = 15;
    const int gapSeventy = 70;
    for (const auto &seq : sequences)
    {
        std::cout << "\nSequence " << seq->sequenceId << ":\n";
        std::cout << std::setw(gapFive) << "Idx"
                  << std::setw(gapTen) << "Type"
                  << std::setw(gapTen) << "EventID"
                  << std::setw(gapTwelve) << "CounterID"
                  << std::setw(gapFifteen) << "ExpectedVal"
                  << std::setw(gapFifteen) << "TimeStamp"<< "\n";
        std::cout << std::string(gapSeventy, '-') << "\n";

        for (size_t i = 0; i < seq->operations.size(); i++)
        {
            const auto &op = seq->operations[i];
            std::cout << std::setw(gapFive) << (op.id == static_cast<size_t>(-1) ? "#" : std::to_string(op.id))
                      << std::setw(gapTen) << OpTypeToString(op.type)
                      << std::setw(gapTen) << op.event->eventId
                      << std::setw(gapTwelve) << (op.event->counterId == -1 ? "N/A" : std::to_string(op.event->counterId))
                      << std::setw(gapFifteen) << (op.event->expectedVal == -1 ? "N/A" : std::to_string(op.event->expectedVal))
                      << std::setw(gapFifteen) << (std::to_string(op.timeStamp))
                      << "\n";
        }
    }
}

int CounterAllocator::FindCounterFromOtherSeq()
{
    int candidateCounter = -1;
    int index = 0;
    int minLastTime = std::numeric_limits<int>::max();
    for (const auto &seq : sequences)
    {
        ALOG_INFO_F("Sequence id = %d, seq->lastUsedCounter = %d, seq->lastTime = %d, minLastTime = %d", index++, seq->lastUsedCounter, seq->lastTime, minLastTime);
        if (seq->lastUsedCounter != -1 && seq->lastTime <= minLastTime) // && seq->lastTime < minLastTime
        {
            candidateCounter = seq->lastUsedCounter;
            // update lastTime to choose counter
            minLastTime = seq->lastTime;
        }
    }
    return candidateCounter;
}

bool CounterAllocator::ProcessSequence(std::shared_ptr<Sequence> seq)
{
    bool modified = false;
    while (seq->processIndex < seq->operations.size())
    {
        auto &op = seq->operations[seq->processIndex];
        if (op.type == OperationType::SET)
        {
            setEventIdTimeMap[op.event->eventId] = op.timeStamp;

            // check if last counter can be reused
            if (seq->lastUsedCounter != -1)
            {
                op.event->counterId = seq->lastUsedCounter;
                op.event->expectedVal = ++counterCurrentVals[op.event->counterId];
                counterCurrentEvent[op.event->counterId] = op.event;
                op.timeStamp += seq->durationTime;

                // update timestamp because of #wait
                setEventIdTimeMap[op.event->eventId] = op.timeStamp;
                seq->processIndex++;
                continue;
            }

            // if no candidate counter, need to add wait
            if (seq->counterCandidates.empty())
            {
                int newCounter = FindCounterFromOtherSeq();
                if (newCounter != -1)
                {
                    AddWaitOperation(seq, newCounter,op);
                    seq->counterCandidates.insert(newCounter);
                    modified = true;
                    continue;
                }
            }

            // select a counter from the candidate counter
            if (!seq->counterCandidates.empty())
            {
                for (auto i : seq->counterCandidates) {
                    ALOG_INFO_F("\nseq->counterCandidates  index i =  %d", i);
                }
                // a naive select the first counter from the candidates, can be optimized
                int selectedCounter = *seq->counterCandidates.begin();
                op.event->counterId = selectedCounter;
                op.event->expectedVal = ++counterCurrentVals[selectedCounter];
                op.timeStamp += seq->durationTime;

                // update timestamp because of #wait
                setEventIdTimeMap[op.event->eventId] = op.timeStamp;
                if (addWaitFlag) {
                    addWaitMap[op.event->eventId][0] = op.event->counterId;
                    addWaitMap[op.event->eventId][1] = op.event->expectedVal;
                    addWaitFlag = 0;
                }

                counterCurrentEvent[selectedCounter] = op.event;
                seq->lastUsedCounter = selectedCounter;
                seq->lastTime = op.timeStamp;

                // remove the select counter from other seq's candidates
                RemoveCounterFromOtherSequences(seq, selectedCounter);
                modified = true;
                seq->processIndex++;
            }
        }

        if (op.type == OperationType::WAIT)
        {
            // 加入#wait后，同seq中后续set和wait的timeStamp都发生了变化，为了保证waitX一定在setX之后完成，
            // 在每次执行wait时，需要判断是否合理，若不合理则把差值补上0+t-t1。
            int set_wait = setEventIdTimeMap[op.event->eventId] - op.timeStamp;
            seq->durationTime += set_wait > 0 ? set_wait : 0;
            op.timeStamp += seq->durationTime;

            // the wait event counter is not allocaetd, switch to other seq
            if (op.event->counterId == -1)
            {
                break;
            }
            if (op.event->expectedVal == counterCurrentVals[op.event->counterId])
            {
                // add the counter of the wait event to candidate
                seq->counterCandidates.insert(op.event->counterId);
            }
            modified = true;
            seq->processIndex++;
        }
    }
    PrintAllocationResults();
    return modified;
}

void CounterAllocator::AllocateCounters()
{
    bool changed;
    do
    {
        changed = false;
        for (auto &seq : sequences) {
            changed = ProcessSequence(seq) ? true : false;
        }
        if (!changed) {
            ALOG_WARN_F("Exist Deadlock");
        }
    } while (changed);
}

void CounterAllocator::RemoveCounterFromOtherSequences(std::shared_ptr<Sequence> currentSeq, int counter)
{
    for (auto &seq : sequences)
    {
        if (seq != currentSeq)
        {
            seq->counterCandidates.erase(counter);
            seq->lastUsedCounter = (seq->lastUsedCounter == counter) ? -1 : seq->lastUsedCounter;
        }
    }
}

void CounterAllocator::AddWaitOperation(std::shared_ptr<Sequence> seq, int counter, TaskOperation setOp)
{
    TaskOperation waitOp;
    waitOp.type = OperationType::WAIT;
    waitOp.event = counterCurrentEvent[counter];
    waitOp.timeStamp = setEventIdTimeMap[waitOp.event->eventId];

    // 用于在json中新增#wait信息
    addWaitMap[setOp.event->eventId] = {setOp.event->counterId, setOp.event->expectedVal, waitOp.event->counterId, waitOp.event->expectedVal};
    addWaitFlag = 1;

    // 加入#wait后，需要把gap的时间加在后续每一个操作上，设置变量duration，后续每个timeStamp会加上duration = timestamp(set_i) - timestamp(set_j)
    seq->durationTime += (waitOp.timeStamp - setOp.timeStamp) > 0 ? waitOp.timeStamp - setOp.timeStamp : 0;
    ALOG_INFO_F("Op event id = %d, durationTime = %d", waitOp.event->eventId, seq->durationTime);
    seq->operations.insert(seq->operations.begin() + seq->processIndex, waitOp);
    seq->processIndex++;
}

void createJson(int counterNum, std::map<int,int> myMap, std::string filename) {
    json document;
    document["num_supported_counters"] = counterNum;

    std::ostringstream mappingStream;
    mappingStream << "{\n";
    bool first = true;
    for (const auto &kv : myMap) {
        if (!first) {
            mappingStream << ",\n";
        }
        first = false;
        mappingStream << "    \"" << kv.first << "\": " << kv.second;
    }
    mappingStream << "\n  }";

    std::ostringstream oss;
    oss << "{\n";
    oss << "  \"num_supported_counters\": " << counterNum << ",\n";
    oss << "  \"event_id_to_counter_mapping\": " << mappingStream.str() << "\n";
    oss << "}";
    std::string outputStr = oss.str();
    std::cout << outputStr << std::endl;

    // Intercept the file name. 
    size_t posSlash = filename.rfind('/');
    size_t posDot = filename.rfind('.');
    std::string baseName;
    if (posSlash != std::string::npos && posDot != std::string::npos && posDot > posSlash) {
        baseName = filename.substr(posSlash + 1, posDot - posSlash - 1);
    }
    std::string outputFile = baseName + std::to_string(counterNum) + ".json";

    std::ofstream ofs(outputFile);
    if (!ofs.is_open()) {
        std::cerr << "Could not open file: " << outputFile << std::endl;
        return;
    }
    ofs << outputStr;
    ofs.close();
}

std::pair<JsonData, int> readJson(const std::string& filename) {
    JsonData jsonData = {0, {}};
    int maxEventId = INT_MIN;

    std::ifstream ifs(filename);
    if (!ifs.is_open()) {
        std::cerr << "Could not open file: " << filename << std::endl;
        return {jsonData, maxEventId};
    }

    json document;
    ifs >> document;
    if (document.contains("cores") && document["cores"].is_array()) {
        const auto& coresArray = document["cores"];
        jsonData.coreNum = coresArray.size();
        ALOG_INFO_F("Number of cores: %d", jsonData.coreNum);

        for (const auto& core : coresArray) {
            Core coreData;
            if (core.contains("core_id") && core["core_id"].is_number_integer()) {
                coreData.coreId = core["core_id"].get<int>();
                std::cout << "core_id: " << coreData.coreId << std::endl;
            }

            if (core.contains("tasks") && core["tasks"].is_array()) {
                const auto& tasks = core["tasks"];
                for (const auto& task : tasks) {
                    Task taskData;
                    if (task.contains("operation") && task["operation"].is_string() &&
                        task.contains("event_id") && task["event_id"].is_number_integer() &&
                        task.contains("time_stamp") && task["time_stamp"].is_number_integer()) {
                        taskData.operation = task["operation"].get<std::string>();
                        taskData.eventId = task["event_id"].get<int>();
                        taskData.timeStamp = task["time_stamp"].get<int>();
                        coreData.tasks.push_back(taskData);
                    }
                    if (task.contains("event_id") && task["event_id"].is_number_integer()) {
                        int eventId = task["event_id"].get<int>();
                        if (eventId > maxEventId) {
                            maxEventId = eventId;
                        }
                    }
                }
            }
            jsonData.cores.push_back(coreData);
        }
    }
    return {jsonData, maxEventId};
}

void addParameters(const std::string& inputFilename,
                   const std::string& outputFilename,
                   std::map<int, std::pair<int, int>> eventPairMap,
                   int counterNum,
                   std::map<int, std::vector<int>> eventVecMap,
                   bool removeFlag) {
    std::ifstream ifs(inputFilename);
    if (!ifs.is_open()) {
        std::cerr << "Could not open file: " << inputFilename << std::endl;
        return;
    }

    json document;
    ifs >> document;
    ifs.close();

    if (document.contains("num_supported_counters") && document["num_supported_counters"].is_number_integer()) {
        document["num_supported_counters"] = counterNum;
    }

    if (document.contains("cores") && document["cores"].is_array()) {
        auto& cores = document["cores"];
        for (auto& core : cores) {
            if (core.contains("core_id") && core["core_id"].is_number_integer()) {
                int origId = core["core_id"].get<int>();
                core["core_id"] = origId;
            }

            if (core.contains("tasks") && core["tasks"].is_array()) {
                auto& tasks = core["tasks"];
                for (size_t j = 0; j < tasks.size(); j++) {
                    auto& task = tasks[j];
                    if (task.contains("operation") && task["operation"].is_string()) {
                        std::string operation = task["operation"].get<std::string>();
                        if (task.contains("event_id") && task["event_id"].is_number_integer()) {
                            int eventId = task["event_id"].get<int>();

                            // 如果 operation 为 "wait" 或 "set"，添加 counter_id 与 expected_value（取自 eventPairMap）
                            if (operation == "wait" || operation == "set") {
                                auto it2 = eventPairMap.find(eventId);
                                if (it2 != eventPairMap.end()) {
                                    task["counter_id"] = it2->second.first;
                                    task["expected_value"] = it2->second.second;
                                }
                            }

                            // 若 removeFlag 为 true，则删除 "task_id" 和 "time_stamp"
                            if (removeFlag) {
                                if (task.contains("task_id"))
                                    task.erase("task_id");
                                if (task.contains("time_stamp"))
                                    task.erase("time_stamp");
                            }

                            // 如果 operation 为 "set" 且 eventVecMap 中有记录，则在当前任务前插入一个新的 wait 任务
                            if (operation == "set" && task.contains("event_id") && (eventVecMap.find(eventId) != eventVecMap.end())) {
                                auto& vec = eventVecMap[eventId];
                                // 构造新的 wait 任务（仅包含需要的字段）
                                json newWaitTask = json::object();
                                newWaitTask["operation"] = "set_wait";
                                if (vec.size() > indexThree) {
                                    newWaitTask["counter_id"] = vec[indexTwo];
                                    newWaitTask["expected_value"] = vec[indexThree];
                                }
                                // 插入新任务到 tasks 数组中，放在当前任务之前
                                tasks.insert(tasks.begin() + j, newWaitTask);
                                // 调整索引，跳过刚插入的新任务
                                j++;
                            }
                        } // end if task contains event_id
                    } // end if task contains "operation"
                } // end for tasks

                // 如果 removeFlag 为 true，则删除每个 task 中的 "event_id"
                if (removeFlag) {
                    for (auto& task : tasks) {
                        if (task.contains("operation") && task.contains("event_id")) {
                            task.erase("event_id");
                        }
                    }
                }
            } // end if core has tasks
        } // end for each core
    }

    // 将修改后的 JSON 用缩进格式写入输出文件
    std::ofstream ofs(outputFilename);
    ofs << document.dump(indexFour);
    ofs.close();
}
} // namespace npu::tile_fwk