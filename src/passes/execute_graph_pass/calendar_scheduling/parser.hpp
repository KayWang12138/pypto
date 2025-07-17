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
 * \file parser.hpp
 * \brief
 */

#ifndef ATC_OPCOMPILER_TILE_FWK_PARSER_HPP
#define ATC_OPCOMPILER_TILE_FWK_PARSER_HPP

#include <iostream>
#include <fstream>
#include "nlohmann/json.hpp"
using json = nlohmann::json;

class Task {
public:
    static std::map<int, std::shared_ptr<Task>> instances;
    int function_id; // Obtained from subGraphId
    int task_id;
    std::string type;
    std::string operation;
    int event_id;
    // Successor tasks in the original input task graph
    std::vector<int> successors;
    std::vector<int> newSuccessors;
    std::vector<int> predecessors;
    int execEnd;
    int execStart;
    int executionTime;
    // Task scheduling priority
    int priority;
    // Core ID assigned after optimized scheduling
    int coreId = -1;

    void from_json(const json &j, std::string core_type)
    {
        function_id = j.value("subGraphId", -1);
        task_id = j.value("taskId", -1);
        type = core_type;
        execEnd = j.value("execEnd", -1);
        execStart = j.value("execStart", -1);
        executionTime = execEnd - execStart;
        for (const auto &s : j.value("successors", json::array())) {
            successors.push_back(s);
        }
    }

    void buildPrimitive(int primitiveId, std::string primitiveOperation, int primitive_eventId)
    {
        task_id = primitiveId;
        operation = primitiveOperation;
        event_id = primitive_eventId;
    }
};
std::map<int, std::shared_ptr<Task>> Task::instances;

class Core {
public:
    int id;                     // Core ID
    std::string coreType;       // Core type
    std::vector<Task> tasks;    // Tasks assigned to the core
    int load = 0;               // Current load of the core (sum of task execution times)
    std::vector<int> getTaskSequence() const {
        std::vector<int> coreSequence;
        for (auto t : tasks) {
            coreSequence.push_back(t.task_id);
        }
        return coreSequence;
    }
};

class TopoInfo {
public:
    std::map<int, std::vector<int>> successorsMap; // taskId -> successors
    static std::shared_ptr<TopoInfo> load(const std::string& filename) {
        std::printf("Loading TopoInfo from %s...\n", filename.c_str());
        std::ifstream inputFile(filename);
        if (!inputFile.is_open()) {
            std::printf("Error: Could not open topo.json!\n");
            return nullptr;
        }

        json j;
        inputFile >> j;

        auto topoInfo = std::make_shared<TopoInfo>();
        for (const auto& taskJson : j) {
            int taskId = taskJson.value("taskId", -1);
            if (taskId == -1) continue;

            for (const auto& succ : taskJson.value("successors", json::array())) {
                topoInfo->successorsMap[taskId].push_back(succ);
            }
        }
        std::printf("Loaded %zu tasks from topo.json\n", topoInfo->successorsMap.size());
        return topoInfo;
    }

    // Get successors of a specific task
    const std::vector<int>& getSuccessors(int taskId) const {
        static const std::vector<int> emptyVec;
        auto it = successorsMap.find(taskId);
        return (it != successorsMap.end()) ? it->second : emptyVec;
    }
};

class CoreExecutionInfo
{
public:
    static std::vector<std::shared_ptr<CoreExecutionInfo>> instances;
    static std::map<int, std::string> coreTypes;
    int core_id;
    std::map<int, std::vector<int>> taskSequences;
    // Mapping from taskId to Task, including set/wait, initialized based on lane graph json
    std::map<int, std::shared_ptr<Task>> taskMap;

    void from_json(const json &j, const std::shared_ptr<TopoInfo>& topoInfo)
    {
        int originalDep = 0;
        int cvDep = 0;
        // ce_json contains information about a blockIdx; build taskMap; build taskSequences
        for (const auto &ce_json : j)
        {
            core_id = ce_json.value("blockIdx", -1);
            std::string core_type_value = ce_json.value("coreType", "");
            if (core_type_value.empty()) {
                std::printf("Error: coreType is missing for blockIdx %d\n", core_id);
                continue; // Skip invalid data
            }
            std::string core_type = core_type_value.substr(0, 3);
            coreTypes[core_id] = core_type;
            // For each task in a blockIdx
            for (const auto &t : ce_json.value("tasks", json::array()))
            {
                int task_id = t.value("taskId", -1);
                taskSequences[core_id].push_back(task_id);
                auto task = std::make_shared<Task>();
                task->from_json(t, core_type);
                // Get successors from topoInfo
                if (topoInfo) {
                    task->successors = topoInfo->getSuccessors(task_id);
                }
                originalDep += task->successors.size();
                taskMap[task_id] = task;
            }
        }
        // Build predecessors and calculate priority
        if (topoInfo) {
            // First ensure all tasks appearing in topo.json are included
            for (const auto& [taskId, successors] : topoInfo->successorsMap) {
                if (taskMap.count(taskId) == 0) {
                    // Create unrecorded task and set its type to "UNKNOWN"
                    auto task = std::make_shared<Task>();
                    task->task_id = taskId;
                    task->successors = successors;
                    task->type = "UNKNOWN";
                    taskMap[taskId] = task;
                }
            }
        }
        for (auto t : taskMap) {
            auto task = t.second;
            for (const auto sid : task->successors) {
                // Ensure successor task exists
                if (taskMap.count(sid) == 0) {
                    // Create unrecorded successor task
                    auto succTask = std::make_shared<Task>();
                    succTask->task_id = sid;
                    succTask->type = "UNKNOWN"; // Set default type
                    taskMap[sid] = succTask;
                }
                auto succTask = taskMap[sid];
                if (task->type != succTask->type) {
                    cvDep++;
                }
                succTask->predecessors.push_back(t.first);
            }
            task->priority = task->predecessors.size() + task->successors.size();
        }
        std::printf("Number of original dependencies: %d\n", originalDep);
        std::printf("Number of CV dependencies: %d\n", cvDep);
    }

    static int load(const std::string& profFilename, const std::string& topoFilename)
    {
        coreTypes.clear();
	    instances.clear();
        // 1. Load topo.json
        auto topoInfo = TopoInfo::load(topoFilename);
        if (!topoInfo) {
            std::printf("Error: Failed to load topo.json!\n");
            return -1;
        }

        // 2. Load tilefwk_prof_data.json
        std::printf("Loading CoreExecutionInfo from %s...\n", profFilename.c_str());
        std::ifstream inputFile(profFilename);
        if (!inputFile.is_open())
        {
            std::printf("Error: Could not open the file!\n");
            return -1;
        }

        json j;
        inputFile >> j;

        auto core_execution_info = std::make_shared<CoreExecutionInfo>();
        core_execution_info->from_json(j, topoInfo); // Pass topoInfo
        instances.push_back(core_execution_info);
        std::printf("Done! Total %lu CoreExecutionInfo loaded.\n", instances.size());
        return 0;
    }
};
std::vector<std::shared_ptr<CoreExecutionInfo>> CoreExecutionInfo::instances;
std::map<int, std::string> CoreExecutionInfo::coreTypes;
#endif // ATC_OPCOMPILER_TILE_FWK_PARSER_HPP