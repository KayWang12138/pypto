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
 * \file calendar_scheduler.cpp
 * \brief
 */

#include <iostream>
#include <fstream>
#include <cstdlib>
#include "nlohmann/json.hpp"
#include <string>
#include <vector>
#include <climits>
#include <queue>
#include <algorithm>
#include <chrono>
#include <set>
#include "util.hpp"
#include "parser.hpp"
#include "calendar_scheduler.h"

using json = nlohmann::json;
namespace {
    // Constants
    constexpr int DISCONNECT = 0;
    constexpr int CONNECTED = 1;
    constexpr double DEFAULT_START_TIME_WEIGHT = 100.0;
    const std::string DEFAULT_SWIM_FILE = "tests/ut/interface/passes/src/tilefwk_prof_data.json";
    const std::string DEFAULT_TOPO_FILE = "tests/ut/interface/passes/src/topo.json";

    struct ProgramOptions {
        bool reassignTasks;
        double startTimeWeight;
        std::string swimPath;
        std::string topoPath;
    };

    struct BestCoreInfo {
        int bestCoreId;
        int minStartTime;
        int minCrossCoreDependencies;
    };

    ProgramOptions parseCommandLineArguments(const std::vector<std::string>& argv) {
        ProgramOptions options;
        options.reassignTasks = false;
        options.startTimeWeight = DEFAULT_START_TIME_WEIGHT;
        options.swimPath = DEFAULT_SWIM_FILE;
        options.topoPath = DEFAULT_TOPO_FILE;

        for (size_t i = 1; i < argv.size(); ++i) {
            const std::string& arg = argv[i];
            if (arg == "--reassign") {
                options.reassignTasks = true;
            } else if (arg == "--weight" && i + 1 < argv.size()) {
                options.startTimeWeight = std::stod(argv[++i]);
            } else if (arg == "--swim" && i + 1 < argv.size()) {
                options.swimPath = argv[++i];
            } else if (arg == "--topo" && i + 1 < argv.size()) {
                options.topoPath = argv[++i];
            }
        }
        return options;
    }

    void updateSrcConnection(Matrix& conn, int srcIdx, int dstId, TaskGraph* graph) {
        set_status(conn, dstId, srcIdx, CONNECTED);
        auto& prevLinks = conn.Row(srcIdx);
        auto& newLinks = conn.Row(dstId);

        for (size_t intId = 0; intId < prevLinks.size(); intId++) {
            uint64_t indirectStatus = prevLinks[intId];
            uint64_t& directStatus = newLinks[intId];

            if (indirectStatus >= CONNECTED) {
                if (directStatus == CONNECTED) {
                    directStatus = indirectStatus + 1;
                    graph->successorMap[intId].erase(dstId);
                } else if (directStatus == DISCONNECT) {
                    directStatus = indirectStatus + 1;
                }
            }
        }
    }

    void deleteRedundantDependency(TaskGraph* graph) {
        const int matrixLen = graph->size();
        Matrix connection(graph->size(), matrixLen);

        auto it = graph->edges.begin();
        while (it != graph->edges.end()) {
            auto edge = *it;
            int srcId = edge.src->id;
            int srcIndex = graph->indexMap[srcId];
            int dstId = edge.dst->id;
            int dstIndex = graph->indexMap[dstId];

            uint64_t connStatus = get_status(connection, dstIndex, srcIndex);
            if (connStatus <= CONNECTED) {
                updateSrcConnection(connection, srcIndex, dstIndex, graph);
            } else {
                graph->successorMap[srcId].erase(dstId);
            }
            it++;
        }
    }

    int calculateCrossCoreDependencies(const Task& task, const std::map<int, std::shared_ptr<Task>>& oriTaskMap) {
        int crossCoreDependencies = 0;

        for (int pred : task.predecessors) {
            const auto& predTask = oriTaskMap.at(pred);
            if (predTask->coreId != -1 && task.coreId != predTask->coreId) {
                crossCoreDependencies++;
            }
        }

        for (int succ : task.successors) {
            const auto& succTask = oriTaskMap.at(succ);
            if (succTask->coreId != -1 && task.coreId != succTask->coreId) {
                crossCoreDependencies++;
            }
        }

        return crossCoreDependencies;
    }

    BestCoreInfo findBestCoreForTask(const Task& task, const std::vector<Core>& cores,
                                   const std::map<int, std::shared_ptr<Task>>& oriTaskMap,
                                   double startTimeWeight) {
        BestCoreInfo result{-1, INT_MAX, INT_MAX};

        for (const Core& core : cores) {
            if (core.coreType != task.type) continue;

            int coreAvailableTime = core.tasks.empty() ? 0 : core.tasks.back().execEnd;
            int maxPredecessorEndTime = 0;
            for (int predecessorId : task.predecessors) {
                maxPredecessorEndTime = std::max(maxPredecessorEndTime,
                                               oriTaskMap.at(predecessorId)->execEnd);
            }

            int taskStartTime = std::max(coreAvailableTime, maxPredecessorEndTime);
            Task tempTask = task;
            tempTask.coreId = core.id;
            int crossCoreDeps = calculateCrossCoreDependencies(tempTask, oriTaskMap);

            double score = startTimeWeight * taskStartTime +
                         (1 - startTimeWeight) * crossCoreDeps;
            double currentMinScore = startTimeWeight * result.minStartTime +
                                   (1 - startTimeWeight) * result.minCrossCoreDependencies;

            if (score < currentMinScore) {
                result = {core.id, taskStartTime, crossCoreDeps};
            }
        }

        return result;
    }

    std::vector<int> topologicalSort(const std::map<int, std::shared_ptr<Task>>& oriTaskMap) {
        std::unordered_map<int, int> inDegree;
        std::queue<int> q;
        std::vector<int> result;

        for (const auto& [id, task] : oriTaskMap) {
            inDegree[id] = task->predecessors.size();
            if (inDegree[id] == 0) {
                q.push(id);
            }
        }

        while (!q.empty()) {
            int currentTask = q.front();
            q.pop();
            result.push_back(currentTask);

            for (int succ : oriTaskMap.at(currentTask)->successors) {
                if (--inDegree[succ] == 0) {
                    q.push(succ);
                }
            }
        }

        return result;
    }

    void processTaskGraph(std::shared_ptr<TaskGraph> taskGraph, const std::vector<std::vector<int>>& allCoreTaskIds, const ProgramOptions& options) {
        taskGraph->updateTaskSequence(allCoreTaskIds);
        taskGraph->addWithinCoreDependency();
        deleteRedundantDependency(taskGraph.get());
        taskGraph->addDependency();

        std::stringstream filename;
        if (options.reassignTasks) {
            filename << "startTimeWeight_" << options.startTimeWeight;
        } else {
            filename << "original";
        }
        taskGraph->outputJson(filename.str());
    }

    std::vector<std::vector<int>> assignTasksToCores(std::map<int, std::shared_ptr<Task>> oriTaskMap, std::vector<Core>& cores, double startTimeWeigh) {
        std::vector<int> results = topologicalSort(oriTaskMap);

        int aicTaskNum = 0;
        int aivTaskNum = 0;
        std::unordered_map<int, int> core_timelines;
        for (int taskId : results) {
            auto task = oriTaskMap[taskId];
            // Skip tasks that are not in tilefwk_prof_data.json
            if (task->type.empty() || task->type == "UNKNOWN") {
                std::printf("Skip task %d (type: %s) as it is not in tilefwk_prof_data.json.\n", taskId, task->type.c_str());
                continue;
            }
            BestCoreInfo info = findBestCoreForTask(*task, cores, oriTaskMap, startTimeWeigh);
            int bestCoreId = info.bestCoreId;
            int minStartTime = info.minStartTime;
            if (bestCoreId != -1) {
                int end_time = minStartTime + task->executionTime;
                task->coreId = bestCoreId;
                task->execStart = minStartTime;
                task->execEnd = end_time;
                core_timelines[bestCoreId] = end_time;
                cores[bestCoreId].tasks.push_back(*task);
            } else {
                std::cout << "Error: No suitable core found for task " <<  taskId << std::endl;
            }
            if (task->type == "AIC") {
                aicTaskNum++;
            } else {
                aivTaskNum++;
            }
        }
        std::printf("AIC task number %d and AIV task number %d.\n", aicTaskNum, aivTaskNum);

        int end_to_end_latency = 0;
        for (const auto& entry : core_timelines) {
            end_to_end_latency = std::max(end_to_end_latency, entry.second);
        }
        std::printf("end_to_end_latency estimatation %d with task size %lu.\n", end_to_end_latency, oriTaskMap.size());
        std::vector<std::vector<int>> allCoreTaskIds;
        for (Core& core : cores) {
            std::vector<Task> coreTasks = core.tasks;
            std::vector<int> coreTaskIds;
            for (Task& task : coreTasks) {
                coreTaskIds.push_back(task.task_id);
            }
            allCoreTaskIds.push_back(coreTaskIds);
        }
        return allCoreTaskIds;
    }

    int runCalendarSchedulerImpl(const std::vector<std::string>& args) {
        // Parse command line arguments
        ProgramOptions options = parseCommandLineArguments(args);
        std::cout << "Attempting to load file from: " << options.topoPath << std::endl;
        CoreExecutionInfo::load(options.swimPath, options.topoPath);

        auto taskGraph = std::make_shared<TaskGraph>();
        std::vector<std::vector<int>> allCoreTaskIds;

        if (options.reassignTasks) {
            std::vector<Core> cores;
            for (const auto& [coreId, coreType] : CoreExecutionInfo::coreTypes) {
                cores.push_back({coreId, coreType, {}});
            }
            allCoreTaskIds = assignTasksToCores(taskGraph->exec->taskMap, cores, options.startTimeWeight);
        } else {
            const auto& taskSequences = taskGraph->exec->taskSequences;
            allCoreTaskIds.resize(taskSequences.empty() ? 0 : taskSequences.rbegin()->first + 1);

            for (const auto& [coreId, taskIds] : taskSequences) {
                allCoreTaskIds[coreId] = taskIds;
            }
        }

        processTaskGraph(taskGraph, allCoreTaskIds, options);
        return 0;
    }

} // namespace
int runCalendarScheduler(const std::vector<std::string>& args) {
    return runCalendarSchedulerImpl(args);
}
