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
 * \file util.hpp
 * \brief
 */

#ifndef TILE_FWK_CALENDAR_SCHEDULING_UTIL_HPP
#define TILE_FWK_CALENDAR_SCHEDULING_UTIL_HPP

#include <vector>
#include <map>
#include <iostream>
#include <fstream>
#include "nlohmann/json.hpp"
#include "parser.hpp"

using json = nlohmann::json;

class Matrix {
private:
    std::vector<std::vector<uint64_t>> data;
    int rows;
    int cols;

public:
    Matrix(int r, int c) : rows(r), cols(c)
    {
        data.resize(rows, std::vector<uint64_t>(cols, 0));
    }

    std::vector<uint64_t> &Row(int r)
    {
        return data[r];
    }

    std::vector<uint64_t> Col(int c)
    {
        std::vector<uint64_t> col;
        for (int i = 0; i < rows; i++) {
            col.push_back(data[i][c]);
        }
        return col;
    }

    uint64_t &operator()(int row, int col)
    {
        return data[row][col];
    }

    Matrix operator+(const Matrix &other)
    {
        if (rows != other.rows || cols != other.cols) {
            throw std::invalid_argument("Matrix dimensions must match");
        }

        Matrix result(rows, cols);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result.data[i][j] = data[i][j] + other.data[i][j];
            }
        }
        return result;
    }

    Matrix operator-(const Matrix &other)
    {
        if (rows != other.rows || cols != other.cols) {
            throw std::invalid_argument("Matrix dimensions must match");
        }

        Matrix result(rows, cols);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result.data[i][j] = data[i][j] - other.data[i][j];
            }
        }
        return result;
    }

    Matrix operator*(const Matrix &other)
    {
        if (rows != other.rows || cols != other.cols) {
            throw std::invalid_argument("Matrix dimensions must match");
        }

        Matrix result(rows, cols);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result.data[i][j] = data[i][j] * other.data[i][j];
            }
        }
        return result;
    }

    Matrix operator/(const Matrix &other)
    {
        if (rows != other.rows || cols != other.cols) {
            throw std::invalid_argument("Matrix dimensions must match");
        }

        Matrix result(rows, cols);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                if (other.data[i][j] == 0) {
                    throw std::invalid_argument("Division by zero");
                }
                result.data[i][j] = data[i][j] / other.data[i][j];
            }
        }
        return result;
    }
};

inline uint64_t get_status(Matrix &conn, int dst_id, int src_id)
{
    uint64_t status = conn(dst_id, src_id);
    return status;
}

inline void set_status(Matrix &conn, int dst_id, int src_id, uint64_t val)
{
    conn(dst_id, src_id) = val;
}

struct Node {
    int id = -1;
    bool is_fake = false;

    Node() = default;
    Node(int nodeId, bool fake = false) : id(nodeId), is_fake(fake) {}
};

struct Edge {
    Node *src = nullptr;
    Node *dst = nullptr;

    Edge() = default;

    Edge(Node *src_, Node *dst_) : src(src_), dst(dst_)
    {}
};

class TaskGraph {
public:
    std::shared_ptr<CoreExecutionInfo> exec;
    std::vector<Node> input_nodes;
    int N = 0;  // 声明时初始化
    int maxTaskId = 0;  // 声明时初始化
    std::map<int, Node> nodes;
    std::vector<Edge> edges;
    std::map<int, std::set<int>> successorMap;
    std::map<int, int> taskToCoreMap;
    std::map<int, std::vector<int>> newTaskSequences;
    std::map<int, std::vector<int>> setWaitMap;
    std::map<int, int> bSetAWaitMap;
    std::map<int, int> waitAfterTaskMap;
    // taskId到自然数序列的映射
    std::map<int, int> indexMap;

    TaskGraph() : exec(CoreExecutionInfo::instances.back())  // 初始化列表
    {
        int index = 0;
        // 第一遍：添加所有节点（包括fake节点）
        for (const auto &t : exec->taskMap) {
            const std::shared_ptr<Task> task = t.second;
            // 检查是否是fake节点（类型为UNKNOWN或不在任何taskSequences中）
            bool isInSequence = false;
            for (const auto& seq : exec->taskSequences) {
                if (std::find(seq.second.begin(), seq.second.end(), task->task_id) != seq.second.end()) {
                    isInSequence = true;
                    break;
                }
            }
            bool isFake = (task->type == "UNKNOWN" || !isInSequence);
            // 添加节点
            add_node(task->task_id, isFake);
            // 维护maxTaskId
            if (task->task_id > maxTaskId) {
                maxTaskId = task->task_id;
            }
            // 为fake节点分配虚拟泳道（固定为-1）
            if (isFake) {
                taskToCoreMap[task->task_id] = -1;  // 修改为固定-1
                std::cout << "Mark as fake node: " << task->task_id << " assigned to virtual core -1\n";
            }
            indexMap[task->task_id] = index++;
        }

        // 第二遍：建立所有依赖关系（包括fake节点的）
        for (const auto &t : exec->taskMap) {
            const std::shared_ptr<Task> task = t.second;

            for (auto succ_id : task->successors) {
                if (nodes.count(task->task_id) && nodes.count(succ_id)) {
                    add_edge(task->task_id, succ_id);
                    successorMap[task->task_id].insert(succ_id);
                } else {
                    std::cout << "Warning: Missing node for edge "
                            << task->task_id << "→" << succ_id << "\n";
                }
            }
        }
        N = exec->taskMap.size();
        printf("task number: %d, maxTaskId: %d.\n", N, maxTaskId);
    }

    // 对于优化AICPU调度策略，产生新的taskSequences
    void updateTaskSequence(std::vector<std::vector<int>> allCoreTaskIds)
    {
        exec->taskSequences.clear();
        for (size_t i = 0; i < allCoreTaskIds.size(); ++i) {
            exec->taskSequences[i] = allCoreTaskIds[i];
        }
    }

    // 将原始任务图taskGraph1上添加由于划分泳道后引入的额外泳道内task依赖添加到successorMap；构建taskToCoreMap
    void addWithinCoreDependency()
    {
        // 处理真实泳道的依赖
        for (auto s: exec->taskSequences) {
            if (s.first < 0) continue;  // 跳过虚拟核心
            if (s.second.empty()) {
                continue;
            }
            for (size_t i = 0; i < s.second.size() - 1; i++)
            {
                int id = s.second[i];
                taskToCoreMap[id] = s.first;
                auto succ =  exec->taskMap[id]->successors;
                // successorMap[id].insert(succ.begin(), succ.end());
                auto _it = std::find(succ.begin(), succ.end(), s.second[i + 1]);
                if (_it == succ.end())
                {
                    successorMap[id].insert(s.second[i + 1]);
                    // printf("add_edge from %d to %d.\n", id, s.second[i + 1]);
                    add_edge(id, s.second[i + 1]);
                }
            }
            int lastId = s.second[s.second.size() - 1];
            taskToCoreMap[lastId] = s.first;
        }
    }
    // 添加泳道间的依赖set/wait（包括虚拟泳道和真实泳道之间的依赖）
    void addDependency()
    {
        printf("=== Start addDependency ===\n");
        // 赋予添加的set/wait task的taskId
        int primitiveId = maxTaskId + 1;
        int globalEventId = 0;
        int primitiveNum = 0;
        newTaskSequences = exec->taskSequences;
        // 初始化虚拟泳道的任务序列
        std::vector<int> virtualCoreTasks;
        for (const auto& node : nodes) {
            if (node.second.is_fake) {
                virtualCoreTasks.push_back(node.first);
                taskToCoreMap[node.first] = -1;  // 确保fake节点分配到虚拟核心
            }
        }
        if (!virtualCoreTasks.empty()) {
            newTaskSequences[-1] = virtualCoreTasks;
        }
        for (auto t: successorMap)
        {
            for (auto succ_id : t.second)
            {
                int srcId = t.first;
                int srcCoreId = taskToCoreMap[srcId];
                int dstCoreId = taskToCoreMap[succ_id];
                // 只有当源和目标在不同泳道时才需要添加依赖（包括虚拟泳道和真实泳道之间）
                if (srcCoreId != dstCoreId)
                {
                    // 检查是否涉及 task0
                    if (srcId == 0 || succ_id == 0) {
                        printf("WARNING: Task 0 involved in cross-core dependency!\n");
                        printf("srcId=%d (core %d), succ_id=%d (core %d)\n",
                            srcId, srcCoreId, succ_id, dstCoreId);
                    }
                    bSetAWaitMap[srcId] = succ_id;
                    primitiveNum ++;
                    int existedEventId = -1;
                    // 在src task后添加set（包括虚拟泳道中的fake节点）
                    std::vector<int> sequences = newTaskSequences[srcCoreId];
                    auto it = std::find(sequences.begin(), sequences.end(), srcId);
                    size_t index = it - sequences.begin();
                    auto tid = primitiveId;
                    // 若src task后已经添加了set，无需再添加set，其余dst task添加wait时使用和已有set同样的event id
                    if (index + 1 < sequences.size() && !exec->taskMap[sequences[index + 1]]->operation.empty() &&
                        exec->taskMap[sequences[index + 1]]->operation == "set")
                    {
                        tid = sequences[index + 1];
                        existedEventId = exec->taskMap[sequences[index + 1]]->event_id;
                    } else {
                        // 添加set
                        sequences.insert(it + 1, tid);
                        newTaskSequences[srcCoreId] = sequences;
                        auto task = std::make_shared<Task>();
                        task->buildPrimitive(tid, "set", globalEventId);
                        exec->taskMap[tid] = task;
                        taskToCoreMap[tid] = srcCoreId;
                        primitiveId++;

                    }                   
                    // 在dst task前添加wait（包括虚拟泳道中的fake节点）
                    std::vector<int> dstSequences = newTaskSequences[dstCoreId];
                    auto it2 = std::find(dstSequences.begin(), dstSequences.end(), succ_id);
                    // 添加wait
                    auto waitId = primitiveId;
                    dstSequences.insert(it2, waitId);
                    newTaskSequences[dstCoreId] = dstSequences;
                    int event_id = existedEventId >= 0 ? existedEventId : globalEventId++;
                    auto waitTask = std::make_shared<Task>();
                    waitTask->buildPrimitive(waitId, "wait", event_id);
                    exec->taskMap[waitId] = waitTask;
                    taskToCoreMap[waitId] = dstCoreId;
                    primitiveId++;
                    setWaitMap[tid].push_back(waitId);
                    waitAfterTaskMap[waitId] = succ_id;
                }
            }
        }
        printf("Number of set,wait pairs: %d\n", primitiveNum);
        printf("=== End addDependency ===\n");
    }

    void outputJson(const std::string& element)
    {
        json cores_json;
        // Output all cores (both real and virtual)
        for (auto& s : newTaskSequences) {
            json core;
            core["core_id"] = s.first;
            core["is_virtual"] = (s.first < 0); // Mark if it's a virtual core
            for (size_t i = 0; i < s.second.size(); i++) {
                int id = s.second[i];
                auto task = exec->taskMap[id];
                if (task->operation == "set") {
                    auto taskBeforeId = s.second[i - 1];
                    auto taskBefore = exec->taskMap[taskBeforeId];
                    json task_json = {
                        {"task_id", id},
                        {"operation", task->operation},
                        {"event_id", task->event_id},
                        {"time_stamp", taskBefore->execEnd},
                        {"is_fake", false} // set/wait are not fake nodes
                    };
                    core["tasks"].push_back(task_json);
                }
                else if (task->operation == "wait") {
                    auto taskAfterId = waitAfterTaskMap[id];
                    auto taskAfter = exec->taskMap[taskAfterId];
                    json task_json = {
                        {"task_id", id},
                        {"operation", task->operation},
                        {"event_id", task->event_id},
                        {"time_stamp", taskAfter->execStart},
                        {"is_fake", false} // set/wait are not fake nodes
                    };
                    core["tasks"].push_back(task_json);
                }
                else {
                    json task_json = {
                        {"task_id", id},
                        {"args", json::array({"args1", "args2"})},
                        {"function_id", task->function_id}
                    };
                    // Only mark as fake if it's on a virtual core
                    if (s.first < 0) {
                        task_json["is_fake"] = true;
                    }
                    core["tasks"].push_back(task_json);
                }
            }
            cores_json["cores"].push_back(core);
        }
        std::string filename = element + ".json";
        std::ofstream file(filename);
        if (file.is_open()) {
            file << cores_json.dump(4);
            file.close();
            std::cout << "JSON文件已成功生成！包含 "
                    << newTaskSequences.size() << " 个核心（"
                    << std::count_if(newTaskSequences.begin(), newTaskSequences.end(),
                        [](const auto& p) { return p.first < 0; })
                    << " 个虚拟核心）" << std::endl;
        } else {
            std::cerr << "无法创建输出文件！" << std::endl;
        }
    }

    void add_node(int id, bool is_fake = false)
    {
        nodes[id] = Node(id, is_fake);
    }

    void add_edge(int src, int dst)
    {
        if (nodes.find(src) != nodes.end() && nodes.find(dst) != nodes.end()) {
            edges.push_back(Edge(&nodes[src], &nodes[dst]));
        } else {
            printf("Warning: src %d or dst %d not exist!\n", src, dst);
        }
    }

    int size()
    {
        return nodes.size();
    }
};
#endif // TILE_FWK_CALENDAR_SCHEDULING_UTIL_HPP