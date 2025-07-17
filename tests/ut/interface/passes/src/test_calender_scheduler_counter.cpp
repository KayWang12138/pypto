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
 * \file test_add_alloc_new.cpp
 * \brief Unit test for AddAlloc pass.
 */
#include <iostream>
#include <fstream>
#include "gtest/gtest.h"
#include "operation/tilefwk_op.h"
#include "interface/function/function.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/program/program.h"
#include "passes/pass_manager.h"
#include "passes/execute_graph_pass/add_alloc.h"
#include "ut_json/ut_json_tool.h"
#include "interface/configs/config_manager.h"
#include <vector>
#include "passes/execute_graph_pass/calendar_scheduler_counter.h"
#include "nlohmann/json.hpp"

using json = nlohmann::json;
using namespace npu::tile_fwk;

class CalenderScheduleCounterTest : public testing::Test {
public:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}

    void SetUp() override {
    }
    void TearDown() override {}
};

TEST_F(CalenderScheduleCounterTest, TestCounterNum71) {
    // Create log file
    std::ofstream logFile("TestGenerateJson.log");
    if (!logFile.is_open()) {
        std::cerr << "Cannot open log file." << std::endl;
    }
    // Redirect standard output to log file
    std::streambuf* originalCoutBuffer = std::cout.rdbuf();
    std::cout.rdbuf(logFile.rdbuf());

    // Set counter number
    // The value range of the counter number is [1, 72], current number is 71
    int counterNum = 71;
    std::string jsonFilePath = "./config/pass/json/llama_4_1_4096_128_0217_startTimeWeight_100.json";

    // Parses and obtains the content in the JSON file and saves the file to the struct JsonData.
    auto jsonInfo = readJson(jsonFilePath);
    JsonData data = jsonInfo.first;
    int eventNum = jsonInfo.second;

    std::vector<std::vector<TaskOperation>> allOperationInfo;

    // print json data
    ALOG_INFO_F("Number of cores: %d", data.coreNum);
    std::map<int, std::shared_ptr<Event>> eventMap;
    for (int i = 0; i <= eventNum; i++) {
        auto event0 = std::make_shared<Event>(i);
        eventMap[i] = event0;
    }

    int maxTime = 0;
    for (const auto& core : data.cores) {
        std::vector<std::tuple<std::string, int, int>> vec2;
        std::vector<TaskOperation> vec3;
        for (const auto& task : core.tasks) {
            maxTime = std::max(maxTime, task.timeStamp);
            if (task.operation == "set") {
                ALOG_INFO("This operation is set");
                TaskOperation op1;
                op1.type = OperationType::SET;
                op1.event = eventMap[task.eventId];
                op1.timeStamp = task.timeStamp;
                vec3.push_back(op1);
            } else if (task.operation == "wait") {
                ALOG_INFO("This operation is wait");
                TaskOperation op2;
                op2.type = OperationType::SET;
                op2.event = eventMap[task.eventId];
                op2.timeStamp = task.timeStamp;
                vec3.push_back(op2);
            }
        }
        allOperationInfo.push_back(vec3);
    }

    CounterAllocator allocator(counterNum);
    for (int i = 0; i < data.coreNum; i++) {
        allocator.AddSequence(allOperationInfo[i], i);
    }

    // print the input sequence
    allocator.DumpInputSequences();

    // allocate counters
    allocator.AllocateCounters();
    // print the allocated counter sequence
    allocator.PrintAllocationResults();
    auto eventCounterMaps = allocator.GetEventCounterMap();
    // create event_id -> counter_id map json
    createJson(counterNum, eventCounterMaps.first, jsonFilePath);
    auto addWaitMap = allocator.GetAddWaitMap();

    std::string output_file = "llama_4_1_4096_128_0217_startTimeWeight_100_costmodel_counter_" + std::to_string(counterNum) + ".json";
    // create json file for cost model
    addParameters(jsonFilePath, output_file, eventCounterMaps.second, counterNum, addWaitMap, true);
    auto setMap = allocator.GetSetEventIdTimeMap();
    for (auto s : setMap) {
        ALOG_INFO_F("Set event id = %d", s.first);
        ALOG_INFO_F("Set timestamp = %d", s.second);
    }

    std::cout.rdbuf(originalCoutBuffer);
    logFile.close();
}

TEST_F(CalenderScheduleCounterTest, TestCounterNum60) {
    // Create log file
    std::ofstream logFile("TestGenerateJson.log");
    if (!logFile.is_open()) {
        std::cerr << "Cannot open log file." << std::endl;
    }
    // Redirect standard output to log file
    std::streambuf* originalCoutBuffer = std::cout.rdbuf();
    std::cout.rdbuf(logFile.rdbuf());

    // Set counter number
    // The value range of the counter number is [1, 72], current number is 60
    int counterNum = 60;
    std::string jsonFilePath = "./config/pass/json/llama_4_1_4096_128_0217_startTimeWeight_100.json";

    // Parses and obtains the content in the JSON file and saves the file to the struct JsonData.
    auto jsonInfo = readJson(jsonFilePath);
    JsonData data = jsonInfo.first;
    int eventNum = jsonInfo.second;

    std::vector<std::vector<TaskOperation>> allOperationInfo;

    // print json data
    ALOG_INFO_F("Number of cores: %d", data.coreNum);
    std::map<int, std::shared_ptr<Event>> eventMap;
    for (int i = 0; i <= eventNum; i++) {
        auto event0 = std::make_shared<Event>(i);
        eventMap[i] = event0;
    }

    int maxTime = 0;
    for (const auto& core : data.cores) {
        std::vector<std::tuple<std::string, int, int>> vec2;
        std::vector<TaskOperation> vec3;
        for (const auto& task : core.tasks) {
            maxTime = std::max(maxTime, task.timeStamp);
            if (task.operation == "set") {
                ALOG_INFO("This operation is set");
                TaskOperation op1;
                op1.type = OperationType::SET;
                op1.event = eventMap[task.eventId];
                op1.timeStamp = task.timeStamp;
                vec3.push_back(op1);
            } else if (task.operation == "wait") {
                ALOG_INFO("This operation is wait");
                TaskOperation op2;
                op2.type = OperationType::SET;
                op2.event = eventMap[task.eventId];
                op2.timeStamp = task.timeStamp;
                vec3.push_back(op2);
            }
        }
        allOperationInfo.push_back(vec3);
    }

    CounterAllocator allocator(counterNum);
    for (int i = 0; i < data.coreNum; i++) {
        allocator.AddSequence(allOperationInfo[i], i);
    }

    // print the input sequence
    allocator.DumpInputSequences();

    // allocate counters
    allocator.AllocateCounters();
    // print the allocated counter sequence
    allocator.PrintAllocationResults();
    auto eventCounterMaps = allocator.GetEventCounterMap();
    // create event_id -> counter_id map json
    createJson(counterNum, eventCounterMaps.first, jsonFilePath);
    auto addWaitMap = allocator.GetAddWaitMap();

    std::string output_file = "llama_4_1_4096_128_0217_startTimeWeight_100_costmodel_counter_" + std::to_string(counterNum) + ".json";
    // create json file for cost model
    addParameters(jsonFilePath, output_file, eventCounterMaps.second, counterNum, addWaitMap, true);
    auto setMap = allocator.GetSetEventIdTimeMap();
    for (auto s : setMap) {
        ALOG_INFO_F("Set event id = %d", s.first);
        ALOG_INFO_F("Set timestamp = %d", s.second);
    }

    std::cout.rdbuf(originalCoutBuffer);
    logFile.close();
}

TEST_F(CalenderScheduleCounterTest, TestInvalidFilePath) {
    // Create log file
    std::ofstream logFile("TestGenerateJson.log");
    if (!logFile.is_open()) {
        std::cerr << "Cannot open log file." << std::endl;
    }
    // Redirect standard output to log file
    std::streambuf* originalCoutBuffer = std::cout.rdbuf();
    std::cout.rdbuf(logFile.rdbuf());

    // Set counter number
    // The value range of the counter number is [1, 72], current number is 60
    int counterNum = 60;
    std::string jsonFilePath = "./config/pass/json/llama_4_1_4096_128_0217_startTimeWeight_10.json";

    // Parses and obtains the content in the JSON file and saves the file to the struct JsonData.
    auto jsonInfo = readJson(jsonFilePath);
    JsonData data = jsonInfo.first;
    int eventNum = jsonInfo.second;

    std::vector<std::vector<TaskOperation>> allOperationInfo;

    // print json data
    ALOG_INFO_F("Number of cores: %d", data.coreNum);
    std::map<int, std::shared_ptr<Event>> eventMap;
    for (int i = 0; i <= eventNum; i++) {
        auto event0 = std::make_shared<Event>(i);
        eventMap[i] = event0;
    }

    int maxTime = 0;
    for (const auto& core : data.cores) {
        std::vector<std::tuple<std::string, int, int>> vec2;
        std::vector<TaskOperation> vec3;
        for (const auto& task : core.tasks) {
            maxTime = std::max(maxTime, task.timeStamp);
            if (task.operation == "set") {
                ALOG_INFO("This operation is set");
                TaskOperation op1;
                op1.type = OperationType::SET;
                op1.event = eventMap[task.eventId];
                op1.timeStamp = task.timeStamp;
                vec3.push_back(op1);
            } else if (task.operation == "wait") {
                ALOG_INFO("This operation is wait");
                TaskOperation op2;
                op2.type = OperationType::SET;
                op2.event = eventMap[task.eventId];
                op2.timeStamp = task.timeStamp;
                vec3.push_back(op2);
            }
        }
        allOperationInfo.push_back(vec3);
    }

    CounterAllocator allocator(counterNum);
    for (int i = 0; i < data.coreNum; i++) {
        allocator.AddSequence(allOperationInfo[i], i);
    }

    // print the input sequence
    allocator.DumpInputSequences();

    // allocate counters
    allocator.AllocateCounters();
    // print the allocated counter sequence
    allocator.PrintAllocationResults();
    auto eventCounterMaps = allocator.GetEventCounterMap();
    // create event_id -> counter_id map json
    createJson(counterNum, eventCounterMaps.first, jsonFilePath);
    auto addWaitMap = allocator.GetAddWaitMap();

    std::string output_file = "llama_4_1_4096_128_0217_startTimeWeight_100_costmodel_counter_" + std::to_string(counterNum) + ".json";
    // create json file for cost model
    addParameters(jsonFilePath, output_file, eventCounterMaps.second, counterNum, addWaitMap, true);
    auto setMap = allocator.GetSetEventIdTimeMap();
    for (auto s : setMap) {
        ALOG_INFO_F("Set event id = %d", s.first);
        ALOG_INFO_F("Set timestamp = %d", s.second);
    }

    std::cout.rdbuf(originalCoutBuffer);
    logFile.close();
}