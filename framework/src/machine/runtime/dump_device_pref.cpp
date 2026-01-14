/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file dump_device_pref.cpp
 * \brief
 */
#include "dump_device_pref.h"
#ifdef BUILD_WITH_CANN

#include "interface/machine/device/tilefwk/aicpu_pref.h"
#include "interface/utils/log.h"
#include "runtime/mem.h"
#include "interface/inner/config.h"
#include "interface/utils/file_utils.h"
#include "machine/device/dynamic/device_utils.h"
#include "interface/configs/config_manager.h"
namespace npu::tile_fwk::dynamic {
constexpr int DUMP_LEVEL_FOUR = 4;
constexpr int PART = 3;
void ContructTaskInfo(const uint32_t &blockNum, json &rootTaskStats,
                     const DeviceArgs &args, const std::vector<void *> &perfData) {
    for (uint32_t i = 0; i < blockNum; i++) {
        void* devPtr = perfData[i];
        size_t dataSize = MAX_DFX_TASK_NUM_PER_CORE * sizeof(TaskStat) + sizeof(Metrics);
        std::vector<uint8_t> hostBuffer(dataSize);
        rtMemcpy(hostBuffer.data(), dataSize, devPtr, dataSize, RT_MEMCPY_DEVICE_TO_HOST);
        Metrics *metric = reinterpret_cast<Metrics*>(hostBuffer.data());
        if (metric->taskCount > MAX_DFX_TASK_NUM_PER_CORE) {metric->taskCount = MAX_DFX_TASK_NUM_PER_CORE;} // Limit to the maximum value 
        TaskStat* taskStats = metric->tasks;
        size_t numTasks = metric->taskCount;
        std::string coreType = (i < args.nrValidAic) ? "AIC" : "AIV";
        json coreObj;
        coreObj["blockIdx"] = i;
        coreObj["coreType"] = coreType;
        json tasksArr = json::array();
        ALOG_ERROR_F("task num : %zu", numTasks);
        for (size_t j = 0; j < numTasks; ++j) {
            if (taskStats[j].execEnd != 0) {
                json taskObj;
                taskObj["seqNo"] = taskStats[j].seqNo;
                taskObj["subGraphId"] = taskStats[j].subGraphId;
                taskObj["taskId"] = taskStats[j].taskId;
                taskObj["execStart"] = taskStats[j].execStart;
                taskObj["execEnd"] = taskStats[j].execEnd;
                tasksArr.push_back(taskObj);
            }
        }
        coreObj["tasks"] = tasksArr;
        if (!tasksArr.empty()) {
            rootTaskStats.push_back(coreObj);
        }
    }
}

void DumpAicoreTaskExectInfo(DeviceArgs &args, const std::vector<void *> &perfData) {
    json root_taskStats = json::array();
    auto block_num = args.GetBlockNum();
    ALOG_INFO("GetBlockNum : %lu",  block_num);
    ContructTaskInfo(block_num, root_taskStats, args, perfData);
    std::string jsonFilePath = npu::tile_fwk::config::LogTopFolder() + "/tilefwk_L1_prof_data.json";
    std::ofstream jsonFile(jsonFilePath);
    jsonFile << root_taskStats << std::endl;
    jsonFile.close();
    ALOG_INFO("tilefwk_L1_prof_data have saved in: %s",  jsonFilePath);
    std::string topo_txt_path = npu::tile_fwk::config::LogTopFolder() + "/dyn_topo.txt";
    std::string program_json_path = npu::tile_fwk::config::LogTopFolder() + "/program.json";
    std::string draw_swim_lane_py_path = GetCurrentSharedLibPath() + "/scripts/draw_swim_lane.py";
    npu::tile_fwk::config::SetRunDataOption(KEY_SWIM_GRAPH_PATH, npu::tile_fwk::config::GetAbsoluteTopFolder() + "/merged_swimlane.json");        

    if (FileExist(program_json_path) && FileExist(topo_txt_path)) {
        ALOG_INFO("The files program.json and dyn_topo.txt exist. Start merging the swimlane.");
        std::string command = "python3 "+ draw_swim_lane_py_path + " \""
                                + jsonFilePath + "\" \""
                                + topo_txt_path + "\" \""
                                + program_json_path + "\" --label_type=1 --time_convert_denominator=50";
        if (system(command.c_str()) != 0) {
           ALOG_WARN("Failed to execute draw_swim_lane.py. Stop merging the swimlane.");
        }
    } else {
        ALOG_WARN("program.json or dyn_topo.txt missing. Stop merging the swimlane.");
    }
}

inline void DevTaskPerfFormat(uint32_t tid, uint32_t type, json &devTaskJson, const AicpuMetrPer *aicpuPer) {
    json per_dev_task;
    for (uint32_t i = 0; i < aicpuPer->perfAicpuTraceDevTaskCnt[tid][DEVTASK_PERF_ARRY_INDEX(type)]; i++) {
        if (type == PERF_TRACE_DEV_TASK_SEND_FIRST_CALLOP_TASK) {
            per_dev_task["name"] = PerfTraceName[type];
        } else {
            per_dev_task["name"] = PerfTraceName[type] + std::to_string(i);
        }
        per_dev_task["end"] = aicpuPer->perfAicpuTraceDevTask[tid][DEVTASK_PERF_ARRY_INDEX(type)][i];
        devTaskJson.push_back(per_dev_task);
    }
}

inline void SparateCore(int total, int idx, int part, const int &offset, std::vector<int> &coreArray) {
    int perCpu = total / part;
    int remain = total % part;
    int start = idx * perCpu + ((idx < remain) ? idx : remain);
    int end = start + perCpu + ((idx < remain) ? 1 : 0);
    for (int i = start; i < end; i++) {
        coreArray[i + offset] = idx;
        ALOG_ERROR_F("CoreArray[%d]: aicpu: %d", i + offset, idx);
    }
}

inline void DumpAicoreDevTask(DeviceArgs &args, json &aicpuPrefArray, const std::vector<void *> &perfData) {
    std::vector<int> coreArray;
    coreArray.resize(args.GetBlockNum());
    for(int i = 0; i < PART; i++) {
        SparateCore(args.nrAic, i, PART, 0, coreArray);
        SparateCore(args.nrAiv, i, PART, args.nrValidAic, coreArray);
    }
    for (uint32_t i = 0; i < args.GetBlockNum(); i++) {
        void* devPtr = perfData[i];
        size_t dataSize = MAX_DFX_TASK_NUM_PER_CORE * sizeof(TaskStat) + sizeof(Metrics);
        std::vector<uint8_t> hostBuffer(dataSize);
        rtMemcpy(hostBuffer.data(), dataSize, devPtr, dataSize, RT_MEMCPY_DEVICE_TO_HOST);
        Metrics *metric = reinterpret_cast<Metrics*>(hostBuffer.data());
        std::string coreType = (i < args.nrValidAic) ? "AIC" : "AIV";
        json aicoreTask;
        aicoreTask["blockIdx"] = i;
        aicoreTask["coreType"] = "SCHED" + std::to_string(coreArray[i]) + "-" + coreType;
        json tasksArr = json::array();
        uint64_t curCycle = 0;
        // ALOG_ERROR_F("=============PERF_TRACE_CORE_MAX: %d", PERF_TRACE_CORE_MAX);
        for (uint32_t type = 0; type < PERF_TRACE_CORE_MAX; type++) {
            ALOG_ERROR_F("=============Type cnt: %d", metric->perfTraceCnt[type]);
            for (uint32_t cnt = 0; cnt < metric->perfTraceCnt[type]; cnt++) {
                json aicoreTaskType;
                curCycle = metric->perfTrace[type][cnt];
                if (curCycle == 0) {
                    break;
                }
                std::string name = AicorePerfTraceName[type];
                if (metric->perfTraceDevTaskId[type][cnt] != INVALID_DEV_TASK_ID) {
                    name = name + "(" + std::to_string(metric->perfTraceDevTaskId[type][cnt]) + ")";
                }
                aicoreTaskType["name"] = name;
                aicoreTaskType["end"] = curCycle;
                tasksArr.push_back(aicoreTaskType);
            }
            metric->perfTraceCnt[type] = 0;
        }
        aicoreTask["tasks"] = tasksArr;
        aicpuPrefArray.push_back(aicoreTask);
    }
}


inline void DumpAicpuDevTask(const DeviceArgs &args, json &aicpuPrefArray) {
    for (uint32_t i = 0; i < args.nrAicpu; i++) {
        auto aicpuPer = PtrToPtr<void, AicpuMetrPer>(ValueToPtr(args.aicpuPerAddr));
        if (aicpuPer == nullptr) {
            ALOG_WARN_F("Aicpu per ptr is null");
            return;
        }
        std::vector<uint8_t> hostBuffer(sizeof(AicpuMetrPer));
        rtMemcpy(hostBuffer.data(), sizeof(AicpuMetrPer), aicpuPer, sizeof(AicpuMetrPer), RT_MEMCPY_DEVICE_TO_HOST);
        AicpuMetrPer *metric = reinterpret_cast<AicpuMetrPer*>(hostBuffer.data());

        json aicpu_per;
        std::string coreType = "\"AICPU\"";
        if (i < MAX_SCHEDULE_AICPU_NUM) {
            coreType = "\"AICPU-SCHED\"";
        } else if (i == CTRL_CPU_THREAD_IDX) {
            coreType = "\"AICPU-CTRL\"";
        }
        aicpu_per["blockIdx"] = i;
        aicpu_per["coryType"] = coreType;
        aicpu_per["freq"] = 50;
        json dev_task_arr_per = json::array();
        for (uint32_t type = 0; type < PERF_TRACE_MAX; type++) {
            if (PerfTraceIsDevTask[type]) {
                DevTaskPerfFormat(i, type, dev_task_arr_per, metric);
                continue;
            }
            if (metric->perfAicpuTrace[i][type] == 0) {
                continue;
            }
            json sch_aicpu_per;
            sch_aicpu_per["name"] = PerfTraceName[type];
            sch_aicpu_per["end"] = metric->perfAicpuTrace[i][type];
            dev_task_arr_per.push_back(sch_aicpu_per);
        }
        aicpu_per["tasks"] = dev_task_arr_per;
        aicpuPrefArray.push_back(dev_task_arr_per);
    }
}


void DumpAicpuPreInfo(DeviceArgs &args, const std::vector<void *> &perfData) {
    json aicpuPrefArray = json::array();
    DumpAicpuDevTask(args, aicpuPrefArray);
    DumpAicoreDevTask(args, aicpuPrefArray, perfData);
    std::string aicpuPerfilePath = npu::tile_fwk::config::LogTopFolder() + "/aicpu_dev_pref.json";
    aicpuPrefArray.dump(DUMP_LEVEL_FOUR);
    if (!DumpFile(aicpuPrefArray.dump(DUMP_LEVEL_FOUR), aicpuPerfilePath)) {
        ALOG_ERROR_F("Contrust custom op json failed");
        return;
    }

    ALOG_ERROR_F("+++++xxxxxxxxxx=========: %s", aicpuPerfilePath.c_str());
}
} // namespce
#endif
