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
 * \file heft_alg.h
 * \brief
 */

// For Heft Graph Partition
#ifndef TILE_FWK_HEFT_H
#define TILE_FWK_HEFT_H

#include "interface/function/function.h"
#include "interface/tensor/logical_tensor.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/program/program.h"
#include "passes/pass_interface/pass.h"

namespace npu::tile_fwk {
using namespace std;
class Processor {
public:
    int computation_cost;
    int est;
    int eft;
};

class Job {
public:
    float computaion_avg; // avg. computation cost across all processors
    float rank;
    int processor_exec; // the processor on which the job was finally scheduled
    int st; // start time in final schedule
    int ft; // finish time in final schedule
    vector<Processor> processors;
    vector<int> parents; // indexes of all the parents of a particular node
    Opcode opcode; // finish time in final schedule
};

class HeftAlgPass : public Pass {
public:
    HeftAlgPass() : Pass("HeftAlgPass") {}
    ~HeftAlgPass() override {
        for (auto &ele : jobs_ptr) {
            delete ele;
        }
        for (auto &ele : processor_ptr) {
            delete ele;
        }
    }

private:
    Status RunOnFunction(Function &function) override;

    void JobConvert( Function &function);

private:

    vector<Job> jobs;

    vector<Job* > jobs_ptr;
    vector<Processor* > processor_ptr;
    int MAX_TIME = 10000000;
    int job_count; // number of jobs
    int processor_count; // number of processors
    vector<vector<int>> communication_cost_dag; // communication cost DAG in adjacency matrix form
    // vector<shared_ptr<Job>> jobs;
    // int job_count;                              // number of jobs
    // int processor_count;                        // number of processors
    // vector<vector<int>> communication_cost_dag; // communication cost DAG in adjacency matrix form
    void initializeData();
    void computationAvgCalculate();
    float rankCalculate(int node);
    vector<int> sortRank();
    int jobInsertion(
        int search_end, vector<vector<bool>> &processor_state, int job_index, int processor_index, int search_start);
    void schedule(vector<int> rank_index_sorted, Function &function);
};
}

#endif // TILE_FWK_HEFT_H