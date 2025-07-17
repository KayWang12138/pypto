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
 * \file heft_alg.cpp
 * \brief
 */

#include "heft_alg.h"
#include "interface/operation/cycles.h"
#include <vector>
namespace npu::tile_fwk {
using namespace std;

// vector<Job> jobs;
// int job_count; // number of jobs
// int processor_count; // number of processors
// vector<vector<int>> communication_cost_dag; // communication cost DAG in adjacency matrix form


// calculate average computation costs of all jobs
void HeftAlgPass::computationAvgCalculate() {
    for (int i = 0; i < job_count; i++) {
        jobs[i].computaion_avg = 0;
        for (int j = 0; j < processor_count; j++) {
            jobs[i].computaion_avg += jobs[i].processors[j].computation_cost;
        }
        jobs[i].computaion_avg = jobs[i].computaion_avg / processor_count;
    }
}

// for every node, recursively find the upper rank
float HeftAlgPass::rankCalculate(int node) {
    float sub_rank = 0; // rank of the dependent task
    float temp_max = 0; // temp_max = maximum of(cost of communication between jobs + rank(i))

    for (int i = 0; i < job_count; i++) { // for every node
        if (communication_cost_dag[node][i] > 0) { // that has an incoming edge from main node
            sub_rank = rankCalculate(i); // calculate its upper rank
            if (sub_rank + communication_cost_dag[node][i] > temp_max) { // if the upper rank is max among others
                temp_max = sub_rank + communication_cost_dag[node][i]; // set this as the max
            }
        }
    }

    return temp_max + jobs[node].computaion_avg; // rank of the main node
}

// sort the jobs w.r.t upper ranks and return in a vector
vector<int> HeftAlgPass::sortRank() {
    vector<int> index(job_count);
    iota(index.begin(), index.end(), 0);
    sort(index.begin(), index.end(), [&](int a, int b) -> bool {
        return jobs[a].rank > jobs[b].rank;
    });

    return index;
}

// searches for gaps in between jobs on a processor and returns where the insertion should be
int HeftAlgPass::jobInsertion(int search_end, vector<vector<bool>> &processor_state, int job_index, int processor_index, int search_start) {
    int t = jobs[job_index].processors[processor_index].computation_cost;
    int counter = 0;
    for (int i = search_start; i < search_end; i++) {
        if (processor_state[processor_index][i] == false) {
            counter++;
        }
        else {
            counter = 0;
        }
        if (counter == t) {
            return i;
        }
    }
    return max(search_end, search_start); // if no gaps were found, resume the scheduling as normal
}

// calculate earliest start and finish times of all jobs
void HeftAlgPass::schedule(vector<int> rank_index_sorted, Function &function) {
    vector<int> processor_free(processor_count, 0); // time at which the processor will be free after its latest job
    int processor_current = -1; // same as job[ii].processor_exec, just for code simplicity
    vector<int> parent_job_delay(job_count, 0); // delay added to processor_free, accounting for finish times of all parent nodes and communication costs
    vector<vector<bool>> processor_state(processor_count, vector<bool>(MAX_TIME, false)); // defines the state of processors at each time unit, max time limit is MAX_TIME

    std::vector<Operation *> opList_ = function.Operations().DuplicatedOpList();
    // scheduling jobs
    for (int i = 0; i < job_count; i++) {
        int ii = rank_index_sorted[i]; // for making the code simpler
        std::cout << "assign job "<<ii << ", opcode " << opList_[ii]->GetOpcodeStr()<<std::endl;
        vector<vector<int>> processor_ready(processor_count); // time when the processor is ready, including communication cost but stores the data for all parent nodes

        for (int j = 0; j < processor_count; j++) { // for each processor w.r.t each job
            for (size_t k = 0; k < jobs[ii].parents.size(); k++) { // for each parent node of the job calculate ready times
                if (jobs[jobs[ii].parents[k]].processor_exec != j) // if processor different from the processor parent was executed on, add communication cost delay
                    processor_ready[j].push_back(communication_cost_dag[jobs[ii].parents[k]][ii] + jobs[jobs[ii].parents[k]].ft);
                else // if on same processor, ignore communication cost
                    processor_ready[j].push_back(jobs[jobs[ii].parents[k]].ft);
            }
            if (processor_ready[j].size() > 0) // final delay on the processor for each job, this considers everythin
                parent_job_delay[j] = *max_element(processor_ready[j].begin(), processor_ready[j].end());
            else
                parent_job_delay[j] = 0;
            jobs[ii].processors[j].est = jobInsertion(processor_free[j], processor_state, ii, j, parent_job_delay[j]);
            jobs[ii].processors[j].eft = jobs[ii].processors[j].est + jobs[ii].processors[j].computation_cost;
        }

        processor_current = min_element(jobs[ii].processors.begin(), jobs[ii].processors.end(), [](Processor &a, Processor &b) -> bool {
            return a.eft < b.eft;
        }) - jobs[ii].processors.begin(); // min_element returns reference to the min element in vector, substracting it with reference of first element gives the index of minimum element

        jobs[ii].st = jobs[ii].processors[processor_current].est;
        jobs[ii].ft = jobs[ii].processors[processor_current].eft;
        jobs[ii].processor_exec = processor_current;
        processor_free[processor_current] = jobs[ii].ft;

        for (int j = jobs[ii].st; j < jobs[ii].ft; j++) { // update the state of processor to reflect that processor was in use for the entire duration of job execution
            processor_state[processor_current][j] = true;
        }
    }
}


Status HeftAlgPass::RunOnFunction(Function &function) {
    JobConvert(function);
    cout << "No. of tasks:" << job_count << endl;
    cout << "No. of processors:" << processor_count << endl;
    cout << "\nThe upward rank values:" << endl;

    computationAvgCalculate();

    for (int i = 0; i < job_count; i++) {
        jobs[i].rank = rankCalculate(i);
        constexpr int six_precision = 6;
        cout << "Task " << (i + 1) << ": " << fixed << setprecision(six_precision) << jobs[i].rank << endl;
    }

    vector<int> rank_index_sorted = sortRank();
    cout << "\nThe order of tasks to be scheduled:" << endl;
    for (int i = 0; i < job_count; i++) {
        cout << (rank_index_sorted[i] + 1) << " ";
    }
    cout << std::endl;
    schedule(rank_index_sorted,function);

    cout << "\n\nEST and EFT on different processors" << endl;
    for (int i = 0; i < job_count; i++) {
        cout << "Task: " << (i + 1) << endl;
        for (int j = 0; j < processor_count; j++) {
            cout << "processor " << (j + 1) << "||est: " << jobs[i].processors[j].est
                 << " eft: " << jobs[i].processors[j].eft << " ||" << endl;
        }
        cout << endl;
    }

    cout << "\nFinal Schedule:" << endl;
    for (int i = 0; i < job_count; i++) {
        cout << "Task " << (i + 1) << " is executed on processor " << (jobs[i].processor_exec + 1) << " from time "
             << jobs[i].st << " to " << jobs[i].ft << endl;
    }

    int i = 0;
    for (auto &op : function.Operations()) {
        op.SetAttribute(OpAttributeKey::color, jobs[i].processor_exec);
        op.UpdateSubgraphID(jobs[i].processor_exec);
        i++;
    }

    int schedule_length; // schedule length has to be calculated since a job with higher upper rank can finish after the
                         // lowest rank job
    schedule_length =
        jobs[max_element(jobs.begin(), jobs.end(), [](Job &a, Job &b) -> bool { return a.ft < b.ft; }) - jobs.begin()]
            .ft;
    cout << "\nHence, the makespan length from the schedule: " << schedule_length << endl;
    return SUCCESS;
}


inline bool IsCopyNode(const Opcode &op) {
    std::set<Opcode> copyOpSet = {
        Opcode::OP_COPY_IN,
        Opcode::OP_L1_COPY_IN_FRACTAL_Z,
        Opcode::OP_L1_COPY_IN,
        Opcode::OP_L1_COPY_IN,
    };
    if (copyOpSet.count(op)) {
        return true;
    }
    return false;
}

int CalCommunicationCost(const Opcode &op) {
    constexpr int NormalDummyDelay = 200;
    constexpr int CopyDummyDelay = 100000;
    if (IsCopyNode(op)) {
        return CopyDummyDelay;
    }
    return NormalDummyDelay;
}

void HeftAlgPass::JobConvert( Function &function) {
    constexpr int CoreNum = 1;
    function.SetTotalSubGraphCount(CoreNum);
    std::unordered_map<int, int> magic2Idx;
    int count = 0;
    std::vector<Operation *> opList_ = function.Operations().DuplicatedOpList();
    for (auto &op : opList_) {
        int magic = op->GetOpMagic();
        magic2Idx[magic] = count;
        count += 1;
    }

    std::vector<std::vector<int>> idxInGraph(opList_.size(), std::vector<int>());
    std::vector<std::vector<int>> idxOutGraph(opList_.size(), std::vector<int>());
    for (size_t i = 0; i < opList_.size(); i++) {
        for (auto &input : opList_[i]->GetIOperands()) {
            // Get operation latency
            std::vector<std::vector<int>> shape;
            for (auto &srcTile : opList_[i]->GetOOperands()) {
                shape.emplace_back(srcTile->shape);
            }
            auto latency_ = GetCycles(opList_[i]->GetOpcodeStr(), shape, opList_[i]->GetOOperands()[0]->tensor->datatype);
            opList_[i]->UpdateLatency(latency_);

            for (auto &parentOpPtr : input->GetProducers()) {
                if (magic2Idx.find(parentOpPtr->GetOpMagic()) != magic2Idx.end()) {
                    idxInGraph[i].push_back(magic2Idx[parentOpPtr->GetOpMagic()]);
                    idxOutGraph[magic2Idx[parentOpPtr->GetOpMagic()]].push_back(i);
                }
            }
        }
    }

    job_count = opList_.size();
    processor_count = CoreNum;

    // fill the vectors with empty data
    for (int i = 0; i < job_count; i++) {
        Job *jb = new Job;
        jobs.push_back(*jb);
        jobs_ptr.push_back(jb);
        for (int j = 0; j < processor_count; j++) {
            Processor *processor = new Processor;
            processor_ptr.push_back(processor);
            jobs[i].processors.push_back(*processor);
        }
    }

    // fill the computation costs of all jobs on all processors
    for (int i = 0; i < processor_count; i++) {
        for (int j = 0; j < job_count; j++) {
            jobs[j].processors[i].computation_cost = opList_[j]->GetLatency();
        }
    }

    // fill the communications directed acyclic graph adjacency matrix
    for (int i = 0; i < job_count; i++) {
        jobs[i].opcode = opList_[i]->GetOpcode();
        vector<int> v(job_count);
        communication_cost_dag.push_back(v);
        for (int j = 0; j < job_count; j++) {
            communication_cost_dag[i][j] = 0;
            if (std::find(idxOutGraph[i].begin(), idxOutGraph[i].end(), j) != idxOutGraph[i].end()) {
                communication_cost_dag[i][j] = CalCommunicationCost(opList_[i]->GetOpcode());
            }
            if (communication_cost_dag[i][j] > 0) {
                jobs[j].parents.push_back(i);
            }
        }
    }
}

} // namespace npu::tile_fwk