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
 * \file test_cost_model.cpp
 * \brief
 */

#include "gtest/gtest.h"
#include <dlfcn.h>

#include "cost_model/simulation/backend.h"
#include "operator/models/llama/llama_def.h"
#include "cost_model/simulation/common/CommonType.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "cost_model/simulation/pv/PvModelFactory.h"
#include "interface/configs/config_manager.h"

using namespace npu::tile_fwk;

class CostModelTest : public testing::Test {
public:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}

    void SetUp() override {
        config::SetPlatformConfig("ENABLE_COST_MODEL", true);
        config::SetSimConfig("BUILD_TASK_BASED_TOPO", true);
        config::SetPlatformConfig(KEY_ONLY_HOST_COMPILE, true);
        Program::GetInstance().Reset();
    }

    void TearDown() override {}
};

void RunLLamaLayerCostModel(const AttentionDims &dimsCfg, float threadhold = 0.001f) {
    (void)threadhold;
    int b = dimsCfg.b;
    int n = dimsCfg.n;
    int s = dimsCfg.s;
    int d = dimsCfg.d;

    PROGRAM("LLAMALAYER") {
        Tensor H(DataType::DT_FP32, {b * s, n * d}, "H");
        Tensor AW(DataType::DT_FP16, {n * d, n * d * 3}, "AW");
        Tensor DW(DataType::DT_FP16, {n * d, n * d}, "DW");
        Tensor FW(DataType::DT_FP16, {n * d, n * d * 3}, "FW");
        Tensor Res(DT_FP32, {b * s, n * d}, "Res");
        ConfigManager::Instance();
        FUNCTION("LLAMA", FunctionType::STATIC, {H, AW, DW, FW, Res}) {
            Res = LlamaLayer(H, AW, DW, FW, dimsCfg, SMALL_DFS_VEC_CFG, DFS_CUBE_CFG);
        }
        config::SetPassStrategy("OOO");
    }
}

TEST_F(CostModelTest, TestProgramJson)
{
    CostModelAgent costModelAgent;
    costModelAgent.SubmitToCostModel(nullptr);
    costModelAgent.RunCostModel();
    costModelAgent.TerminateCostModel();
}

TEST_F(CostModelTest, TestComm)
{
    CostModelAgent costModelAgent;

    ALOG_INFO("Init CostModel Communication Simulation.");
    costModelAgent.costModel = std::make_shared<CostModel::CostModelInterface>();
    std::vector<std::string> configs;
    auto folder = config::LogTopFolder() + "/" + ("CostModelSimulationOutput");
    configs.push_back("-o");
    configs.push_back(folder);
    configs.push_back("-m");
    configs.push_back("1");
    configs.push_back("-s");
    configs.push_back("Worker.layerNum=5");
    configs.push_back("-s");
    configs.push_back("Worker.useFixedRandomSeed=true");
    costModelAgent.costModel->BuildCostModel(configs);
    costModelAgent.RunCostModel();
    costModelAgent.TerminateCostModel();
}

TEST_F(CostModelTest, llama_1_1_128_128)
{
    AttentionDims dimsCfg = {1, 1, 128, 128, DFT_SINGLE_M, DFT_SINGLE_N};
    RunLLamaLayerCostModel(dimsCfg);
}

TEST_F(CostModelTest, llama_1_1_256_128)
{
    AttentionDims dimsCfg = {1, 1, 256, 128, DFT_SINGLE_M, DFT_SINGLE_N};
    RunLLamaLayerCostModel(dimsCfg);
}

TEST_F(CostModelTest, llama_1_1_512_128)
{
    AttentionDims dimsCfg = {1, 1, 512, 128, DFT_SINGLE_M, DFT_SINGLE_N};
    RunLLamaLayerCostModel(dimsCfg);
}

TEST_F(CostModelTest, llama_1_1_1024_128)
{
    AttentionDims dimsCfg = {1, 1, 1024, 128, DFT_SINGLE_M, DFT_SINGLE_N};
    RunLLamaLayerCostModel(dimsCfg);
}

void RunAttentionPostCostModel()
{
    config::SetPlatformConfig(KEY_ONLY_HOST_COMPILE, true);
    int b = 1;
    int n = 2;
    int s = 128;
    int d = 512;
    int v_head =128;
    int h = 256;
    std::vector<int64_t> inShape = {b, n, s, d}; // (b, n, s, d)
    Tensor attnPostIn(DT_FP32, inShape, "attnPostIn");
    Tensor kvBProjWV(DT_FP32, {n, d, v_head}, "kvBProjWV");
    Tensor oProjW(DT_FP32, {n * v_head, h}, "oProjW");
    Tensor atten_output;
    ConfigManager::Instance();
    FUNCTION("AttentionPost") {
        int new_b = attnPostIn->shape[0];
        int new_n = attnPostIn->shape[1];
        int new_s = attnPostIn->shape[2];
        DataType dType = attnPostIn->Datatype();
        Program::GetInstance().GetTileShape().SetVecTileShapes({1, 1, 32, d});
        Tensor atten_res1 = Reshape(Transpose(attnPostIn, {1, 2}), {new_b * new_s, new_n, d});
        Program::GetInstance().GetTileShape().SetVecTileShapes({32, 1, d});
        Tensor atten_res2 = Transpose(atten_res1, {0, 1});
        // [n,bs,kvLoraRank] * [n, kvLoraRank, vHeadDim] = [n,bs,vHeadDim]
        Program::GetInstance().GetTileShape().SetVecTileShapes(128, 128);
        Program::GetInstance().GetTileShape().SetCubeTileShapes({32, 32}, {128, 128}, {128, 128});
        Tensor mm7_res = Matrix::BatchMatmul(dType, atten_res2, kvBProjWV);
        // Tensor mm7_res = Matrix::BatchMatmul(dType, atten_res2, kvBProjWV);
        Program::GetInstance().GetTileShape().SetVecTileShapes({1, 128, 128});
        Tensor mm7_res1 = Transpose(mm7_res, {0, 1});
        Tensor mm7_res2 = Reshape(mm7_res1, {new_b, new_s, new_n * v_head});

        // [b,s, n*vHeadDim] @ [n*vHeadDim, H] = [b,s,h]
        Tensor attn_out_w = Unsqueeze(oProjW, 0);
        atten_output = Matrix::BatchMatmul(dType, mm7_res2, attn_out_w);
    }
}

TEST_F(CostModelTest, TestGenCCESimulation)
{
    std::string soPath = "libtile_fwk_simulation_ca.so";
    void* handle = dlopen(soPath.c_str(), RTLD_LAZY);
    if (!handle) {
        std::cout << "can not load library: " << std::string(dlerror()) << std::endl;
        return;
    }

    typedef uint64_t (*GetCceInputFunc)(const std::vector<std::string>&);
    std::string Func_name = "GetCceInput";
    GetCceInputFunc GetCceInput = (GetCceInputFunc)dlsym(handle, Func_name.c_str());
    if (!GetCceInput) {
        std::cerr << "Error finding symbol: " << dlerror() << std::endl;
        dlclose(handle);
        return;
    }
    std::vector<std::string> program = {
        "copy_gm_to_ubuf float 14067728 14077728 0 1 16 0 0"
    };
    GetCceInput(program);
}

TEST_F(CostModelTest, TestAttentionPostAccuracy1)
{
    int accuracylevel = 1;
    config::SetSimConfig("ACCURACY_LEVEL", accuracylevel);
    RunAttentionPostCostModel();
}

TEST_F(CostModelTest, TestAttentionPostPvModel)
{
    config::SetSimConfig("PV_LEVEL", static_cast<int>(CostModel::PVModelLevel::PV_UT));
    RunAttentionPostCostModel();
}

TEST_F(CostModelTest, TestAttentionPostAccuracyFromJson)
{
    config::SetPlatformConfig("ENABLE_COST_MODEL", false);
    RunAttentionPostCostModel();

    std::string jPath = config::LogTopFolder() + "/program.json";
    config::SetPlatformConfig("ENABLE_COST_MODEL", true);;
    config::SetSimConfig("AGENT_JSON_PATH", jPath);
    CostModelAgent costModelAgent;
    costModelAgent.SubmitToCostModel(nullptr);
    costModelAgent.RunCostModel();
    costModelAgent.TerminateCostModel();
}

TEST_F(CostModelTest, TestGenCalendarSchedule)
{
    int accuracylevel = 1;
    config::SetSimConfig("ACCURACY_LEVEL", accuracylevel);
    std::vector<std::string> arg = config::GetSimConfig("args", arg);
    arg.emplace_back("Model.genCalendarScheduleCpp=true");
    config::SetSimConfig("args", arg);
    RunAttentionPostCostModel();
}

TEST_F(CostModelTest, TestGenCalendarScheduleFronJson)
{
    std::string jsonPath("./config/fixed_task_topo_2.json");
    std::vector<std::string> arg = config::GetSimConfig("args", arg);
    arg.emplace_back("Model.simulationFixedLatencyTask=true");
    arg.emplace_back("Model.fixedLatencyTaskInfoPath=" + jsonPath);
    arg.emplace_back("Model.genCalendarScheduleCpp=true");
    config::SetSimConfig("args", arg);

    CostModelAgent costModelAgent;
    costModelAgent.SubmitToCostModel(nullptr);
    costModelAgent.RunCostModel();
    costModelAgent.TerminateCostModel();
}

TEST_F(CostModelTest, TestAttentionPostCVMIXMode)
{
    int accuracylevel = 1;
    config::SetSimConfig("ACCURACY_LEVEL", accuracylevel);
    std::vector<std::string> arg = config::GetSimConfig("args", arg);
    arg.emplace_back("Model.cubeVecMixMode=true");
    config::SetSimConfig("args", arg);
    RunAttentionPostCostModel();
}

TEST_F(CostModelTest, TestAttentionPostSimulationSchedule)
{
    int accuracylevel = 1;
    config::SetSimConfig("ACCURACY_LEVEL", accuracylevel);
    std::vector<std::string> arg;
    arg.emplace_back("Model.statisticReportToFile=true");
    arg.emplace_back("Model.deviceArch=A2A3");
    arg.emplace_back("Model.useOOOPassSeq=false");
    config::SetSimConfig("args", arg);
    RunAttentionPostCostModel();
}

TEST_F(CostModelTest, TestAttentionPostFunctional)
{
    int accuracylevel = 1;
    config::SetSimConfig("SIM_MODE", int(CostModel::SimMode::EMULATOR));
    config::SetSimConfig("ACCURACY_LEVEL", accuracylevel);
    RunAttentionPostCostModel();
}

TEST_F(CostModelTest, TestErrorInput)
{
    std::string name = "TEST";
    auto newFunc = std::make_shared<Function>(npu::tile_fwk::Program::GetInstance(), name, name, nullptr);
    std::vector<int64_t> shape = {1, 1};
    auto outcast = std::make_shared<LogicalTensor>(*newFunc, DT_FP32, shape);
    newFunc->outCasts_.push_back(outcast);
    newFunc->inCasts_.push_back(outcast);
    CostModelAgent costModelAgent;
    costModelAgent.SubmitToCostModel(newFunc.get());
}

TEST_F(CostModelTest, TestFixedLatencyTasks)
{
    std::string jsonPath("./config/fixed_task_topo.json");
    std::vector<std::string> arg = config::GetSimConfig("args", arg);
    arg.emplace_back("Model.simulationFixedLatencyTask=true");
    arg.emplace_back("Model.fixedLatencyTaskInfoPath=" + jsonPath);
    config::SetSimConfig("args", arg);

    CostModelAgent costModelAgent;
    costModelAgent.SubmitToCostModel(nullptr);
    costModelAgent.RunCostModel();
    costModelAgent.TerminateCostModel();
}

TEST_F(CostModelTest, TestSimulationOnly)
{
    CostModelAgent costModelAgent;
    costModelAgent.SubmitToCostModel(nullptr);
    costModelAgent.RunCostModel();
    costModelAgent.TerminateCostModel();
}

TEST_F(CostModelTest, TestAttentionPostAccuracy2)
{
    int accuracylevel = 2;
    config::SetSimConfig("ACCURACY_LEVEL", accuracylevel);
    RunAttentionPostCostModel();
}

TEST_F(CostModelTest, TestAttentionPostL2Cache)
{
    int accuracylevel = 1;
    config::SetSimConfig("ACCURACY_LEVEL", accuracylevel);
    std::vector<std::string> arg;
    arg.emplace_back("Model.statisticReportToFile=false");
    arg.emplace_back("Model.deviceArch=A2A3");
    arg.emplace_back("Model.mteUseL2Cache=true");
    config::SetSimConfig("args", arg);
    RunAttentionPostCostModel();
}

TEST_F(CostModelTest, TestBuildBasedOnConfigs)
{
    ALOG_INFO("Init CostModel Communication Simulation.");
    std::string configPath("./config/test_config.conf");
    std::vector<std::string> configs;
    configs.push_back("--conf");
    configs.push_back(configPath);

    CostModelAgent costModelAgent;
    costModelAgent.costModel = std::make_shared<CostModel::CostModelInterface>();
    costModelAgent.costModel->BuildCostModel(configs);

    configs.clear();
    configs.push_back("-m");
    configs.push_back("1");
    configs.push_back("--conf");
    configs.push_back(configPath);
    costModelAgent.costModel = std::make_shared<CostModel::CostModelInterface>();
    costModelAgent.costModel->BuildCostModel(configs);
}

TEST_F(CostModelTest, TestCoreMachineDeadlock)
{
    int accuracylevel = 1;
    config::SetSimConfig("ACCURACY_LEVEL", accuracylevel);
    std::vector<std::string> arg = config::GetSimConfig("args", arg);
    arg.emplace_back("Model.testDeadLock=true");
    arg.emplace_back("Core.bufferBackPressure=true");
    arg.emplace_back("Pipe.ubSizeThreshold=256");
    arg.emplace_back("Pipe.l1SizeThreshold=256");
    arg.emplace_back("Pipe.l0aSizeThreshold=256");
    arg.emplace_back("Pipe.l0bSizeThreshold=256");
    arg.emplace_back("Pipe.l0cSizeThreshold=256");
    config::SetSimConfig("args", arg);
    RunAttentionPostCostModel();
}

TEST_F(CostModelTest, TestAttentionPostBf16Real) {
    config::SetPlatformConfig(KEY_ONLY_HOST_COMPILE, true);
    int b = 32;
    int n = 32;
    int s = 1;
    int d = 512;
    int v_head =128;
    int h = 7168;

    int tile16 = 16;
    int tile128 = 128;
    int tile8 = 8;
    int tile1024 = 1024;

    std::vector<int64_t> inShape = {b, n, s, d}; // (b, n, s, d)
    Tensor attnPostIn(DT_BF16, inShape, "attnPostIn");
    Tensor kvBProjWV(DT_BF16, {n, d, v_head}, "kvBProjWV");
    Tensor oProjW(DT_BF16, {n * v_head, h}, "oProjW");
    Tensor atten_output;
    ConfigManager::Instance();
    FUNCTION("AttentionPost") {
        DataType dType = attnPostIn->Datatype();
        Program::GetInstance().GetTileShape().SetVecTileShapes({2, n, 1, d});
        Tensor atten_res0 = Transpose(attnPostIn, {1, 2});
        Program::GetInstance().GetTileShape().SetVecTileShapes({4, 1, tile16, tile128});
        Tensor atten_res1 = Reshape(atten_res0, {b * s, n, d});
        Program::GetInstance().GetTileShape().SetVecTileShapes({tile8, tile8, d});
        Tensor t2_res = Transpose(atten_res1, {0, 1});

        Program::GetInstance().GetTileShape().SetCubeTileShapes({tile16, tile16}, {tile128, tile128}, {tile128, tile128});  // M 16对齐
        // [n,bs,kvLoraRank] * [n, kvLoraRank, vHeadDim] = [n,bs,vHeadDim]
        Tensor bmm4_res = Matrix::BatchMatmul(dType, t2_res, kvBProjWV);

        Program::GetInstance().GetTileShape().SetVecTileShapes(tile16, tile16, v_head); // 必须切，但是尾轴不能切
        Tensor t3_res = Transpose(bmm4_res, {0, 1}); // [bs,n,v_head]

        Program::GetInstance().GetTileShape().SetVecTileShapes({tile16, tile16, v_head});
        Tensor r2_res = Reshape(t3_res, {b * s, n*v_head});

        // [b,s, n*v_head] @ [n*v_head, h] = [b,s,h]
        Program::GetInstance().GetTileShape().SetCubeTileShapes({tile16, tile16}, {tile128, tile128}, {tile128, tile128});
        Tensor bmm5_res = Matrix::Matmul<false, false>(dType, r2_res, oProjW);

        Program::GetInstance().GetTileShape().SetVecTileShapes({b, tile1024});
        atten_output = Reshape(bmm5_res, {b, s, h});
    }
    std::cout << Program::GetInstance().Dump() << std::endl;
}

void RunConcat()
{
    Program::GetInstance().GetTileShape().SetVecTileShapes(16, 6, 6, 6);
    config::SetPlatformConfig(KEY_ONLY_HOST_COMPILE, true);

    std::vector<int64_t> shape1 = {10, 10, 10, 10};
    std::vector<int64_t> shape2 = {20, 10, 10, 10};
    int axis = 0;
    Tensor params1(DT_FP32, shape1, "params1");
    Tensor params2(DT_FP32, shape2, "params2");
    Tensor res;

    FUNCTION("A") {
        res = Concat(std::vector<Tensor>{params1, params2}, axis);
    }
}

TEST_F(CostModelTest, TestGlobalCalendar)
{
    std::string jsonPath("./config/global.calendar.json");
    std::string inputPath("./config/fixed_task_topo.json");
    CostModel::CalendarMode calendarMode = CostModel::CalendarMode::GLOBAL_COUNTER;
    std::vector<std::string> arg;
    arg.emplace_back("Model.simulationFixedLatencyTask=true");
    arg.emplace_back("Model.fixedLatencyTaskInfoPath=" + inputPath);
    arg.emplace_back("Model.calendarFile=" + jsonPath);
    arg.emplace_back("Model.calendarMode=" +  std::to_string(static_cast<int>(calendarMode)));
    config::SetSimConfig("args", arg);
    RunConcat();
}

TEST_F(CostModelTest, TestReplayDispatch)
{
    std::string jsonPath("./config/_swim.json");
    std::string agentPath("./config/_program.json");
    config::SetSimConfig("ACCURACY_LEVEL", 1);
    config::SetSimConfig("AGENT_JSON_PATH", agentPath);
    std::vector<std::string> arg;
    arg.emplace_back("Model.deviceArch=A2A3");
    arg.emplace_back("Model.statisticReportToFile=false");
    arg.emplace_back("Model.replayDispatchMode=1");
    arg.emplace_back("Model.replayFile=" + jsonPath);
    config::SetSimConfig("args", arg);
    RunAttentionPostCostModel();
}

TEST_F(CostModelTest, TestReplayAll)
{
    std::string jsonPath("./config/_swim.json");
    std::string agentPath("./config/_program.json");
    config::SetSimConfig("ACCURACY_LEVEL", 1);
    config::SetSimConfig("AGENT_JSON_PATH", agentPath);
    std::vector<std::string> arg;
    arg.emplace_back("Model.deviceArch=A2A3");
    arg.emplace_back("Model.statisticReportToFile=false");
    arg.emplace_back("Model.replayAllMode=1");
    arg.emplace_back("Model.replayFile=" + jsonPath);
    config::SetSimConfig("args", arg);
    RunAttentionPostCostModel();
}

TEST_F(CostModelTest, TestDynamic)
{
    std::string jsonPath("./config/_dynamic.json");
    config::SetSimConfig("BUILD_TASK_BASED_TOPO", false);
    config::SetSimConfig("JSON_PATH", jsonPath);
    config::SetSimConfig("EXECUTE_CYCLE_THRESHOLD", 200000);
    config::SetSimConfig("ACCURACY_LEVEL", 1);
    CostModelAgent costModelAgent;
    costModelAgent.SubmitToCostModel(nullptr);
    costModelAgent.RunCostModel();
    costModelAgent.TerminateCostModel();
}

TEST_F(CostModelTest, TestLeafFunctionMode)
{
    config::SetSimConfig("SIM_MODE", int(CostModel::SimMode::LEAF_FUNCTION));
    config::SetSimConfig("ACCURACY_LEVEL", 1);
    std::vector<std::string> arg;
    arg.emplace_back("Model.deviceArch=A2A3");
    arg.emplace_back("Model.statisticReportToFile=false");
    config::SetSimConfig("args", arg);
    RunAttentionPostCostModel();
}


class CostModelDynTest : public testing::Test {
public:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}

    void SetUp() override {
        cacheEnable = config::GetHostConfig(KEY_ENABLE_BINARY_CACHE, false);
        config::SetHostConfig(KEY_ENABLE_BINARY_CACHE, false);
        oriEnableAihacBackend = config::GetPlatformConfig(KEY_ENABLE_AIHAC_BACKEND, oriEnableAihacBackend);
        config::SetPlatformConfig(KEY_ENABLE_AIHAC_BACKEND, true);
        Program::GetInstance().Reset();
        constexpr int level = 2;
        EnablePVModel(level);
    }

    void TearDown() override {
        config::SetHostConfig(KEY_ENABLE_BINARY_CACHE, cacheEnable);
        config::SetPlatformConfig(KEY_ENABLE_AIHAC_BACKEND, oriEnableAihacBackend);
        ResetPVModelConfig();
    }

    void EnablePVModel(int level)
    {
        oriPvLevel = config::GetSimConfig("PV_LEVEL", 0);
        config::SetSimConfig("PV_LEVEL", level);
    }

    void ResetPVModelConfig()
    {
        config::SetSimConfig("PV_LEVEL", oriPvLevel);
    }

protected:
    bool oriEnableAihacBackend = false;
    int oriPvLevel = 0;
    bool cacheEnable = false;
};

void CostModelTestLoopViewAssemble(const Tensor &t0, const Tensor &t1, const Tensor &blockTable, Tensor &out, int s) {
    FUNCTION("main", FunctionType::DYNAMIC, {t0, t1, blockTable}, {out}) {
        LOOP("L0", FunctionType::DYNAMIC_LOOP, i, LoopRange(GetInputShapeDim(t0, 0) / s)) {
            SymbolicScalar idx = GetInputDataInt32Dim2(blockTable, i, 0);
            Tensor t0s = View(t0, {s, s}, {idx * s, 0});

            Tensor qi(DT_FP32, {s, 2*s}, "qi");
            Assemble(t1, {0, 0}, qi);
            Assemble(t0s, {0, s}, qi);

            Tensor ki(DT_FP32, {s, 2*s}, "ki");
            Assemble(t0s, {0, 0}, ki);
            Assemble(t1, {0, s}, ki);

            Tensor t2 = Matrix::Matmul<false, true>(DataType::DT_FP32, qi, ki);
            // conat((t0s + t1, t1)) @ concat (t0s, t1)^T
            Assemble(t2, {idx * s, 0}, out);
        }
    }
}

TEST_F(CostModelDynTest, TestDD) {
    config::SetHostConfig(KEY_ONLY_CODEGEN, true);
    constexpr int tilingX = 32;
    constexpr int tilingY = 32;
    Program::GetInstance().GetTileShape().SetVecTileShapes(tilingX, tilingY);
    constexpr int tilingM = 32;
    constexpr int tilingN = 32;
    constexpr int tilingK = 32;
    Program::GetInstance().GetTileShape().SetCubeTileShapes({tilingM, tilingM}, {tilingN, tilingN}, {tilingK, tilingK});

    std::vector<uint8_t> devProgBinary;

    int s = 32;
    int n = 8;
    Tensor t0(DT_FP32, {n * s, s}, "t0");  // [32*8, 32]
    Tensor t1(DT_FP32, {s, s}, "t1");  // [32, 32]
    Tensor blockTable{
        DT_INT32, {n, 1},
         "blockTable"
    };
    Tensor out(DT_FP32, {n * s, s}, "out");
    CostModelTestLoopViewAssemble(t0, t1, blockTable, out, s);

    auto func = Program::GetInstance().GetLastFunction();
    auto pv = CostModel::PvModelFactory::CreateDyn();
    pv->Codegen(func);
}
