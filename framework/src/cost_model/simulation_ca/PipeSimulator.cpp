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
 * \file PipeSimulator.cpp
 * \brief
 */

#include "cost_model/simulation_ca/PipeSimulator.h"

#include <algorithm>
#include <fstream>
#include <unordered_set>

#include "cost_model/simulation/config/EnvConfig.h"
#include "SimulatorAdaptor.h"
#include "cost_model/simulation/base/ModelLogger.h"
#include "cost_model/simulation_ca/A2A3/SimulatorA2A3.h"
#include "interface/function/function.h"
#include "codegen/codegen.h"
#include "codegen/codegen_op.h"
#include "codegen/codegen_cce.h"
#include "codegen/cloudnpu/codegen_cloudnpu.h"
#include "codegen/cloudnpu/codegen_op_cloudnpu.h"
#include "cost_model/simulation/arch/SimplifiedMemoryAllocator.h"
#include "cost_model/simulation/arch/PipeSimulatorFast.h"
#include "cost_model/simulation/arch/A2A3/PostSimulatorA2A3.h"
#include "interface/utils/file_utils.h"

namespace CostModel
{
    template class PipeSimulator<SimulatorA2A3>;

    static std::string GenerateBuf(const TileOpPtr &tileOp)
    {
        SimplifiedMemoryAllocator memoryAllocator;
        auto locToOffsetMap = GenRealizeIdMap(tileOp->funcPtr->parentFunction->GetParameter());
        CodeGenCtx ctx;
        CodeGenCloudNPU cga(ctx);
        cga.GenAllocForLocalBuffer(*(tileOp->operation), memoryAllocator);
        CodeGenOpCloudNPU cop(memoryAllocator, tileOp->funcPtr->parentFunction->GetFunctionType(), locToOffsetMap,
            tileOp->funcPtr->parentFunction->IsUnderDynamicFunction());
        cop.Init(*tileOp->operation);
        return cop.GenOpCode();
    }

    static bool GenerateCode(const std::string& buffer, std::string fileName)
    {
        std::ofstream os(fileName);
        if (!os.is_open()) {
            MLOG_ERROR("can't open ", fileName);
            return false;
        }

        os << "#include <iostream>" << std::endl;
        os << "#include \"mock_inst.h\"" << std::endl;
        os << "#include \"vector.h\"" << std::endl;
        os << "#include \"cube.h\"" << std::endl;
        os << "#include \"mte.h\"" << std::endl;
        os << "int main(int argc, char **argv) {" << std::endl;
        os << "char charArray1[65536] = {0};" << std::endl;
        os << "char* charArrayPtr1 = charArray1;" << std::endl;
        os << "char charArray2[65536] = {0};" << std::endl;
        os << "char* charArrayPtr2 = charArray2;" << std::endl;
        os << "char charArray3[256] = {0};" << std::endl;
        os << "char* charArrayPtr3 = charArray3;" << std::endl;
        os << "char charArray4[256] = {0};" << std::endl;
        os << "char* charArrayPtr4 = charArray4;" << std::endl;
        os << "    " << buffer;
        os << "    return 0;" << std::endl;
        os << "}" << std::endl;
        os.close();
        return true;
    }

    std::vector<std::string> RunExeAndCaptureOutput(const std::string& exePath) {
        std::vector<std::string> outputLines;
        FILE* pipe = popen(exePath.c_str(), "r");
        if (!pipe) {
            MLOG_ERROR("run error");
            return outputLines;
        }

        const int bufferSize = 4096;
        char buffer[bufferSize];
        while (fgets(buffer, bufferSize, pipe) != nullptr) {
            std::string line(buffer);
            if (!line.empty() && line[line.length() - 1] == '\n') {
                line.erase(line.length() - 1);
            }
            outputLines.push_back(line);
        }

        pclose(pipe);
        return outputLines;
    }

    static std::vector<std::string> CompileAndRunCode(const std::string &source, const EnvConfig &config)
    {
        std::string incPath = GetCurrentSharedLibPath() + "/include";
        std::string cPlusPlus = "g++";

        std::string executable = source.substr(0, source.size() - 4);
        std::string cmd = config.cPlusPlus + " -w -std=c++17 " + source + " -o " + executable +
                          " -I " + incPath + "/tileop/a2a3" +
                          " -I " + incPath + "/mock"; //  + ">/dev/null 2>&1"
        int result = std::system(cmd.c_str());
        if (result != 0) {
            MLOG_ERROR("compile error: ", cmd);
            return {};
        }
#ifdef _WIN32
        std::replace(executable.begin(), executable.end(), '/', '\\');
        const std::string exePath = executable + ".exe";
        std::vector<std::string> outputLines = RunExeAndCaptureOutput(exePath);
#else
        std::vector<std::string> outputLines = RunExeAndCaptureOutput(executable);
#endif
        return outputLines;
    }

    std::string ReplaceGMStr(const std::string &str) {
        std::regex pattern(R"(\(\(__gm__ GMTensorInfo\*\)\(param\) \+ \d+\)->Addr)");  // 正则表达式匹配目标格式
        std::string result = std::regex_replace(str, pattern, "charArray1");
        result = std::regex_replace(result, std::regex("GMStackBase"), "charArray1");
        std::regex getParamPattern(R"(GET_PARAM_ADDR\(param, \d+, \d+\))");
        result = std::regex_replace(result, getParamPattern, "charArray2");
        std::regex oriAddrPattern(R"(\(\(__gm__ GMTensorInfo\*\)\(oriAddrParam\) \+ \d+\)->Addr)");
        result = std::regex_replace(result, oriAddrPattern, "charArray3");
        std::regex runtimeCoaPattern(R"(RUNTIME_COA_GET_PARAM_OFFSET\(\d+,\d+,\d+\))");
        result = std::regex_replace(result, runtimeCoaPattern, "0");
        return result;
    }

    template <typename Simulator>
    uint64_t PipeSimulator<Simulator>::Simulate(const TileOpPtr& tileOp)
    {
        if(tileOp->opcode == "RESHAPE" || tileOp->opcode == "VIEW" || tileOp->opcode == "ASSEMBLE" || tileOp->opcode == "NOP") {
            MLOG_INFO("ignore reshape op");
            return 0;
        }
        std::string buf = GenerateBuf(tileOp);
        if (buf == "" || buf == "CODEGEN_ERROR") {
            MLOG_ERROR("can't generate buf");
            return 0;
        }
        int notSize = 3;
        if (buf.substr(0, notSize) == "NOT") {  // NOT HANDLED OP
            return 0;
        }

        buf = ReplaceGMStr(buf);

        auto it = tileopLatencyCacheMp.find(buf);
        if (it != tileopLatencyCacheMp.end()) {
            MLOG_INFO("sim: ", it->first, "latency: ", it->second, "\n");
            return it->second;
        }

        std::string simulatorDir = "simulator";
        if (!CreateDir(simulatorDir)) {
            MLOG_ERROR("can't create simulator dir");
            return 0;
        }
        auto func = tileOp->funcPtr;
        std::string fileHeader(simulatorDir + "/" + tileOp->opcode + std::to_string(tileOp->taskId) + "-" + std::to_string(func->magic) + "-" + std::to_string(tileOp->magic));
        std::string fileName(fileHeader + ".cpp");
        bool success = GenerateCode(buf, fileName);
        if (!success) {
            MLOG_ERROR("can't generate code, buf:%s", buf.c_str());
            return tileopLatencyCacheMp[buf] = 0;
        }

        EnvConfig config;
        std::vector<std::string> program = CompileAndRunCode(fileName, config);
        if (program.empty()) {
            MLOG_ERROR("can't run code, buf:", buf.c_str());
            return tileopLatencyCacheMp[buf] = 1;
        }

        SimulatorAdaptor sa;
        auto np = sa.Rewrite(program);

        int gmLatency = 0;
        Simulator sm;
        for (auto str :np) {
            if (str.find("copy_gm_to") != str.npos) {
                gmLatency = sm.rGmLatency;
            }
            if (str.find("to_gm") != str.npos) {
                gmLatency = sm.wGmLatency;
            }
        }

        tileopLatencyCacheMp[buf] = sm.Run(np) + gmLatency;
        it = tileopLatencyCacheMp.find(buf);
        MLOG_INFO("sim: ", it->first, "latency: ", it->second);
        return it->second;
    }

    template <typename Simulator>
    uint64_t PipeSimulator<Simulator>::PostSimulate(const CostModel::TileOpPtr &tileOp)
    {
        auto result = Simulate(tileOp);
        if (result == 0) {
            auto ptr = std::make_unique<PipeSimulatorFast<PostSimulatorA2A3>>();
            return ptr->PostSimulate(tileOp);
        }
        return result;
    }

    extern "C" UnifiedPipeMachinePtr CreatePipeSimulatorSimulatorA2A3() {
        return UnifiedPipeMachinePtr(new PipeSimulator<SimulatorA2A3>());
    }
} // namespace CostModel
