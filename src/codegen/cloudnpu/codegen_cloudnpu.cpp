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
 * \file codegen.cpp
 * \brief
 */

#include "codegen_op_cloudnpu.h"
#include "interface/utils/log.h"
#include "codegen/parallel_execute.h"
#include "interface/utils/file_utils.h"
#include "interface/tensor/logical_tensor.h"
#include "interface/function/function.h"
#include "interface/configs/config_manager.h"
#include "securec.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/program/program.h"
#include "codegen_vf.h"
#include "codegen_cloudnpu.h"

#include <cstring>

#include <nlohmann/json.hpp>

namespace npu::tile_fwk {
#ifdef SRCPATH
constexpr const char *SRC_PATH = SRCPATH;
#else
constexpr const char *SRC_PATH = ".";
#endif

using SubstMap = std::map<std::string, std::string>;
std::string StringSubstitute(std::string const &in, SubstMap const &subst) {
    const char *tokenHead = "${";
    const char *tokenTail = "}$";
    constexpr size_t tokenSepLen = 2;

    std::ostringstream out;
    size_t pos = 0;
    for (;;) {
        size_t substPos = in.find(tokenHead, pos);
        size_t endPos = in.find(tokenTail, substPos);
        if (endPos == std::string::npos) {
            break;
        }

        out.write(&*in.begin() + pos, substPos - pos);

        substPos += tokenSepLen;
        auto substIter = subst.find(in.substr(substPos, endPos - substPos));
        if (substIter == subst.end()) {
            throw std::runtime_error("undefined substitution");
        }

        out << substIter->second;
        pos = endPos + tokenSepLen;
    }
    out << in.substr(pos, std::string::npos);
    return out.str();
}

bool CompareStrings(const std::string &s1, const std::string &s2) {
    std::string str1 = s1;
    std::string str2 = s2;
    transform(str1.begin(), str1.end(), str1.begin(), ::tolower);
    transform(str2.begin(), str2.end(), str2.begin(), ::tolower);

    return str1 < str2;
}

bool CodeGenCloudNPU::IsCube(const OperationsViewer &operationList) const {
    auto isL1CopyIn = [](const Operation &op) {
        return op.GetOpcode() == Opcode::OP_COPY_IN && !(op.oOperand.empty()) &&
               op.oOperand[0]->GetMemoryTypeOriginal() == npu::tile_fwk::MemoryType::MEM_L1;
    };

    for (const auto &oper : operationList) {
        if (isL1CopyIn(oper)) {
            return true;
        }
    }

    return false;
}

void PrintOperand(const std::string &operIO, std::shared_ptr<LogicalTensor> operand) {
    ALOG_INFO_F("insert %s magic: %d, tensor: %s, memory map is: ", operIO.c_str(), operand->GetMagic(),
        operand->Dump().c_str());
    for (auto kv : operand->memorymap) {
        ALOG_INFO_F(
            "subgraph id: %d, range is [%d, %d, %d]\n", kv.first, kv.second.start, kv.second.end, kv.second.memId);
    }
}

bool HasAllocAttr(const std::shared_ptr<LogicalTensor> &tensor) {
    bool needAlloc = false;
    tensor->GetAttr(OpAttributeKey::needAlloc, needAlloc);
    return needAlloc;
}

std::string CodeGenCloudNPU::GenCodeImpl(Function &subFunc, Function &topFunc) {
    OperationsViewer operationList = subFunc.Operations();
    if (operationList.IsEmpty()) {
        ALOG_ERROR("operationList is empty");
        return {};
    }

    ALOG_INFO_F("Function to codegen:\n %s\n", topFunc.Dump().c_str());

    SymbolManager memAlloc;
    std::string allocSourceRegion;
    std::string tileOpSourceRegion;
    auto locToOffsetMap = GenRealizeIdMap(subFunc.GetParameter());

    for (const auto &op : operationList) {
        ALOG_INFO_F(
            "======================== Op CodeGenNPU Start ========================\nGen OP IS: %s", op.Dump().c_str());
        Opcode opcode = op.GetOpcode();
        if (SKIP_OPCODE.find(opcode) != SKIP_OPCODE.end()) {
            ALOG_INFO_F("ignore this op\n------------------------ Op CodeGenNPU Finish -----------------------");
            continue;
        }

        std::string allocSourceCode = GenAllocForLocalBuffer(op, memAlloc);

        CodeGenOpCloudNPU cop(memAlloc, topFunc.GetFunctionType(), locToOffsetMap, topFunc.IsUnderDynamicFunction());
        auto success = cop.Init(op);
        if (!success) {
            ALOG_INFO_F(": failed to init CodeGenOpCloudNPU from an operation: %s \n", op.Dump().c_str());
            break;
        }

        std::string tileOpSourceCode = cop.GenOpCode();
        ASSERT(tileOpSourceCode.find("CG_ERROR") == tileOpSourceCode.npos) << "gen op invalid" << op.Dump();

        allocSourceRegion += allocSourceCode;
        tileOpSourceRegion += tileOpSourceCode;

        if (!allocSourceCode.empty()) {
            ALOG_INFO_F(": extra alloc generated(moved up to alloc region): %s", allocSourceCode.c_str());
        }
        ALOG_INFO_F(": op codegen result: \n, %s", tileOpSourceCode.c_str());
        ALOG_INFO_F("------------------------ Op CodeGenNPU Finish -----------------------");
    }

    std::string dynParamDef = GenDynParamForExpr(subFunc);
    std::string programCode = allocSourceRegion + dynParamDef + tileOpSourceRegion;
    return programCode;
}

std::string CodeGenCloudNPU::GenAllocForLocalBuffer(const Operation &op, SymbolManager &memAlloc) const {
    std::string allocSourceCode{};
    auto genExtraAllocForTensor = [this, &memAlloc, &op](const std::shared_ptr<LogicalTensor> &operand) -> std::string {
        if (HasAllocAttr(operand)) {
            ALOG_INFO_F("operand has an alloc attr, need to gen extra alloc\n%s", operand->Dump().c_str());
            std::optional<std::string> allocCodeMaybe = GenExtraAlloc(memAlloc, operand, op);
            if (allocCodeMaybe.has_value()) {
                return allocCodeMaybe.value();
            }
        }
        return "";
    };
    for (const std::shared_ptr<LogicalTensor> &operand : op.GetIOperands()) {
        // NEXTNEXT "inverseMap_.emplace" should be deleted later when gaoxiang prepared
        memAlloc.AddToTensorMap(operand->GetMagic(), operand);
        PrintOperand("IOperand", operand);
        allocSourceCode += genExtraAllocForTensor(operand);
    }
    for (const std::shared_ptr<LogicalTensor> &operand : op.GetOOperands()) {
        // NEXTNEXT "inverseMap_.emplace" should be deleted later when gaoxiang prepared
        memAlloc.AddToTensorMap(operand->GetMagic(), operand);
        PrintOperand("OOperand", operand);
        allocSourceCode += genExtraAllocForTensor(operand);
    }

    return allocSourceCode;
}

// GET_PARAM_OFFSET_BY_IDX(param, n, base, dim, idx)
// GET_PARAM_VALID_SHAPE_BY_IDX(param, n, base, dim, idx)
std::string CodeGenCloudNPU::GenDynParamForExpr(const npu::tile_fwk::Function &func) const {
    unsigned isSupportUnaligned = ConfigManager::Instance().GetCodeGenConfig(KEY_SUPPORT_DYNAMIC_UNALIGNED, false);
    if (!isSupportUnaligned) {
        return {};
    }
    std::string dynParamList;
    for (const auto &dynParam : func.GetDynParamTable()) {
        std::string dynParamExpr = "uint64_t " + dynParam.first + " = ";
        DynParamInfo info = dynParam.second;
        if (info.type == DynParamInfoType::VALID_SHAPE) {
            dynParamExpr += GET_PARAM_VALID_SHAPE_BY_IDX;
        } else if (info.type == DynParamInfoType::OFFSET) {
            dynParamExpr += GET_PARAM_OFFSET_BY_IDX;
        }
        dynParamExpr += "(param, " + std::to_string(info.tensorIndex) + ", " +
                        std::to_string(info.tensorBaseAddrCoaIndex) + ", " + std::to_string(info.dimSize) + ", " +
                        std::to_string(info.dimIndex) + ");\n";
        dynParamList += dynParamExpr;
    }
    return dynParamList;
}

std::string CodeGenCloudNPU::GetParamType(const Function &func) {
    if (isUnderDynamicFunction) {
        return GM_PARAM_TYPE_FOR_DYN;
    }
    return func.GetFunctionType() == FunctionType::DYNAMIC_LOOP_PATH ? GM_PARAM_TYPE_FOR_DYN : GM_PARAM_TYPE_FOR_STATIC;
}

std::string CodeGenCloudNPU::GenFuncCodeAfterReplace(
    const Function &func, std::pair<uint64_t, Function *> subFuncPair, const std::string &subProgramCode) {
    std::string tpl = R"!!!(
#include "TileOpImpl.h"

// funcHash: ${funcHash}$

[aicore] void ${FunctionName}$_${ProgramId}$(${ParamType}$* param, uint64_t GMStackBase, __gm__ int64_t *hcclContext, __gm__ GMTensorInfo* oriAddrParam) {
${SubProgCode}$
}
)!!!";
    SubstMap substMap = {
        {"FunctionName",                   func.GetMagicName()},
        {   "ProgramId",     std::to_string(subFuncPair.first)},
        { "SubProgCode",                        subProgramCode},
        {   "ParamType",                    GetParamType(func)},
        {    "funcHash", subFuncPair.second->GetFunctionHash()}
    };

    std::string funCode = StringSubstitute(tpl, substMap);
    return funCode;
}

void CodeGenCloudNPU::GenCode(
    const std::string &jsonPath, const std::map<uint64_t, std::list<InvokeParaOffset>> &invokeParaOffset) {
    std::ifstream file(jsonPath);
    ASSERT(file.good()) << "Json file: " << jsonPath << " open failed!!!";
    Json jsonData;
    try {
        file >> jsonData;
    } catch (const std::exception &e) {
        ASSERT(false) << "Json file: " << jsonPath << " parsing error: " << e.what();
    }
    ALOG_INFO_F("Start GenOpCode by Json");
    Program::GetInstance().LoadJson(jsonData);
    Function *func = Program::GetInstance().GetCurrentFunction();
    ASSERT(func->rootFunc_ != nullptr) << "func can not be nullptr";
    GenCode(*func, invokeParaOffset);
}

void CodeGenCloudNPU::GenCode(
    Function &topFunc, [[maybe_unused]] const std::map<uint64_t, std::list<InvokeParaOffset>> &invokeParaOffset) {
    std::deque<std::function<void(void)>> tasks;
    for (auto &subFuncPair : topFunc.rootFunc_->programs_) {
        std::function task = [this, subFuncPair, &topFunc]() {
            ALOG_INFO_F(" ----- subprogram id [%d] -----", subFuncPair.first);
            auto subFunc = subFuncPair.second;
            if (subFunc->IsAicpuSubFunction()) {
                subFunc->SetCoreType(CoreType::AICPU);
                subFunc->SetBinPath("");
                return;
            }
            isUnderDynamicFunction = subFunc->IsUnderDynamicFunction();
            std::string subProgramCode = GenCodeImpl(*subFunc, topFunc);
            std::string funCode = GenFuncCodeAfterReplace(topFunc, subFuncPair, subProgramCode);
            bool isCube = IsCube(subFunc->Operations());
            std::string coreType = isCube ? "_aic" : "_aiv";
            std::stringstream ss;
            ss << ctx.ccePath << "/" << topFunc.GetMagicName() << "_" << topFunc.GetFunctionHash() << "_" << subFuncPair.first
               << coreType << "_rankId_" << npu::tile_fwk::stubs::DeviceStub::GetCurrentDeviceId();

            std::string fileNameStub = ss.str();
            std::string inputFile = fileNameStub + ".cpp";
            std::string outputFile = fileNameStub + ".o";
            std::string configJson = fileNameStub + ".json";

            // vf codegen
            std::string vfFile = fileNameStub + ".h";
            VFCodegen vfCodegen;
            auto vfCodeRet = vfCodegen.GenCode(subFunc, vfFile);
            if (vfCodeRet) {
                funCode = "#include \"" + vfFile + "\"\n" + funCode;
            }

            // expression fusion
            if (npu::tile_fwk::ConfigManager::Instance().GetCodeGenConfig(npu::tile_fwk::KEY_CODEGEN_EXPRESSION_FUSION, false)) {
                std::string expressionFileName = "../kernel_aicpu/expression.h";
                funCode = "#include \"" + expressionFileName + "\"\n" + funCode;
            }

            bool needDump = false;
            if (npu::tile_fwk::ConfigManager::Instance().GetCodeGenConfig(npu::tile_fwk::KEY_CODEGEN_FORCE_DUMP_CCE_ON_EXIST, true)) {
                // force dump, default is true
                needDump = true;
            } else {
                // not force dump
                if (npu::tile_fwk::FileExist(inputFile)) {
                    needDump = false;
                } else {
                    needDump = true;
                }
            }
            bool ret = true;
            if (needDump) {
                ret = DumpCCE(inputFile, funCode);
                ASSERT(ret) << "Dump cce code failed!!";
            }

            int errCode = CompileCCE(inputFile, outputFile, isCube, "");
            ASSERT(errCode == 0) << "CompileCCE failed. errCode = " << errCode << ", cce file: " << inputFile;

            ret = GenConfigJson(
                configJson, inputFile, outputFile, topFunc.GetMagicName(), subFunc->GetStackWorkespaceSize());
            ASSERT(ret) << "Gen config json failed!!";

            // update bin path
            subFunc->SetBinPath(outputFile);
            subFunc->SetCoreType(isCube ? CoreType::AIC : CoreType::AIV);
        };
        tasks.push_back(task);
    }
    unsigned threadNum = ConfigManager::Instance().GetCodeGenConfig(KEY_PARALLEL_THREAD_NUM, 1u);
    util::ParallelExecuteAndWait(threadNum, tasks);
}

bool CodeGenCloudNPU::DumpCCE(const std::string &name, const std::string &code) const {
    std::ofstream file;
    file.open(name);
    file << code;
    return !file.fail();
}

bool CodeGenCloudNPU::GenConfigJson(const std::string &configJson, const std::string &cppName,
    const std::string &binName, const std::string &kernelName, int workspaceSize) const {
    std::ofstream file;
    file.open(configJson);
    file << "{\n"
         << R"(    "kernelFile": ")" << cppName << "\",\n"
         << R"(    "kernelBin": ")" << binName << "\",\n"
         << R"(    "kernelName": ")" << kernelName + "_main"
         << "\",\n"
         << "    \"workspaceSize\": " << workspaceSize << "\n"
         << "}";
    return !file.fail();
}

std::optional<std::string> CodeGenCloudNPU::GenExtraAlloc(
    SymbolManager &memAlloc, const std::shared_ptr<LogicalTensor> &tensor, const npu::tile_fwk::Operation &op) const {
    const auto &memMap = tensor->memorymap;
    if (memMap.find(tensor->subGraphID) == memMap.end()) {
        ALOG_ERROR_F("%s: can not find subgGraphID(%d) in the memorymap of op:", __FUNCTION__, tensor->subGraphID);
        ALOG_ERROR_F("    %s", op.Dump().c_str());
        return std::nullopt;
    }

    auto memType = tensor->GetMemoryTypeOriginal();
    if (npu::tile_fwk::OPERAND_TYPE_TO_MEMORY_TYPE.find(memType) == npu::tile_fwk::OPERAND_TYPE_TO_MEMORY_TYPE.end()) {
        ALOG_ERROR_F("%s: invalid memory type(%d) of tensor tensor: ", __FUNCTION__, static_cast<size_t>(memType));
        ALOG_ERROR_F("    %s", tensor->Dump().c_str());
        return std::nullopt;
    }

    const npu::tile_fwk::TileRange &range = memMap.at(tensor->subGraphID);
    auto bufferType = npu::tile_fwk::OPERAND_TYPE_TO_MEMORY_TYPE.at(memType);

    return GenAlloc(memAlloc, bufferType, tensor->Datatype(), range);
}

std::string GenAllocVarName(const char *prefix, const npu::tile_fwk::TileRange &range) {
    std::stringstream ss;
    ss << prefix
       // range start/end are always positive
       << "_S" << range.start << "_E" << range.end;

    return ss.str();
}

std::string CodeGenCloudNPU::GenAlloc(SymbolManager &manager, SymbolManager::BufferType bufferType,
    npu::tile_fwk::DataType dataType, const npu::tile_fwk::TileRange &range) const {
    if ((BUFFER_TYPE_TO_PREFIX.count(bufferType) == 0) ||
        (npu::tile_fwk::OPERAND_TYPE_TO_ADDR_TYPE.count(bufferType) == 0)) {
        ALOG_ERROR_F("%s: invalid bufferType: %d", __FUNCTION__, static_cast<size_t>(bufferType));
        ASSERT(false);
        return "";
    }

    const char *prefix = BUFFER_TYPE_TO_PREFIX.at(bufferType);
    const std::string &addrSpaceQualifier = npu::tile_fwk::OPERAND_TYPE_TO_ADDR_TYPE.at(bufferType);

    std::string allocVarName = GenAllocVarName(prefix, range);

    // must conform to CodeGenOpCloudNPU::createAllocKey
    SymbolManager::AllocKey key = SymbolManager::AllocKey(bufferType, range.start, range.end);
    bool reuse = manager.BindAddrWithVariableName(key, allocVarName);
    if (reuse) {
        return "";
    }

    ALOG_INFO_F(
        "%s: bind key to name: %s->%s", __FUNCTION__, npu::tile_fwk::FormatAllocKey(key).c_str(), allocVarName.c_str());

    std::string dataTypeStr = DataType2CCEStr(dataType);

    char buffer[256] = "CG_ERROR";
    int ret = sprintf_s(buffer, sizeof(buffer), "%s %s *%s = (%s %s *)get_imm(0x%x); // size: %x \n",
        dataTypeStr.c_str(), addrSpaceQualifier.c_str(), allocVarName.c_str(), dataTypeStr.c_str(),
        addrSpaceQualifier.c_str(), static_cast<unsigned>(range.start), static_cast<unsigned>(range.Size()));
    ASSERT(ret >= 0) << "GenAlloc sprintf_s failed ";
    return buffer;
}

int CheckInjectStr(char cmdStr[], size_t strLen) {
    if (cmdStr == nullptr) {
        return -1;
    }
    char filtChar[] = {';', '|', '`', '>', '<'};
    for (size_t i = 0; i < strLen; ++i) {
        for (const auto &c : filtChar) {
            if (cmdStr[i] == c) {
                return -1;
            }
        }
    }
    return 0;
}

int CodeGenCloudNPU::CompileCCE(
    const std::string &srcFile, const std::string &objFile, bool isCube, const std::string &compileOptions) const {
    std::string coreType = isCube ? "dav-c220-cube" : "dav-c220-vec";

    char ccecCmd[2048];
    std::string includePath = ctx.IsIncludePathEmpty() ? SRC_PATH : ctx.includePath;
    int ret = snprintf_s(ccecCmd, sizeof(ccecCmd), sizeof(ccecCmd) - 1,
        "ccec %s -c -O3 -g -x cce -std=c++17 "
        "--cce-aicore-only "
        "--cce-aicore-arch=%s "
        "-mllvm -cce-aicore-stack-size=0x8000 "
        "-mllvm -cce-aicore-function-stack-size=0x8000 "
        "-mllvm -cce-aicore-record-overflow=false "
        "-mllvm -cce-aicore-addr-transform "
        "-mllvm -cce-aicore-dcci-insert-for-scalar=false "
        "-I%s/include/tileop/a2a3 "
        "-I%s/src/machine/kernel "
        "-I%s/src/ "
        "-o %s "
        "%s",
        compileOptions.c_str(), coreType.c_str(), includePath.c_str(), includePath.c_str(), includePath.c_str(),
        objFile.c_str(), srcFile.c_str());
    if (ret < 0) {
        ALOG_INFO_F("CompileCCE snprintf_s failed %d", ret);
    }

    ALOG_INFO_F("compile kernel...\n%s", ccecCmd);
    if (CheckInjectStr(ccecCmd, strlen(ccecCmd)) != 0) {
        ALOG_INFO_F("CheckInjectStr failed...\n");
        return -1;
    }
    ret = std::system(ccecCmd);
    if (ret != 0) {
        ALOG_INFO_F("CompileCce ccec failed %d", ret);
    }
    return ret;
}

} // namespace npu::tile_fwk
