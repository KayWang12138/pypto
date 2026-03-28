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

#include "codegen_op_litenpu.h"

#include <cstring>
#include <nlohmann/json.hpp>

#include "interface/utils/log.h"
#include "codegen/utils/parallel_execute.h"
#include "interface/utils/file_utils.h"
#include "interface/tensor/logical_tensor.h"
#include "interface/function/function.h"
#include "interface/configs/config_manager.h"
#include "securec.h"
#include "tilefwk/tilefwk.h"
#include "interface/program/program.h"
#include "interface/utils/op_info_manager.h"
#include "codegen_litenpu.h"

namespace npu::tile_fwk {
#ifdef SRCPATH
constexpr const char *SRC_PATH = SRCPATH;
#else
constexpr const char *SRC_PATH = ".";
#endif

const std::string ENV_ASCEND_HOME_PATH = "ASCEND_HOME_PATH";
const std::string ENV_PTO_TILE_LIB_CODE_PATH = "PTO_TILE_LIB_CODE_PATH";
constexpr const int64_t CODE_RESERVED_SIZE = 1024 * 1024;

bool CodeGenLiteNPU::IsCube(const OperationsViewer &operationList) const {
    auto isL1CopyIn = [](const Operation &op) {
        return op.GetOpcode() == Opcode::OP_COPY_IN && !(op.oOperand.empty()) &&
               op.oOperand[ID0]->GetMemoryTypeOriginal() == MemoryType::MEM_L1;
    };

    for (const auto &oper : operationList) {
        if (isL1CopyIn(oper)) {
            return true;
        }
    }

    return false;
}

void CodeGenLiteNPU::PrintOperand(const std::string &operIO, std::shared_ptr<LogicalTensor> operand) const {
    CODEGEN_LOGI("insert %s magic: %d, tensor: %s, memory map is: ", operIO.c_str(), operand->GetMagic(),
        operand->Dump().c_str());
    CODEGEN_LOGI(
        "range is [%zu, %zu, %d]\n", operand->memoryrange.start, operand->memoryrange.end, operand->memoryrange.memId);
}

bool CodeGenLiteNPU::HasAllocAttr(const std::shared_ptr<LogicalTensor> &tensor) const {
    bool needAlloc = false;
    tensor->GetAttr(OpAttributeKey::needAlloc, needAlloc);
    return needAlloc;
}

void CodeGenLiteNPU::GenFuncBody(Function &subFunc, Function &topFunc, std::ostringstream &oss) const {
    OperationsViewer operationList = subFunc.Operations(false);
    if (operationList.IsEmpty()) {
        CODEGEN_LOGW("operationList from PASS is empty, func magic name: %s, func hash: %s",
            subFunc.GetMagicName().c_str(), subFunc.GetFunctionHash().c_str());
    }

    CODEGEN_LOGI("TopFunc Type is %s\nFunction to codegen:\n %s\n", topFunc.GetFunctionTypeStr().c_str(),
        topFunc.Dump().c_str());

    std::shared_ptr<SymbolManager> symbolMgr = std::make_shared<SymbolManager>();
    FloatSpecValMgrLite floatSpecValMgr;
    std::string allocSourceRegion;
    allocSourceRegion.reserve(CODE_RESERVED_SIZE);
    std::string tileOpSourceRegion;
    tileOpSourceRegion.reserve(CODE_RESERVED_SIZE);
    auto locToOffsetMap = GenRealizeIdMap(subFunc.GetParameter());
    for (const auto &op : operationList) {
        CODEGEN_LOGI(
            "======================== Op CodeGenNPU Start ========================\nGen OP IS: %s", op.Dump().c_str());
        Opcode opcode = op.GetOpcode();
        if (SKIP_OPCODE_FOR_CODEGEN.find(opcode) != SKIP_OPCODE_FOR_CODEGEN.end()) {
            CODEGEN_LOGI("ignore this op\n------------------------ Op CodeGenNPU Finish -----------------------");
            continue;
        }

        std::string allocSourceCode = GenAllocForLocalBuffer(op, symbolMgr);
        floatSpecValMgr.UpdateByOp(op);

        CodeGenOpLiteNPU cop({symbolMgr, topFunc, subFunc, op, locToOffsetMap, ctx.isMainBlock});
        std::string tileOpSourceCode = cop.GenOpCode();
        ASSERT(tileOpSourceCode.find("CG_ERROR") == tileOpSourceCode.npos)
            << "Generate code of op failed, op is " << op.Dump();

        allocSourceRegion.append(allocSourceCode);

        for (auto &c : op.GetCommentList()) {
            tileOpSourceRegion.append("/*").append(c).append("*/\n");
        }
        tileOpSourceRegion.append(tileOpSourceCode);

        if (!allocSourceCode.empty()) {
            CODEGEN_LOGI(": extra alloc generated(moved up to alloc region): %s", allocSourceCode.c_str());
        }
        CODEGEN_LOGI("------------------------ Op CodeGenNPU Finish -----------------------");
    }
    floatSpecValMgr.PrintFloatSpecVal(oss);
    oss << allocSourceRegion << symbolMgr->GenUsingList() << symbolMgr->GenTileTensorDefList() << tileOpSourceRegion;
}

std::string CodeGenLiteNPU::GenAllocForLocalBuffer(const Operation &op, const std::shared_ptr<SymbolManager> &symbolMgr) const {
    std::string allocSourceCode{};
    auto genExtraAllocForTensor = [this, &symbolMgr, &op](
                                      const std::shared_ptr<LogicalTensor> &operand) -> std::string {
        if (CodeGenLiteNPU::HasAllocAttr(operand)) {
            ALOG_INFO_F("operand has an alloc attr, need to gen extra alloc\n%s", operand->Dump().c_str());
            std::optional<std::string> allocCodeMaybe = GenExtraAlloc(symbolMgr, operand);
            if (allocCodeMaybe.has_value()) {
                return allocCodeMaybe.value();
            }
        }
        return "";
    };
    for (const std::shared_ptr<LogicalTensor> &operand : op.GetIOperands()) {
        symbolMgr->AddToTensorMap(operand->GetMagic(), operand);
        CodeGenLiteNPU::PrintOperand("IOperand", operand);
        allocSourceCode += genExtraAllocForTensor(operand);
    }
    for (const std::shared_ptr<LogicalTensor> &operand : op.GetOOperands()) {
        symbolMgr->AddToTensorMap(operand->GetMagic(), operand);
        CodeGenLiteNPU::PrintOperand("OOperand", operand);
        allocSourceCode += genExtraAllocForTensor(operand);
    }

    return allocSourceCode;
}

std::string CodeGenLiteNPU::GetParamType(const Function &func) const {
    if (isUnderDynamicFunction_) {
        return GM_PARAM_TYPE_FOR_DYN;
    }
    return func.GetFunctionType() == FunctionType::DYNAMIC_LOOP_PATH ? GM_PARAM_TYPE_FOR_DYN : GM_PARAM_TYPE_FOR_STATIC;
}

void CodeGenLiteNPU::GenCode(
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

std::string GetDtype(DataType dtype) {
    switch (dtype) {
        case DataType::DT_UINT8: return "uint8_t";
        case DataType::DT_INT8: return "int8_t";
        case DataType::DT_INT16: return "int16_t";
        case DataType::DT_INT32: return "int32_t";
        case DataType::DT_INT64: return "int64_t";
        case DataType::DT_FP16: return "half";
        case DataType::DT_FP32: return "float";
        default: return "unknown";
    }
}
using SubstMap = std::map<std::string, std::string>;
static std::string StringSubstitute(std::string const &in, SubstMap const &subst) {
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

static bool CompareStrings(const std::string &s1, const std::string &s2) {
    std::string str1 = s1;
    std::string str2 = s2;
    transform(str1.begin(), str1.end(), str1.begin(), ::tolower);
    transform(str2.begin(), str2.end(), str2.begin(), ::tolower);

    return str1 < str2;
}

std::map<int, std::string> GenParamsSymbolMap(const SubfuncParam &subFuncParam,
    std::vector<std::string> &params, std::map<std::string, std::string> &dTypeMap) {
    auto &tensorInvokeArgs = subFuncParam.tensorsArgs_;
    auto &incastInvokeArgs = subFuncParam.inCastArgs_;
    auto &outcastInvokeArgs = subFuncParam.outCastArgs_;

    std::map<int, std::string> symbolMap;
    std::set<std::string> paramsSet;
    auto f = [&paramsSet, &dTypeMap, &symbolMap](size_t offset, auto &invokeArgs) {
        ALOG_INFO_F("start offset is %d, arg size is %d", offset, invokeArgs.size());
        for (size_t i = 0; i < invokeArgs.size(); i++) {
            size_t paramOff = (offset + i);
            uint32_t paramLoc = invokeArgs[i].paramLoc;
            ALOG_DEBUG("paramLoc ", paramLoc, " --> offset ", paramOff);
            ALOG_INFO_F(" paramLoc is %d, paramOff is %d, SymDDRId is %d, SymName is %s", paramLoc, paramOff,
                invokeArgs[i].symDDRId, invokeArgs[i].symName,
                invokeArgs[i].symbol, static_cast<size_t>(invokeArgs[i].dataType));
            symbolMap.insert({paramLoc, invokeArgs[i].symbol});
            paramsSet.insert(invokeArgs[i].symbol);
            dTypeMap[invokeArgs[i].symbol] = GetDtype(invokeArgs[i].dataType);
        }
    };

    ALOG_INFO_F("---  start tensorInvokeArgs paramLoc map ---- ");
    f(0, tensorInvokeArgs);
    ALOG_INFO_F("---  start incastInvokeArgs paramLoc map ---- ");
    f(tensorInvokeArgs.size(), incastInvokeArgs);
    ALOG_INFO_F("---  start outcastInvokeArgs paramLoc map ---- ");
    f(tensorInvokeArgs.size() + incastInvokeArgs.size(), outcastInvokeArgs);
    for (auto &t : paramsSet) {
        params.push_back(t);
    }
    std::sort(params.begin(), params.end(), CompareStrings);
    return symbolMap;
}

std::string CodeGenLiteNPU::GenFuncGlobalCodeAfterReplace(
    const Function &func, std::pair<uint64_t, Function *> subFuncPair, const std::string &subProgramCode) {
    std::string tpl = R"!!!(
#include "TileOpImpl.h"

extern "C" __global__ [aicore] void ${FunctionName}$_main(${GlobalParams}$) {
    ${SubProgCode}$
}
)!!!";
    std::vector<std::string> inOutParams;
    std::map<std::string, std::string> dTypeMap;
    auto symbolMap = GenParamsSymbolMap(subFuncPair.second->GetParameter(), inOutParams, dTypeMap);

    std::string globalParams = "";
    std::string subParams = "";
    for (auto &p : inOutParams) {
        globalParams += "__gm__ " + dTypeMap[p] + "* " + "__restrict__ " + p + ", ";
        subParams += p + ", ";
    }
    if (subFuncPair.second->GetStackWorkespaceSize() > 0) {
        globalParams += "__gm__ int8_t* __restrict__ workspace, ";
        subParams += "workspace, ";
    }

    SubstMap substMap = {
        {"FunctionName",                                  func.GetMagicName()},
        {   "ProgramId",                    std::to_string(subFuncPair.first)},
        { "SubProgCode",                                       subProgramCode},
        { "GlobalParams",   globalParams.substr(0, globalParams.length() - 2)},
        { "SubParams",            subParams.substr(0, subParams.length() - 2)},
    };

    std::string funCode = StringSubstitute(tpl, substMap);

    // GM replace
    for (auto &ele : symbolMap) {
        std::string oldStr = "RealizedGM" + std::to_string(ele.first) + ".Addr";
        std::string newStr = ele.second;
        size_t pos = 0;
        while ((pos = funCode.find(oldStr, pos)) != std::string::npos) {
            funCode.replace(pos, oldStr.length(), newStr);
            pos += newStr.length();
        }
    }
    return funCode;
}

std::vector<std::string> CodeGenLiteNPU::GetInOutParams(std::pair<uint64_t, Function *> subFuncPair) {
    std::vector<std::string> inOutParams;
    std::map<std::string, std::string> dTypeMap;
    auto symbolMap = GenParamsSymbolMap(subFuncPair.second->GetParameter(), inOutParams, dTypeMap);

    return inOutParams;
}

void CodeGenLiteNPU::GenConfigJson(const std::string &jsonName, const std::string &cppName, const std::string &binName,
    const std::string &kernelName, const int &workspaceSize, const std::vector<std::string> &argNames,
    const int &blockDim) const {
    std::ofstream file;
    file.open(jsonName);

    file << "{\n"
         << "   \"kernelFile\": \"" << cppName << "\",\n"
         << "   \"kernelBin\": \"" << binName << "\",\n"
         << "   \"kernelName\": \"" << kernelName + "_main" << "\",\n"
         << "   \"workspaceSize\": " << workspaceSize << ",\n"
         << "   \"blockDim\": " << blockDim << ",\n"
         << "   \"argNames\": [";

    for (size_t i = 0; i < argNames.size(); i++) {
        file << "\"" << argNames[i] << "\"";

        if (i < argNames.size() - 1) {
            file << ", ";
        }
    }

    file << "]\n}";
}

void CodeGenLiteNPU::GenCode(
    Function &topFunc, [[maybe_unused]] const std::map<uint64_t, std::list<InvokeParaOffset>> &invokeParaOffset) {
    std::deque<std::function<void(void)>> tasks;
    for (auto &subFuncPair : topFunc.rootFunc_->programs_) {
        std::function task = [this, subFuncPair, &topFunc]() {
            CODEGEN_LOGI(" ----- subprogram id [%lu] -----", subFuncPair.first);
            auto subFunc = subFuncPair.second;
            if (HandleForAICpuSubFunc(*subFunc)) {
                return;
            }
            std::vector<std::string> inOutParams = GetInOutParams(subFuncPair);
            bool isCube = subFunc->IsCube();
            CompileInfo_LiteNPU compileInfo(topFunc, ctx, subFuncPair, isCube, subFunc->IsUnderDynamicFunction());
            std::ostringstream leafKernelFunc;
            GenFuncBody(*subFunc, topFunc, leafKernelFunc);
            std::string funcCode = GenFuncGlobalCodeAfterReplace(topFunc, subFuncPair, leafKernelFunc.str());
#ifdef BUILD_WITH_CANN
            if (std::getenv(ENV_ASCEND_HOME_PATH.c_str()) != nullptr) {
                DumpCCE(compileInfo.GetCCEAbsPath(), funcCode);
                DoCompileCCE(compileInfo, ""); // TODO: currently has issue
                int blockDim = 1; // TODO: currently only support one block dim
                int jsonWorkspaceSize = 0; // TODO...
                GenConfigJson(compileInfo.GetJsonAbsPath(), compileInfo.GetCCEAbsPath(), compileInfo.GetBinAbsPath(),
                    topFunc.GetMagicName(), jsonWorkspaceSize, inOutParams, blockDim);
            }
#endif
            UpdateSubFunc(subFuncPair, compileInfo);
        };
        tasks.push_back(task);
    }
    unsigned threadNum = ConfigManager::Instance().GetCodeGenConfig(KEY_PARALLEL_COMPILE, 1u);
    ParallelExecuteAndWait(threadNum, tasks);
}

void CodeGenLiteNPU::UpdateSubFunc(std::pair<uint64_t, Function *> subFuncPair, const CompileInfo_LiteNPU &compileInfo) const {
    auto subFunc = subFuncPair.second;
    std::shared_ptr<LeafFuncAttribute> attr = std::make_shared<LeafFuncAttribute>();
    attr->kernelName = compileInfo.GetKernelName();
    attr->binPath = compileInfo.GetBinAbsPath();
    attr->kernelDeclare = compileInfo.GetFuncDeclare();
    CoreType coreType = compileInfo.IsCube() ? CoreType::AIC : CoreType::AIV;
    attr->coreType = coreType;
    subFunc->SetLeafFuncAttribute(attr);
}

bool CodeGenLiteNPU::IsNeedDumpCCE(const std::string &inputFile) const {
    if (ConfigManager::Instance().GetCodeGenConfig(KEY_FORCE_OVERWRITE, true)) {
        // force dump, default is true
        return true;
    }
    // not force dump
    if (FileExist(inputFile)) {
        return false;
    }
    return true;
}

void CodeGenLiteNPU::DumpCCE(const std::string &fileName, const std::string &code) const {
    if (!IsNeedDumpCCE(fileName)) {
        return;
    }

    std::ofstream cceFile;
    try {
        // 开启异常：failbit/badbit 触发 std::ofstream::failure 异常
        cceFile.exceptions(std::ofstream::failbit | std::ofstream::badbit);
        cceFile.open(fileName);
        cceFile << code;
        cceFile.flush();
        cceFile.close();
    } catch (const std::ofstream::failure &e) {
        CODEGEN_LOGE("CCE file operation failed: %s, error: %s, errno: %d", fileName.c_str(), e.what(), errno);
        cceFile.close();
        std::remove(fileName.c_str());
        return;
    }
}

std::optional<std::string> CodeGenLiteNPU::GenExtraAlloc(
    const std::shared_ptr<SymbolManager> &symbolMgr, const std::shared_ptr<LogicalTensor> &tensor) const {
    auto memType = tensor->GetMemoryTypeOriginal();
    if (OPERAND_TYPE_TO_MEMORY_TYPE.find(memType) == OPERAND_TYPE_TO_MEMORY_TYPE.end()) {
        ALOG_ERROR_F("%s: invalid memory type(%d) of tensor tensor: ", __FUNCTION__, static_cast<size_t>(memType));
        ALOG_ERROR_F("    %s", tensor->Dump().c_str());
        return std::nullopt;
    }

    const TileRange &memRange = tensor->memoryrange;
    auto bufferType = OPERAND_TYPE_TO_MEMORY_TYPE.at(memType);

    return GenAlloc(symbolMgr, bufferType, tensor->Datatype(), memRange);
}

std::pair<std::string, std::string> CodeGenLiteNPU::GenAllocVarName(const std::string &prefix, const TileRange &range) const {
    std::ostringstream ss;
    ss << prefix
       // range start/end are always positive
       << "_S" << range.start << "_E" << range.end;

    return std::make_pair(ss.str(), ss.str() + "_T");
}

std::string CodeGenLiteNPU::GenAlloc(
    const std::shared_ptr<SymbolManager> &sm, BufferType bufferType, DataType dataType, const TileRange &range) const {
    if ((BUFFER_TYPE_TO_PREFIX.count(bufferType) == 0) || (OPERAND_TYPE_TO_ADDR_TYPE.count(bufferType) == 0)) {
        ALOG_ERROR_F("%s: invalid bufferType: %d", __FUNCTION__, static_cast<size_t>(bufferType));
        ASSERT(false);
        return "";
    }

    const std::string prefix = BUFFER_TYPE_TO_PREFIX.at(bufferType);
    const std::string &addrSpaceQualifier = OPERAND_TYPE_TO_ADDR_TYPE.at(bufferType);
    auto [allocVarName, allocVarNameTileTensor] = GenAllocVarName(prefix, range);

    // must conform to CodeGenOplitenpu::createAllocKey
    AllocKey key = AllocKey(bufferType, range.start, range.end);
    bool reuse = sm->BindAddrWithVariableName(key, allocVarName, allocVarNameTileTensor);
    if (reuse) {
        return "";
    }

    ALOG_INFO_F(
        "%s: bind key to name: %s->%s", __FUNCTION__, sm->FormatAllocKey(key).c_str(), allocVarName.c_str());

    std::string dataTypeStr = DataType2CCEStr(dataType);

    std::ostringstream oss;
    oss << dataTypeStr << " " << addrSpaceQualifier << " *" << allocVarName << " = (" << dataTypeStr << " "
        << addrSpaceQualifier << " *)get_imm(0x" << std::hex << static_cast<unsigned>(range.start) << "); // size: 0x"
        << std::hex << static_cast<unsigned>(range.Size()) << "\n";

    return oss.str();
}

int CodeGenLiteNPU::CheckInjectStr(const char cmdStr[], size_t strLen) const {
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

void CodeGenLiteNPU::DoCompileCCE(const CompileInfo_LiteNPU &compileInfo, const std::string &compileOptions) const {
    if (config::GetHostOption<int64_t>(COMPILE_STAGE) == CS_CODEGEN_INSTRUCTION) {
        CODEGEN_LOGI("Compile stage terminates after codegen instruction.");
        return;
    }
    auto [ret, ccecCmd] = CompileCCE(compileInfo, compileOptions);
    ASSERT(ret == 0) << "CompileCCE failed. errCode = " << ret << ", cce file: " << compileInfo.GetCCEAbsPath()
                     << "\n******** bisheng compiling cmd start ********\n"
                     << ccecCmd << "\n******** bisheng compiling cmd end ********\n";
}

std::string CodeGenLiteNPU::GetIncludePathByRelative() const {
    std::string curExePath = GetCurRunningPath();
    ALOG_INFO_F("curExePath is %s", curExePath.c_str());
    std::string includePath = curExePath + "/../include/";
    ALOG_INFO_F("includePath relative is %s", includePath.c_str());
    return includePath;
}

std::string CodeGenLiteNPU::GetIncludePathByLib() const {
    std::string libPath = GetCurrentSharedLibPath();
    if (libPath.empty()) {
        return "";
    }

    std::string includePath = libPath + "/../../include/";
    ALOG_INFO_F("includePath by lib is %s", includePath.c_str());

    if (IsPathExist(includePath)) {
        return includePath;
    }

    return "";
}

std::string CodeGenLiteNPU::GetIncludePathByEnv() const {
    const char *homePath = std::getenv(ENV_ASCEND_HOME_PATH.c_str());
    if (homePath == nullptr) {
        return "";
    }

    std::string includePath = std::string(homePath) + "/include/tile_fwk/";
    if (IsPathExist(includePath)){
        return includePath;
    }

    return "";
}

std::string CodeGenLiteNPU::GetIncludePathForCompileCCE() const {
    if (!ctx.IsIncludePathEmpty()) {
        ALOG_INFO_F("include path from ctx is %s", ctx.includePath.c_str());
        return ctx.includePath;
    }

    std::string includePathByLib = GetIncludePathByLib();
    ALOG_INFO_F("includePathByLib is %s", includePathByLib.c_str());
    if (!includePathByLib.empty()) {
        return includePathByLib;
    }

    std::string includePathByEnv = GetIncludePathByEnv();
    ALOG_INFO_F("includePathByEnv is %s", includePathByEnv.c_str());
    if (!includePathByEnv.empty()) {
        return includePathByEnv;
    }

    std::string includePathByRel = GetIncludePathByRelative();
    ALOG_INFO_F("includePathByRel is %s", includePathByRel.c_str());
    if (!includePathByRel.empty()) {
        return includePathByRel;
    }

    ASSERT(false) << "include path for compiling cce is unavailable";
    return "";
}

std::string CodeGenLiteNPU::GetPtoTileLibPathByEnv() const {
    if (!ConfigManager::Instance().GetCodeGenConfig(KEY_CODEGEN_SUPPORT_TILE_TENSOR, false)) {
        return "";
    }

    // Priority 1: Obtain pto-isa from the patch specified by the environment variable "PTO_TILE_LIB_CODE_PATH".
    const char *homePath = std::getenv(ENV_PTO_TILE_LIB_CODE_PATH.c_str());
    if (homePath != nullptr) {
        std::string envPath = std::string(homePath) + "/include";
        ASSERT(IsPathExist(envPath + "/pto")) << "Pto-isa path " << envPath << "/pto not found! please check.";
        return envPath;
    }

    // Priority 2: Obtain pto-isa from the installed cann package.
    homePath = std::getenv(ENV_ASCEND_HOME_PATH.c_str());
    if (homePath != nullptr) {
        std::string cannPath = std::string(homePath) + "/include";
        ASSERT(IsPathExist(cannPath + "/pto")) << "Pto-isa path " << cannPath << "/pto not found! please check.";
        return cannPath;
    }

    ASSERT(false) << "Pto-isa path not found. please install pto-isa properly.";
    return "";
}

// TODO: modify for kirin...
void CodeGenLiteNPU::BuildArchOptions(std::ostringstream &oss, const CompileInfo_LiteNPU &compileInfo) const {
    (void)compileInfo; // TODO...
    // const std::string corePredefine = compileInfo.IsCube() ? "-D__AIC__" : "-D__AIV__";

    // std::vector<std::string> compileOpts{corePredefine};
    std::vector<std::string> compileOpts;
    if (ConfigManager::Instance().GetCodeGenConfig(KEY_CODEGEN_SUPPORT_TILE_TENSOR, false)) {
        compileOpts.emplace_back("-DSUPPORT_TILE_TENSOR");
    }

    compileOpts.emplace_back("--cce-aicore-only");
    // std::string coreArch = GetCoreArch(compileInfo); // TODO: support lite npu...
    std::string coreArch = "dav-l311";
    compileOpts.emplace_back("--cce-aicore-arch=" + coreArch);

    std::string allCompileOpts = JoinString(compileOpts, " ");
    oss << allCompileOpts << " ";
}

// TODO: modify for kirin...
void CodeGenLiteNPU::BuildIncludes(std::ostringstream &oss) const {
    // used for compiling cce
    std::string includePath = GetIncludePathForCompileCCE();
    oss << "-I" << includePath << "/tilefwk "
        << "-I" << includePath << "/tileop "
        << "-I" << includePath << "/tileop/arch32 "
        << "-I" << includePath << " ";

    std::string ptoTileLibPath = GetPtoTileLibPathByEnv();
    if (!ptoTileLibPath.empty()) {
        oss << "-I" << ptoTileLibPath << " ";
    }
}

void CodeGenLiteNPU::BuildExtraOptions(std::ostringstream &oss, const std::string &compileOptions) const {
    oss << "-mllvm -cce-aicore-function-stack-size=16384 "
        << "-mllvm -cce-aicore-record-overflow=false "
        << "-mllvm -cce-aicore-addr-transform "
        << "-mllvm -cce-aicore-dcci-insert-for-scalar=false ";
    oss << compileOptions << " ";
}

std::pair<int, std::string> CodeGenLiteNPU::CompileCCE(
    const CompileInfo_LiteNPU &compileInfo, const std::string &compileOptions) const {
    std::ostringstream oss;
    oss << "bisheng -c -O3 -g -x cce -std=c++17 -w ";
    BuildArchOptions(oss, compileInfo);
    BuildIncludes(oss);
    BuildExtraOptions(oss, compileOptions);

    const std::string srcFile = compileInfo.GetCCEAbsPath();
    const std::string objFile = compileInfo.GetBinAbsPath();
    oss << "-o " << objFile << " " << srcFile;

    std::string ccecCmd = oss.str();

    CODEGEN_LOGI_FULL("compile kernel...\n%s", ccecCmd.c_str());

    int ret = CheckInjectStr(ccecCmd.c_str(), ccecCmd.length());
    ASSERT(ret == 0) << "CheckInjectStr failed. errCode = " << ret;

    ret = std::system(ccecCmd.c_str());
    if (ret != 0) {
        CODEGEN_LOGE("Compile cce kernel failed, ret = %d\ncompile cmd is:\n %s", ret, ccecCmd.c_str());
    }

    return {ret, ccecCmd};
}

bool CodeGenLiteNPU::HandleForAICpuSubFunc(Function &subFunc) {
    if (!subFunc.IsAicpuSubFunction().first) {
        return false;
    }

    std::shared_ptr<LeafFuncAttribute> attr = std::make_shared<LeafFuncAttribute>();
    attr->coreType = CoreType::AICPU;
    subFunc.SetLeafFuncAttribute(attr);
    return true;
}

void FloatSpecValMgrLite::UpdateByOp(const Operation &op) {
    std::vector<Element> eles;
    if (op.HasAttr(OpAttributeKey::scalar)) {
        eles.emplace_back(op.GetElementAttribute(OpAttributeKey::scalar));
    }
    if (op.HasAttr(OpAttributeKey::vectorScalar)) {
        auto vecScalars = op.GetVectorElementAttribute(OpAttributeKey::vectorScalar);
        eles.insert(eles.end(), vecScalars.begin(), vecScalars.end());
    }

    if (eles.empty()) {
        return;
    }

    for (const auto &e : eles) {
        if (e.GetDataType() != DataType::DT_FP16 && e.GetDataType() != DataType::DT_FP32) {
            continue;
        }
        double value = e.Cast<float>();
        if (std::isinf(value) || std::isnan(value)) {
            floatSpecVals_.insert({e.GetDataType(), value});
        }
    }
}

void FloatSpecValMgrLite::PrintFloatSpecVal(std::ostringstream &oss) {
    // print statement like: union {float f; uint32_t u;} float_inf = {.u = 0x7F800000};
    for (const auto &fs : floatSpecVals_) {
        std::string dtypeCCE = DataType2CCEStr(fs.dtype);
        oss << "union " << "{" << dtypeCCE << " f; " << "uint32_t u;} " << fs.GetFsVarName()
            << " = {.u = " << fs.GetFsValueStr() << "};\n";
    }
}

}