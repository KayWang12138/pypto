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
#include "codegen_vf.h"
#include "interface/utils/op_info_manager.h"
#include "codegen_cloudnpu.h"

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
               op.oOperand[ID0]->GetMemoryTypeOriginal() == MemoryType::MEM_L1;
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

std::string CodeGenCloudNPU::GenInclude(const VFCodeGen &vfCg) const {
    std::ostringstream include;
    // expression fusion
    if (ConfigManager::Instance().GetCodeGenConfig(KEY_CODEGEN_EXPRESSION_FUSION, false)) {
        uint64_t tilingKey = OpInfoManager::GetInstance().GetOpTilingKey();
        std::string expFileName = "../kernel_aicpu/expression_" + std::to_string(tilingKey) + ".h";
        // expression.h depend on __TILE_FWK_AICORE__
        include << "#define __TILE_FWK_AICORE__ 1\n#include \"" << expFileName << "\"\n";
    }

    if (vfCg.IsGenSuccess()) {
        include << vfCg.GetVFHeaderForInclude() << "\n\n";
    }

    include << "#include \"TileOpImpl.h\"\n\n";

    return include.str();
}

std::string CodeGenCloudNPU::GenCommentBeforeFuncHeader(Function &subFunc) {
    std::ostringstream comment;
    comment << "// funcHash: " << subFunc.GetFunctionHash() << "\n\n";
    return comment.str();
}

std::string CodeGenCloudNPU::GenKernelName(Function &topFunc, uint64_t programId) {
    std::ostringstream kernelName;
    kernelName << topFunc.GetMagicName() << "_" << programId;
    uint64_t tilingKey = OpInfoManager::GetInstance().GetNewSubTilingKey();
    kernelName << "_" << std::to_string(tilingKey);
    return kernelName.str();
}

std::string CodeGenCloudNPU::GenFuncHeader(uint64_t programId, Function &topFunc, CompileInfo &compileInfo) const {
    std::ostringstream funcHeader;
    funcHeader << "extern \"C\" [aicore] void ";
    // kernel name
    auto kernelName = GenKernelName(topFunc, programId);
    compileInfo.SetKernelName(kernelName);
    funcHeader << kernelName;
    // kernel func param
    std::string paramType = GetParamType(topFunc);
    funcHeader << "(" << paramType
               << "* param, int64_t GMStackBase, __gm__ int64_t *hcclContext, __gm__ GMTensorInfo* oriAddrParam)";
    auto funcDec = funcHeader.str() + ";";
    compileInfo.SetFuncDeclare(funcDec);
    funcHeader <<  " {\n";
    return funcHeader.str();
}

std::string CodeGenCloudNPU::GenFuncBodyBefore(const std::pair<uint64_t, Function *> &subFuncPair,
    Function &topFunc, const VFCodeGen &vfCg, CompileInfo &compileInfo) const {
    std::ostringstream codeBefore;
    codeBefore << GenInclude(vfCg);
    codeBefore << GenCommentBeforeFuncHeader(*subFuncPair.second);
    codeBefore << GenFuncHeader(subFuncPair.first, topFunc, compileInfo);
    return codeBefore.str();
}

std::string CodeGenCloudNPU::GenFuncEnd() {
    return "}\n";
}

std::string CodeGenCloudNPU::GenFuncBody(Function &subFunc, Function &topFunc) const {
    OperationsViewer operationList = subFunc.Operations(false);
    if (operationList.IsEmpty()) {
        ALOG_ERROR("operationList is empty");
        return {};
    }

    ALOG_INFO_F("Function to codegen:\n %s\n", topFunc.Dump().c_str());

    SymbolManager symbolMgr;
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

        std::string allocSourceCode = GenAllocForLocalBuffer(op, symbolMgr);

        CodeGenOpCloudNPU cop(symbolMgr, topFunc.GetFunctionType(), locToOffsetMap, topFunc.IsUnderDynamicFunction());
        auto success = cop.Init(op);
        if (!success) {
            ALOG_INFO_F(": failed to init CodeGenOpCloudNPU from an operation: %s \n", op.Dump().c_str());
            break;
        }
        cop.UpdateTileTensorInfo();
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
    std::string usingType = symbolMgr.GenUsingList();
    std::string tileTensorDef = symbolMgr.GenTileTensorDefList();
    std::ostringstream oss;
    oss << allocSourceRegion << dynParamDef << usingType << tileTensorDef << tileOpSourceRegion;
    std::string programCode = oss.str();
    return programCode;
}

std::string CodeGenCloudNPU::GenAllocForLocalBuffer(const Operation &op, SymbolManager &symbolMgr) const {
    std::string allocSourceCode{};
    auto genExtraAllocForTensor = [this, &symbolMgr, &op](
                                      const std::shared_ptr<LogicalTensor> &operand) -> std::string {
        if (HasAllocAttr(operand)) {
            ALOG_INFO_F("operand has an alloc attr, need to gen extra alloc\n%s", operand->Dump().c_str());
            std::optional<std::string> allocCodeMaybe = GenExtraAlloc(symbolMgr, operand, op);
            if (allocCodeMaybe.has_value()) {
                return allocCodeMaybe.value();
            }
        }
        return "";
    };
    for (const std::shared_ptr<LogicalTensor> &operand : op.GetIOperands()) {
        symbolMgr.AddToTensorMap(operand->GetMagic(), operand);
        PrintOperand("IOperand", operand);
        allocSourceCode += genExtraAllocForTensor(operand);
    }
    for (const std::shared_ptr<LogicalTensor> &operand : op.GetOOperands()) {
        symbolMgr.AddToTensorMap(operand->GetMagic(), operand);
        PrintOperand("OOperand", operand);
        allocSourceCode += genExtraAllocForTensor(operand);
    }

    return allocSourceCode;
}

// GET_PARAM_OFFSET_BY_IDX(param, n, base, dim, idx)
// GET_PARAM_VALID_SHAPE_BY_IDX(param, n, base, dim, idx)
std::string CodeGenCloudNPU::GenDynParamForExpr(const Function &func) const {
    unsigned isSupportUnaligned = ConfigManager::Instance().GetCodeGenConfig(KEY_SUPPORT_DYNAMIC_UNALIGNED, false);
    if (!isSupportUnaligned) {
        return {};
    }
    std::string dynParamList;
    for (const auto &dynParam : func.GetDynParamTable()) {
        std::string dynParamExpr = "uint64_t " + dynParam.first + " = ";
        DynParamInfo info = dynParam.second;
        if (info.dim.IsValid()) {
            dynParamExpr += SymbolicExpressionTable::BuildExpression(info.dim) + "; //";
        }
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

std::string CodeGenCloudNPU::GetParamType(const Function &func) const {
    if (isUnderDynamicFunction_) {
        return GM_PARAM_TYPE_FOR_DYN;
    }
    return func.GetFunctionType() == FunctionType::DYNAMIC_LOOP_PATH ? GM_PARAM_TYPE_FOR_DYN : GM_PARAM_TYPE_FOR_STATIC;
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
            if (HandleForAICpuSubFunc(*subFunc)) {
                return;
            }
            isUnderDynamicFunction_ = subFunc->IsUnderDynamicFunction();
            bool isCube = IsCube(subFunc->Operations());
            CompileInfo compileInfo(topFunc, ctx.cceDir, subFuncPair.first, isCube, isUnderDynamicFunction_);
            VFCodeGen vfCodeGen;
            vfCodeGen.GenCode(subFunc, compileInfo.GetVFHeaderAbsPath());
            std::ostringstream leafKernelFunc;
            leafKernelFunc << GenFuncBodyBefore(subFuncPair, topFunc, vfCodeGen, compileInfo);
            leafKernelFunc << GenFuncBody(*subFunc, topFunc);
            leafKernelFunc << GenFuncEnd();
            DumpCCE(compileInfo.GetCCEAbsPath(), leafKernelFunc.str());
            DoCompileCCE(compileInfo, "");
            UpdateSubFunc(subFuncPair, compileInfo);
        };
        tasks.push_back(task);
    }
    unsigned threadNum = ConfigManager::Instance().GetCodeGenConfig(KEY_PARALLEL_THREAD_NUM, 1u);
    ParallelExecuteAndWait(threadNum, tasks);
}

void CodeGenCloudNPU::UpdateSubFunc(std::pair<uint64_t, Function *> subFuncPair, const CompileInfo &compileInfo) const {
    auto subFunc = subFuncPair.second;
    std::shared_ptr<LeafFuncAttribute> attr = std::make_shared<LeafFuncAttribute>();
    attr->kernelName = compileInfo.GetKernelName();
    attr->binPath = compileInfo.GetBinAbsPath();
    attr->kernelDeclare = compileInfo.GetFuncDeclare();
    CoreType coreType = compileInfo.IsCube() ? CoreType::AIC : CoreType::AIV;
    attr->coreType = coreType;
    subFunc->SetLeafFuncAttribute(attr);
}

bool CodeGenCloudNPU::IsNeedDumpCCE(const std::string &inputFile) const {
    if (ConfigManager::Instance().GetCodeGenConfig(KEY_CODEGEN_FORCE_DUMP_CCE_ON_EXIST, true)) {
        // force dump, default is true
        return true;
    }
    // not force dump
    if (FileExist(inputFile)) {
        return false;
    }
    return true;
}

void CodeGenCloudNPU::DumpCCE(const std::string &fileName, const std::string &code) const {
    if (!IsNeedDumpCCE(fileName)) {
        return;
    }

    std::ofstream file;
    file.open(fileName);
    file << code;
    bool ret = !file.fail();
    ASSERT(ret) << "Dump cce code failed!!";
}

std::optional<std::string> CodeGenCloudNPU::GenExtraAlloc(
    SymbolManager &symbolMgr, const std::shared_ptr<LogicalTensor> &tensor, const Operation &op) const {
    const auto &memMap = tensor->memorymap;
    if (memMap.find(tensor->subGraphID) == memMap.end()) {
        ALOG_ERROR_F("%s: can not find subgGraphID(%d) in the memorymap of op:", __FUNCTION__, tensor->subGraphID);
        ALOG_ERROR_F("    %s", op.Dump().c_str());
        return std::nullopt;
    }

    auto memType = tensor->GetMemoryTypeOriginal();
    if (OPERAND_TYPE_TO_MEMORY_TYPE.find(memType) == OPERAND_TYPE_TO_MEMORY_TYPE.end()) {
        ALOG_ERROR_F("%s: invalid memory type(%d) of tensor tensor: ", __FUNCTION__, static_cast<size_t>(memType));
        ALOG_ERROR_F("    %s", tensor->Dump().c_str());
        return std::nullopt;
    }

    const TileRange &range = memMap.at(tensor->subGraphID);
    auto bufferType = OPERAND_TYPE_TO_MEMORY_TYPE.at(memType);

    return GenAlloc(symbolMgr, bufferType, tensor->Datatype(), range);
}

std::string GenAllocVarName(const std::string &prefix, const TileRange &range) {
    std::ostringstream ss;
    ss << prefix
       // range start/end are always positive
       << "_S" << range.start << "_E" << range.end;

    return ss.str();
}

std::string CodeGenCloudNPU::GenAlloc(
    SymbolManager &manager, BufferType bufferType, DataType dataType, const TileRange &range) const {
    if ((BUFFER_TYPE_TO_PREFIX.count(bufferType) == 0) || (OPERAND_TYPE_TO_ADDR_TYPE.count(bufferType) == 0)) {
        ALOG_ERROR_F("%s: invalid bufferType: %d", __FUNCTION__, static_cast<size_t>(bufferType));
        ASSERT(false);
        return "";
    }

    const std::string prefix = BUFFER_TYPE_TO_PREFIX.at(bufferType);
    const std::string &addrSpaceQualifier = OPERAND_TYPE_TO_ADDR_TYPE.at(bufferType);
    std::string allocVarName = GenAllocVarName(prefix, range);

    // must conform to CodeGenOpCloudNPU::createAllocKey
    AllocKey key = AllocKey(bufferType, range.start, range.end);
    bool reuse = manager.BindAddrWithVariableName(key, allocVarName);
    if (reuse) {
        return "";
    }

    ALOG_INFO_F(
        "%s: bind key to name: %s->%s", __FUNCTION__, manager.FormatAllocKey(key).c_str(), allocVarName.c_str());

    std::string dataTypeStr = DataType2CCEStr(dataType);

    std::ostringstream oss;
    oss << dataTypeStr << " " << addrSpaceQualifier << " *" << allocVarName << " = (" << dataTypeStr << " "
        << addrSpaceQualifier << " *)get_imm(0x" << std::hex << static_cast<unsigned>(range.start) << "); // size: 0x"
        << std::hex << static_cast<unsigned>(range.Size()) << "\n";

    return oss.str();
}

int CheckInjectStr(const char cmdStr[], size_t strLen) {
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

void CodeGenCloudNPU::DoCompileCCE(const CompileInfo &compileInfo, const std::string &compileOptions) const {
    if (!compileInfo.IsNeedCompileCCE()) {
        return;
    }
    int errCode = CompileCCE(compileInfo, compileOptions);
    ASSERT(errCode == 0) << "CompileCCE failed. errCode = " << errCode << ", cce file: " << compileInfo.GetCCEAbsPath();
}

int CodeGenCloudNPU::CompileCCE(const CompileInfo &compileInfo, const std::string &compileOptions) const {
    const std::string srcFile = compileInfo.GetCCEAbsPath();
    const std::string objFile = compileInfo.GetBinAbsPath();

    std::string coreType = compileInfo.IsCube() ? "dav-c220-cube" : "dav-c220-vec";
    std::string includePath = ctx.IsIncludePathEmpty() ? SRC_PATH : ctx.includePath;

    std::ostringstream oss;
    oss << "ccec " << compileOptions << " -c -O3 -g -x cce -std=c++17 "
        << "--cce-aicore-only "
        << "--cce-aicore-arch=" << coreType << " "
        << "-mllvm -cce-aicore-stack-size=0x8000 "
        << "-mllvm -cce-aicore-function-stack-size=0x8000 "
        << "-mllvm -cce-aicore-record-overflow=false "
        << "-mllvm -cce-aicore-addr-transform "
        << "-mllvm -cce-aicore-dcci-insert-for-scalar=false "
        << "-I" << includePath << "/include/tileop/a2a3 "
        << "-I" << includePath << "/src/machine/kernel "
        << "-I" << includePath << "/src/ "
        << "-I" << includePath << "/include/tilefwk "
        << "-I" << includePath << "/include/ "
        << "-o " << objFile << " " << srcFile;

    std::string ccecCmd = oss.str();

    ALOG_INFO_F("compile kernel...\n%s", ccecCmd.c_str());

    int ret = CheckInjectStr(ccecCmd.c_str(), ccecCmd.length());
    ASSERT(ret == 0) << "CheckInjectStr failed. errCode = " << ret;

    ret = std::system(ccecCmd.c_str());
    if (ret != 0) {
        ALOG_INFO_F("CompileCce ccec failed %d", ret);
    }
    return ret;
}

bool CodeGenCloudNPU::HandleForAICpuSubFunc(Function &subFunc) {
    if (!subFunc.IsAicpuSubFunction()) {
        return false;
    }

    std::shared_ptr<LeafFuncAttribute> attr = std::make_shared<LeafFuncAttribute>();
    attr->coreType = CoreType::AICPU;
    subFunc.SetLeafFuncAttribute(attr);
    return true;
}

} // namespace npu::tile_fwk
