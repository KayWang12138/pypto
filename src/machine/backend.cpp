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
 * \file backend.cpp
 * \brief
 */

#include "backend.h"
#include "interface/operation/operation.h"
#include "machine/host/machine_agent.h"
#include "machine/dump/kernel_dump_utils.h"
#include "machine/host/machine_compiler.h"
#include "machine/cache_manager/cache_manager.h"
#include "machine/utils/dynamic/dev_encode.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/program/program.h"
#include "interface/configs/config_manager.h"
#include "codegen/codegen.h"
#include "interface/utils/common.h"
#include "interface/utils/file_utils.h"

using namespace npu::tile_fwk::dynamic;
namespace npu::tile_fwk {
static constexpr size_t TABSIZE = 2;
constexpr int ALIGN_SIZE_8 = 8;
extern "C" int32_t Initialize() {
    CacheManager::Instance().Initialize();
    return 0;
}

extern "C" bool MatchCache(const std::string &cacheKey) {
    return CacheManager::Instance().MatchBinCache(cacheKey);
}

extern "C" int32_t Execute(MachineTask *task, FunctionCache &cache) {
    if (config::GetPlatformConfig(KEY_ONLY_HOST_COMPILE, false)) {
        ALOG_INFO("draw graph switch enabled, push finish queue.");
        return 0;
    }
    auto deviceAgentTask = std::make_unique<DeviceAgentTask>(task);
    auto function = deviceAgentTask->compileTask->GetFunction();
    deviceAgentTask->SetAsync(false);
    deviceAgentTask->SetOpOriginArgsInfo(function->GetOpOriginArgsInfo());
    deviceAgentTask->compileInfo.distTilingManager = function->GetDistTilingManager();
    deviceAgentTask->compileInfo.commGroups = Program::GetInstance().GetCommGroupRecorder().Output();
    // recover task info and bin
    if (task->GetCacheReuseType() == CacheReuseType::Bin) {
        if (!CacheManager::Instance().RecoverTask(task->GetCacheKey(), deviceAgentTask.get())) {
            ALOG_WARN_F("Fail to recover task from cache[%s].", task->GetCacheKey().c_str());
            return 0;
        }
    } else {
        if(function->GetRootFunction()) {
            /* calc workspace size and every sub function invoke entry para offset */
            CalcFunctionInvokeWorkespace(nullptr, function, deviceAgentTask->compileInfo);
        }

        deviceAgentTask->compileInfo.PrintDistributed();
        deviceAgentTask->compileInfo.workSpaceStackSize = function->GetStackWorkespaceSize();

        (void)GenCode(deviceAgentTask->compileTask, deviceAgentTask->compileInfo.invokeParaOffset, cache);
        function = deviceAgentTask->compileTask->GetFunction();
        /* finish compile add function cache */
        cache.Insert(function->GetFunctionHash(), *function);
        deviceAgentTask->SetFunctionCache(cache.Get(function->GetFunctionHash()));
        if (function->IsFunctionType(FunctionType::STATIC)) {
            deviceAgentTask->Validate();
            deviceAgentTask->UpdateCompileInfo();
        }
        // save compile result on disk
        CacheManager::Instance().SaveTaskFile(deviceAgentTask.get());
    }

    if (config::GetHostConfig(KEY_DUMP_BIN_AND_JSON, false)) {
        if (!KernelDumpUtils::DumpKernelFile(deviceAgentTask.get(),
                    config::GetHostConfig(KEY_DUMP_KERNEL_NAME, ""),
                    config::GetHostConfig(KEY_DUMP_BIN_AND_JSON_PATH, ""))) {
            ALOG_ERROR_F("Dump ast bin failed");
        }
    }

    if (config::GetHostConfig(KEY_ONLY_CODEGEN, false)) {
        ALOG_INFO("only gen code switch enabled, push finish queue.");
        return 0;
    }

    MachineAgent agent;
    agent.AgentProc(deviceAgentTask.get());
#ifndef AC_ENABLE_FRAMEWORK_WITHOUT_CANN
    rtSetDevice(npu::tile_fwk::stubs::DeviceStub::GetCurrentDeviceId());
#endif
    MachinePipe piple;
    piple.PipeProc(deviceAgentTask.get());
    return 0;
}

static std::vector<Function *> GetCalleeList(FunctionCache &cache, Function *func) {
    std::vector<Function *> calleeList;

    std::vector<std::shared_ptr<CallOpAttribute>> callopAttrList = func->GetCallopAttrList();
    for (auto &callopAttr : callopAttrList) {
        auto hash = callopAttr->GetCalleeHash();
        Function *cacheFunction = cache.GetCacheFunction(hash);
        if (cacheFunction != nullptr) {
            calleeList.push_back(cacheFunction);
        } else {
            ALOG_ERROR_F("Cannot find cache %lu", hash.GetHash());
        }
    }
    return calleeList;
}

static void FindAllExpression(FunctionCache &cache, Linker &linker, Function *func) {
    if (func->IsDynloop()) {
        auto dynloopAttr = func->GetDynloopAttribute();
        auto ss = SymbolicScalar(dynloopAttr->iterSymbolName);
        linker.AddSymbol(func, ss);
    }
    if (func->IsFunctionTypeAndGraphType({FunctionType::DYNAMIC, FunctionType::DYNAMIC_LOOP, FunctionType::DYNAMIC_LOOP_PATH}, GraphType::TENSOR_GRAPH)) {
        ALOG_INFO("Compile control:", func->Dump());
        for (auto &callee : GetCalleeList(cache, func)) {
            FindAllExpression(cache, linker, callee);
        }
    } else if (func->GetGraphType() == GraphType::TILE_GRAPH) {
        ALOG_INFO("Compile tile:", func->Dump());
        Function *root = func->GetRootFunction();
        FindAllExpression(cache, linker, root);
    } else if (func->GetGraphType() == GraphType::ROOT_GRAPH) {
        ALOG_INFO("Compile root:", func->Dump());
        linker.AddFunction(func);
        for (auto &callopAttr : func->GetCallopAttrList()) {
            for (auto &arg : callopAttr->GetLinearArgList()) {
                linker.AddExpression(func, arg);
            }
            auto hash = callopAttr->GetCalleeHash();
            Function *leafFunc = cache.GetCacheFunction(hash);
            FindAllExpression(cache, linker, leafFunc);
        }
    } else if (func->GetGraphType() == GraphType::LEAF_GRAPH) {
    } else {
        ASSERT(false) << "Impossible function type: " << GetFunctionTypeNameDict().Find(func->GetFunctionType());
    }
}

static void AlignUpTo(std::vector<uint8_t> &code, int align, uint8_t padding) {
    while (code.size() % align != 0) {
        code.push_back(padding);
    }
}

static void SimplifySlots(IncastOutcastLink &inoutLink, DyndevFunctionAttribute *attr,
                          std::unordered_map<Function *, Function *> &rootTileDict) {
    std::vector<bool> slotUsed(inoutLink.totalSlot);

    for (Function *devRoot : attr->devRootList) {
        Function *devTile = rootTileDict[devRoot];

        ASSERT(inoutLink.ioslotDict.count(devTile));
        IncastOutcastSlot &ioslot = inoutLink.ioslotDict[devTile];

        for (auto &incastSlots : ioslot.incastSlot) {
            ASSERT(!incastSlots.empty()) << "devTile: " << devTile->GetMagicName();
            incastSlots.resize(1); // meaningless to maintain multi incast slots
            slotUsed[incastSlots.front()] = true;
        }
    }

    for (int slotIdx : inoutLink.inputSlotIndexList) {
        slotUsed[slotIdx] = true;
    }

    for (int slotIdx : inoutLink.outputSlotIndexList) {
        slotUsed[slotIdx] = true;
    }

    for (Function *devRoot : attr->devRootList) {
        Function *devTile = rootTileDict[devRoot];

        ASSERT(inoutLink.ioslotDict.count(devTile));
        IncastOutcastSlot &ioslot = inoutLink.ioslotDict[devTile];

        for (auto &outcastSlots : ioslot.outcastSlot) {
            ASSERT(!outcastSlots.empty()) << "devTile: " << devTile->GetMagicName();
            bool outcastSlotFound = false;
            for (auto &outcastSlot : outcastSlots) {
                outcastSlotFound = outcastSlotFound || slotUsed[outcastSlot];                
            }
            if (!outcastSlotFound) {
                slotUsed[outcastSlots.front()] = true;
            }
        }
    }

    std::unordered_map<int, int> slotIdxMapping;
    for (int i = 0; i < inoutLink.totalSlot; i++) {
        if (slotUsed[i]) {
            slotIdxMapping.emplace(i, slotIdxMapping.size());
        }
    }

    auto replaceSlotIdx = [&slotIdxMapping](std::vector<int> &slots) {
        for (int &slot : slots) {
            slot = slotIdxMapping.count(slot) ? slotIdxMapping[slot] : -1;
        }
        slots.erase(std::remove(slots.begin(), slots.end(), -1), slots.end());
    };

    inoutLink.totalSlot = slotIdxMapping.size();
    for (Function *devRoot : attr->devRootList) {
        Function *devTile = rootTileDict[devRoot];

        ASSERT(inoutLink.ioslotDict.count(devTile));
        IncastOutcastSlot &ioslot = inoutLink.ioslotDict[devTile];

        for (auto &incastSlots : ioslot.incastSlot) {
            replaceSlotIdx(incastSlots);
        }

        for (auto &outcastSlots : ioslot.outcastSlot) {
            replaceSlotIdx(outcastSlots);
        }
    }

    replaceSlotIdx(inoutLink.inputSlotIndexList);
    replaceSlotIdx(inoutLink.outputSlotIndexList);
    replaceSlotIdx(inoutLink.assembleSlotIndexList);

    for (auto &slot : inoutLink.inplaceSlotIndexList) {
        if (slot != -1)
            slot = slotIdxMapping[slot];
    }
}

static std::string BuildControlFlowCallee(Function *func) {
    std::ostringstream oss;
    oss << "#name:" << func->GetRawName() << " #hash:" << func->GetFunctionHash() << " #magic:" << func->GetFuncMagic();
    return oss.str();
}

static void BuildControlFlow(FunctionCache &cache, Linker &linker, const std::string &sectionName,
    Function *func, std::vector<Function *> &devRootList,
    std::unordered_map<Function *, Function *> &rootTileDict, std::ostringstream &oss, int indent,
    IntermediateVariableTable *curIvt = nullptr) {
    auto funcType = func->GetFunctionType();
    if (funcType == FunctionType::DYNAMIC) {
        IntermediateVariableTable ivt("intermediateVariable");
        oss << "#include <stdint.h>\n"
            << "#include \"machine/utils/dynamic/codegen/codegen.h\"\n"
            << linker.symbolTable.BuildSymbolList();
        const std::vector<std::string> &inputNameList = Program::GetInstance().GetTensorSlotManager()->GetInputNameList();
        const std::vector<std::string> &outputNameList = Program::GetInstance().GetTensorSlotManager()->GetOutputNameList();
        for (size_t i = 0; i < inputNameList.size(); i++) {
            oss << "#define " << AddArgPrefix(inputNameList[i]) << " " << i << "\n";
        }
        for (size_t i = 0; i < outputNameList.size(); i++) {
            oss << "#define " << AddArgPrefix(outputNameList[i]) << " " << i << "\n";
        }
        oss << "#define LOOP(idx, b, e, s) for (uint64_t idx = (b), idxEnd = (e), idxStep = (s); idx < idxEnd; idx += idxStep)\n"
            << "namespace npu::tile_fwk {\n"
            << "// " << BuildControlFlowCallee(func) << "\n"
            << "__attribute__((section(\"" << sectionName
            << "\")))\n"
            << "uint64_t ControlFlowEntry(void *ctx, uint64_t *symbolTable, CallRootEntryType callRootList[3], DevStartArgsBase *startArgs) {\n";
        for (auto &callee : GetCalleeList(cache, func)) {
            BuildControlFlow(cache, linker, sectionName, callee, devRootList, rootTileDict, oss, indent + 1, &ivt);
        }
        oss << std::setw((indent + 1) * TABSIZE) << ' ' << "callRootList[CallRootStage::T_CALLROOT_STITCH](ctx, RUNTIME_FINISH_FUNCKEY); // Notify finish \n";
        oss << std::setw((indent + 1) * TABSIZE) << ' ' << "return 0;\n";
        oss << "}\n";
        oss << "} // namespace npu::tile_fwk\n";
    } else if (func->IsFunctionTypeAndGraphType(FunctionType::DYNAMIC_LOOP, GraphType::TENSOR_GRAPH)) {
        ASSERT(curIvt != nullptr);
        std::vector<IntermediateVariableTable::IntermediateVariableInfo> intermediateVariables;
        std::function<void(const std::shared_ptr<DynloopFunctionPathNode> &)> intermediateVariableFind =
            [&linker, &curIvt, &intermediateVariables, &intermediateVariableFind](const std::shared_ptr<DynloopFunctionPathNode> &node){
            if (!node->cond.IsValid()) {
                if (node->root->GetGraphType() == GraphType::TILE_GRAPH) {
                    auto curFunc = node->root->GetRootFunction();
                    ASSERT(curFunc != nullptr);
                    ASSERT(linker.rootExpressionTableDict.count(curFunc));
                    SymbolicExpressionTable &table = linker.rootExpressionTableDict[curFunc];
                    for (auto &[expr, rawSS] : table.expressionIndexTable) {
                        (void)expr;
                        auto result = curIvt->GetAllIntermediateVariables(*rawSS.first);
                        intermediateVariables.insert(intermediateVariables.end(), result.begin(), result.end());
                    }
                }
                return;
            }
            auto result = curIvt->GetAllIntermediateVariables(*node->cond.Raw());
            intermediateVariables.insert(intermediateVariables.end(), result.begin(), result.end());
            if (node->branchNodeList[1] != nullptr) {
                if (node->branchNodeList[0] != nullptr) {
                    intermediateVariableFind(node->branchNodeList[1]);
                    intermediateVariableFind(node->branchNodeList[0]);
                } else {
                    intermediateVariableFind(node->branchNodeList[1]);
                }
            } else {
                if (node->branchNodeList[0] != nullptr) {
                    intermediateVariableFind(node->branchNodeList[0]);
                } else {
                    ASSERT(false) << "Both conds is nullptr!";
                }
            }
        };

        std::function<void(const std::shared_ptr<DynloopFunctionPathNode> &, int)> condBuilder =
            [&cache, &linker, &sectionName, &devRootList, &rootTileDict, &oss, &condBuilder, &curIvt](
                const std::shared_ptr<DynloopFunctionPathNode> &node, int condIndent) {
                if (!node->cond.IsValid()) {
                    BuildControlFlow(
                        cache, linker, sectionName, node->root, devRootList, rootTileDict, oss, condIndent, curIvt);
                } else {
                    std::string cond = curIvt->BuildExpression(node->cond);
                    if (node->branchNodeList[1] != nullptr) {
                        if (node->branchNodeList[0] != nullptr) {
                            oss << std::setw(condIndent * TABSIZE) << ' ' << "if (" << cond << ") {" << "\n";
                            condBuilder(node->branchNodeList[1], condIndent + 1);
                            oss << std::setw(condIndent * TABSIZE) << ' ' << "} else {" << "\n";
                            condBuilder(node->branchNodeList[0], condIndent + 1);
                            oss << std::setw(condIndent * TABSIZE) << ' ' << "}" << "\n";
                        } else {
                            condBuilder(node->branchNodeList[1], condIndent);
                        }
                    } else {
                        if (node->branchNodeList[0] != nullptr) {
                            condBuilder(node->branchNodeList[0], condIndent);
                        } else {
                            ASSERT(false) << "Both conds is nullptr!";
                        }
                    }
                }
            };
        auto attr = func->GetDynloopAttribute();
        ASSERT(attr != nullptr);
        auto result1 = curIvt->GetAllIntermediateVariables(*attr->Begin().Raw());
        auto result2 = curIvt->GetAllIntermediateVariables(*attr->End().Raw());
        intermediateVariables.insert(intermediateVariables.end(), result1.begin(), result1.end());
        intermediateVariables.insert(intermediateVariables.end(), result2.begin(), result2.end());
        std::string iterBegin = curIvt->BuildExpression(attr->Begin());
        std::string iterEnd = curIvt->BuildExpression(attr->End());
        std::string iterStep = SymbolicExpressionTable::BuildExpression(attr->Step());
        if (attr->submitBeforeLoop) {
            oss << std::setw(indent * TABSIZE) << ' ' << "callRootList[CallRootStage::T_CALLROOT_STITCH](ctx, RUNTIME_FINISH_FUNCKEY); // force submit before LOOP \n";
        }
        oss << std::setw(indent * TABSIZE) << ' ' << "// hash=" << func->GetFunctionHash() << "\n";
        // Outer intermediate variables
        for (const auto &[name, expression] : intermediateVariables) {
            oss << std::setw(indent * TABSIZE) << ' ' << "uint64_t " << name << " = " << expression << ";" << std::endl;
        }
        intermediateVariables.clear();
        oss << std::setw(indent * TABSIZE) << ' ' << "LOOP(INDEX, " << iterBegin << ", " << iterEnd << ", " << iterStep << ") {\n";
        oss << std::setw((indent + 1) * TABSIZE) << ' ' << "VALUE_" << attr->iterSymbolName << " = INDEX;\n";

        auto pathNode = attr->BuildPathNode();
        ALOG_INFO("Paths: \n", pathNode->Dump());
        // Inner intermediate variables
        auto curIvtCheckPoint = *curIvt;
        intermediateVariableFind(pathNode);
        for (const auto &[name, expression] : intermediateVariables) {
            oss << std::setw((indent + 1) * TABSIZE) << ' ' << "uint64_t " << name << " = " << expression << ";" << std::endl;
        }
        std::vector<Function *> calleeList = GetCalleeList(cache, func);
        std::sort(calleeList.begin(), calleeList.end());

        std::vector<Function *> pathRootList;
        for (size_t i = 0; i < attr->pathList.size(); i++) {
            pathRootList.push_back(attr->pathList[i].root);
        }
        std::sort(pathRootList.begin(), pathRootList.end());

        ASSERT(calleeList == pathRootList);
        condBuilder(pathNode, indent + 1);
        oss << std::setw(indent * TABSIZE) << ' ' << "}\n";
        *curIvt = curIvtCheckPoint;
    } else if (func->IsFunctionTypeAndGraphType(FunctionType::DYNAMIC_LOOP_PATH, GraphType::TENSOR_GRAPH)) {
        oss << std::setw(indent * TABSIZE) << ' ' << "// " << BuildControlFlowCallee(func) << "\n";
        auto curIvtCheckPoint = *curIvt;
        for (auto &callee : GetCalleeList(cache, func)) {
            BuildControlFlow(cache, linker, sectionName, callee, devRootList, rootTileDict, oss, indent + 1, curIvt);
        }
        *curIvt = curIvtCheckPoint;
    } else if (func->GetGraphType() == GraphType::TILE_GRAPH) {
        oss << std::setw(indent * TABSIZE) << ' ' << "// " << BuildControlFlowCallee(func) << "\n";
        Function *root = func->GetRootFunction();
        rootTileDict[root] = func;
        BuildControlFlow(cache, linker, sectionName, root, devRootList, rootTileDict, oss, indent, curIvt);
    } else if (func->GetGraphType() == GraphType::ROOT_GRAPH) {
        int key = devRootList.size();
        oss << std::setw(indent * TABSIZE) << ' ' << "// " << BuildControlFlowCallee(func) << "\n";
        oss << std::setw(indent * TABSIZE) << ' ' << "uint64_t *exprList" << key << " = (uint64_t *)callRootList[CallRootStage::T_CALLROOT_ALLOC](ctx, " << key << "ULL);\n";
        if (linker.rootExpressionTableDict.count(func) != 0) {
            SymbolicExpressionTable &table = linker.rootExpressionTableDict[func];
            for (auto &[expr, rawSS] : table.expressionIndexTable) {
                (void)expr;
                oss << std::setw(indent * TABSIZE) << ' ' << "exprList" << key << "[" << rawSS.second << "] = " << curIvt->BuildExpression(SymbolicScalar(rawSS.first)) << ";\n";
            }
        }

        oss << std::setw(indent * TABSIZE) << ' ' << "callRootList[CallRootStage::T_CALLROOT_STITCH](ctx, " << key << "ULL);\n";
        devRootList.push_back(func);
    } else {
        ASSERT(false) << "Impossible function type: " << GetFunctionTypeNameDict().Find(funcType);
    }
}

static std::string Arm64TargetTool(const std::string &bin) {
    // ARM arch compiler
#ifdef ASCEND_CANN_ROOT_PATH
    return std::string(ASCEND_CANN_ROOT_PATH) + "/toolkit/toolchain/hcc/bin/aarch64-target-linux-gnu-" + bin;
#else
    (void) bin;
    ASSERT(false) << "CANN environment not found";
    return "";
#endif // ifdef ASCEND_CANN_ROOT_PATH
}

static void FillL2PrefetchInfo(std::shared_ptr<DyndevFunctionAttribute> attr) {
    uint64_t idx = 0;
    for (auto &param : attr->startArgsInputTensorList) {
        const auto &tensor = param.get();
        auto asc_tensor = tensor.GetStorage();
        if (asc_tensor != nullptr && tensor->NeedPrefetch()) {
          attr->l2InfoList.emplace_back(L2Info(tensor->MemorySize(), idx));
        }
        idx++;
    }
    ALOG_INFO_F("Need prefetch tensor size is:%zu.\n", attr->l2InfoList.size());
    return;
}

static void SetDyndevProgBinary(Function *function) {
    if (function == nullptr || function->GetDyndevAttribute() == nullptr) {
        return;
    }
    std::shared_ptr<DyndevFunctionAttribute> dynAttrPtr = function->GetDyndevAttribute();
    uint64_t size = 0;
    dynamic::EncodeDevAscendProgram(function, size, nullptr);
    dynAttrPtr->devProgBinary.resize(size);

    dynamic::DevAscendProgram *devProg = reinterpret_cast<dynamic::DevAscendProgram *>(&dynAttrPtr->devProgBinary[0]);
    dynamic::EncodeDevAscendProgram(function, size, devProg);

    devProg->Reloc(-reinterpret_cast<int64_t>(devProg));
    ALOG_INFO_F("Dev prog binary size is:%zu.\n", dynAttrPtr->devProgBinary.size());
}

static void CompileDyndevFunction(Function *function, FunctionCache &cache) {
    Linker linker;
    FindAllExpression(cache, linker, function);
    linker.BuildAndLoad();

    std::shared_ptr<DyndevFunctionAttribute> attr = function->GetDyndevAttribute();
    ASSERT(attr != nullptr);
    FillL2PrefetchInfo(attr);
    attr->symbolTable = linker.symbolTable;
    attr->rootExpressionTableDict = linker.rootExpressionTableDict;

    ALOG_INFO("SymbolTable: ", linker.symbolTable.Dump());
    for (auto &[func, exprTable] : linker.rootExpressionTableDict) {
        ALOG_INFO("ExpressionTable: name:", func->GetMagicName(), "\n", exprTable.Dump());
    }

    auto slotManager = Program::GetInstance().GetTensorSlotManager();
    attr->inoutLink = slotManager->BuildIncastOutcastLink(function->GetRawName());

    std::ostringstream oss;
    std::unordered_map<Function *, Function *> rootTileDict;
    BuildControlFlow(cache, linker, "ast2", function, attr->devRootList, rootTileDict, oss, 0);
    std::string controlFlowSource = oss.str();

#ifdef __x86_64__
    std::string cflags = "-mno-sse2 -mno-sse";
#else
    std::string cflags = "";
#endif
    attr->hostControlFlowBinary = CompileAndLoadSection(
        controlFlowSource, config::LogTopFolder() + "/controlFlow_host.cpp",
        "g++", "objcopy", "ast2", cflags);
    AlignUpTo(attr->hostControlFlowBinary, 0x8, 0);
#ifdef ASCEND_CANN_ROOT_PATH
    if (ToolchainExist(Arm64TargetTool("g++"))) {
        attr->devControlFlowBinary = CompileAndLoadSection(
            controlFlowSource, config::LogTopFolder() + "/controlFlow_dev.cpp",
            Arm64TargetTool("g++"), Arm64TargetTool("objcopy"), "ast2");
    } else {
        // brk #0
        attr->devControlFlowBinary = std::vector<uint8_t>{0xd4, 0x20, 0x00, 0x00};
    }
    AlignUpTo(attr->devControlFlowBinary, 0x8, 0);
#else
    (void) Arm64TargetTool;
    attr->devControlFlowBinary = attr->hostControlFlowBinary;
#endif // ifdef ASCEND_CANN_ROOT_PATH

    std::map<std::string, Function *> leafDict;
    for (size_t i = 0; i < attr->devRootList.size(); i++) {
        Function *devRoot = attr->devRootList[i];
        for (auto &[hash, leaf] : devRoot->programs_) {
            (void) hash;
            if (!leafDict.count(leaf->GetRawName())) {
                leafDict[leaf->GetRawName()] = leaf;
                ALOG_INFO("Dyndev.codegen: ", leaf->GetRawName());
            }
        }
    }
    attr->cceCodeList.resize(leafDict.size() + 1);
    attr->cceCodeInfo.resize(leafDict.size() + 1);

    struct EncodeDevAscendFunctionParam encodeDevAscendFunctionParam = {};

    /* cceIdx 0 for dummy callop */
    attr->cceCodeInfo[0].coreType = static_cast<uint32_t>(CoreType::HUB);
    attr->cceCodeInfo[0].psgId = 0;
    attr->cceCodeInfo[0].funcHash = 0;
    encodeDevAscendFunctionParam.calleeHashIndexDict[0] = 0;

    int leafIndex = 1;
    for (auto &[name, leaf] : leafDict) {
        attr->cceCodeList[leafIndex] = LoadBinData(leaf->GetBinPath());
        AlignUpTo(attr->cceCodeList[leafIndex], ALIGN_SIZE_8, 0);
        encodeDevAscendFunctionParam.calleeHashIndexDict[leaf->ComputeHash().GetHash()] = leafIndex;
        attr->devLeafIndex2Hash[leafIndex] = leaf->GetFunctionHash().GetHash();
        ALOG_INFO("Dyndev.codegen: [", leafIndex, "] hash=", leaf->ComputeHash(), " name=", name,
            " binpath=", leaf->GetBinPath());
        attr->cceCodeInfo[leafIndex].coreType = static_cast<uint32_t>(leaf->GetCoreType());
        if (leaf->IsDummyFunction())
            attr->cceCodeInfo[leafIndex].coreType = static_cast<uint32_t>(CoreType::HUB);
        attr->cceCodeInfo[leafIndex].psgId = leaf->GetProgramId();
        attr->cceCodeInfo[leafIndex].funcHash = leaf->GetFunctionHash().GetHash();
        leafIndex++;
    }
    encodeDevAscendFunctionParam.cceCodeInfoList = attr->cceCodeInfo;
    encodeDevAscendFunctionParam.inoutLink = &attr->inoutLink;

    SimplifySlots(attr->inoutLink, attr.get(), rootTileDict);
    attr->devEncodeList.resize(attr->devRootList.size());
    for (size_t i = 0; i < attr->devRootList.size(); i++) {
        Function *devRoot = attr->devRootList[i];
        ALOG_INFO("Dyndev.encode: ", devRoot->GetRawName());

        ASSERT(rootTileDict.count(devRoot));
        Function *devTile = rootTileDict[devRoot];

        ASSERT(attr->inoutLink.ioslotDict.count(devTile));
        IncastOutcastSlot *slot = &attr->inoutLink.ioslotDict[devTile];

        encodeDevAscendFunctionParam.symbolTable = &linker.symbolTable;
        if (linker.rootExpressionTableDict.count(devRoot) != 0) {
            encodeDevAscendFunctionParam.expressionTable = &linker.rootExpressionTableDict[devRoot];
        }
        encodeDevAscendFunctionParam.devRoot = devRoot;
        encodeDevAscendFunctionParam.slot = slot;

        uint64_t size = 0;
        EncodeDevAscendFunction(encodeDevAscendFunctionParam, size, nullptr);

        attr->devEncodeList[i].resize(size);
        DevAscendFunction *funcBin = reinterpret_cast<DevAscendFunction *>(&attr->devEncodeList[i][0]);
        funcBin->funcKey = i;
        funcBin->stackWorkSpaceSize = devTile->GetStackWorkespaceSize();
        EncodeDevAscendFunction(encodeDevAscendFunctionParam, size, funcBin);
        funcBin->Reloc(-reinterpret_cast<int64_t>(funcBin), true);
    }

    for (auto &[name, index] : attr->symbolTable.symbolIndexTable) {
        if (symbolHandlerIndexDict.count(name)) {
            attr->startArgsSymbolHandlerList.emplace_back(symbolHandlerIndexDict.find(name)->second, index);
        }
    }

    // save dev prog binary
    SetDyndevProgBinary(function);
}

MachineTask *GenCode(
    MachineTask *task, const std::map<uint64_t, std::list<InvokeParaOffset>> &invokeParaOffset, FunctionCache &cache) {
    npu::tile_fwk::CodeGen codeGen;
    auto function = task->GetFunction();
    /* each leafFunction inside is compiled to a standalone object file.
     * the filepath of the object file is updated to the binPath_ member.
     */
    if (config::GetCodeGenConfig(npu::tile_fwk::KEY_CODEGEN_BY_JSON, false)) {
        std::string jsonPath = config::LogTopFolder() + "/program.json";
        Program::GetInstance().DumpJsonFile(jsonPath);
        codeGen.GenCode(jsonPath, invokeParaOffset);
        task->SetFunction(Program::GetInstance().GetCurrentFunction());
    } else if (function->GetGraphType() == GraphType::TILE_GRAPH) {
        codeGen.GenCode(*function, invokeParaOffset);
    } else {
        if (function->IsFunctionType(FunctionType::DYNAMIC)) {
            CompileDyndevFunction(function, cache);
        }
    }

    return task;
}
} // namespace npu::tile_fwk
