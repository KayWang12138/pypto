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

#include "machine/host/backend.h"
#include "tilefwk/tilefwk.h"
#include "codegen/codegen.h"
#include "interface/inner/tilefwk.h"
#include "interface/program/program.h"
#include "interface/operation/operation.h"
#include "interface/configs/config_manager.h"
#include "interface/utils/common.h"
#include "interface/utils/file_utils.h"
#include "machine/dump/kernel_dump_utils.h"
#include "machine/host/machine_compiler.h"
#include "machine/cache_manager/cache_manager.h"
#include "machine/utils/dynamic/dev_encode.h"
#include "machine/host/device_agent_task.h"
#include "kernel/aicore_compiler.h"
#include "interface/utils/op_info_manager.h"

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
    auto deviceAgentTask = std::make_shared<DeviceAgentTask>(task);
    auto function = deviceAgentTask->compileTask->GetFunction();
    deviceAgentTask->SetAsync(false);
    deviceAgentTask->SetOpOriginArgsInfo(function->GetOpOriginArgsInfo());
    deviceAgentTask->compileInfo.distTilingManager = function->GetDistTilingManager();
    deviceAgentTask->compileInfo.commGroups = Program::GetInstance().GetCommGroupRecorder().Output();
    std::string kernelPath;
    // recover task info and bin
    if (task->GetCacheReuseType() == CacheReuseType::Bin) {
        if (!CacheManager::Instance().RecoverTask(task->GetCacheKey(), deviceAgentTask.get())) {
            ALOG_WARN_F("Fail to recover task from cache[%s].", task->GetCacheKey().c_str());
            return 0;
        }
    } else {
        if(function->IsFunctionType(FunctionType::STATIC) && function->GetRootFunction()) {
            /* calc workspace size and every sub function invoke entry para offset */
            CalcFunctionInvokeWorkespace(nullptr, function, deviceAgentTask->compileInfo);
        }

        deviceAgentTask->compileInfo.PrintDistributed();
        deviceAgentTask->compileInfo.workSpaceStackSize = function->GetStackWorkespaceSize();

        if (npu::tile_fwk::ConfigManager::Instance().GetCodeGenConfig(npu::tile_fwk::KEY_CODEGEN_EXPRESSION_FUSION, false)) {
            if (function->GetGraphType() == GraphType::TILE_GRAPH) {
                // When expression fusion, don't need tile graph codegen.
                return 0;
            }
        }
        (void)GenCode(deviceAgentTask->compileTask, deviceAgentTask->compileInfo.invokeParaOffset, cache, kernelPath);
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
        if (!KernelDumpUtils::DumpKernelFile(deviceAgentTask.get(), config::GetHostConfig(KEY_DUMP_KERNEL_NAME, ""),
                    config::GetHostConfig(KEY_DUMP_BIN_AND_JSON_PATH, ""), kernelPath)) {
            ALOG_ERROR_F("Dump ast bin failed");
        }
    }

    if (config::GetHostConfig(KEY_ONLY_CODEGEN, false)) {
        ALOG_INFO("only gen code switch enabled, push finish queue.");
        return 0;
    }

    gDeviceAgentTaskPtr = deviceAgentTask;
    return 0;
}

static std::string GetEmitPath(const std::string &name) {
    std::string dirPath;
    if (npu::tile_fwk::ConfigManager::Instance().GetCodeGenConfig(KEY_CODEGEN_DUMP_TO_OUTPUT, true)) {
        dirPath = config::LogTopFolder() + "/" + name;
    } else {
        dirPath = name;
    }
    return dirPath;
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
        linker.AddSymbol(ss);
    }
    if (func->IsFunctionTypeAndGraphType({FunctionType::DYNAMIC, FunctionType::DYNAMIC_LOOP, FunctionType::DYNAMIC_LOOP_PATH}, GraphType::TENSOR_GRAPH)) {
        ALOG_INFO("Compile control:", func->Dump());
        for (auto &callee : GetCalleeList(cache, func)) {
            FindAllExpression(cache, linker, callee);
        }
        if (func->IsFunctionTypeAndGraphType(FunctionType::DYNAMIC_LOOP, GraphType::TENSOR_GRAPH)) {
            auto attr = func->GetDynloopAttribute();
            linker.AddPrimaryExpressionForLoopBes(func, attr->Begin());
            linker.AddPrimaryExpressionForLoopBes(func, attr->End());
            linker.AddPrimaryExpressionForLoopBes(func, attr->Step());

            for (auto &path : attr->GetPathList()) {
                for (auto &cond : path.GetPathCondList()) {
                    linker.AddPrimaryExpressionForLoopIf(func, cond.GetCond());
                }
            }
        }
    } else if (func->GetGraphType() == GraphType::TILE_GRAPH) {
        ALOG_INFO("Compile tile:", func->Dump());
        Function *root = func->GetRootFunction();
        FindAllExpression(cache, linker, root);
    } else if (func->GetGraphType() == GraphType::ROOT_GRAPH) {
        ALOG_INFO("Compile root:", func->Dump());
        for(auto outCast : func->GetOutcast()) {
            for (auto dynShapeValidShapeI : outCast->tensor->GetDynRawShape())
                linker.AddPrimaryExpressionForDevRootCoa(func, dynShapeValidShapeI);
        }
        for (auto &callopAttr : func->GetCallopAttrList()) {
            for (auto &arg : callopAttr->GetLinearArgList()) {
                linker.AddPrimaryExpressionForDevRootCoa(func, arg);
            }
            auto hash = callopAttr->GetCalleeHash();
            Function *leafFunc = cache.GetCacheFunction(hash);
            FindAllExpression(cache, linker, leafFunc);
        }
    } else if (func->GetGraphType() == GraphType::LEAF_GRAPH) {
        for (auto &op : func->Operations()) {
            if (op.GetOpcode() == Opcode::OP_VEC_DUP) {
                if (op.HasAttr(OpAttributeKey::dynScalar)) {
                    auto dynScalar = op.GetSymbolicScalarAttribute(OpAttributeKey::dynScalar);
                    linker.AddPrimaryExpressionForDevLeafOp(func, &op, dynScalar);
                }
            }
        }
    } else {
        ASSERT(false) << "Impossible function type: " << GetFunctionTypeNameDict().Find(func->GetFunctionType());
    }
}

static void AlignUpTo(std::vector<uint8_t> &code, int align, uint8_t padding) {
    while (code.size() % align != 0) {
        code.push_back(padding);
    }
}

static void ReplaceSlotIndex(DyndevFunctionAttribute *attr, std::vector<bool>& slotUsed) {
    IncastOutcastLink &inoutLink = attr->inoutLink;
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
    for (Function *devRoot : attr->funcGroup.devRootList) {
        Function *devTile = attr->rootTileDict[devRoot];

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
    replaceSlotIdx(inoutLink.partialUpdateSlotIdexList);
    for (auto &slot : inoutLink.inplaceSlotIndexList) {
        if (slot != -1)
            slot = slotIdxMapping[slot];
    }
}

static void SimplifySlots(DyndevFunctionAttribute *attr) {
    IncastOutcastLink &inoutLink = attr->inoutLink;
    std::vector<bool> slotUsed(inoutLink.totalSlot);

    for (Function *devRoot : attr->funcGroup.devRootList) {
        Function *devTile = attr->rootTileDict[devRoot];

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

    for (Function *devRoot : attr->funcGroup.devRootList) {
        Function *devTile = attr->rootTileDict[devRoot];

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

    ReplaceSlotIndex(attr, slotUsed);
}

static void BuildSlotRootIncastOutcastDict(DyndevFunctionAttribute *attr) {
    IncastOutcastLink &inoutLink = attr->inoutLink;
    for (size_t i = 0; i < attr->funcGroup.devRootList.size(); i++) {
        Function *devRoot = attr->funcGroup.devRootList[i];
        Function *devTile = attr->rootTileDict[devRoot];

        ASSERT(inoutLink.ioslotDict.count(devTile));
        IncastOutcastSlot &ioslot = inoutLink.ioslotDict[devTile];
        for (size_t incastIndex = 0; incastIndex < ioslot.incastSlot.size(); incastIndex++) {
            for (auto &slotIndex : ioslot.incastSlot[incastIndex]) {
                attr->slotRootIncastDict[slotIndex][devRoot] = incastIndex;
            }
        }
        for (size_t outcastIndex = 0; outcastIndex < ioslot.outcastSlot.size(); outcastIndex++) {
            for (auto &slotIndex : ioslot.outcastSlot[outcastIndex]) {
                attr->slotRootOutcastDict[slotIndex][devRoot] = outcastIndex;
            }
        }
    }
}

static void BuildRootFuncKeyDict(DyndevFunctionAttribute *attr) {
    for (size_t i = 0; i < attr->funcGroup.devRootList.size(); i++) {
        int funcKey = (int)i;
        Function *devRoot = attr->funcGroup.devRootList[i];
        attr->rootFuncKeyDict[devRoot] = funcKey;
    }
}
static std::string BuildControlFlowCallee(Function *func) {
    std::ostringstream oss;
    oss << "#name:" << func->GetRawName() << " #hash:" << func->GetFunctionHash() << " #magic:" << func->GetFuncMagic();
    return oss.str();
}

static void BuildControlFlow(FunctionCache &cache, Linker &linker, const std::string &sectionName,
    Function *func,
    DyndevFunctionAttribute::FunctionGroup &group,
    std::unordered_map<Function *, Function *> &rootTileDict,
    std::ostringstream &controlFlowOss,
    std::ostringstream &expressionOss,
    int indent, const std::string &expName) {
    auto funcType = func->GetFunctionType();
    if (funcType == FunctionType::DYNAMIC) {
        controlFlowOss
            << "#define __TILE_FWK_AICPU__ 1\n"
            << "#include <stdint.h>\n"
            << "#include \"" << expName << "\"\n"
            << "#include \"machine/utils/dynamic/codegen/aicpu_runtime.h\"\n";
        expressionOss
            << "\n/* Symbol table list */\n"
            << linker.GetSymbolTable()->BuildSymbolList();
        const std::vector<std::string> &inputNameList = Program::GetInstance().GetTensorSlotManager()->GetInputNameList();
        const std::vector<std::string> &outputNameList = Program::GetInstance().GetTensorSlotManager()->GetOutputNameList();

        expressionOss << "\n/* Input tensor list */\n";
        for (size_t i = 0; i < inputNameList.size(); i++) {
            expressionOss << "#define " << AddArgPrefix(inputNameList[i]) << " " << i << "\n";
        }

        expressionOss << "\n/* Output tensor list */\n";
        for (size_t i = 0; i < outputNameList.size(); i++) {
            expressionOss << "#define " << AddArgPrefix(outputNameList[i]) << " " << i << "\n";
        }

        controlFlowOss << "#define LOOP(idx, b, e, s) for (uint64_t idx = (b), idxEnd = (e), idxStep = (s); idx < idxEnd; idx += idxStep)\n"
            << "namespace npu::tile_fwk {\n"
            << "// " << BuildControlFlowCallee(func) << "\n"
            << "__attribute__((section(\"" << sectionName
            << "\")))\n"
            << "uint64_t ControlFlowEntry(void *ctx, uint64_t *symbolTable, CallRootEntryType callRootList[3], DevStartArgsBase *startArgs) {\n";
        for (auto &callee : GetCalleeList(cache, func)) {
            BuildControlFlow(cache, linker, sectionName, callee, group, rootTileDict, controlFlowOss, expressionOss, indent + 1, expName);
        }
        controlFlowOss << std::setw((indent + 1) * TABSIZE) << ' ' << "callRootList[CallRootStage::T_CALLROOT_STITCH](ctx, RUNTIME_FINISH_FUNCKEY); // Notify finish \n";
        controlFlowOss << std::setw((indent + 1) * TABSIZE) << ' ' << "return 0;\n";
        controlFlowOss << "}\n";
        controlFlowOss << "} // namespace npu::tile_fwk\n";
    } else if (func->IsFunctionTypeAndGraphType(FunctionType::DYNAMIC_LOOP, GraphType::TENSOR_GRAPH)) {
        std::function<void(const std::shared_ptr<DynloopFunctionPathNode> &, int)> condBuilder =
            [&cache, &linker, &sectionName, &group, &rootTileDict, &controlFlowOss, &expressionOss, &condBuilder,
             &expName] (const std::shared_ptr<DynloopFunctionPathNode> &node, int condIndent) {
                if (!node->cond.IsValid()) {
                    BuildControlFlow(cache, linker, sectionName, node->root, group, rootTileDict, controlFlowOss, expressionOss, condIndent, expName);
                } else {
                    std::string cond = SymbolicExpressionTable::BuildExpression(node->cond);
                    if (node->branchNodeList[1] != nullptr) {
                        if (node->branchNodeList[0] != nullptr) {
                            controlFlowOss << std::setw(condIndent * TABSIZE) << ' ' << "if (" << cond << ") {" << "\n";
                            condBuilder(node->branchNodeList[1], condIndent + 1);
                            controlFlowOss << std::setw(condIndent * TABSIZE) << ' ' << "} else {" << "\n";
                            condBuilder(node->branchNodeList[0], condIndent + 1);
                            controlFlowOss << std::setw(condIndent * TABSIZE) << ' ' << "}" << "\n";
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
        controlFlowOss << std::setw(indent * TABSIZE) << ' ' << "// hash=" << func->GetFunctionHash() << "\n";
        auto attr = func->GetDynloopAttribute();
        ASSERT(attr != nullptr);
        if (attr->submitBeforeLoop) {
            controlFlowOss << std::setw(indent * TABSIZE) << ' ' << "callRootList[CallRootStage::T_CALLROOT_STITCH](ctx, RUNTIME_FINISH_FUNCKEY); // force submit before LOOP \n";
        }        
        std::string iterBegin = SymbolicExpressionTable::BuildExpression(attr->Begin());
        std::string iterEnd = SymbolicExpressionTable::BuildExpression(attr->End());
        std::string iterStep = SymbolicExpressionTable::BuildExpression(attr->Step());
        std::string iterVar = "VAR_" + attr->iterSymbolName;
        controlFlowOss << std::setw(indent * TABSIZE) << ' ' << "LOOP(" << iterVar << ", " << iterBegin << ", " << iterEnd << ", " << iterStep << ") {\n";
        controlFlowOss << std::setw((indent + 1) * TABSIZE) << ' ' << "VALUE_" << attr->iterSymbolName << " = " << iterVar << ";\n";

        auto pathNode = attr->BuildPathNode();
        ALOG_INFO("Paths: \n", pathNode->Dump());                
        std::vector<Function *> calleeList = GetCalleeList(cache, func);
        std::sort(calleeList.begin(), calleeList.end());

        std::vector<Function *> pathRootList;
        for (size_t i = 0; i < attr->pathList.size(); i++) {
            pathRootList.push_back(attr->pathList[i].root);
        }
        std::sort(pathRootList.begin(), pathRootList.end());

        ASSERT(calleeList == pathRootList);
        condBuilder(pathNode, indent + 1);
        controlFlowOss << std::setw(indent * TABSIZE) << ' ' << "}\n";
    } else if (func->IsFunctionTypeAndGraphType(FunctionType::DYNAMIC_LOOP_PATH, GraphType::TENSOR_GRAPH)) {
        controlFlowOss << std::setw(indent * TABSIZE) << ' ' << "// " << BuildControlFlowCallee(func) << "\n";
        for (auto &callee : GetCalleeList(cache, func)) {
            BuildControlFlow(cache, linker, sectionName, callee, group, rootTileDict, controlFlowOss, expressionOss, indent + 1, expName);
        }
    } else if (func->GetGraphType() == GraphType::TILE_GRAPH) {
        controlFlowOss << std::setw(indent * TABSIZE) << ' ' << "// " << BuildControlFlowCallee(func) << "\n";
        Function *root = func->GetRootFunction();
        rootTileDict[root] = func;
        BuildControlFlow(cache, linker, sectionName, root, group, rootTileDict, controlFlowOss, expressionOss, indent, expName);
    } else if (func->GetGraphType() == GraphType::ROOT_GRAPH) {
        ASSERT(group.devRootList.count(func));
        int devRootKey = group.devRootList.GetIndex(func);
        controlFlowOss << std::setw(indent * TABSIZE) << ' ' << "// " << BuildControlFlowCallee(func) << "\n";
        controlFlowOss << std::setw(indent * TABSIZE) << ' ' << "uint64_t *exprList" << devRootKey << " = (uint64_t *)callRootList[CallRootStage::T_CALLROOT_ALLOC](ctx, " << devRootKey << "ULL);\n";

        SymbolicExpressionTable *exprTable = linker.LookupDevRootCoa(func);
        if (exprTable != nullptr) {
            for (auto &expr : exprTable->GetPrimaryExpressionSet()) {
                auto index = exprTable->GetPrimaryExpressionSet().GetIndex(expr);
                auto exprStr = exprTable->BuildExpression(expr);
                controlFlowOss << std::setw(indent * TABSIZE) << ' ' << "exprList" << devRootKey << "[" << index << "] = " << exprStr << ";\n";
            }
        }
        controlFlowOss << std::setw(indent * TABSIZE) << ' ' << "callRootList[CallRootStage::T_CALLROOT_STITCH](ctx, " << devRootKey << "ULL);\n";
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
        if (asc_tensor == nullptr) {
          idx++;
          attr->disableL2List.emplace_back(0);
          continue;
        }
        if (tensor->GetCachePolicy(CachePolicy::PREFETCH)) {
          attr->l2InfoList.emplace_back(L2Info(tensor->MemorySize(), idx));
        }
        if (tensor->GetCachePolicy(CachePolicy::NONE_CACHEABLE)) {
          attr->disableL2List.emplace_back(1);
        } else {
          attr->disableL2List.emplace_back(0);
        }
        idx++;
    }
    for (auto &param : attr->startArgsOutputTensorList) {
        const auto &tensor = param.get();
        auto asc_tensor = tensor.GetStorage();
        if (asc_tensor == nullptr) {
          idx++;
          attr->disableL2List.emplace_back(0);
          continue;
        }
        if (tensor->GetCachePolicy(CachePolicy::NONE_CACHEABLE)) {
          attr->disableL2List.emplace_back(1);
        } else {
          attr->disableL2List.emplace_back(0);
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

    if (config::GetPassDefaultConfig(npu::tile_fwk::KEY_PRINT_FUNCTION, false)) {
        devProg->DumpFile(config::LogTopFolder() + "/program.tifwkbintxt");
        std::string loopDirPath = config::LogTopFolder() + "/loop";
        CreateMultiLevelDir(loopDirPath);
        for (size_t index = 0; index < dynAttrPtr->funcGroup.loopList.size(); index++) {
            Function *func = dynAttrPtr->funcGroup.loopList[index];
            func->DumpFile(loopDirPath + "/" + func->GetMagicName() + ".tifwkgr");
        }
    }
    devProg->RelocProgram(-reinterpret_cast<int64_t>(devProg));
    ALOG_INFO_F("Dev prog binary size is:%zu.\n", dynAttrPtr->devProgBinary.size());
}

std::vector<SymbolicExpressionTable *> GetAllExpressionTable(DyndevFunctionAttribute::ExpressionTableDictGroup &exprTableGroup) {
    std::vector<SymbolicExpressionTable *> exprTableList;
    for (auto &[func, exprTable] : exprTableGroup.loopBesDict)  {
        (void)func;
        exprTableList.push_back(&exprTable);
    }
    for (auto &[func, ifDict] : exprTableGroup.loopIfDict) {
        (void)func;
        for (auto &[expr, exprTable] : ifDict) {
            (void)expr;
            exprTableList.push_back(&exprTable);
        }
    }
    for (auto &[func, exprTable] : exprTableGroup.devRootCoaDict) {
        (void)func;
        exprTableList.push_back(&exprTable);
    }
    for (auto &[func, opDict] : exprTableGroup.loopIfDict) {        
        (void)func;
        for (auto &[op, exprTable] : opDict) {
            (void)op;
            exprTableList.push_back(&exprTable);
        }
    }
    return exprTableList;
}

static void ConstructCodeInfo(struct EncodeDevAscendFunctionParam &encodeDevAscendFunctionParam,
    std::map<std::string, Function *> &leafDict, std::shared_ptr<DyndevFunctionAttribute> attr) {
    attr->cceCodeInfo.resize(leafDict.size() + 1);
    /* cceIdx 0 for dummy callop */
    attr->cceCodeInfo[0].coreType = static_cast<uint32_t>(CoreType::HUB);
    attr->cceCodeInfo[0].psgId = 0;
    attr->cceCodeInfo[0].funcHash = 0;
    encodeDevAscendFunctionParam.calleeHashIndexDict[0] = 0;

    int leafIndex = 1;
    for (auto &[name, leaf] : leafDict) {
      auto leafFuncAttr = leaf->GetLeafFuncAttribute();
      ASSERT(leafFuncAttr != nullptr);

      encodeDevAscendFunctionParam.calleeHashIndexDict[leaf->ComputeHash().GetHash()] = leafIndex;
      attr->devLeafIndex2Hash[leafIndex] = leaf->GetFunctionHash().GetHash();
      ALOG_INFO("Dyndev.codegen: [", leafIndex, "] hash=", leaf->ComputeHash(), " name=", name, " binpath=",
                leafFuncAttr->binPath);
      attr->cceCodeInfo[leafIndex].coreType = static_cast<uint32_t>(leafFuncAttr->coreType);
      if (leaf->IsDummyFunction())
        attr->cceCodeInfo[leafIndex].coreType = static_cast<uint32_t>(CoreType::HUB);
      attr->cceCodeInfo[leafIndex].psgId = leaf->GetProgramId();
      attr->cceCodeInfo[leafIndex].funcHash = leaf->GetFunctionHash().GetHash();
      leafIndex++;
    }
    encodeDevAscendFunctionParam.cceCodeInfoList = attr->cceCodeInfo;
    return;
}

static void CompileDyndevFunction(Function *function, FunctionCache &cache, const std::string &ccePath,
                                  std::string &kernelPath) {
    std::shared_ptr<DyndevFunctionAttribute> attr = function->GetDyndevAttribute();
    ASSERT(attr != nullptr);

    Linker linker(attr->symbolTable, attr->funcGroup, attr->exprTableDictGroup);
    FindAllExpression(cache, linker, function);    

    FillL2PrefetchInfo(attr);

    auto slotManager = Program::GetInstance().GetTensorSlotManager();
    attr->inoutLink = slotManager->BuildIncastOutcastLink(function->GetRawName());

    std::ostringstream controlFlowOss;
    std::ostringstream expressionOss;

    expressionOss << "#ifndef TILE_FWK_EXPRESSION_H" << "\n"
                  << "#define TILE_FWK_EXPRESSION_H" << "\n";
    auto &exprTableGroup = linker.GetExpressionTableDictGroup();
    std::vector<SymbolicExpressionTable *> exprTableList = GetAllExpressionTable(exprTableGroup);
    linker.GetSymbolTable()->NormalizeForSymbol();
    for (auto exprTable : exprTableList) {
        exprTable->NormalizeForSymbolTable(*linker.GetSymbolTable());
        expressionOss << exprTable->BuildExpressionList();
    }
    uint64_t tilingKey = OpInfoManager::GetInstance().GetOpTilingKey();
    const std::string expName = "expression_" + std::to_string(tilingKey) + ".h";
    BuildControlFlow(cache, linker, "ast2", function, attr->funcGroup, attr->rootTileDict, controlFlowOss,
                     expressionOss, 0, expName);
    expressionOss << "#endif/*TILE_FWK_EXPRESSION_H*/" << "\n";
    std::string controlFlowSource = controlFlowOss.str();
    std::string expressionSource = expressionOss.str();
    SimplifySlots(attr.get());
    BuildSlotRootIncastOutcastDict(attr.get());
    BuildRootFuncKeyDict(attr.get());

#ifdef __x86_64__
    std::string cflags = "-mno-sse2 -mno-sse";
#else
    std::string cflags = "";
#endif

    std::string aicpuDirPath = GetEmitPath("kernel_aicpu");
    npu::tile_fwk::CreateMultiLevelDir(aicpuDirPath);

    DumpFile(expressionSource, aicpuDirPath + "/" + expName);
    
    attr->hostControlFlowBinary = CompileAndLoadSection(
        controlFlowSource, aicpuDirPath + "/controlFlow_host.cpp",
        "g++", "objcopy", "ast2", cflags);
    AlignUpTo(attr->hostControlFlowBinary, 0x8, 0);
#ifdef ASCEND_CANN_ROOT_PATH
    if (ToolchainExist(Arm64TargetTool("g++"))) {
        attr->devControlFlowBinary = CompileAndLoadSection(
            controlFlowSource, aicpuDirPath + "/controlFlow_dev.cpp",
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
    for (auto &devRoot : attr->funcGroup.devRootList) {
        if (npu::tile_fwk::ConfigManager::Instance().GetCodeGenConfig(npu::tile_fwk::KEY_CODEGEN_EXPRESSION_FUSION, false)) {
            Function *devTile = attr->rootTileDict[devRoot];
            npu::tile_fwk::CodeGenCtx codeGenCtx("", GetEmitPath("kernel_aicore"));
            npu::tile_fwk::CodeGen codeGen(codeGenCtx);
            codeGen.GenCode(*devTile, {});
        }

        for (auto &[hash, leaf] : devRoot->programs_) {
            (void) hash;
            if (!leafDict.count(leaf->GetRawName())) {
                leafDict[leaf->GetRawName()] = leaf;
                ALOG_INFO("Dyndev.codegen: ", leaf->GetRawName());
            }
        }
    }

    struct EncodeDevAscendFunctionParam encodeDevAscendFunctionParam = {};
    ConstructCodeInfo(encodeDevAscendFunctionParam, leafDict, attr);

    encodeDevAscendFunctionParam.inoutLink = &attr->inoutLink;

    int ret = CompileAICoreKernel(leafDict, encodeDevAscendFunctionParam, ccePath, kernelPath);
    if (ret != 0) {
      ALOG_ERROR_F("Compile dynamic aicore.o failed.");
      return;
    }
    attr->kernelBinary = LoadFile(kernelPath);
    ALOG_DEBUG_F("KernelBinary size %zu.", attr->kernelBinary.size());

    attr->devEncodeList.resize(attr->funcGroup.devRootList.size());
    for (auto &devRoot : attr->funcGroup.devRootList) {
        int devRootKey = attr->funcGroup.devRootList.GetIndex(devRoot);
        ALOG_INFO("Dyndev.encode: ", devRoot->GetRawName());

        ASSERT(attr->rootTileDict.count(devRoot));
        Function *devTile = attr->rootTileDict[devRoot];

        ASSERT(attr->inoutLink.ioslotDict.count(devTile));
        IncastOutcastSlot *slot = &attr->inoutLink.ioslotDict[devTile];

        encodeDevAscendFunctionParam.symbolTable = linker.GetSymbolTable();
        if (linker.GetExpressionTableDictGroup().devRootCoaDict.count(devRoot) != 0) {
            encodeDevAscendFunctionParam.expressionTable = &linker.GetExpressionTableDictGroup().devRootCoaDict.find(devRoot)->second;
        }
        encodeDevAscendFunctionParam.devRoot = devRoot;
        encodeDevAscendFunctionParam.slot = slot;

        uint64_t size = 0;
        EncodeDevAscendFunction(encodeDevAscendFunctionParam, size, nullptr);

        attr->devEncodeList[devRootKey].resize(size);
        DevAscendFunction *funcBin = reinterpret_cast<DevAscendFunction *>(&attr->devEncodeList[devRootKey][0]);
        funcBin->rootHash = devRoot->GetFunctionHash().GetHash();
        funcBin->funcKey = devRootKey;
        funcBin->stackWorkSpaceSize = devTile->GetStackWorkespaceSize();
        EncodeDevAscendFunction(encodeDevAscendFunctionParam, size, funcBin);
        funcBin->Reloc(-reinterpret_cast<int64_t>(funcBin), true);
    }

    for (size_t index = 0; index < attr->symbolTable.GetSymbolTable().size(); index++) {
        std::string name = attr->symbolTable.GetSymbolTable()[index];
        if (symbolHandlerIndexDict.count(name)) {
            attr->startArgsSymbolHandlerList.emplace_back(symbolHandlerIndexDict.find(name)->second, index);
        }
    }

    // save dev prog binary
    SetDyndevProgBinary(function);
}

MachineTask *GenCode(
    MachineTask *task, const std::map<uint64_t, std::list<InvokeParaOffset>> &invokeParaOffset, FunctionCache &cache,
    std::string &kernelPath) {
    npu::tile_fwk::CodeGenCtx codeGenCtx("", GetEmitPath("kernel_aicore"));
    npu::tile_fwk::CreateMultiLevelDir(codeGenCtx.cceDir);

    npu::tile_fwk::CodeGen codeGen(codeGenCtx);
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
        if (!npu::tile_fwk::ConfigManager::Instance().GetCodeGenConfig(npu::tile_fwk::KEY_CODEGEN_EXPRESSION_FUSION, false)) {
            codeGen.GenCode(*function, invokeParaOffset);
        }
    } else {
        if (function->IsFunctionType(FunctionType::DYNAMIC)) {
            std::string cce_path = RealPath(codeGenCtx.cceDir) + "/";
            CompileDyndevFunction(function, cache, cce_path, kernelPath);
        }
    }

    return task;
}
} // namespace npu::tile_fwk
