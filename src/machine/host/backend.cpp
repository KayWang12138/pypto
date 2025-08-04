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

        if (npu::tile_fwk::ConfigManager::Instance().GetCodeGenConfig(npu::tile_fwk::KEY_CODEGEN_EXPRESSION_FUSION, false)) {
            if (function->GetGraphType() == GraphType::TILE_GRAPH) {
                // When expression fusion, don't need tile graph codegen.
                return 0;
            }
        }        
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

static void FindAllExpression(FunctionCache &cache, Linker &linker, Function *func, 
                              DyndevFunctionAttribute::FunctionGroup &group) {
    if (func->IsDynloop()) {
        auto dynloopAttr = func->GetDynloopAttribute();
        auto ss = SymbolicScalar(dynloopAttr->iterSymbolName);
        linker.AddSymbol(func, ss);
    }
    if (func->IsFunctionTypeAndGraphType({FunctionType::DYNAMIC, FunctionType::DYNAMIC_LOOP, FunctionType::DYNAMIC_LOOP_PATH}, GraphType::TENSOR_GRAPH)) {
        ALOG_INFO("Compile control:", func->Dump());
        for (auto &callee : GetCalleeList(cache, func)) {
            FindAllExpression(cache, linker, callee, group);
        }
        if (func->IsFunctionTypeAndGraphType(FunctionType::DYNAMIC_LOOP, GraphType::TENSOR_GRAPH)) {
            group.loopList.Insert(func);
            auto attr = func->GetDynloopAttribute();
            linker.AddPrimaryExpression(func, attr->Begin());
            linker.AddPrimaryExpression(func, attr->End());
            linker.AddPrimaryExpression(func, attr->Step());

            for (auto &path : attr->GetPathList()) {
                for (auto &cond : path.GetPathCondList()) {
                    linker.AddPrimaryExpression(func, cond.GetCond());
                }
            }
        }
    } else if (func->GetGraphType() == GraphType::TILE_GRAPH) {
        ALOG_INFO("Compile tile:", func->Dump());
        Function *root = func->GetRootFunction();
        FindAllExpression(cache, linker, root, group);
    } else if (func->GetGraphType() == GraphType::ROOT_GRAPH) {
        ALOG_INFO("Compile root:", func->Dump());
        group.devRootList.Insert(func);
        for (auto &callopAttr : func->GetCallopAttrList()) {
            for (auto &arg : callopAttr->GetLinearArgList()) {
                linker.AddPrimaryExpression(func, arg);
            }
            auto hash = callopAttr->GetCalleeHash();
            Function *leafFunc = cache.GetCacheFunction(hash);
            FindAllExpression(cache, linker, leafFunc, group);
        }
    } else if (func->GetGraphType() == GraphType::LEAF_GRAPH) {
        group.devLeafList.Insert(func);
        for (auto &op : func->Operations()) {
            if (op.GetOpcode() == Opcode::OP_VEC_DUP) {
                if (op.HasAttr(OpAttributeKey::dynScalar)) {
                    auto dynScalar = op.GetSymbolicScalarAttribute(OpAttributeKey::dynScalar);
                    linker.AddPrimaryExpression(func, dynScalar);
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

static void SimplifySlots(IncastOutcastLink &inoutLink, DyndevFunctionAttribute *attr,
                          std::unordered_map<Function *, Function *> &rootTileDict) {
    std::vector<bool> slotUsed(inoutLink.totalSlot);

    for (Function *devRoot : attr->group.devRootList) {
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

    for (Function *devRoot : attr->group.devRootList) {
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
    for (Function *devRoot : attr->group.devRootList) {
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
    Function *func, npu::tile_fwk::DyndevFunctionAttribute::FunctionGroup &group,
    std::unordered_map<Function *, Function *> &rootTileDict,
    std::ostringstream &controlFlowOss,
    std::ostringstream &expressionOss,
    int indent) {
    auto funcType = func->GetFunctionType();
    if (funcType == FunctionType::DYNAMIC) {
        controlFlowOss
            << "#include <stdint.h>\n"
            << "#include \"expression.h\"\n"
            << "#include \"machine/utils/dynamic/codegen/codegen.h\"\n"
            << linker.symbolTable.BuildSymbolList();
        const std::vector<std::string> &inputNameList = Program::GetInstance().GetTensorSlotManager()->GetInputNameList();
        const std::vector<std::string> &outputNameList = Program::GetInstance().GetTensorSlotManager()->GetOutputNameList();
        for (size_t i = 0; i < inputNameList.size(); i++) {
            controlFlowOss << "#define " << AddArgPrefix(inputNameList[i]) << " " << i << "\n";
        }
        for (size_t i = 0; i < outputNameList.size(); i++) {
            controlFlowOss << "#define " << AddArgPrefix(outputNameList[i]) << " " << i << "\n";
        }
        controlFlowOss << "#define LOOP(idx, b, e, s) for (uint64_t idx = (b), idxEnd = (e), idxStep = (s); idx < idxEnd; idx += idxStep)\n"
            << "namespace npu::tile_fwk {\n"
            << "// " << BuildControlFlowCallee(func) << "\n"
            << "__attribute__((section(\"" << sectionName
            << "\")))\n"
            << "uint64_t ControlFlowEntry(void *ctx, uint64_t *symbolTable, CallRootEntryType callRootList[3], DevStartArgsBase *startArgs) {\n";
        for (auto &callee : GetCalleeList(cache, func)) {
            BuildControlFlow(cache, linker, sectionName, callee, group, rootTileDict, controlFlowOss, expressionOss, indent + 1);
        }
        controlFlowOss << std::setw((indent + 1) * TABSIZE) << ' ' << "callRootList[CallRootStage::T_CALLROOT_STITCH](ctx, RUNTIME_FINISH_FUNCKEY); // Notify finish \n";
        controlFlowOss << std::setw((indent + 1) * TABSIZE) << ' ' << "return 0;\n";
        controlFlowOss << "}\n";
        controlFlowOss << "} // namespace npu::tile_fwk\n";
    } else if (func->IsFunctionTypeAndGraphType(FunctionType::DYNAMIC_LOOP, GraphType::TENSOR_GRAPH)) {
        std::function<void(const std::shared_ptr<DynloopFunctionPathNode> &, int)> condBuilder =
            [&cache, &linker, &sectionName, &group, &rootTileDict, &controlFlowOss, &expressionOss, &condBuilder](
                const std::shared_ptr<DynloopFunctionPathNode> &node, int condIndent) {
                if (!node->cond.IsValid()) {
                    BuildControlFlow(
                        cache, linker, sectionName, node->root, group, rootTileDict, controlFlowOss, expressionOss, condIndent);
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
        auto attr = func->GetDynloopAttribute();
        ASSERT(attr != nullptr);
        std::string iterBegin = SymbolicExpressionTable::BuildExpression(attr->Begin());
        std::string iterEnd = SymbolicExpressionTable::BuildExpression(attr->End());
        std::string iterStep = SymbolicExpressionTable::BuildExpression(attr->Step());
        if (attr->submitBeforeLoop) {
            controlFlowOss << std::setw(indent * TABSIZE) << ' ' << "callRootList[CallRootStage::T_CALLROOT_STITCH](ctx, RUNTIME_FINISH_FUNCKEY); // force submit before LOOP \n";
        }
        controlFlowOss << std::setw(indent * TABSIZE) << ' ' << "// hash=" << func->GetFunctionHash() << "\n";
        controlFlowOss << std::setw(indent * TABSIZE) << ' ' << "LOOP(INDEX, " << iterBegin << ", " << iterEnd << ", " << iterStep << ") {\n";
        controlFlowOss << std::setw((indent + 1) * TABSIZE) << ' ' << "VALUE_" << attr->iterSymbolName << " = INDEX;\n";

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
            BuildControlFlow(cache, linker, sectionName, callee, group, rootTileDict, controlFlowOss, expressionOss, indent + 1);
        }
    } else if (func->GetGraphType() == GraphType::TILE_GRAPH) {
        controlFlowOss << std::setw(indent * TABSIZE) << ' ' << "// " << BuildControlFlowCallee(func) << "\n";
        Function *root = func->GetRootFunction();
        rootTileDict[root] = func;
        BuildControlFlow(cache, linker, sectionName, root, group, rootTileDict, controlFlowOss, expressionOss, indent);
    } else if (func->GetGraphType() == GraphType::ROOT_GRAPH) {
        ASSERT(group.devRootList.count(func));
        int devRootKey = group.devRootList.GetIndex(func);
        controlFlowOss << std::setw(indent * TABSIZE) << ' ' << "// " << BuildControlFlowCallee(func) << "\n";
        controlFlowOss << std::setw(indent * TABSIZE) << ' ' << "uint64_t *exprList" << devRootKey << " = (uint64_t *)callRootList[CallRootStage::T_CALLROOT_ALLOC](ctx, " << devRootKey << "ULL);\n";
        if (linker.rootExpressionTableDict.count(func) != 0) {
            SymbolicExpressionTable &table = linker.rootExpressionTableDict[func];
            for (auto &[expr, index] : table.expressionIndexTable) {
                controlFlowOss << std::setw(indent * TABSIZE) << ' ' << "exprList" << devRootKey << "[" << index << "] = " << expr << ";\n";
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
    std::shared_ptr<DyndevFunctionAttribute> attr = function->GetDyndevAttribute();
    ASSERT(attr != nullptr);

    Linker linker;
    FindAllExpression(cache, linker, function, attr->group);
    linker.BuildAndLoad();

    FillL2PrefetchInfo(attr);
    attr->symbolTable = linker.symbolTable;
    attr->rootExpressionTableDict = linker.rootExpressionTableDict;

    ALOG_INFO("SymbolTable: ", linker.symbolTable.Dump());
    for (auto &[func, exprTable] : linker.rootExpressionTableDict) {
        ALOG_INFO("ExpressionTable: name:", func->GetMagicName(), "\n", exprTable.Dump());
    }

    auto slotManager = Program::GetInstance().GetTensorSlotManager();
    attr->inoutLink = slotManager->BuildIncastOutcastLink(function->GetRawName());

    std::ostringstream controlFlowOss;
    std::ostringstream expressionOss;

    std::unordered_map<Function *, Function *> rootTileDict;
    expressionOss << "#ifndef TILE_FWK_EXPRESSION_H" << "\n"
                  << "#define TILE_FWK_EXPRESSION_H" << "\n";
    for (auto &[func, builder] : linker.rootExpressionTableBuilderDict) {
        std::string prefix;
        if (attr->group.loopList.count(func)) {
            prefix = npu::tile_fwk::SymbolicExpressionTableBuilder::GetExprNameLoopPrefix(attr->group.loopList.GetIndex(func));
        } else if (attr->group.devRootList.count(func)) {
            prefix = npu::tile_fwk::SymbolicExpressionTableBuilder::GetExprNameRootPrefix(attr->group.devRootList.GetIndex(func));
        } else if (attr->group.devLeafList.count(func)) {
            prefix = npu::tile_fwk::SymbolicExpressionTableBuilder::GetExprNameLeafPrefix(attr->group.devLeafList.GetIndex(func));
        } else {
            ASSERT(false) << "Unknown function: " << func->GetMagicName();
        }        
        std::string title = "name=" + func->GetRawName() + " hash=" + std::to_string(func->GetFunctionHash().GetHash());
        expressionOss << builder.BuildExpressionList(prefix, title);
    }    
    BuildControlFlow(cache, linker, "ast2", function, attr->group, rootTileDict, controlFlowOss, expressionOss, 0);
    expressionOss << "#endif/*TILE_FWK_EXPRESSION_H*/" << "\n";
    std::string controlFlowSource = controlFlowOss.str();
    std::string expressionSource = expressionOss.str();

#ifdef __x86_64__
    std::string cflags = "-mno-sse2 -mno-sse";
#else
    std::string cflags = "";
#endif

    std::string aicpuDirPath = GetEmitPath("kernel_aicpu");
    npu::tile_fwk::CreateMultiLevelDir(aicpuDirPath);

    DumpFile(expressionSource, aicpuDirPath + "/expression.h");
    
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
    for (auto &devRoot : attr->group.devRootList) {
        if (npu::tile_fwk::ConfigManager::Instance().GetCodeGenConfig(npu::tile_fwk::KEY_CODEGEN_EXPRESSION_FUSION, false)) {
            Function *devTile = rootTileDict[devRoot];
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
    attr->devEncodeList.resize(attr->group.devRootList.size());
    for (auto &devRoot : attr->group.devRootList) {
        int devRootKey = attr->group.devRootList.GetIndex(devRoot);
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

        attr->devEncodeList[devRootKey].resize(size);
        DevAscendFunction *funcBin = reinterpret_cast<DevAscendFunction *>(&attr->devEncodeList[devRootKey][0]);
        funcBin->funcKey = devRootKey;
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
    npu::tile_fwk::CodeGenCtx codeGenCtx("", GetEmitPath("kernel_aicore"));
    npu::tile_fwk::CreateMultiLevelDir(codeGenCtx.ccePath);

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
        codeGen.GenCode(*function, invokeParaOffset);        
    } else {
        if (function->IsFunctionType(FunctionType::DYNAMIC)) {
            CompileDyndevFunction(function, cache);
        }
    }

    return task;
}
} // namespace npu::tile_fwk
