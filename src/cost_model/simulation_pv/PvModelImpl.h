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
 * \file PvModelImpl.h
 * \brief
 */

#pragma once

#include <vector>
#include <string>
#include <list>
#include <regex>
#include "interface/utils/file_utils.h"
#include "cost_model/simulation/pv/PvModel.h"
#include "cost_model/simulation_pv/PvMemAllocator.h"
#include "codegen/cloudnpu/codegen_cloudnpu.h"


constexpr size_t INVALID_ARG_INDEX = 0xFFFFFFFF;

namespace CostModel {
inline uint64_t CalcShapeSizeFunc(const std::vector<int> &shape) {
    uint64_t size = 1;
    for (auto &i : shape) {
        size *= i;
    }
    return size;
}

struct InvokeParaOffset {
    uint8_t *rawTensorAddr{nullptr}; // 原始input output tensor基地址, 如果是子图间workspace incast outcast 则为null
    uint64_t offset{0};
    uint64_t rawTensorOffset{0};
    bool isTensorParam{false};
    uint64_t rawShapeSize{0};
    int rawMagic{0};
    std::string rawSymbol{""};
    size_t opOriginArgsSeq{INVALID_ARG_INDEX}; // map origin args seq no
    int funcitonMagic{-1};
    int8_t ioIndex{-1};
    int8_t paramType{-1};
    std::vector<int> tensorShape;
    int opMagic{0};
    npu::tile_fwk::DataType datatype{npu::tile_fwk::DataType::DT_INT32};
    std::vector<int> rawTensorShape;
    void LogRawTensor(std::shared_ptr<npu::tile_fwk::RawTensor> rawTensor) {
        auto rawShape = rawTensor->GetRawShape();
        rawShapeSize = CalcShapeSizeFunc(rawShape) * BytesOf(rawTensor->GetDataType());
        rawMagic = rawTensor->GetRawMagic();
        rawSymbol = rawTensor->GetSymbol();
        datatype = rawTensor->GetDataType();
    }
};

struct PvModelInvoke {
    uint64_t programFunctionCnt; // 同构后的funciton 个数
    uint64_t coreFunctionCnt;
    uint64_t workSpaceStackSize{0}; // ooo 调度use stack workspace
    uint64_t invokeParaWorkSpaceSize{0};
    size_t invokeOffsetSize{0};
    std::vector<std::string> commGroups;
    std::map<uint64_t, std::list<InvokeParaOffset>> invokeParaOffset; // map esgid to all para list
    std::map<uint64_t, uint64_t> coreFunctionIdToProgramId;           // 对应graph 里 esgid map psgid
    std::vector<uint64_t> readyAicIdVec;
    std::vector<uint64_t> readyAivIdVec;
    std::vector<uint64_t> coreFuncBinOffset;
    std::vector<std::vector<uint64_t>> invokeArgsOffset; // esgid to all para offset
    std::vector<std::vector<int64_t>>
        invokeTensorsIdx; // esgid to all para tensorIdx: input0 input1 ... output0 output1, -1 means workspace
    std::vector<uint64_t> coreFunctionInvokeEntryOffset;
    std::vector<uint64_t> coreFunctionTensorInfoOffset;
    std::vector<uint64_t> coreTensorNum;
};

class PvModelTask {
public:
    std::vector<uint8_t> stack;
    uint64_t stackAddr;
    uint64_t stackSize;
    uint64_t hcclContextAddr;
    uint64_t hcclContextSize;
    std::vector<uint8_t> workspace;
    uint64_t workspaceAddr;
    uint64_t workspaceSize;
    PvModelInvoke invoke;
    std::map<uint64_t, uint64_t> binAddr;
    std::map<uint64_t, std::string> objPath;
    std::map<uint64_t, std::string> binPath;
    std::map<uint64_t, npu::tile_fwk::CoreType> binType;
    std::vector<npu::tile_fwk::OriArgInfo> oriArgs;
    std::vector<std::vector<uint8_t>> args;
    std::vector<uint64_t> oriArgsAddr;
    std::map<uint64_t, uint64_t> oriArgsMap;
    std::map<int, uint64_t> stubOutRawTensorAddr;
};

template <typename SystemConfig, typename CaseConfig>
class PvModelImpl : public PvModel {
private:
    std::string arch_;
    npu::tile_fwk::Function *func_;
    std::string dir_;
    std::string funcDir_;
    PvData *data_;
    PvModelTask task_;
    int level_;
    std::unique_ptr<PvMemAllocator> allocator_;

public:
    PvModelImpl(std::string arch) : arch_(arch) {}
    void Submit(npu::tile_fwk::Function *func, PvData *data, int level, std::string dir);
    void Run(int esgId, int psgId);

private:
    void Prepare(npu::tile_fwk::Function *func);
    void CodeGen(npu::tile_fwk::Function *func);
    void BinGen(npu::tile_fwk::Function *func);
    void CalcInvokeWorkespace(npu::tile_fwk::Function *function, PvModelInvoke &invoke);
    uint64_t GetBinSize(std::string path);
    void DumpBin(std::vector<uint8_t> bytes, uint64_t size, std::string path);
    void ReadBin(std::string path, std::vector<uint8_t> &bytes);
    void PrepareInvoke(int esgId, std::vector<uint64_t> &invokeOffsetVec, std::vector<uint64_t> &invokeOffsetOriVec);
    void SetUp(int esgId, int psgId, std::string esgDir);
    void RunModel(std::string esgDir);
    void TearDown(std::string esgDir);
};

class PvModelCodegen {
public:
    static void AddGlobalAttr(std::string srcPath) {
        const std::string searchStr = "[aicore]";                         
        const std::string replaceStr = "extern \"C\" __global__ [aicore]";

        std::ifstream file(srcPath);
        if (!file.is_open()) {
            return;
        }

        std::stringstream buffer;
        buffer << file.rdbuf();
        std::string content = buffer.str();
        file.close();

        content = ReplaceAll(content, searchStr, replaceStr);

        std::ofstream outFile(srcPath);
        if (!outFile.is_open()) {
            return;
        }

        outFile << content;
        outFile.close();
    }

    static void AddKernelEntry(std::string srcPath) {
        std::ifstream file(srcPath);
        if (!file.is_open()) {
            return;
        }

        std::stringstream buffer;
        buffer << file.rdbuf();
        std::string content = buffer.str();
        file.close();

        std::ofstream outFile(srcPath);
        if (!outFile.is_open()) {
            return;
        }

        outFile << content;
        auto name = ExtractFunctionName(content);

        std::string entry = R"!!!(
extern "C" __global__ [aicore] void PvModelKernelEntry(__gm__ npu::tile_fwk::DynFuncData *funcData, __gm__ uint64_t *opAttrs, __gm__ uint64_t *exprTbl, uint64_t GMStackBase, __gm__ int64_t *hcclContext) {
    CoreFuncParam param = {funcData, opAttrs, exprTbl};
    {KernelName}(&param, GMStackBase, hcclContext, (__gm__ GMTensorInfo*)NULL);
}

)!!!";
        entry = ReplaceAll(entry, "{KernelName}", name);
        outFile << entry;
        outFile.close();
    }

private:
    static std::string ReplaceAll(std::string str, const std::string &from, const std::string &to) {
        size_t startPos = 0;
        while ((startPos = str.find(from, startPos)) != std::string::npos) {
            str.replace(startPos, from.length(), to);
            startPos += to.length();
        }
        return str;
    }

    static std::string ExtractFunctionName(const std::string &code) {
        std::string functionName;
        std::regex functionPattern(R"(\b\w+\s+(\w+)\s*\([^)]*\))");
        std::smatch match;

        std::string::const_iterator searchStart(code.cbegin());
        std::regex_search(searchStart, code.cend(), match, functionPattern);
        if (match.size() > 1) {
            functionName = match[1].str();
        }

        return functionName;
    }
};

// Dynamic
template <typename SystemConfig, typename CaseConfig>
class DynPvModelImpl : public DynPvModel {
private:
    std::string arch_;
    npu::tile_fwk::Function *func_;
    std::string dir_;
    std::string funcDir_;
    int level_;
    std::unique_ptr<PvMemAllocator> allocator_;
    struct DataMap {
        uint64_t hostPtr;
        uint64_t devPtr;
        uint64_t size;
    };
    std::vector<DataMap> data_;
    std::vector<std::vector<uint8_t>> storage_;

    struct PvModelCceBin {
        uint32_t psgId;
        uint64_t funcHash;
        npu::tile_fwk::CoreType coreType;
        std::string srcPath;
        std::string binPath;
        PvModelCceBin(uint32_t p, uint64_t h, npu::tile_fwk::CoreType t, std::string s = "", std::string b = "") : psgId(p), funcHash(h), coreType(t), srcPath(s), binPath(b) {
        }
    };
    std::vector<PvModelCceBin> cceBin;

public:
    explicit DynPvModelImpl(std::string arch) : arch_(arch) { 
        allocator_ = std::make_unique<PvMemAllocator>(); 
    }

    void Codegen(npu::tile_fwk::Function *func) {
        auto attr = func->GetDyndevAttribute();
        std::map<std::string, npu::tile_fwk::Function *> leafDict;
        for (size_t i = 0; i < attr->funcGroup.devRootList.size(); i++) {
            npu::tile_fwk::Function *devRoot = attr->funcGroup.devRootList[i];
            for (auto &[hash, leaf] : devRoot->programs_) {
                (void) hash;
                if (!leafDict.count(leaf->GetRawName())) {
                    leafDict[leaf->GetRawName()] = leaf;
                }
            }
        }

        cceBin.emplace_back(PvModelCceBin(0, 0, npu::tile_fwk::CoreType::HUB));
        int Len2 = 2;
        int Len3 = 3;
        for (auto &[name, leaf] : leafDict) {
            (void) name;
            if (leaf->IsDummyFunction()) {
                cceBin.emplace_back(PvModelCceBin(leaf->GetProgramId(), leaf->GetFunctionHash().GetHash(), npu::tile_fwk::CoreType::HUB));
            } else {
                auto leafFuncAttr = leaf->GetLeafFuncAttribute();
                auto binPath = leafFuncAttr == nullptr ? "" : leafFuncAttr->binPath;
                auto orgSrcPath = binPath.substr(0, binPath.length() - 1) + "cpp";
                auto srcPath = binPath.substr(0, binPath.length() - Len2) + "_pvmodel.cpp";
                npu::tile_fwk::CopyFile(orgSrcPath, srcPath);
                PvModelCodegen::AddKernelEntry(srcPath);

                auto objPath = srcPath.substr(0, srcPath.length() - Len3) + "o";
                npu::tile_fwk::CodeGenCtx ctx;
                npu::tile_fwk::CodeGenCloudNPU cga(ctx);
                auto coreType = leafFuncAttr == nullptr ? npu::tile_fwk::CoreType::INVALID : leafFuncAttr->coreType;
                bool isCube = coreType == npu::tile_fwk::CoreType::AIC;
                npu::tile_fwk::CompileInfo compileInfo(
                    *func, ctx.ccePath, leaf->GetProgramId(), isCube, leaf->IsUnderDynamicFunction());
                compileInfo.SetCCEAbsPath(srcPath);
                compileInfo.SetBinAbsPath(objPath);
                cga.CompileCCE(compileInfo, "");

                binPath = srcPath.substr(0, srcPath.length() - Len3) + "bin";
                constexpr int cmdLen = 2048;
                char cmd[cmdLen];
                (void)snprintf_s(cmd, sizeof(cmd), sizeof(cmd)-1, "llvm-objcopy -O -binary -j .text %s %s", objPath.c_str(), binPath.c_str());
                (void)std::system(cmd);

                cceBin.emplace_back(
                    PvModelCceBin(leaf->GetProgramId(), leaf->GetFunctionHash().GetHash(), coreType, srcPath, binPath));
            }
        }
    }

    uint8_t *CopyToDev(const uint8_t *data, uint64_t size) {
        std::vector<uint8_t> s(data, data + size);
        uint8_t *hostPtr = s.data();
        storage_.emplace_back(std::move(s));
        uint64_t devPtr = allocator_->AllocArg(size);
        DataMap m = {reinterpret_cast<uint64_t>(hostPtr), devPtr, size};
        data_.emplace_back(m);
        std::cout << "[PVMODEL]arg map host: " << std::hex << m.hostPtr << ", dev: " << std::hex << m.devPtr << ", size: " << m.size << std::endl;
        return hostPtr;
    }

    void CopyFromDev(uint8_t *data, uint8_t *devPtr, uint64_t size) { memcpy_s(data, size, devPtr, size); }

    uint8_t *AllocDev(size_t size) {
        std::vector<uint8_t> s(size, 0);
        uint8_t *hostPtr = s.data();
        storage_.emplace_back(std::move(s));
        uint64_t devPtr = allocator_->AllocWorkspace(size);
        DataMap m = {reinterpret_cast<uint64_t>(hostPtr), devPtr, size};
        data_.emplace_back(m);
        std::cout << "[PVMODEL]workspace map host: " << std::hex << m.hostPtr << ", dev: " << std::hex << m.devPtr << ", size: " << m.size << std::endl;
        return hostPtr;
    }
};

} // namespace CostModel
