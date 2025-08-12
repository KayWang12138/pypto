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
 * \file function_cache.cpp
 * \brief
 */

#include "function_cache.h"
#include <elf.h>
#include <cstdio>
#include "interface/utils/common.h"
#include "interface/utils/file_utils.h"
#include "securec.h"
#include "topo_processor.h"
#include "interface/utils/log.h"

namespace npu::tile_fwk {
std::optional<CacheValue> FunctionCache::Get(HashKey key) {
    std::lock_guard<std::mutex> cLockGuard(lock_);
    getCnt_++;
    if (auto it = cache_.find(key); it != cache_.end()) {
        hitCnt_++;
        return it->second;
    } else {
        return std::nullopt;
    }
}

void FunctionCache::UpdateTopoCache(const Function &func, CacheValue &value) {
    uint64_t totalSize = 0;
    uint32_t topoNum = func.topoInfo_.topology_.size();
    totalSize += (topoNum * sizeof(uint64_t));
    for (uint32_t i = 0; i < topoNum; i++) {
        uint32_t depNum = func.topoInfo_.topology_[i].outGraph.size();
        uint32_t extParamNum = func.topoInfo_.topology_[i].extParamNum;
        totalSize += sizeof(CoreFunctionTopo) + sizeof(uint64_t) * depNum + sizeof(uint64_t) * extParamNum;
    }

    uint8_t* topoCachePtr = new uint8_t[totalSize + sizeof(uint64_t)];
    *reinterpret_cast<uint64_t*>(topoCachePtr) = totalSize;

    uint64_t* offsetPtr = reinterpret_cast<uint64_t*>(topoCachePtr + sizeof(uint64_t));
    uint8_t* topoPtr = reinterpret_cast<uint8_t*>(reinterpret_cast<uint64_t*>(topoCachePtr) + topoNum + 1);
    uint64_t curCoreFuncOffset = sizeof(uint64_t) + topoNum * sizeof(uint64_t);
    for (uint32_t i = 0; i < topoNum; i++) {
        offsetPtr[i] = curCoreFuncOffset;
        CoreFunctionTopo* tempPtr = reinterpret_cast<CoreFunctionTopo*>(topoPtr);
        tempPtr->coreType = static_cast<uint64_t>(func.GetSubFuncInvokeInfo(i).GetGraphType());
        ASSERT((tempPtr->coreType == static_cast<uint64_t>(CoreType::AIV)) ||
               (tempPtr->coreType == static_cast<uint64_t>(CoreType::AIC)) ||
               (tempPtr->coreType == static_cast<uint64_t>(CoreType::HUB)) ||
               (tempPtr->coreType == static_cast<uint64_t>(CoreType::AICPU)));
        tempPtr->psgId = func.GetSubFuncInvokeInfo(i).GetProgramId();
        tempPtr->readyCount = func.topoInfo_.topology_[i].readyState;
        tempPtr->depNum = func.topoInfo_.topology_[i].outGraph.size();
        tempPtr->extParamNum = func.topoInfo_.topology_[i].extParamNum;
        tempPtr->extType = func.topoInfo_.topology_[i].extType;
        ALOG_DEBUG_F("[function cache]topo %u, readycount:%ld, depnum:%lu, coreType:%lu, extType:%u",
            i, tempPtr->readyCount, tempPtr->depNum, tempPtr->coreType, tempPtr->extType);
        uint32_t j = 0;
        for (auto& ele : func.topoInfo_.topology_[i].outGraph) {
            tempPtr->depIds[j] = ele;
            j++;
            ALOG_DEBUG_F("[function cache]depend %u", ele);
        }
        for (auto& ele : func.topoInfo_.topology_[i].extParams) {
            tempPtr->depIds[j++] = static_cast<uint64_t>(ele);
        }
        uint32_t tempLength = sizeof(CoreFunctionTopo) + sizeof(uint64_t) * (tempPtr->depNum + tempPtr->extParamNum);
        curCoreFuncOffset += tempLength;
        topoPtr += tempLength;
    }
    value.topoCache = reinterpret_cast<CoreFunctionTopoCache*>(topoCachePtr);
    value.header.coreFunctionNum = topoNum;
    ASSERT(topoNum != 0);

    TopoProcessor processor(reinterpret_cast<CoreFunctionTopoCache*>(topoCachePtr), topoNum);
    std::tuple<CoreFunctionTopoCache*, uint64_t> newTopo = processor.MergeBatchDepend(10, 1);
    value.topoCache = reinterpret_cast<CoreFunctionTopoCache*>(std::get<0>(newTopo));
    if (value.topoCache != reinterpret_cast<CoreFunctionTopoCache*>(topoCachePtr)) {
        delete[] topoCachePtr;
    }
    value.header.virtualFunctionNum = std::get<1>(newTopo);
}

std::vector<uint8_t> LoadBinData(const std::string &binPath) {
    std::vector<uint8_t> text;

    uint32_t fileSize = GetFileSize(binPath);
    std::vector<char> buf(fileSize);
    std::ifstream file(binPath);
    file.read(buf.data(), fileSize);

    auto elfHeader = reinterpret_cast<Elf64_Ehdr *>(buf.data());
    if (elfHeader->e_ident[EI_MAG0] != ELFMAG0 || elfHeader->e_ident[EI_MAG1] != ELFMAG1 ||
        elfHeader->e_ident[EI_MAG2] != ELFMAG2 || elfHeader->e_ident[EI_MAG3] != ELFMAG3) {
        return text;
    }

    auto sectionHeaders = reinterpret_cast<Elf64_Shdr *>(reinterpret_cast<uint64_t>(elfHeader) + elfHeader->e_shoff);
    auto shstrHeader = &sectionHeaders[elfHeader->e_shstrndx];
    auto strtbl = buf.data() + shstrHeader->sh_offset;
    for (int i = 0; i < elfHeader->e_shnum; i++) {
        auto section = &sectionHeaders[i];
        auto sectionName = strtbl + section->sh_name;
        if (strcmp(sectionName, ".text") == 0) {
            text.resize(section->sh_size);
            memcpy_s(text.data(), section->sh_size, buf.data() + section->sh_offset, section->sh_size);
            break;
        }
    }

    return text;
}

void FunctionCache::UpdateBinCache(const Function &func, CacheValue &value) {
    std::map<uint64_t, std::vector<uint8_t>> binMap;
    uint64_t totalSize = 0;

    for (auto &ele : func.programs_) {
        auto leafFuncAttr = ele.second->GetLeafFuncAttribute();
        ASSERT(leafFuncAttr != nullptr);
        auto binPath = leafFuncAttr->binPath;
        if (!RealPath(binPath).empty()) {
            auto binData = LoadBinData(binPath);
            assert(binData.size() != 0);
            totalSize += binData.size() + sizeof(uint64_t);
            binMap[ele.first] = std::move(binData);
        } else if (leafFuncAttr->coreType == CoreType::AICPU) {
            std::vector<uint8_t> binData(0, 0);
            totalSize += binData.size() + sizeof(uint64_t);
            binMap[ele.first] = std::move(binData);
        } else {
            ALOG_ERROR("bin path %s is not existed", binPath.c_str());
            abort();
        }
    }

    uint64_t progNum = func.programs_.size();
    totalSize += progNum * sizeof(uint64_t);

    uint8_t *buf = new uint8_t[totalSize + sizeof(uint64_t)];
    value.binCache = reinterpret_cast<CoreFunctionBinCache *>(buf);
    value.header.programFuncionNum = progNum;
    value.binCache->dataSize = totalSize;

    auto binOffsets = reinterpret_cast<uint64_t *>(buf + sizeof(CoreFunctionBinCache));
    uint64_t curOffset = sizeof(uint64_t) + progNum * sizeof(uint64_t);
    for (auto &ele : func.programs_) {
        *binOffsets++ = curOffset;
        auto *funcBin = reinterpret_cast<CoreFunctionBin *>(buf + curOffset);
        auto binData = binMap[ele.first];
        funcBin->size = binData.size();
        memcpy_s(funcBin->data, binData.size(), binData.data(), funcBin->size);
        curOffset += sizeof(CoreFunctionBin) + funcBin->size;
    }
}

void FunctionCache::UpdateReadyFunction(const Function &func, CacheValue &value) {
    uint64_t readyNum = func.GetAllReadySubGraphCount();
    uint64_t totalSize = readyNum * sizeof(ReadyCoreFunction);
    uint8_t* readyFuncPtr = new uint8_t[totalSize + sizeof(uint64_t)];

    *reinterpret_cast<uint64_t*>(readyFuncPtr) = totalSize;
    ReadyCoreFunction* listPtr = reinterpret_cast<ReadyCoreFunction*>(reinterpret_cast<uint64_t*>(readyFuncPtr) + 1);
    // ReadyCoreFunctionCache
    size_t index = 0;
    for (size_t i = 0; i < func.GetReadySubGraphCount(CoreType::AIC); i++) {
        listPtr[index].id = func.GetReadySubGraphId(CoreType::AIC, i);
        listPtr[index].coreType = static_cast<uint64_t>(CoreType::AIC);
        index++;
    }

    for (size_t i = 0; i < func.GetReadySubGraphCount(CoreType::AIV); i++) {
        listPtr[index].id = func.GetReadySubGraphId(CoreType::AIV, i);
        listPtr[index].coreType = static_cast<uint64_t>(CoreType::AIV);
        index++;
    }

    for (size_t i = 0; i < func.GetReadySubGraphCount(CoreType::AICPU); i++) {
        listPtr[index].id = func.GetReadySubGraphId(CoreType::AICPU, i);
        listPtr[index].coreType = static_cast<uint64_t>(CoreType::AICPU);
        index++;
    }

    value.readyListCache = reinterpret_cast<ReadyCoreFunctionCache*>(readyFuncPtr);
    value.header.readyCoreFunctionNum = readyNum;
    ASSERT(value.header.readyCoreFunctionNum != 0);
}


void FunctionCache::Insert(const HashKey& key, Function &func) {
    CacheValue cacheVal;
    if (func.IsFunctionTypeAndGraphType({FunctionType::DYNAMIC_LOOP, FunctionType::DYNAMIC_LOOP_PATH, FunctionType::STATIC}, {GraphType::TENSOR_GRAPH, GraphType::TILE_GRAPH})) {
        if (func.rootFunc_) {
            UpdateTopoCache(*func.rootFunc_, cacheVal);
            UpdateBinCache(*func.rootFunc_, cacheVal);
            UpdateReadyFunction(*func.rootFunc_, cacheVal);
        }
        cacheVal.cacheFunction = &func;
    } else {
        if (func.GetGraphType() == GraphType::LEAF_GRAPH) {
            cacheVal.cacheFunction = &func;
        } else {
            return;
        }
    }
    Insert(key, cacheVal);
}

void FunctionCache::Insert(const HashKey& key, CacheValue value) {
    std::lock_guard<std::mutex> cLockGuard(lock_);
    cache_[key] = value;
}

size_t FunctionCache::Size() {
    std::lock_guard<std::mutex> cLockGuard(lock_);
    return cache_.size();
}

std::string FunctionCache::GetHitRate() {
    std::lock_guard<std::mutex> cLockGuard(lock_);
    std::string temp = std::to_string(hitCnt_) + "/" + std::to_string(getCnt_);
    return temp;
}

Function *FunctionCache::GetCacheFunction(const HashKey &key) {
    std::lock_guard<std::mutex> cLockGuard(lock_);
    getCnt_++;
    if (auto it = cache_.find(key); it != cache_.end()) {
        hitCnt_++;
        return it->second.cacheFunction;
    } else {
        return nullptr;
    }
}

void FunctionCache::Reset() {
    std::lock_guard<std::mutex> cLockGuard(lock_);
    for (auto& ele : cache_) {
        if (ele.second.topoCache != nullptr) {
            delete[] reinterpret_cast<uint8_t*>(ele.second.topoCache);
            ele.second.topoCache = nullptr;
        }
        if (ele.second.binCache != nullptr) {
            delete[] reinterpret_cast<uint8_t*>(ele.second.binCache);
            ele.second.binCache = nullptr;
        }
        if (ele.second.readyListCache != nullptr) {
            delete[] reinterpret_cast<uint8_t*>(ele.second.readyListCache);
            ele.second.readyListCache = nullptr;
        }
    }
    cache_.clear();
}

FunctionCache::~FunctionCache() {
    for (auto& ele : cache_) {
        if (ele.second.topoCache != nullptr) {
            delete[] reinterpret_cast<uint8_t*>(ele.second.topoCache);
            ele.second.topoCache = nullptr;
        }
        if (ele.second.binCache != nullptr) {
            delete[] reinterpret_cast<uint8_t*>(ele.second.binCache);
            ele.second.binCache = nullptr;
        }
        if (ele.second.readyListCache != nullptr) {
            delete[] reinterpret_cast<uint8_t*>(ele.second.readyListCache);
            ele.second.readyListCache = nullptr;
        }
    }
}

} // namespace npu::tile_fwk
