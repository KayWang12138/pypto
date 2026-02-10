/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file platform.cpp
 * \brief
 */

#include <fstream>
#include "tilefwk/platform.h"
#include "parser/internal_parser.h"
#include "parser/platform_parser.h"
#include "simulation_platform/simulation_platform.h"

namespace npu::tile_fwk {
const std::string version = "version";
const std::string socVersionInfo = "Soc_version";
const std::string shortSocVersion = "Short_SoC_version";
const std::string npuArchInfo = "NpuArch";
const std::string socInfo = "SoCInfo";
const std::string aiCoreCnt = "ai_core_cnt";
const std::string cubeCoreCnt = "cube_core_cnt";
const std::string vectorCoreCnt = "vector_core_cnt";
const std::string aiCpuCnt = "ai_cpu_cnt";
const std::string aiCoreSpec = "AICoreSpec";
const std::string l0aSize = "l0_a_size";
const std::string l0bSize = "l0_b_size";
const std::string l0cSize = "l0_c_size";
const std::string l1Size = "l1_size";
const std::string ubSize = "ub_size";
const std::string aic = "AIC";
const std::string aiv = "AIV";
const std::string aicVersion = "AIC_version";
const std::string aivVersion = "AIV_version";
const std::string ccecAicVersion = "CCEC_AIC_version";
const std::string ccecAivVersion = "CCEC_AIV_version";
const std::string ccecCubeVersion = "CCEC_CUBE_version";
const std::string ccecVectorVersion = "CCEC_VECTOR_version";
const std::string iniFile = "platformInfo.ini";
const std::unordered_map<std::string, NPUArch> npuArchMap = {
    {"1001", NPUArch::DAV_1001},
    {"2201", NPUArch::DAV_2201},
    {"3510", NPUArch::DAV_3510},
};

NPUArch StringToNPUArch(const std::string& npuArch) {
    auto it = npuArchMap.find(npuArch);
    if (it != npuArchMap.end()) {
        return it->second;
    }
    return NPUArch::DAV_2201;
}

void *GetSymbol(const char *sym) {
    void *ptr = nullptr;
    std::string soPath = GetCurrentSharedLibPath() + "/libtile_fwk_platform_ops.so";
    std::string stubPath = GetCurrentSharedLibPath() + "/libtile_fwk_platform_ops_stub.so";
    void* handle = dlopen(soPath.c_str(), RTLD_LAZY);
    void* stubhandle = dlopen(stubPath.c_str(), RTLD_LAZY);
    if (handle != nullptr) {
        ptr = dlsym(handle, sym);
    }
    if (ptr == nullptr) {
        ptr = dlsym(stubhandle, sym);
    }
    return ptr;
}

bool PlatformParser::FilterCCECVersion(const std::string& key, std::string &coreType) const {
    const std::string prefix = "CCEC_";
    const std::string suffix = "_version";
    const size_t prefixLen = prefix.length();
    const size_t suffixLen = suffix.length();
    if (key.length() >= (prefixLen + suffixLen) &&
        key.substr(0, prefixLen) == prefix &&
        key.substr(key.length() - suffixLen) == suffix) {
        coreType = key.substr(prefixLen, key.length() - prefixLen - suffixLen);
        return true;
    } else {
        return false;
    }
}

bool PlatformParser::GetSizeVal(const std::string& column, const std::string& key, size_t& val) const {
    std::string valStr;
    const size_t max_size_t = std::numeric_limits<size_t>::max();
    if (!GetStringVal(column, key, valStr)) {
        return false;
    }
    val = 0UL;

    constexpr int    kRadix10    = 10;
    constexpr int    kMaxDigit10 = kRadix10 - 1;

    for (const char &c : valStr) {
        int digit = c - '0';
        if (digit < 0 || digit > kMaxDigit10) {
            return false;
        }
        if (val > (max_size_t - digit) / kRadix10) {
            return false;
        }
        val = val * kRadix10 + digit;
    }
    return true;
}

bool PlatformParser::GetCCECVersion(std::unordered_map<std::string, std::string>& ccecVersion) const {
    const std::vector<std::string> ccecVersions = {ccecAicVersion, ccecAivVersion, ccecCubeVersion, ccecVectorVersion};
    ccecVersion.clear();
    std::string coreType;
    std::string versionVal;
    for (const auto &curVersion : ccecVersions) {
        if (FilterCCECVersion(curVersion, coreType) && GetStringVal(version, curVersion, versionVal)) {
            ccecVersion[coreType] = versionVal;
        }
    }
    return true;
}

bool PlatformParser::GetCoreVersion(std::unordered_map<std::string, std::string>& curVersion) const {
    curVersion.clear();
    std::string versionVal;
    if (GetStringVal(version, aicVersion, versionVal)) {
        curVersion[aic] = versionVal;
    }
    if (GetStringVal(version, aivVersion, versionVal)) {
        curVersion[aiv] = versionVal;
    }
    return true;
}

bool CmdParser::GetStringVal(const std::string& column, const std::string& key, std::string& val) const {
    using GetSocSpecFunc = bool (*)(const std::string &, const std::string &, std::string &);
    std::string socSpecFuncName = "GetrtSocSpec";
    auto socSpecFunc = (GetSocSpecFunc)GetSymbol(socSpecFuncName.c_str());
    if (socSpecFunc(column, key, val)) {
        return true;
    }
    (void)column;
    (void)key;
    (void)val;
    return false;
}


size_t Core::GetMemorySize(MemoryType type) const {
    auto it = memories_.find(type);
    if (it != memories_.end()) {
        return it->second.size;
    }
    return 0;
}

size_t Die::GetMemoryLimit(MemoryType type) const {
    size_t aic_limit = core_wrap_.GetAICMemorySize(type);
    size_t aiv_limit = core_wrap_.GetAIVMemorySize(type);
    if(aic_limit == 0 && aiv_limit == 0) {
        // ERROR Note
        return 0;
    }
    return aic_limit == 0 ? aiv_limit : aic_limit;
}

bool Die::SetMemoryPath(const std::vector<std::pair<MemoryType, MemoryType>>& dataPaths) {
    for (const auto &pathDesc : dataPaths) {
        if (pathDesc.first != MemoryType::MEM_UNKNOWN && pathDesc.second != MemoryType::MEM_UNKNOWN) {
            memoryGraph_.AddPath(pathDesc.first, pathDesc.second);
        }
    }
    return true;
}

bool Die::FindNearestPath(MemoryType from, MemoryType to, std::vector<MemoryType> &paths) const {
    auto res = memoryGraph_.FindNearestPath(from, to, paths);
    if (res == true) {
        return true;
    }
    paths.clear();
    return false;
}

void SoC::SetNPUArch(const std::string& versionStr) {
    version_ = StringToNPUArch(versionStr);
}

size_t SoC::GetAICPUNum() const {
    size_t rtAiCpuNum;
    using GetAiCpuNumFunc = bool (*)(size_t &);
    std::string AiCpuNumFuncName = "GetrtAICPUNum";
    auto socVerFunc = (GetAiCpuNumFunc)GetSymbol(AiCpuNumFuncName.c_str());
    if (socVerFunc(rtAiCpuNum)) {
        return rtAiCpuNum;
    } else {
        return ai_cpu_cnt_;
    }
}

void SoC::SetCoreVersion(const std::unordered_map<std::string, std::string>& ver) {
    for (const auto &pair : ver) {
        if (pair.first == "AIC") {
            GetAICCore().SetVersion(pair.second);
        } else if (pair.first == "AIV") {
            GetAIVCore().SetVersion(pair.second);
        }
    }
}

void SoC::SetCCECVersion(const std::unordered_map<std::string, std::string>& ver) {
    for (const auto &pair : ver) {
        if (pair.first == "AIC") {
            GetAICCore().SetCCECVersion(pair.second);
        } else if (pair.first == "AIV") {
            GetAIVCore().SetCCECVersion(pair.second);
        }
    }
}

std::string SoC::GetCoreVersion(std::string CoreType) {
    if (CoreType == "AIC") {
        return GetAICCore().GetVersion();
    } else if (CoreType == "AIV") {
        return GetAIVCore().GetVersion();
    } else {
        return "UNKNOWN_CORE";
    }
}

std::string SoC::GetCCECVersion(std::string CoreType) {
    if (CoreType == "AIC") {
        return GetAICCore().GetCCECVersion();
    } else if (CoreType == "AIV") {
        return GetAIVCore().GetCCECVersion();
    } else {
        return "UNKNOWN_CORE";
    }
}

void MemoryGraph::AddPath(MemoryType from, MemoryType to) {
    if (from == to) {
        return;
    }
    std::shared_ptr<MemoryNode> fromNode = GetNode(from);
    std::shared_ptr<MemoryNode> toNode = GetNode(to);
    if ((fromNode == nullptr) || (toNode == nullptr)) {
        return;
    }
    fromNode->AddDest(toNode);
}

std::shared_ptr<MemoryNode> MemoryGraph::GetNode(MemoryType type) {
    std::shared_ptr<MemoryNode> node;
    if (nodes.count(type) != 0) {
        node = nodes[type];
        return node;
    }
    node = std::make_shared<MemoryNode>();
    if (node == nullptr) {
        return nullptr;
    }
    node->type = type;
    nodes.insert({type, node});
    return node;
}

void MemoryGraph::DFS(MemoryType target, const std::shared_ptr<MemoryNode> &node, std::vector<MemoryType> &candidate, std::vector<MemoryType> &paths) const {
    for (auto &dest : node->dests) {
        if (std::find(candidate.begin(), candidate.end(), dest) != candidate.end()) {
            continue;
        }
        candidate.push_back(dest);
        if (dest != target) {
            DFS(target, nodes.at(dest), candidate, paths);
            candidate.pop_back();
            continue;
        }
        if ((!paths.empty()) && (paths.size() <= candidate.size())) {
            candidate.pop_back();
            continue;
        }
        paths.clear();
        for (auto &t : candidate) {
            paths.push_back(t);
        }
        candidate.pop_back();
    }
}

bool MemoryGraph::FindNearestPath(MemoryType from, MemoryType to, std::vector<MemoryType> &paths) const {
    if (nodes.count(from) == 0) {
        return false;
    }
    if (nodes.count(to) == 0) {
        return false;
    }
    std::vector<MemoryType> candidate = {from};
    paths.clear();
    const auto it = nodes.find(from);
    DFS(to, it->second, candidate, paths);
    return true;
}

void Platform::LoadPlatformInfo(const PlatformParser &parser) {
    std::string archType;
    std::string socVersion;
    std::unordered_map<std::string, std::string> versionInfo;
    if (parser.GetStringVal(version, npuArchInfo, archType)) {
        GetSoc().SetNPUArch(archType);
    }
    if (parser.GetStringVal(version, shortSocVersion, socVersion)) {
        GetSoc().SetShortSocVersion(socVersion);
    }
    if (parser.GetCCECVersion(versionInfo)) {
        GetSoc().SetCCECVersion(versionInfo);
    }
    if (parser.GetCoreVersion(versionInfo)) {
        GetSoc().SetCoreVersion(versionInfo);
    }
    size_t coreNum;
    if (parser.GetSizeVal(socInfo, aiCoreCnt, coreNum)) {
        GetSoc().SetAICoreNum(coreNum);
    }
    if (parser.GetSizeVal(socInfo, cubeCoreCnt, coreNum)) {
        GetSoc().SetAICCoreNum(coreNum);
    }
    if (parser.GetSizeVal(socInfo, vectorCoreCnt, coreNum)) {
        GetSoc().SetAIVCoreNum(coreNum);
    }
    if (parser.GetSizeVal(socInfo, aiCpuCnt, coreNum)) {
        GetSoc().SetAICPUNum(coreNum);
    }
    size_t memoryLimit;
    if (parser.GetSizeVal(aiCoreSpec, l0aSize, memoryLimit)) {
        GetAICCore().AddMemory(MemoryInfo(MemoryType::MEM_L0A, memoryLimit));
    }
    if (parser.GetSizeVal(aiCoreSpec, l0bSize, memoryLimit)) {
        GetAICCore().AddMemory(MemoryInfo(MemoryType::MEM_L0B, memoryLimit));
    }
    if (parser.GetSizeVal(aiCoreSpec, l0cSize, memoryLimit)) {
        GetAICCore().AddMemory(MemoryInfo(MemoryType::MEM_L0C, memoryLimit));
    }
    if (parser.GetSizeVal(aiCoreSpec, l1Size, memoryLimit)) {
        GetAIVCore().AddMemory(MemoryInfo(MemoryType::MEM_L1, memoryLimit));
    }
    if (parser.GetSizeVal(aiCoreSpec, ubSize, memoryLimit)) {
        GetAIVCore().AddMemory(MemoryInfo(MemoryType::MEM_UB, memoryLimit));
    }
}

void Platform::ObtainPlatformInfo() {
    std::string socVersion;
    using GetSocVerFunc = bool (*)(std::string &);
    std::string socVerFuncName = "GetrtSocVersion";
    auto socVerFunc = (GetSocVerFunc)GetSymbol(socVerFuncName.c_str());
    if (socVerFunc(socVersion)) {
        npu::tile_fwk::CmdParser cmdparser;
        LoadPlatformInfo(cmdparser);
    } else {
        std::string srcPath;
        SimulationPlatform simulationPlatform;
        simulationPlatform.GetSimulationPlatformRealPath(srcPath);
        npu::tile_fwk::INIParser iniparser;
        iniparser.Initialize(srcPath);
        LoadPlatformInfo(iniparser);
    }
    std::vector<std::pair<MemoryType, MemoryType>> dataPath;
    InternalParser internalParser = InternalParser(NPUArchToString(GetSoc().GetNPUArch()));
    internalParser.LoadInternalInfo();
    if (internalParser.GetDataPath(dataPath)) {
        GetDie().SetMemoryPath(dataPath);
    }
}
}