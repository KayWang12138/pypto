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
 * \file runtime.cpp
 * \brief
 */

#include "pybind_common.h"

#include <climits>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <dlfcn.h>
#include <iostream>
#include <initializer_list>
#include <memory>
#include <mutex>
#include <sys/stat.h>
#include <string>
#include <thread>
#include <chrono>
#include <unordered_map>
#include <utility>
#include <vector>
#include "tilefwk/tile_shape.h"
#include "interface/interpreter/raw_tensor_data.h"
#include "interface/tileop/distributed/hccl_context.h"
#include "machine/runtime/device_launcher_binding.h"
#include "machine/runtime/emulation_launcher.h"
#include "machine/host/perf_analysis.h"
#include "machine/runtime/mc2_tiling.h"
#ifdef BUILD_WITH_CANN
#include "hccl/hccl.h"
#include "hccl/hccl_types.h"
#include "acl/acl.h"
#include "runtime/rt.h"
extern "C" HcclResult HcomGetCommHandleByGroup(const char *group, HcclComm *commHandle);
#endif
#ifdef BUILD_WITH_CANN_SHMEM
#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmissing-field-initializers"
#endif
#include "shmem.h"
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif
#endif

using namespace npu::tile_fwk;
using namespace npu::tile_fwk::dynamic;

namespace {
#ifdef BUILD_WITH_CANN
extern "C" int HcclAllocComResourceByTiling(void* comm, void *stream, void *mc2Tiling, void **commContext);

std::mutex g_ctxMutex;
std::unordered_map<uint64_t, uint64_t> g_hcclContextCache;

#ifdef BUILD_WITH_CANN_SHMEM
bool IsShmemGroupName(const std::string &groupName)
{
    return groupName.find("shmem_group") != std::string::npos;
}
#endif

#if defined(__GNUC__)
__attribute__((format(printf, 1, 2)))
#endif
void ShmemLog(const char *format, ...)
{
    FILE *commLog = fopen("/tmp/pypto_commgroup.log", "a");
    if (commLog == nullptr) {
        return;
    }
    va_list args;
    va_start(args, format);
    vfprintf(commLog, format, args);
    va_end(args);
    fclose(commLog);
}

#ifdef BUILD_WITH_CANN
void *GetHcclLibHandle()
{
    static void *handle = []() {
        void *lib = dlopen("libhccl.so", RTLD_LAZY | RTLD_NOLOAD);
        if (lib == nullptr) {
            lib = dlopen("libhccl.so", RTLD_LAZY | RTLD_GLOBAL);
        }
        return lib;
    }();
    return handle;
}

template <typename Fn>
Fn LoadHcclFunc(const char *name)
{
    auto handle = GetHcclLibHandle();
    if (handle == nullptr) {
        return nullptr;
    }
    return reinterpret_cast<Fn>(dlsym(handle, name));
}
#endif

#ifdef BUILD_WITH_CANN_SHMEM
constexpr uint64_t SHMEM_ALLOC_SIZE = 1UL << 30;
constexpr uint64_t SHMEM_HALF_SIZE = 1UL << 29;
constexpr uint64_t SHMEM_LOCAL_MEM_SIZE = 1UL << 31;
constexpr uint64_t SHMEM_HANDLE_PTR_MIN = 1UL << 20;
constexpr const char *DEFAULT_SHMEM_UID_PATH = "/tmp/pypto_shmem_uid";
constexpr const char *DEFAULT_SHMEM_HOME = "/usr/local/Ascend/shmem/1.0.0/shmem";

bool PathExists(const std::string &path)
{
    struct stat st {};
    return stat(path.c_str(), &st) == 0;
}

void EnsureShmemBootstrapPath()
{
    const char *home = std::getenv("SHMEM_HOME_PATH");
    std::string base = (home && home[0] != '\0') ? home : DEFAULT_SHMEM_HOME;
    std::string libPath = base + "/lib";
    if (!PathExists(libPath)) {
        std::string candidate = base + "/shmem/lib";
        if (PathExists(candidate)) {
            libPath = candidate;
        }
    }
    const char *ld = std::getenv("LD_LIBRARY_PATH");
    if (ld != nullptr && std::strstr(ld, libPath.c_str()) != nullptr) {
        return;
    }
    std::string newLd = libPath;
    if (ld != nullptr && ld[0] != '\0') {
        newLd.append(":").append(ld);
    }
    (void)setenv("LD_LIBRARY_PATH", newLd.c_str(), 1);
}

int64_t ReadEnvAny(const std::initializer_list<const char *> &names);
int GetWorldSizeFromEnv();
std::string GetShmemSessionIdFromEnv(const char *addrName, const char *portName, int portOffset);
std::string StripShmemIpPortScheme(const std::string &value);

int GetDefaultShmemPortOffset(int world)
{
    int offset = 11;
    if (world > 0) {
        int extra = world * 2;
        if (extra > 1000) {
            extra %= 1000;
        }
        offset += extra;
    }
    return offset;
}

void EnsureShmemBootstrapSession()
{
    if (std::getenv("ACLSHMEM_UID_SESSION_ID") != nullptr || std::getenv("ACLSHMEM_UID_SOCK_IFNAME") != nullptr) {
        return;
    }
    int world = GetWorldSizeFromEnv();
    if (world <= 0) {
        return;
    }
    std::string sessionId;
    const char *ipPortEnv = std::getenv("PYPTO_SHMEM_IP_PORT");
    if (ipPortEnv == nullptr || ipPortEnv[0] == '\0') {
        ipPortEnv = std::getenv("ACLSHMEM_IP_PORT");
    }
    if (ipPortEnv != nullptr && ipPortEnv[0] != '\0') {
        sessionId = StripShmemIpPortScheme(ipPortEnv);
    }
    if (sessionId.empty()) {
        sessionId = GetShmemSessionIdFromEnv("ACLSHMEM_MASTER_ADDR", "ACLSHMEM_MASTER_PORT", 0);
    }
    if (sessionId.empty()) {
        int offset = GetDefaultShmemPortOffset(world);
        sessionId = GetShmemSessionIdFromEnv("MASTER_ADDR", "MASTER_PORT", offset);
    }
    if (sessionId.empty()) {
        int localWorld = ReadEnvAny({"LOCAL_WORLD_SIZE", "OMPI_COMM_WORLD_LOCAL_SIZE", "MPI_LOCALNRANKS"});
        if (localWorld > 0 && localWorld == world) {
            sessionId = "127.0.0.1:19777";
        }
    }
    if (!sessionId.empty()) {
        const char *uidPathEnv = std::getenv("SHMEM_UID_PATH");
        if (uidPathEnv == nullptr || uidPathEnv[0] == '\0') {
            std::string sanitized;
            sanitized.reserve(sessionId.size());
            for (char c : sessionId) {
                if ((c >= '0' && c <= '9') || (c >= 'A' && c <= 'Z') ||
                    (c >= 'a' && c <= 'z') || c == '.' || c == '-') {
                    sanitized.push_back(c);
                } else {
                    sanitized.push_back('_');
                }
            }
            std::string uidPath = std::string(DEFAULT_SHMEM_UID_PATH) + "_" + sanitized;
            (void)setenv("SHMEM_UID_PATH", uidPath.c_str(), 0);
        }
    }
    if (!sessionId.empty()) {
        (void)setenv("ACLSHMEM_UID_SESSION_ID", sessionId.c_str(), 0);
    }
}

std::string NormalizeShmemIpPort(const char *value)
{
    if (value == nullptr || value[0] == '\0') {
        return {};
    }
    std::string ipPort(value);
    if (ipPort.rfind("tcp://", 0) != 0 && ipPort.rfind("tcp6://", 0) != 0) {
        ipPort.insert(0, "tcp://");
    }
    return ipPort;
}

std::string StripShmemIpPortScheme(const std::string &value)
{
    if (value.rfind("tcp://", 0) == 0) {
        return value.substr(6);
    }
    if (value.rfind("tcp6://", 0) == 0) {
        return value.substr(7);
    }
    return value;
}

std::string FormatShmemIpPort(const std::string &address, int port)
{
    if (address.find(':') != std::string::npos && address.find(']') == std::string::npos) {
        return "tcp6://[" + address + "]:" + std::to_string(port);
    }
    return "tcp://" + address + ":" + std::to_string(port);
}

std::string FormatShmemSessionId(const std::string &address, int port)
{
    if (address.find(':') != std::string::npos && address.find(']') == std::string::npos) {
        return "[" + address + "]:" + std::to_string(port);
    }
    return address + ":" + std::to_string(port);
}

std::string GetShmemIpPortFromEnv(const char *addrName, const char *portName, int portOffset)
{
    const char *addr = std::getenv(addrName);
    const char *port = std::getenv(portName);
    if (addr == nullptr || port == nullptr || addr[0] == '\0' || port[0] == '\0') {
        return {};
    }
    char *end = nullptr;
    long portVal = std::strtol(port, &end, 10);
    if (end == port || (end != nullptr && *end != '\0') || portVal <= 0 || portVal > INT_MAX - portOffset) {
        return {};
    }
    portVal += portOffset;
    if (portVal <= 0 || portVal > UINT16_MAX) {
        return {};
    }
    return FormatShmemIpPort(addr, static_cast<int>(portVal));
}

std::string GetShmemSessionIdFromEnv(const char *addrName, const char *portName, int portOffset)
{
    const char *addr = std::getenv(addrName);
    const char *port = std::getenv(portName);
    if (addr == nullptr || port == nullptr || addr[0] == '\0' || port[0] == '\0') {
        return {};
    }
    char *end = nullptr;
    long portVal = std::strtol(port, &end, 10);
    if (end == port || (end != nullptr && *end != '\0') || portVal <= 0 || portVal > INT_MAX - portOffset) {
        return {};
    }
    portVal += portOffset;
    if (portVal <= 0 || portVal > UINT16_MAX) {
        return {};
    }
    return FormatShmemSessionId(addr, static_cast<int>(portVal));
}

std::string GetDefaultShmemIpPort(int world)
{
    const char *env = std::getenv("PYPTO_SHMEM_IP_PORT");
    if (env == nullptr || env[0] == '\0') {
        env = std::getenv("ACLSHMEM_IP_PORT");
    }
    std::string fromEnv = NormalizeShmemIpPort(env);
    if (!fromEnv.empty()) {
        return fromEnv;
    }
    std::string fromMaster = GetShmemIpPortFromEnv("ACLSHMEM_MASTER_ADDR", "ACLSHMEM_MASTER_PORT", 0);
    if (!fromMaster.empty()) {
        return fromMaster;
    }
    int offset = GetDefaultShmemPortOffset(world);
    fromMaster = GetShmemIpPortFromEnv("MASTER_ADDR", "MASTER_PORT", offset);
    if (!fromMaster.empty()) {
        return fromMaster;
    }
    int localWorld = ReadEnvAny({"LOCAL_WORLD_SIZE", "OMPI_COMM_WORLD_LOCAL_SIZE", "MPI_LOCALNRANKS"});
    if (localWorld > 0 && localWorld == world) {
        return "tcp://127.0.0.1:19777";
    }
    return {};
}

bool LoadOrCreateShmemUniqueId(int rank, shmem_uniqueid_t &uid)
{
    const char *uidPath = std::getenv("SHMEM_UID_PATH");
    if (uidPath == nullptr || uidPath[0] == '\0') {
        uidPath = DEFAULT_SHMEM_UID_PATH;
    }
    if (rank == 0) {
        int getRet = aclshmemx_get_uniqueid(&uid);
        if (getRet != ACLSHMEM_SUCCESS) {
            ShmemLog("[pypto] shmem_get_uniqueid failed ret=%d\n", getRet);
            return false;
        }
        FILE *file = fopen(uidPath, "wb");
        if (file == nullptr) {
            ShmemLog("[pypto] shmem uid write failed path=%s\n", uidPath);
            return false;
        }
        size_t wrote = fwrite(&uid, 1, sizeof(uid), file);
        fclose(file);
        if (wrote != sizeof(uid)) {
            ShmemLog("[pypto] shmem uid write size mismatch wrote=%zu\n", wrote);
            return false;
        }
        return true;
    }

    for (int attempt = 0; attempt < 200; ++attempt) {
        FILE *file = fopen(uidPath, "rb");
        if (file != nullptr) {
            size_t read = fread(&uid, 1, sizeof(uid), file);
            fclose(file);
            if (read == sizeof(uid)) {
                return true;
            }
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    ShmemLog("[pypto] shmem uid read timeout path=%s\n", uidPath);
    return false;
}

int64_t ReadEnvInt(const char *name)
{
    if (name == nullptr) {
        return -1;
    }
    const char *value = std::getenv(name);
    if (value == nullptr || value[0] == '\0') {
        return -1;
    }
    char *end = nullptr;
    long parsed = std::strtol(value, &end, 10);
    if (end == value || (end != nullptr && *end != '\0')) {
        return -1;
    }
    return static_cast<int64_t>(parsed);
}

int64_t ReadEnvAny(const std::initializer_list<const char *> &names)
{
    for (const char *name : names) {
        int64_t value = ReadEnvInt(name);
        if (value >= 0) {
            return value;
        }
    }
    return -1;
}

int GetRankFromEnv(uint64_t fallbackHandle)
{
    int64_t rank = ReadEnvAny({"RANK", "RANK_ID", "OMPI_COMM_WORLD_RANK", "PMI_RANK", "SLURM_PROCID", "LOCAL_RANK"});
    if (rank < 0) {
        int64_t tileRank = TileShape::Current().GetDistRankId();
        if (tileRank >= 0 && tileRank < INT16_MAX) {
            rank = tileRank;
        }
    }
    if (rank < 0 && fallbackHandle < SHMEM_HANDLE_PTR_MIN) {
        rank = static_cast<int64_t>(fallbackHandle);
    }
    return static_cast<int>(rank);
}

int GetWorldSizeFromEnv()
{
    int64_t world = ReadEnvAny({"WORLD_SIZE", "RANK_SIZE", "HCCL_WORLD_SIZE", "OMPI_COMM_WORLD_SIZE",
        "PMI_SIZE", "SLURM_NTASKS"});
    if (world <= 0) {
        const auto &rankTile = TileShape::Current().GetDistTileRank();
        if (rankTile[1] > 0) {
            world = rankTile[1];
        }
    }
    return static_cast<int>(world);
}

uint64_t AllocShmemContextLocked(uint64_t hcclHandle)
{
    EnsureShmemBootstrapPath();
    EnsureShmemBootstrapSession();
    static std::once_flag tlsOnce;
    std::call_once(tlsOnce, []() {
        int tlsRet = shmem_set_conf_store_tls(false, "", 0);
        if (tlsRet != ACLSHMEM_SUCCESS) {
            ShmemLog("[pypto] shmem_set_conf_store_tls failed ret=%d\n", tlsRet);
        }
    });
    int initStatus = shmem_init_status();
    if (initStatus != ACLSHMEM_STATUS_IS_INITIALIZED) {
        int rank = GetRankFromEnv(hcclHandle);
        int world = GetWorldSizeFromEnv();
        if (rank < 0 || world <= 0) {
            ShmemLog("[pypto] shmem init skipped: rank=%d world=%d handle=%lu\n", rank, world, hcclHandle);
            return 0;
        }
        bool inited = false;
        bool enableUniqueId = (std::getenv("PYPTO_SHMEM_USE_UNIQUEID") != nullptr);
        if (enableUniqueId) {
            shmem_uniqueid_t uid{};
            if (LoadOrCreateShmemUniqueId(rank, uid)) {
                shmem_init_attr_t attributes{};
                int setRet = shmem_set_attr_uniqueid_args(rank, world, SHMEM_LOCAL_MEM_SIZE, &uid, &attributes);
                if (setRet != ACLSHMEM_SUCCESS) {
                    ShmemLog("[pypto] shmem_set_attr_uniqueid_args failed ret=%d\n", setRet);
                } else {
                    attributes.option_attr.data_op_engine_type = ACLSHMEM_DATA_OP_MTE;
                    int initRet = shmem_init_attr(ACLSHMEMX_INIT_WITH_UNIQUEID, &attributes);
                    if (initRet == ACLSHMEM_SUCCESS) {
                        inited = true;
                    } else {
                        ShmemLog("[pypto] shmem_init_attr uniqueid failed ret=%d\n", initRet);
                    }
                }
            }
        }
        if (!inited) {
            std::string ipPort = GetDefaultShmemIpPort(world);
            if (ipPort.empty()) {
                ShmemLog("[pypto] shmem default init skipped: ip_port missing\n");
                return 0;
            }
            shmem_init_attr_t attributes{};
            shmem_uniqueid_t defaultUid{};
            defaultUid.version = ACLSHMEM_UNIQUEID_VERSION;
            int setRet = shmem_set_attr_uniqueid_args(rank, world, SHMEM_LOCAL_MEM_SIZE, &defaultUid, &attributes);
            if (setRet != ACLSHMEM_SUCCESS) {
                ShmemLog("[pypto] shmem_set_attr_uniqueid_args default failed ret=%d\n", setRet);
                return 0;
            }
            std::strncpy(attributes.ip_port, ipPort.c_str(), sizeof(attributes.ip_port) - 1);
            attributes.ip_port[sizeof(attributes.ip_port) - 1] = '\0';
            attributes.option_attr.data_op_engine_type = ACLSHMEM_DATA_OP_MTE;
            attributes.option_attr.sockFd = -1;
            int initRet = shmem_init_attr(ACLSHMEMX_INIT_WITH_DEFAULT, &attributes);
            if (initRet != ACLSHMEM_SUCCESS) {
                ShmemLog("[pypto] shmem_init_attr default failed ret=%d\n", initRet);
                return 0;
            }
        }
        for (int attempt = 0; attempt < 200; ++attempt) {
            if (shmem_init_status() == ACLSHMEM_STATUS_IS_INITIALIZED) {
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
        if (shmem_init_status() != ACLSHMEM_STATUS_IS_INITIALIZED) {
            ShmemLog("[pypto] shmem init status not initialized\n");
            return 0;
        }
    }

    int rank = shmem_my_pe();
    int world = shmem_n_pes();
    if (rank < 0 || world <= 0) {
        int envRank = GetRankFromEnv(hcclHandle);
        int envWorld = GetWorldSizeFromEnv();
        if (rank < 0) {
            rank = envRank;
        }
        if (world <= 0) {
            world = envWorld;
        }
    }
    if (rank < 0 || world <= 0 || world > static_cast<int>(TileOp::AICPU_MAX_RANK_NUM_V1)) {
        ShmemLog("[pypto] shmem rank/world invalid: rank=%d world=%d\n", rank, world);
        return 0;
    }

    uint64_t baseAddr = 0;
    if (hcclHandle >= SHMEM_HANDLE_PTR_MIN) {
        baseAddr = hcclHandle;
    } else {
        void *shmemPtr = shmem_malloc(SHMEM_ALLOC_SIZE);
        if (shmemPtr == nullptr) {
            ShmemLog("[pypto] shmem_malloc failed size=%lu\n", SHMEM_ALLOC_SIZE);
            return 0;
        }
        aclshmem_barrier_all();
        baseAddr = reinterpret_cast<uint64_t>(shmemPtr);
    }

    TileOp::HcclCombinOpParam hostParam{};
    hostParam.rankId = static_cast<uint32_t>(rank);
    hostParam.rankNum = static_cast<uint32_t>(world);
    hostParam.winSize = SHMEM_HALF_SIZE;
    hostParam.winExpSize = SHMEM_HALF_SIZE;
    hostParam.padding[0] = TileOp::HCCL_CONTEXT_MAGIC;
    for (int pe = 0; pe < world; ++pe) {
        void *winIn = shmem_ptr(reinterpret_cast<void *>(baseAddr), pe);
        void *winExp = shmem_ptr(reinterpret_cast<void *>(baseAddr + SHMEM_HALF_SIZE), pe);
        if (winIn == nullptr || winExp == nullptr) {
            ShmemLog("[pypto] shmem_ptr failed for pe=%d base=0x%lx\n", pe, baseAddr);
            return 0;
        }
        hostParam.windowsIn[pe] = reinterpret_cast<uint64_t>(winIn);
        hostParam.windowsOut[pe] = reinterpret_cast<uint64_t>(winIn);
        hostParam.windowsExp[pe] = reinterpret_cast<uint64_t>(winExp);
    }

    void *commContext = nullptr;
    auto mallocRet = rtMalloc(&commContext, sizeof(hostParam), RT_MEMORY_HBM, 0);
    if (mallocRet != RT_ERROR_NONE || commContext == nullptr) {
        ShmemLog("[pypto] rtMalloc hcclContext failed ret=%d\n", mallocRet);
        return 0;
    }
    auto memcpyRet = rtMemcpy(commContext, sizeof(hostParam), &hostParam, sizeof(hostParam), RT_MEMCPY_HOST_TO_DEVICE);
    if (memcpyRet != RT_ERROR_NONE) {
        ShmemLog("[pypto] rtMemcpy hcclContext failed ret=%d\n", memcpyRet);
        (void)rtFree(commContext);
        return 0;
    }
    uint64_t contextVal = reinterpret_cast<uint64_t>(commContext);
    g_hcclContextCache[hcclHandle] = contextVal;
    ShmemLog("[pypto] shmemContext rank=%d world=%d base=0x%lx ctx=0x%lx\n", rank, world, baseAddr, contextVal);
    return contextVal;
}
#endif

#ifdef BUILD_WITH_CANN
uint64_t BuildHcclCombinContextFromOpRes(void *commContext)
{
    if (commContext == nullptr) {
        return 0;
    }
    bool debugHccl = (std::getenv("PYPTO_HCCL_DEBUG") != nullptr);
    auto opRes = std::make_unique<TileOp::HcclOpResParam>();
    auto memcpyRet = rtMemcpy(opRes.get(), sizeof(*opRes), commContext, sizeof(*opRes), RT_MEMCPY_DEVICE_TO_HOST);
    if (memcpyRet != RT_ERROR_NONE) {
        ShmemLog("[pypto] rtMemcpy HcclOpResParam failed ret=%d\n", memcpyRet);
        return 0;
    }
    if (opRes->rankSize == 0 || opRes->rankSize > TileOp::AICPU_MAX_RANK_NUM_V1) {
        ShmemLog("[pypto] HcclOpResParam rankSize invalid: %u\n", opRes->rankSize);
        return 0;
    }
    TileOp::HcclCombinOpParam hostParam{};
    hostParam.rankId = opRes->localUsrRankId;
    hostParam.rankNum = opRes->rankSize;
    hostParam.winSize = opRes->winSize;
    hostParam.winExpSize = opRes->winExpSize;
    hostParam.padding[0] = TileOp::HCCL_CONTEXT_MAGIC;
    std::vector<uint8_t> filled(static_cast<size_t>(hostParam.rankNum), 0U);
    if (debugHccl) {
        ShmemLog("[pypto] opRes rankId=%u rankNum=%u winSize=%lu winExpSize=%lu localIn=0x%lx localOut=0x%lx localExp=0x%lx\n",
            opRes->localUsrRankId, opRes->rankSize, opRes->winSize, opRes->winExpSize,
            opRes->localWindowsIn, opRes->localWindowsOut, opRes->localWindowsExp);
        ShmemLog("[pypto] opRes remoteResNum=%u\n", opRes->remoteResNum);
    }
    for (uint32_t rank = 0; rank < hostParam.rankNum; ++rank) {
        if (rank == hostParam.rankId) {
            hostParam.windowsIn[rank] = opRes->localWindowsIn;
            hostParam.windowsOut[rank] = opRes->localWindowsOut;
            hostParam.windowsExp[rank] = opRes->localWindowsExp;
            filled[rank] = 1U;
            continue;
        }
    }
    auto applyRel = [&](const TileOp::HcclRankRelationResV2 &rel, uint32_t srcIdx) {
        if (rel.remoteUsrRankId >= hostParam.rankNum) {
            if (debugHccl) {
                ShmemLog("[pypto] opRes rel idx=%u remoteUsrRankId=%u out of range\n", srcIdx, rel.remoteUsrRankId);
            }
            return;
        }
        if (rel.remoteUsrRankId == hostParam.rankId) {
            return;
        }
        if (filled[rel.remoteUsrRankId] != 0U) {
            return;
        }
        hostParam.windowsIn[rel.remoteUsrRankId] = rel.windowsIn;
        hostParam.windowsOut[rel.remoteUsrRankId] = rel.windowsOut;
        hostParam.windowsExp[rel.remoteUsrRankId] = rel.windowsExp;
        filled[rel.remoteUsrRankId] = 1U;
        if (debugHccl) {
            ShmemLog("[pypto] opRes rel idx=%u rank=%u winIn=0x%lx winOut=0x%lx winExp=0x%lx\n",
                srcIdx, rel.remoteUsrRankId, rel.windowsIn, rel.windowsOut, rel.windowsExp);
        }
    };
    uint32_t maxRemoteRes = opRes->remoteResNum;
    if (maxRemoteRes > TileOp::AICPU_MAX_RANK_NUM) {
        maxRemoteRes = TileOp::AICPU_MAX_RANK_NUM;
    }
    for (uint32_t idx = 0; idx < maxRemoteRes; ++idx) {
        uint64_t remotePtr = opRes->remoteRes[idx].nextDevicePtr;
        if (debugHccl) {
            ShmemLog("[pypto] opRes remote idx=%u hostPtr=0x%lx devPtr=0x%lx\n",
                idx, opRes->remoteRes[idx].nextHostPtr, opRes->remoteRes[idx].nextDevicePtr);
        }
        if (remotePtr == 0) {
            continue;
        }
        TileOp::HcclRankRelationResV2 rel{};
        auto relRet = rtMemcpy(&rel, sizeof(rel), reinterpret_cast<void *>(remotePtr),
            sizeof(rel), RT_MEMCPY_DEVICE_TO_HOST);
        if (relRet != RT_ERROR_NONE) {
            ShmemLog("[pypto] rtMemcpy HcclRankRelationResV2 failed idx=%u ret=%d\n", idx, relRet);
            continue;
        }
        applyRel(rel, idx);
    }
    for (uint32_t rank = 0; rank < hostParam.rankNum; ++rank) {
        if (rank == hostParam.rankId || filled[rank] != 0U) {
            continue;
        }
        uint64_t remotePtr = opRes->remoteRes[rank].nextDevicePtr;
        if (debugHccl) {
            ShmemLog("[pypto] opRes remote rank=%u ptr=0x%lx\n", rank, remotePtr);
        }
        if (remotePtr == 0) {
            continue;
        }
        TileOp::HcclRankRelationResV2 rel{};
        auto relRet = rtMemcpy(&rel, sizeof(rel), reinterpret_cast<void *>(remotePtr),
            sizeof(rel), RT_MEMCPY_DEVICE_TO_HOST);
        if (relRet != RT_ERROR_NONE) {
            ShmemLog("[pypto] rtMemcpy HcclRankRelationResV2 failed rank=%u ret=%d\n", rank, relRet);
            continue;
        }
        applyRel(rel, rank);
    }
    if (debugHccl) {
        for (uint32_t rank = 0; rank < hostParam.rankNum; ++rank) {
            if (rank == hostParam.rankId || filled[rank] != 0U) {
                continue;
            }
            ShmemLog("[pypto] opRes missing rank=%u windows\n", rank);
        }
    }

    void *newContext = nullptr;
    auto mallocRet = rtMalloc(&newContext, sizeof(hostParam), RT_MEMORY_HBM, 0);
    if (mallocRet != RT_ERROR_NONE || newContext == nullptr) {
        ShmemLog("[pypto] rtMalloc HcclCombinOpParam failed ret=%d\n", mallocRet);
        return 0;
    }
    auto copyRet = rtMemcpy(newContext, sizeof(hostParam), &hostParam, sizeof(hostParam), RT_MEMCPY_HOST_TO_DEVICE);
    if (copyRet != RT_ERROR_NONE) {
        ShmemLog("[pypto] rtMemcpy HcclCombinOpParam failed ret=%d\n", copyRet);
        (void)rtFree(newContext);
        return 0;
    }
    return reinterpret_cast<uint64_t>(newContext);
}
#endif

uint64_t AllocHcclContext(uint64_t hcclHandle, const std::string &groupName, void *aicoreStream)
{
    bool debugHccl = (std::getenv("PYPTO_HCCL_DEBUG") != nullptr);
    auto logAttempt = [&](const char *tag, int retVal, void *ctx, uint32_t mode) {
        if (!debugHccl) {
            return;
        }
        FILE *commLog = fopen("/tmp/pypto_commgroup.log", "a");
        if (commLog != nullptr) {
            fprintf(commLog, "[pypto] %s ret=%d ctx=%p mode=%u group=%s\n",
                tag, retVal, ctx, mode, groupName.c_str());
            fclose(commLog);
        }
    };
    if (groupName.empty()) {
        return 0;
    }
    std::lock_guard<std::mutex> lock(g_ctxMutex);
    auto it = g_hcclContextCache.find(hcclHandle);
    if (it != g_hcclContextCache.end()) {
        return it->second;
    }
#ifdef BUILD_WITH_CANN
    auto aclRet = aclInit(nullptr);
    if (debugHccl && aclRet != ACL_SUCCESS && aclRet != ACL_ERROR_REPEAT_INITIALIZE) {
        FILE *commLog = fopen("/tmp/pypto_commgroup.log", "a");
        if (commLog != nullptr) {
            fprintf(commLog, "[pypto] aclInit failed ret=%d\n", static_cast<int>(aclRet));
            fclose(commLog);
        }
    }
#endif
#ifdef BUILD_WITH_CANN_SHMEM
    if (IsShmemGroupName(groupName)) {
        return AllocShmemContextLocked(hcclHandle);
    }
#endif
    if (hcclHandle == 0) {
        if (debugHccl) {
            FILE *commLog = fopen("/tmp/pypto_commgroup.log", "a");
            if (commLog != nullptr) {
                fprintf(commLog, "[pypto] hcclHandle=0 for group=%s\n", groupName.c_str());
                fclose(commLog);
            }
        }
        return 0;
    }
    HcclComm commHandle = reinterpret_cast<HcclComm>(hcclHandle);
#ifdef BUILD_WITH_CANN
    auto readEnvInt = [](const char *name) -> int {
        if (name == nullptr) {
            return -1;
        }
        const char *value = std::getenv(name);
        if (value == nullptr || value[0] == '\0') {
            return -1;
        }
        char *end = nullptr;
        long parsed = std::strtol(value, &end, 10);
        if (end == value || (end != nullptr && *end != '\0')) {
            return -1;
        }
        return static_cast<int>(parsed);
    };
    auto readEnvAny = [&](const std::initializer_list<const char *> &names) -> int {
        for (const char *name : names) {
            int value = readEnvInt(name);
            if (value > 0) {
                return value;
            }
        }
        return -1;
    };
    auto createGroup = LoadHcclFunc<HcclResult (*)(const char *, uint32_t, uint32_t *)>("HcomCreateGroup");
    if (createGroup != nullptr) {
        int world = readEnvAny({"WORLD_SIZE", "RANK_SIZE", "HCCL_WORLD_SIZE", "OMPI_COMM_WORLD_SIZE",
            "PMI_SIZE", "SLURM_NTASKS"});
        if (world <= 0) {
            const auto &rankTile = TileShape::Current().GetDistTileRank();
            if (rankTile[1] > 0) {
                world = static_cast<int>(rankTile[1]);
            }
        }
        if (world > 0) {
            std::vector<uint32_t> ranks(static_cast<size_t>(world));
            for (int i = 0; i < world; ++i) {
                ranks[static_cast<size_t>(i)] = static_cast<uint32_t>(i);
            }
            auto retCreate = createGroup(groupName.c_str(), static_cast<uint32_t>(world), ranks.data());
            logAttempt("HcomCreateGroup", static_cast<int>(retCreate), nullptr, 0);
        }
    }
    HcclComm hcomHandle = nullptr;
    HcclResult hcomRet = HcomGetCommHandleByGroup(groupName.c_str(), &hcomHandle);
    if (hcomRet == HCCL_SUCCESS && hcomHandle != nullptr) {
        commHandle = hcomHandle;
    } else if (debugHccl) {
        FILE *commLog = fopen("/tmp/pypto_commgroup.log", "a");
        if (commLog != nullptr) {
            fprintf(commLog, "[pypto] HcomGetCommHandleByGroup failed group=%s ret=%d handle=0x%lx\n",
                groupName.c_str(), static_cast<int>(hcomRet), reinterpret_cast<uint64_t>(hcomHandle));
            fclose(commLog);
        }
    }
#endif
    void *commContext = nullptr;
    int ret = -1;
    Mc2CommConfigV2 commConfigV2 = {};
    if (MakeMc2TilingStructV2(commConfigV2, groupName) == 0) {
        ret = HcclAllocComResourceByTiling(reinterpret_cast<void *>(commHandle),
            aicoreStream, &commConfigV2, &commContext);
        logAttempt("HcclAllocComResourceByTiling(v2)", ret, commContext, 0);
    }
    if (ret != 0 || commContext == nullptr) {
        commContext = nullptr;
        Mc2CommConfig commConfig = {};
        if (MakeMc2TilingStruct(commConfig, groupName) == 0) {
            ret = HcclAllocComResourceByTiling(reinterpret_cast<void *>(commHandle),
                aicoreStream, &commConfig, &commContext);
            logAttempt("HcclAllocComResourceByTiling(v1)", ret, commContext, 0);
        }
    }
    if (ret != 0 || commContext == nullptr) {
        commContext = nullptr;
        Mc2CommConfig commConfigRetry = {};
        if (MakeMc2TilingStruct(commConfigRetry, groupName) == 0) {
            ret = HcclAllocComResourceByTiling(reinterpret_cast<void *>(commHandle), nullptr,
                &commConfigRetry, &commContext);
            logAttempt("HcclAllocComResourceByTiling(v1,null)", ret, commContext, 0);
        }
        if (ret != 0 || commContext == nullptr) {
            commContext = nullptr;
            Mc2CommConfigV2 commConfigV2Retry = {};
            if (MakeMc2TilingStructV2(commConfigV2Retry, groupName) == 0) {
                ret = HcclAllocComResourceByTiling(reinterpret_cast<void *>(commHandle), nullptr,
                    &commConfigV2Retry, &commContext);
                logAttempt("HcclAllocComResourceByTiling(v2,null)", ret, commContext, 0);
            }
        }
    }
    if (ret != 0 || commContext == nullptr) {
        using AllocResFn = HcclResult (*)(HcclComm, uint32_t, void **);
        auto allocRes = LoadHcclFunc<AllocResFn>("HcclAllocComResource");
        if (allocRes != nullptr) {
            for (uint32_t mode : {0U, 1U}) {
                commContext = nullptr;
                auto allocRet = allocRes(commHandle, mode, &commContext);
                logAttempt("HcclAllocComResource", static_cast<int>(allocRet), commContext, mode);
                if (allocRet == HCCL_SUCCESS && commContext != nullptr) {
                    ret = 0;
                    break;
                }
            }
        }
    }
    if (ret != 0 || commContext == nullptr) {
        using CreateResFn = HcclResult (*)(const char *, uint32_t, void **);
        auto createRes = LoadHcclFunc<CreateResFn>("HcclCreateComResource");
        if (createRes != nullptr) {
            for (uint32_t mode : {0U, 1U}) {
                commContext = nullptr;
                auto createRet = createRes(groupName.c_str(), mode, &commContext);
                logAttempt("HcclCreateComResource", static_cast<int>(createRet), commContext, mode);
                if (createRet == HCCL_SUCCESS && commContext != nullptr) {
                    ret = 0;
                    break;
                }
            }
        }
    }
    if (ret != 0 || commContext == nullptr) {
        using HcomCreateResFn = HcclResult (*)(HcclComm, uint32_t, bool, void **, bool);
        auto hcomCreate = LoadHcclFunc<HcomCreateResFn>("HcomCreateComResourceByComm");
        if (hcomCreate != nullptr) {
            for (uint32_t mode : {0U, 1U}) {
                commContext = nullptr;
                auto createRet = hcomCreate(commHandle, mode, false, &commContext, true);
                logAttempt("HcomCreateComResourceByComm", static_cast<int>(createRet), commContext, mode);
                if (createRet == HCCL_SUCCESS && commContext != nullptr) {
                    ret = 0;
                    break;
                }
            }
        }
    }
    if ((ret != 0 || commContext == nullptr) && debugHccl) {
        FILE *commLog = fopen("/tmp/pypto_commgroup.log", "a");
        if (commLog != nullptr) {
            fprintf(commLog, "[pypto] HcclAllocComResourceByTiling failed group=%s ret=%d ctx=%p\n",
                groupName.c_str(), ret, commContext);
            fclose(commLog);
        }
    }
#ifdef BUILD_WITH_CANN_SHMEM
    if (ret != 0 || commContext == nullptr) {
        uint64_t shmemContext = AllocShmemContextLocked(hcclHandle);
        if (shmemContext != 0) {
            return shmemContext;
        }
    }
#endif
    if (ret != 0 || commContext == nullptr) {
        void *fallbackContext = reinterpret_cast<void *>(commHandle);
        if (fallbackContext == nullptr) {
            return 0;
        }
        uint64_t contextVal = 0;
        TileOp::HcclCombinOpParam probe{};
        auto probeRet = rtMemcpy(&probe, sizeof(probe), fallbackContext, sizeof(probe), RT_MEMCPY_DEVICE_TO_HOST);
        bool looksCombin = (probeRet == RT_ERROR_NONE && probe.padding[0] == TileOp::HCCL_CONTEXT_MAGIC &&
            probe.rankNum > 0 && probe.rankNum <= TileOp::AICPU_MAX_RANK_NUM_V1 && probe.rankId < probe.rankNum);
        if (looksCombin) {
            contextVal = reinterpret_cast<uint64_t>(fallbackContext);
        } else {
            uint64_t converted = BuildHcclCombinContextFromOpRes(fallbackContext);
            if (converted != 0) {
                contextVal = converted;
            }
        }
        if (contextVal != 0) {
            g_hcclContextCache[hcclHandle] = contextVal;
            return contextVal;
        }
        return 0;
    }
    if (ret == 0 && commContext != nullptr) {
        uint64_t contextVal = reinterpret_cast<uint64_t>(commContext);
        TileOp::HcclCombinOpParam probe{};
        auto probeRet = rtMemcpy(&probe, sizeof(probe), commContext, sizeof(probe), RT_MEMCPY_DEVICE_TO_HOST);
        bool looksCombin = (probeRet == RT_ERROR_NONE && probe.padding[0] == TileOp::HCCL_CONTEXT_MAGIC &&
            probe.rankNum > 0 && probe.rankNum <= TileOp::AICPU_MAX_RANK_NUM_V1 && probe.rankId < probe.rankNum);
        if (!looksCombin) {
            uint64_t converted = BuildHcclCombinContextFromOpRes(commContext);
            if (converted != 0) {
                contextVal = converted;
            } else {
                ShmemLog("[pypto] Hccl context conversion failed, using original context\n");
            }
        }
        g_hcclContextCache[hcclHandle] = contextVal;
        static bool logged = false;
        if (!logged) {
            FILE *commLog = fopen("/tmp/pypto_commgroup.log", "a");
            if (commLog != nullptr) {
                fprintf(commLog, "[pypto] hcclContext=0x%lx\n", contextVal);
                fclose(commLog);
            }
            TileOp::HcclCombinOpParam hostParam{};
            auto memcpyRet = rtMemcpy(&hostParam, sizeof(hostParam), reinterpret_cast<void *>(contextVal),
                sizeof(hostParam), RT_MEMCPY_DEVICE_TO_HOST);
            if (memcpyRet == RT_ERROR_NONE) {
                FILE *ctxLog = fopen("/tmp/pypto_commgroup.log", "a");
                if (ctxLog != nullptr) {
                    fprintf(ctxLog, "[pypto] hcclContext rankId=%u rankNum=%u winSize=%lu winExpSize=%lu\n",
                        hostParam.rankId, hostParam.rankNum, hostParam.winSize, hostParam.winExpSize);
                    fclose(ctxLog);
                }
            } else {
                FILE *errLog = fopen("/tmp/pypto_commgroup.log", "a");
                if (errLog != nullptr) {
                    fprintf(errLog, "[pypto] hcclContext rtMemcpy failed ret=%d\n", memcpyRet);
                    fclose(errLog);
                }
            }
            logged = true;
        }
        if (debugHccl) {
            TileOp::HcclCombinOpParam debugParam{};
            auto debugRet = rtMemcpy(&debugParam, sizeof(debugParam), reinterpret_cast<void *>(contextVal),
                sizeof(debugParam), RT_MEMCPY_DEVICE_TO_HOST);
            if (debugRet == RT_ERROR_NONE) {
                fprintf(stderr,
                    "[pypto] hcclHandle=0x%lx ctx=0x%lx rankId=%u rankNum=%u winSize=%lu winExpSize=%lu padding0=0x%x\n",
                    hcclHandle, contextVal, debugParam.rankId, debugParam.rankNum, debugParam.winSize,
                    debugParam.winExpSize, debugParam.padding[0]);
                for (uint32_t rank = 0; rank < debugParam.rankNum; ++rank) {
                    fprintf(stderr, "[pypto] hcclContext rank=%u winIn=0x%lx winOut=0x%lx winExp=0x%lx\n",
                        rank, debugParam.windowsIn[rank], debugParam.windowsOut[rank], debugParam.windowsExp[rank]);
                }
            } else {
                fprintf(stderr, "[pypto] hcclHandle=0x%lx ctx=0x%lx rtMemcpy failed ret=%d\n",
                    hcclHandle, contextVal, debugRet);
            }
        }
        return contextVal;
    }
    return 0;
}
#endif
} // namespace

namespace pypto {

void CopyToHost(const DeviceTensorData &devTensor, DeviceTensorData &hostTensor) {
    CopyDevToHost(devTensor, hostTensor);
}

void CopyToDev(const DeviceTensorData &devTensor, DeviceTensorData &hostTensor) {
    CopyHostToDev(devTensor, hostTensor);
}

void SetVerifyData(const std::vector<DeviceTensorData> &inputs,
                   const std::vector<DeviceTensorData> &outputs,
                   const std::vector<DeviceTensorData> &goldens) {
    ProgramData::GetInstance().Reset();
    for (size_t i = 0; i < inputs.size(); i++) {
        auto rawData = RawTensorData::CreateTensor(
            inputs[i].GetDataType(), inputs[i].GetShape(), (uint8_t *)inputs[i].GetAddr());
        ProgramData::GetInstance().AppendInput(rawData);
    }
    for (size_t i = 0; i < outputs.size(); i++) {
        auto rawData = std::make_shared<RawTensorData>(
            outputs[i].GetDataType(), outputs[i].GetShape());
        ProgramData::GetInstance().AppendOutput(rawData);
    }
    for (size_t i = 0; i < goldens.size(); i++) {
        if (goldens[i].GetAddr() == 0) {
            ProgramData::GetInstance().AppendGolden(nullptr);
        } else {
            auto rawData = RawTensorData::CreateTensor(
            goldens[i].GetDataType(), goldens[i].GetShape(), (uint8_t *)goldens[i].GetAddr());
            ProgramData::GetInstance().AppendGolden(rawData);
        }
    }
}

static std::string ValidateFunctionAndIO(Function *func, const std::vector<DeviceTensorData> &inputs,
                                   const std::vector<DeviceTensorData> &outputs) {
    if (!func->IsFunctionTypeAndGraphType(FunctionType::DYNAMIC, GraphType::TENSOR_GRAPH)) {
        return "Invalid function format";
    }

    auto attr = func->GetDyndevAttribute();
    if (attr == nullptr) {
        return "Invalid function format";
    }
    FILE *commLog = fopen("/tmp/pypto_commgroup.log", "a");
    if (commLog != nullptr) {
        fprintf(commLog, "[pypto] commGroupNames size=%zu\n", attr->commGroupNames.size());
        for (const auto &name : attr->commGroupNames) {
            fprintf(commLog, "[pypto] commGroupName: %s\n", name.c_str());
        }
        fclose(commLog);
    }

    auto inputSize = attr->startArgsInputLogicalTensorList.size();
    auto outputSize = attr->startArgsOutputLogicalTensorList.size();
    if (inputSize != inputs.size() || outputSize != outputs.size()) {
        return "mismatch input/output";
    }
    return "";
}

static void InitializeInputOutputData(const std::vector<DeviceTensorData> &inputs,
                               const std::vector<DeviceTensorData> &outputs) {
    for (size_t i = 0; i < inputs.size(); i++) {
        auto rawData = RawTensorData::CreateTensor(inputs[i].GetDataType(), inputs[i].GetShape(), (uint8_t *)inputs[i].GetAddr());
        ProgramData::GetInstance().AppendInput(rawData);
    }
    for (size_t i = 0; i < outputs.size(); i++) {
        auto rawData = std::make_shared<RawTensorData>(outputs[i].GetDataType(), outputs[i].GetShape());
        ProgramData::GetInstance().AppendOutput(rawData);
    }
}

std::string DeviceRunOnceDataFromHost(
    const std::vector<DeviceTensorData> &inputs, const std::vector<DeviceTensorData> &outputs) {
    if (config::GetHostOption<int64_t>(COMPILE_STAGE) != CS_ALL_COMPLETE) {
        return "";
    }
    ProgramData::GetInstance().Reset();
    Function *func = Program::GetInstance().GetLastFunction();
    auto errorMsg = ValidateFunctionAndIO(func, inputs, outputs);
    if (!errorMsg.empty()) {
        return errorMsg;
    }

    InitializeInputOutputData(inputs, outputs);

    DevControlFlowCache* hostCache = nullptr;
    if (config::GetRuntimeOption<int64_t>(STITCH_CFGCACHE_SIZE) != 0) {
        DeviceLauncherConfig config;
        DeviceLauncher::DeviceLauncherConfigFillDeviceInfo(config);
        EmulationLauncher::BuildControlFlowCache(func, inputs, outputs, &hostCache, config);
    }

    if (config::GetDebugOption<int>(CFG_RUNTIME_DBEUG_MODE) == 1 && EmulationLauncher::EmulationRunOnce(func, hostCache) != 0) {
        return "emulation run failed";
    }

    if (DeviceRunOnce(func, reinterpret_cast<uint8_t*>(hostCache)) != 0) {
        return "device run failed";
    }

    if (hostCache) {
        free(hostCache);
    }

    for (size_t i = 0; i < outputs.size(); i++) {
        auto output = ProgramData::GetInstance().GetOutputData(i);
        StringUtils::DataCopy(outputs[i].GetAddr(), output->GetDataSize(), output->data(), output->GetDataSize());
    }

    if (HasInplaceArgs(Program::GetInstance().GetLastFunction()) || outputs.size() == 0) {
        for (size_t i = 0; i < inputs.size(); i++) {
            auto input = ProgramData::GetInstance().GetInputData(i);
            StringUtils::DataCopy(inputs[i].GetAddr(), input->GetDataSize(), input->data(), input->GetDataSize());
        }
    }
    return "";
}

std::string OperatorDeviceRunOnceDataFromDevice([[maybe_unused]] py::int_ pythonOperatorPython,
    [[maybe_unused]] const std::vector<DeviceTensorData> &inputs, [[maybe_unused]] const std::vector<DeviceTensorData> &outputs,
    [[maybe_unused]] py::int_ incomingStreamPython, [[maybe_unused]] py::int_ workspaceData,
    [[maybe_unused]] py::int_ devCtrlCache) {

    if (config::GetHostOption<int64_t>(COMPILE_STAGE) != CS_ALL_COMPLETE) {
        return "";
    }
    HOST_PERF_TRACE_START();
    HOST_PERF_EVT_BEGIN(EventPhase::RunDevice);

#ifdef BUILD_WITH_CANN
    auto opAddr = static_cast<uintptr_t>(pythonOperatorPython);
    if (opAddr == 0) {
        return "invalid operator";
    }

    ExportedOperator *op = reinterpret_cast<ExportedOperator *>(opAddr);
    Function *func = op->GetFunction();
    auto errorMsg = ValidateFunctionAndIO(func, inputs, outputs);
    if (!errorMsg.empty()) {
        return errorMsg;
    }
    auto attr = func->GetDyndevAttribute();
    bool debugHccl = (std::getenv("PYPTO_HCCL_DEBUG") != nullptr);
    if (debugHccl && attr != nullptr) {
        FILE *commLog = fopen("/tmp/pypto_commgroup.log", "a");
        if (commLog != nullptr) {
            fprintf(commLog, "[pypto] device commGroupNames size=%zu\n", attr->commGroupNames.size());
            for (const auto &name : attr->commGroupNames) {
                fprintf(commLog, "[pypto] device commGroupName: %s\n", name.c_str());
            }
            fclose(commLog);
        }
    }

    if (config::GetDebugOption<int>(CFG_RUNTIME_DBEUG_MODE) == 1) {
        DeviceLauncherConfig config;
        DeviceLauncher::DeviceLauncherConfigFillDeviceInfo(config);
        if (EmulationLauncher::EmulationLaunchDeviceTensorData(func, inputs, outputs, config) != 0) {
            return "emulation run failed";
        }
    }

    auto incomingStream = static_cast<uintptr_t>(incomingStreamPython);
    if (incomingStream == 0) {
        return "invalid incoming stream";
    }

    auto aicoreStream = incomingStream;
    auto aicpuStream = DeviceGetAicpuStream();
    auto workspaceDataAddr = static_cast<uintptr_t>(workspaceData);
    auto launcherConfig = DeviceLauncherConfig::CreateConfigWithWorkspaceAddr(workspaceDataAddr);
    auto ctrlCache = static_cast<uintptr_t>(devCtrlCache);
    std::vector<int64_t> hcclHandles;
    std::vector<std::string> hcclGroupNames;
    bool hasHandles = false;
    bool hasNames = false;
    auto globalScope = ConfigManagerNg::GetInstance().GlobalScope();
    if (globalScope != nullptr) {
        if (globalScope->HasConfig("global.distributed.hccl_handle")) {
            try {
                hcclHandles = globalScope->GetConfigAllType<std::vector<int64_t>>(
                    "global.distributed.hccl_handle");
                hasHandles = true;
            } catch (const std::exception &e) {
                if (debugHccl) {
                    ShmemLog("[pypto] read global.distributed.hccl_handle failed: %s\n", e.what());
                }
            }
        }
        if (globalScope->HasConfig("global.distributed.hccl_group_name")) {
            try {
                hcclGroupNames = globalScope->GetConfigAllType<std::vector<std::string>>(
                    "global.distributed.hccl_group_name");
                hasNames = true;
            } catch (const std::exception &e) {
                if (debugHccl) {
                    ShmemLog("[pypto] read global.distributed.hccl_group_name failed: %s\n", e.what());
                }
            }
        }
    }
    if (debugHccl) {
        FILE *commLog = fopen("/tmp/pypto_commgroup.log", "a");
        if (commLog != nullptr) {
            fprintf(commLog,
                "[pypto] device options hasHandles=%d hasNames=%d handles=%zu names=%zu\n",
                static_cast<int>(hasHandles), static_cast<int>(hasNames),
                hcclHandles.size(), hcclGroupNames.size());
            fclose(commLog);
        }
    }
    if (!attr->commGroupNames.empty() && hasHandles && hasNames && !hcclHandles.empty()) {
        if (hcclHandles.size() != hcclGroupNames.size()) {
            return "hccl handle and group name size mismatch";
        }
        std::unordered_map<std::string, uint64_t> nameToHandle;
        nameToHandle.reserve(hcclHandles.size());
        for (size_t i = 0; i < hcclHandles.size(); ++i) {
            nameToHandle[hcclGroupNames[i]] = static_cast<uint64_t>(hcclHandles[i]);
        }
        std::vector<uint64_t> hcclContext;
        hcclContext.reserve(attr->commGroupNames.size());
        for (const auto &groupName : attr->commGroupNames) {
            uint64_t handle = 0;
            auto it = nameToHandle.find(groupName);
            if (it != nameToHandle.end()) {
                handle = it->second;
            }
            hcclContext.push_back(AllocHcclContext(handle, groupName, reinterpret_cast<void *>(aicoreStream)));
        }
        bool valid = !hcclContext.empty();
        for (auto ctx : hcclContext) {
            if (ctx == 0) {
                valid = false;
                break;
            }
        }
        if (!valid) {
            return "hccl context init failed";
        }
        launcherConfig.hcclContext = hcclContext;
    }
    int rc = ExportedOperatorDeviceLaunchOnceWithDeviceTensorData(op, inputs, outputs,
        aicpuStream, aicoreStream, false, reinterpret_cast<uint8_t *>(ctrlCache), launcherConfig);
    if (rc < 0) {
        return "device run failed";
    }
#endif

    HOST_PERF_EVT_END(EventPhase::RunDevice);
    return "";
}

uint64_t GetWorkSpaceSize(uintptr_t opAddr, const std::vector<DeviceTensorData> &inputs,
    const std::vector<DeviceTensorData> &outputs) {
    ExportedOperator *op = reinterpret_cast<ExportedOperator *>(opAddr);
    if (op) {
        return op->GetWorkSpaceSize(inputs, outputs);
    }
    return 0;
}

std::string OperatorDeviceSynchronize(py::int_ incomingStreamPython) {
    auto incomingStream = static_cast<uintptr_t>(incomingStreamPython);
    if (incomingStream == 0) {
        return "invalid incoming stream";
    }

    auto aicpuStream = incomingStream;
    auto aicoreStream = DeviceGetAicoreStream();
    int rc = DeviceSynchronize(aicpuStream, aicoreStream);
    if (rc < 0) {
        return "device sync failed";
    }
    return "";
}

#ifdef BUILD_WITH_CANN
py::bytes HcclGetRootInfoBytes()
{
    using GetRootInfoFn = HcclResult (*)(HcclRootInfo *);
    auto func = LoadHcclFunc<GetRootInfoFn>("HcclGetRootInfo");
    if (func == nullptr) {
        throw std::runtime_error("HcclGetRootInfo symbol not found");
    }
    HcclRootInfo rootInfo{};
    auto ret = func(&rootInfo);
    if (ret != HCCL_SUCCESS) {
        throw std::runtime_error("HcclGetRootInfo failed");
    }
    return py::bytes(reinterpret_cast<const char *>(&rootInfo), sizeof(rootInfo));
}

uint64_t HcclCommInitRootInfoBytes(const py::bytes &rootInfoBytes, int rank, int world)
{
    using CommInitFn = HcclResult (*)(uint32_t, const HcclRootInfo *, uint32_t, HcclComm *);
    auto func = LoadHcclFunc<CommInitFn>("HcclCommInitRootInfo");
    if (func == nullptr) {
        throw std::runtime_error("HcclCommInitRootInfo symbol not found");
    }
    std::string data = rootInfoBytes;
    if (data.size() != sizeof(HcclRootInfo)) {
        throw std::runtime_error("HcclRootInfo size mismatch");
    }
    HcclRootInfo rootInfo{};
    std::memcpy(&rootInfo, data.data(), sizeof(rootInfo));
    HcclComm comm = nullptr;
    auto ret = func(static_cast<uint32_t>(world), &rootInfo, static_cast<uint32_t>(rank), &comm);
    if (ret != HCCL_SUCCESS || comm == nullptr) {
        throw std::runtime_error("HcclCommInitRootInfo failed");
    }
    return reinterpret_cast<uint64_t>(comm);
}

std::string HcclGetCommNameFromHandle(uint64_t handle)
{
    using GetCommNameFn = HcclResult (*)(HcclComm, char *);
    auto func = LoadHcclFunc<GetCommNameFn>("HcclGetCommName");
    if (func == nullptr) {
        throw std::runtime_error("HcclGetCommName symbol not found");
    }
    char name[COMM_NAME_MAX_LENGTH] = {};
    auto ret = func(reinterpret_cast<HcclComm>(handle), name);
    if (ret != HCCL_SUCCESS) {
        throw std::runtime_error("HcclGetCommName failed");
    }
    return std::string(name);
}

uint32_t GetHcclRootInfoSize()
{
    return HCCL_ROOT_INFO_BYTES;
}
#endif

void DeviceInit() {
    DeviceLauncherInit();
}

void DeviceFini() {
    DeviceLauncherFini();
}

uintptr_t OperatorBegin() {
    ExportedOperator *op = ExportedOperatorBegin();
    auto opAddr = reinterpret_cast<uintptr_t>(op);
    return opAddr;
}

std::string OperatorEnd(uintptr_t opAddr) {
    ExportedOperator *op = reinterpret_cast<ExportedOperator *>(opAddr);
    ExportedOperatorEnd(op);
    return "";
}

int64_t BuildCache(uintptr_t opAddr, const std::vector<DeviceTensorData> &inputList,
        const std::vector<DeviceTensorData> &outputList, [[maybe_unused]] bool isCapturing) {
    ExportedOperator *op = reinterpret_cast<ExportedOperator *>(opAddr);
    if (config::GetRuntimeOption<int64_t>(STITCH_CFGCACHE_SIZE) != 0) {
        DeviceLauncherConfig config;
        DeviceLauncher::DeviceLauncherConfigFillDeviceInfo(config);
        uint8_t* ctrlCache = op->FindCtrlFlowCache(inputList, outputList);
        if (ctrlCache == nullptr) {
            HOST_PERF_EVT_BEGIN(EventPhase::BuildCtrlFlowCache);
            DevControlFlowCache* hostCache = nullptr;
            if (EmulationLauncher::BuildControlFlowCache(op->GetFunction(),
                inputList, outputList, &hostCache, config) != 0) {
                return 0;
            }

#ifdef BUILD_WITH_CANN
            if (isCapturing) {
                ChangeCaptureModeRelax();
            }

            if (hostCache) {
                ctrlCache = CopyHostToDev(reinterpret_cast<uint8_t*>(hostCache),
                    reinterpret_cast<DevControlFlowCache*>(hostCache)->allCacheSize);
                free(hostCache);
            }

            if (isCapturing) {
                ChangeCaptureModeGlobal();
            }
#else
            ctrlCache = reinterpret_cast<uint8_t*>(hostCache);
#endif

            if (ctrlCache) {
                op->InsertCtrlFlowCache(inputList, outputList, ctrlCache);
            }
            HOST_PERF_EVT_END(EventPhase::BuildCtrlFlowCache);
        }

        return ctrlCache == nullptr ? 0 : reinterpret_cast<int64_t>(ctrlCache);
    }

    return 0;
}

void BindRuntime(py::module &m) {
    m.def("DeviceInit", &DeviceInit);
    m.def("DeviceFini", &DeviceFini);
    m.def("DeviceRunOnceDataFromHost", &DeviceRunOnceDataFromHost);
    m.def("OperatorDeviceRunOnceDataFromDevice", &OperatorDeviceRunOnceDataFromDevice);
    m.def("OperatorDeviceSynchronize", &OperatorDeviceSynchronize);
    m.def("GetWorkSpaceSize", &GetWorkSpaceSize);
    m.def("OperatorBegin", OperatorBegin);
    m.def("OperatorEnd", OperatorEnd);
    m.def("SetVerifyData", &SetVerifyData);
    m.def("BuildCache", BuildCache);
    m.def("CopyToHost", &CopyToHost);
#ifdef BUILD_WITH_CANN
    m.def("GetHcclRootInfoSize", &GetHcclRootInfoSize);
    m.def("HcclGetRootInfo", &HcclGetRootInfoBytes);
    m.def("HcclCommInitRootInfo", &HcclCommInitRootInfoBytes);
    m.def("HcclGetCommName", &HcclGetCommNameFromHandle);
#endif
    m.def("CopyToDev", &CopyToDev);

    py::class_<DeviceTensorData>(m, "DeviceTensorData")
        .def(py::init<DataType, uintptr_t, const std::vector<int64_t> &>(), py::arg("dtype"), py::arg("addr"),
            py::arg("shape"))
        .def("GetDataPtr", &DeviceTensorData::GetAddr)
        .def("GetShape", &DeviceTensorData::GetShape)
        .def("GetDataType", &DeviceTensorData::GetDataType);
}
} // namespace pypto
