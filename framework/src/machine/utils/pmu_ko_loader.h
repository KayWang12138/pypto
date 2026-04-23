/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file pmu_ko_loader.h
 * \brief 内嵌 PMU 用户态访问内核模块加载器
 *
 *  将 pmu_user_access.ko 二进制硬编码到代码中，
 *  在 AICPU device 启动时保存为文件并 insmod 加载。
 *
 *  使用方式：
 *    1. 运行 embed_ko_to_code.sh 生成 pmu_user_access_ko_embedded.h
 *    2. 在 device 初始化代码中调用 PmuKoLoader::LoadEmbeddedKo()
 */

#pragma once

#ifdef __DEVICE__

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unistd.h>
#include <sys/stat.h>
#include <errno.h>

#ifdef __linux__
#include <sys/syscall.h>
#include <fcntl.h>
#endif

#include "machine/utils/device_log.h"

namespace npu::tile_fwk {

/**
 * @brief PMU 内核模块加载器
 *
 * 将内嵌的 .ko 二进制保存到文件并加载到内核。
 */
struct PmuKoLoader {
    static const char* DefaultKoPath()
    {
        return "/tmp/pmu_user_access.ko";
    }

    /**
     * @brief 加载内嵌的内核模块
     *
     * @param koData 内核模块二进制数据指针
     * @param koLen 数据长度
     * @param koPath 保存路径（默认 "/tmp/pmu_user_access.ko"）
     * @return 0 成功，负值失败
     */
    static int LoadEmbeddedKo(const unsigned char* koData,
                              size_t koLen,
                              const char* koPath = nullptr)
    {
        const char* targetPath = (koPath == nullptr) ? DefaultKoPath() : koPath;
        if (koData == nullptr || koLen == 0) {
            DEV_WARN("[PMU_KO] Invalid embedded KO input, data=%p, len=%zu", koData, koLen);
            return -1;
        }

        if (IsModuleLoaded()) {
            DEV_INFO("[PMU_KO] Module already loaded, skip");
            return 0;
        }

        int ret = WriteKoFile(koData, koLen, targetPath);
        if (ret != 0) {
            DEV_WARN("[PMU_KO] Failed to write embedded KO to %s, ret=%d", targetPath, ret);
            return -2;
        }

        ret = LoadKoModuleFromFile(targetPath);
        if (ret != 0) {
            DEV_WARN("[PMU_KO] Failed to load %s, ret=%d", targetPath, ret);
            return -3;
        }

        DEV_INFO("[PMU_KO] Module loaded from embedded image successfully");
        return 0;
    }

    /**
     * @brief 从 device 文件系统中的 .ko 文件加载模块。
     *
     * 主要用于调试。正式路径应使用 LoadEmbeddedKo()，确保 .ko 来自编译进 device
     * 二进制的内嵌数组。
     */
    static int LoadKoModuleFromFile(const char* koPath)
    {
        if (koPath == nullptr) {
            return -1;
        }
        if (IsModuleLoaded()) {
            DEV_INFO("[PMU_KO] Module already loaded, skip");
            return 0;
        }

        int ret = InsmodViaSystem(koPath);
        if (ret == 0) {
            return 0;
        }

#ifdef __linux__
        ret = LoadViaSyscall(koPath);
        if (ret == 0) {
            return 0;
        }
#endif

        return -1;
    }

    /**
     * @brief 检查内核模块是否已加载
     *
     * @return true 已加载，false 未加载
     */
    static bool IsModuleLoaded()
    {
        FILE* fp = fopen("/proc/modules", "r");
        if (fp) {
            char line[256];
            while (fgets(line, sizeof(line), fp)) {
                if (strncmp(line, "pmu_user_access ", strlen("pmu_user_access ")) == 0) {
                    fclose(fp);
                    return true;
                }
            }
            fclose(fp);
        }
        return false;
    }

    /**
     * @brief 卸载内核模块
     */
    static void UnloadModule()
    {
        if (!IsModuleLoaded()) {
            return;
        }
        system("rmmod pmu_user_access 2>/dev/null");
    }

private:
    static int WriteKoFile(const unsigned char* koData, size_t koLen, const char* koPath)
    {
        FILE* fp = fopen(koPath, "wb");
        if (fp == nullptr) {
            DEV_WARN("[PMU_KO] Failed to open %s for write, errno=%d", koPath, errno);
            return -1;
        }

        size_t written = fwrite(koData, 1, koLen, fp);
        int closeRet = fclose(fp);
        if (written != koLen) {
            DEV_WARN("[PMU_KO] Write incomplete: %zu/%zu bytes", written, koLen);
            return -2;
        }
        if (closeRet != 0) {
            DEV_WARN("[PMU_KO] Failed to close %s, errno=%d", koPath, errno);
            return -3;
        }
        if (chmod(koPath, 0644) != 0) {
            DEV_WARN("[PMU_KO] Failed to chmod %s, errno=%d", koPath, errno);
            return -4;
        }

        DEV_INFO("[PMU_KO] Written %zu bytes to %s", koLen, koPath);
        return 0;
    }

    static bool AppendShellQuotedPath(char* cmd, size_t cmdSize, const char* koPath)
    {
        int offset = snprintf(cmd, cmdSize, "insmod '");
        if (offset < 0 || static_cast<size_t>(offset) >= cmdSize) {
            return false;
        }

        for (const char* p = koPath; *p != '\0'; ++p) {
            const char* text = (*p == '\'') ? "'\\''" : nullptr;
            if (text != nullptr) {
                int ret = snprintf(cmd + offset, cmdSize - static_cast<size_t>(offset), "%s", text);
                if (ret < 0 || static_cast<size_t>(ret) >= cmdSize - static_cast<size_t>(offset)) {
                    return false;
                }
                offset += ret;
            } else {
                if (static_cast<size_t>(offset + 1) >= cmdSize) {
                    return false;
                }
                cmd[offset++] = *p;
                cmd[offset] = '\0';
            }
        }

        int ret = snprintf(cmd + offset, cmdSize - static_cast<size_t>(offset), "' 2>/dev/null");
        return ret >= 0 && static_cast<size_t>(ret) < cmdSize - static_cast<size_t>(offset);
    }

    /**
     * @brief 执行 insmod 加载内核模块
     *
     * @param koPath 内核模块路径
     * @return 0 成功，负值失败
     */
    static int InsmodViaSystem(const char* koPath)
    {
        char cmd[256];
        if (!AppendShellQuotedPath(cmd, sizeof(cmd), koPath)) {
            DEV_WARN("[PMU_KO] KO path is too long: %s", koPath);
            return -1;
        }

        int ret = system(cmd);
        if (ret == 0) {
            DEV_INFO("[PMU_KO] insmod %s succeeded", koPath);
            return 0;
        }

        DEV_WARN("[PMU_KO] insmod %s failed, system ret=%d", koPath, ret);
        return -1;
    }

#ifdef __linux__
    /**
     * @brief 通过 init_module syscall 加载内核模块
     *
     * 直接读取文件内容并通过 syscall 加载，不依赖 shell。
     */
    static int LoadViaSyscall(const char* koPath)
    {
        // 读取文件内容
        FILE* fp = fopen(koPath, "rb");
        if (!fp) {
            return -1;
        }

        // 获取文件大小
        fseek(fp, 0, SEEK_END);
        long size = ftell(fp);
        fseek(fp, 0, SEEK_SET);

        if (size <= 0) {
            fclose(fp);
            return -1;
        }

        // 分配内存读取文件
        void* koBuf = malloc(static_cast<size_t>(size));
        if (!koBuf) {
            fclose(fp);
            return -1;
        }

        size_t readSize = fread(koBuf, 1, static_cast<size_t>(size), fp);
        fclose(fp);

        if (readSize != static_cast<size_t>(size)) {
            free(koBuf);
            return -1;
        }

        // init_module(void* module_image, unsigned long len, const char* param_values)
        int ret = syscall(__NR_init_module, koBuf, static_cast<unsigned long>(size), "");

        free(koBuf);

        if (ret != 0) {
            DEV_WARN("[PMU_KO] init_module syscall failed, errno=%d", errno);
        } else {
            DEV_INFO("[PMU_KO] init_module syscall succeeded");
        }

        return ret;
    }
#endif
};

/**
 * @brief PMU KO 初始化辅助函数（使用内嵌数据）
 *
 * 需要先通过 embed_ko_to_code.sh 生成 pmu_user_access_ko_embedded.h，
 * 并在调用此函数前 #include 该头文件。
 *
 * 使用示例：
 * @code
 * #include "pmu_user_access_ko_embedded.h"  // 由脚本生成
 * #include "pmu_ko_loader.h"
 *
 * int ret = npu::tile_fwk::PmuInitEmbeddedKo();
 * @endcode
 *
 * @return 0 成功，负值失败
 */
inline int PmuInitEmbeddedKo()
{
#if defined(PMU_USER_ACCESS_KO_EMBEDDED)
    return PmuKoLoader::LoadEmbeddedKo(GetEmbeddedPmuKoData(), GetEmbeddedPmuKoLen());
#else
    DEV_WARN("[PMU_KO] No embedded KO data. Please run embed_ko_to_code.sh and include the generated header.");
    return -1;
#endif
}

} // namespace npu::tile_fwk

#endif // __DEVICE__