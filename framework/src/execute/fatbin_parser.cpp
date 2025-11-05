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
 * \file fatbin_parser.cpp
 * \brief
 */

#include "fatbin_parser.h"
#include <fstream>
#include <fcntl.h>
#include <climits>
#include <unistd.h>
#include <sys/stat.h>
#include <dirent.h>
#include <dlfcn.h>
#include <ftw.h>
#include <fcntl.h>
#include "interface/utils/log.h"

namespace npu::tile_fwk {
namespace {
std::string RealPath(const std::string &path) {
    if (path.empty()) {
        ALOG_INFO("path string is nullptr.");
        return "";
    }
    if (path.size() >= PATH_MAX) {
        ALOG_INFO("file path ", path.c_str(), " is too long.");
        return "";
    }

    // PATH_MAX is the system marco, indicate the maximum length for file path
    // pclint check one param in stack can not exceed 1K bytes
    char resovedPath[PATH_MAX] = {0x00};

    std::string res;

    // path not exists or not allowed to read return nullptr
    // path exists and readable, return the resoved path
    if (realpath(path.c_str(), resovedPath) != nullptr) {
        res = resovedPath;
    } else {
        ALOG_INFO("path ", path.c_str(), " is not exist.");
    }
    return res;
}

bool ReadBytesFromFile(const std::string &filePath, std::vector<char> &buffer) {
    std::string realPath = RealPath(filePath);
    if (realPath.empty()) {
        ALOG_WARN("Bin file path[%s] is not valid.", filePath.c_str());
        return false;
    }

    std::ifstream ifStream(realPath.c_str(), std::ios::binary | std::ios::ate);
    if (!ifStream.is_open()) {
        ALOG_WARN("read file %s failed.", filePath.c_str());
        return false;
    }
    try {
        std::streamsize size = ifStream.tellg();
        if (size <= 0) {
            ifStream.close();
            ALOG_WARN("file length <= 0, not valid.");
            return false;
        }

        if (size > INT_MAX) {
            ifStream.close();
            ALOG_WARN("File size %ld is out of limit: %d.", size, INT_MAX);
            return false;
        }
        ifStream.seekg(0, std::ios::beg);

        buffer.resize(size);
        ifStream.read(&buffer[0], size);
        ALOG_DEBUG("Release file(%s) handle.", realPath.c_str());
        ifStream.close();
        ALOG_DEBUG("Read size:%ld.", size);
    } catch (const std::ifstream::failure& e) {
        ALOG_WARN("Fail to read file %s. Exception: %s.", filePath.c_str(), e.what());
        ifStream.close();
        return false;
    }
    return true;
}
}

bool FatbinParser::ParseFatbin(const std::string &bin_file_path, const uint64_t &config_key, size_t &subkernl_index,
                               std::vector<uint8_t> &op_binary_bin, std::vector<uint8_t> &kernel_bin) {
    std::vector<char> fatbin;
    if (!ReadBytesFromFile(bin_file_path, fatbin)) {
        ALOG_ERROR_F("Failed to read bin file, file path is %s.", bin_file_path.c_str());
        return false;
    }
    ALOG_DEBUG_F("Fatbin data size is %zu.", fatbin.size());

    std::vector<uint8_t> subkernel;
    if (!MatchSubkernel(fatbin, config_key, subkernl_index, subkernel)) {
        ALOG_ERROR_F("Failed to match subkernel.");
        return false;
    }

    if (!ParseSubkernel(subkernel, op_binary_bin, kernel_bin)) {
        ALOG_ERROR_F("Failed to parse subkernel.");
        return false;
    }
    return true;
}

bool FatbinParser::MatchSubkernel(const std::vector<char> &fatbin, const uint64_t &config_key,
                                  size_t &subkernl_index, std::vector<uint8_t> &subkernel) {
    const uint8_t* fatbin_ptr = reinterpret_cast<const uint8_t*>(fatbin.data());
    uint64_t config_key_num = 0;
    if (memcpy_s(&config_key_num, sizeof(uint64_t), fatbin_ptr, sizeof(uint64_t)) != EOK) {
        ALOG_ERROR_F("Failed to get config key num.");
        return false;
    }
    FatbinHeadInfo fatbin_head_info(config_key_num);
    if (memcpy_s(fatbin_head_info.configKeyList.data(), sizeof(uint64_t) * config_key_num,
            fatbin_ptr + sizeof(uint64_t), sizeof(uint64_t) * config_key_num) != EOK) {
        ALOG_ERROR_F("Failed to get config key list.");
        return false;
    }
    if (memcpy_s(fatbin_head_info.binOffsets.data(), sizeof(size_t) * config_key_num,
            fatbin_ptr + sizeof(uint64_t) + sizeof(uint64_t) * config_key_num,
            sizeof(size_t) * config_key_num) != EOK) {
        ALOG_ERROR_F("Failed to get bin offset list.");
        return false;
    }
    if (fatbin_head_info.configKeyList.size() != config_key_num ||
        fatbin_head_info.binOffsets.size() != config_key_num) {
        ALOG_ERROR_F("Config key list size %zu or bin offset list size %zu is not equal to config key num.",
                     fatbin_head_info.configKeyList.size(), fatbin_head_info.binOffsets.size(), config_key_num);
        return false;
    }
    size_t subkernel_bin_size = 0;
    for (size_t i = 0; i < config_key_num; ++i) {
        if (fatbin_head_info.configKeyList[i] == config_key) {
            if (i == config_key_num - 1) {
                subkernel_bin_size = fatbin.size() - fatbin_head_info.binOffsets[i];
            } else {
                subkernel_bin_size =
                    fatbin_head_info.binOffsets[i + 1] - fatbin_head_info.binOffsets[i];
            }
            subkernl_index = i;
            break;
        }
    }
    if (subkernel_bin_size == 0) {
        ALOG_ERROR_F("Failed to match config key %lu from fatbin.", config_key);
        return false;
    }
    subkernel.resize(subkernel_bin_size);
    if (memcpy_s(subkernel.data(), subkernel_bin_size, fatbin_ptr + fatbin_head_info.binOffsets[subkernl_index],
            subkernel_bin_size) != EOK) {
        ALOG_ERROR_F("Failed to get subkernel.");
        return false;
    }
    return true;
}

bool FatbinParser::ParseSubkernel(const std::vector<uint8_t> &subkernel, std::vector<uint8_t> &op_binary_bin,
                                  std::vector<uint8_t> &kernel_bin) {
    ALOG_DEBUG_F("Subkernel data size is %zu.", subkernel.size());
    const uint8_t* subkernel_bin = reinterpret_cast<const uint8_t*>(subkernel.data());
    KernelHeader kernel_header;
    if (memcpy_s(&kernel_header, sizeof(KernelHeader), subkernel_bin, sizeof(KernelHeader)) != EOK) {
        ALOG_ERROR_F("Failed to get subkernel header.");
        return false;
    }
    op_binary_bin.resize(kernel_header.dataSize[static_cast<size_t>(KernelContextType::OpBinary)]);
    if (memcpy_s(op_binary_bin.data(), kernel_header.dataSize[static_cast<size_t>(KernelContextType::OpBinary)],
                 subkernel_bin + kernel_header.dataOffset[static_cast<size_t>(KernelContextType::OpBinary)],
                 kernel_header.dataSize[static_cast<size_t>(KernelContextType::OpBinary)]) != EOK) {
        ALOG_ERROR_F("Failed to get op binary bin.");
        return false;
    }
    ALOG_DEBUG_F("Op binary bin data size is %zu.", op_binary_bin.size());
    kernel_bin.resize(kernel_header.dataSize[static_cast<size_t>(KernelContextType::Kernel)]);
    if (memcpy_s(kernel_bin.data(), kernel_header.dataSize[static_cast<size_t>(KernelContextType::Kernel)],
                 subkernel_bin + kernel_header.dataOffset[static_cast<size_t>(KernelContextType::Kernel)],
                 kernel_header.dataSize[static_cast<size_t>(KernelContextType::Kernel)]) != EOK) {
        ALOG_ERROR_F("Failed to get kernel bin.");
        return false;
    }
    ALOG_DEBUG_F("Kernel data size is %zu.", kernel_bin.size());
    return true;
};
}