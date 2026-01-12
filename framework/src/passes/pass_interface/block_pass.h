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
 * \file pass_for_block.h
 * \brief
 */

#ifndef PASSES_PASS_FOR_BLOCK_H_
#define PASSES_PASS_FOR_BLOCK_H_

#include <string>
#include <atomic>
#include "interface/inner/pre_def.h"
#include "interface/configs/config_manager.h"
#include "tilefwk/platform.h"
#include "ir/function.h"

namespace npu::tile_fwk {
class BlockPass {
public:
    explicit BlockPass(std::string name);
    virtual ~BlockPass() = default;
    Status Run(pto::Function &function, const std::string &strategy,
               const std::string &identifier, size_t runtimeIdx = 0);
    virtual Status PreCheck(pto::Function &function);

    virtual Status PostCheck(pto::Function &function);
    const std::string &LogFolder(const std::string &topFolder, size_t i) const;
    const std::string &GetName() const { return name_; }
    void SetPassConfigs(const PassConfigs &config) {
        passDfxconfigs_ = config;
    }
    std::vector<NPUArch> &GetSupportedArches() {
        return supportedArches_;
    }
    void SetSupportedArches(const std::vector<NPUArch> &supportedArches) {
        supportedArches_ = supportedArches;
    }

protected:
    virtual Status RunOnFunction(pto::Function &function) = 0;
    template <typename T>
    ConvertedConfigType<T> GetConfig(const std::string &key, const T &defaultValue) {
        return config::GetPassConfig(strategy_, identifier_, key, defaultValue);
    }
    virtual Status CreateLogFolder(const std::string &topFolder, size_t i) const;
    virtual Status PrintFunction(pto::Function& function, const std::string &logFolder, bool beforeFunction);
    virtual Status DumpFunctionJson(pto::Function& function, const std::string &logFolder, bool beforeFunction);
    virtual Status DumpGraphJson(pto::Function& function, const std::string &fileName);
    virtual Status CreateGraphFolder(pto::Function &function);
    virtual Status PreRun(pto::Function &function);
    virtual Status PostRun(pto::Function &function);
    // folderPath: dump路径
    virtual void DoHealthCheckBefore(pto::Function &function, const std::string &folderPath);
    virtual void DoHealthCheckAfter(pto::Function &function, const std::string &folderPath);
    mutable PassConfigs passDfxconfigs_;
    // 获取dump的文件名，如果是leaffunction，后面两个参数需要配置
    std::string GetDumpFilePrefix(pto::Function& function, bool before = false,
                                  pto::Function* subFunction = nullptr, int subFuncId = -1);

private:
    mutable std::string identifier_;
    mutable std::string strategy_;
    size_t passRuntimeIndex_;
    mutable std::string passFolder_{"."};
    std::string name_;
    std::string graphFolder_;
    std::vector<NPUArch> supportedArches_;
};
} // namespace npu::tile_fwk
#endif  // PASSES_PASS_FOR_BLOCK_H_
