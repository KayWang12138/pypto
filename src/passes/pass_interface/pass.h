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
 * \file pass.h
 * \brief
 */

#ifndef PASSES_PASS_H_
#define PASSES_PASS_H_

#include <string>
#include <atomic>
#include "common/pre_def.h"
#include "interface/configs/config_manager.h"

namespace npu::tile_fwk {

class Pass {
public:
    explicit Pass(std::string name);
    virtual ~Pass() = default;
    Status Run(Function &function, const std::string &strategy,
               const std::string &identifier, size_t runtimeIdx = 0);
    virtual Status PreCheck(Function &function);

    virtual Status PostCheck(Function &function);
    const std::string &LogFolder(const std::string &topFolder, size_t i) const;
    const std::string &GetName() const { return name_; }
    void SetPassConfigs(const PassConfigs &config) {
        passDfxconfigs_ = config;
    }

protected:
    virtual Status RunOnFunction(Function &function) = 0;
    template <typename T>
    ConvertedConfigType<T> GetConfig(const std::string &key, const T &defaultValue) {
        return config::GetPassConfig(strategy_, identifier_, key, defaultValue);
    }
    virtual Status CreateLogFolder(const std::string &topFolder, size_t i) const;
    virtual Status PrintFunction(Function& function, const std::string &logFolder, bool beforeFunction);
    virtual Status DumpFunctionJson(Function& function, const std::string &logFolder, bool beforeFunction);
    virtual Status PreRun(Function &function);
    virtual Status PostRun(Function &function);
    // folderPath: dump路径
    virtual void DoHealthCheckBefore(Function &function, const std::string &folderPath);
    virtual void DoHealthCheckAfter(Function &function, const std::string &folderPath);
    mutable PassConfigs passDfxconfigs_;
    // 获取dump的文件名，如果是leaffunction，后面两个参数需要配置
    std::string GetDumpFilePrefix(Function& function, bool before = false,
                                  Function* subFunction = nullptr, int subFuncId = -1);

private:
    mutable std::string identifier_;
    mutable std::string strategy_;
    size_t passRuntimeIndex_;
    mutable std::string passFolder_{"."};
    std::string name_;
};
} // namespace npu::tile_fwk
#endif  // PASSES_PASS_H_
