/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!\file monitor_config.cpp
 * \brief Implementation of compiler stage configuration
 */

#include "interface/compiler_monitor/monitor_config.h"
#include "interface/utils/log.h"
#include <algorithm>

namespace npu::tile_fwk {

// Static member initialization
StageConfig::StageConfig()
    : mode_(Mode::COARSE) {
    LoadCoarseStages();
}

StageConfig& StageConfig::Instance() {
    static StageConfig instance;
    return instance;
}

void StageConfig::LoadConfig(Mode mode, const std::vector<std::string>& custom_stages) {
    mode_ = mode;
    stages_.clear();
    stage_set_.clear();

    switch (mode) {
        case Mode::COARSE:
            LoadCoarseStages();
            break;
        case Mode::FINE:
            LoadFineStages();
            break;
        case Mode::CUSTOM:
            LoadCustomStages(custom_stages);
            break;
        default:
            ALOG_WARN("Unknown stage mode, defaulting to COARSE");
            LoadCoarseStages();
            break;
    }

    ALOG_INFO("StageConfig loaded with mode: %s, %zu stages",
              GetModeString(mode_), stages_.size());
}

void StageConfig::LoadCoarseStages() {
    // 6 major compilation stages (coarse-grained)
    stages_ = {
        Stage("FrontendParser", 0),
        Stage("TensorGraphPass", 0),
        Stage("TileGraphPass", 0),
        Stage("BlockGraphPass", 0),
        Stage("CodeGen", 0),
        Stage("BinaryGeneration", 0)
    };

    for (const auto& stage : stages_) {
        stage_set_[stage.name] = true;
    }
}

void StageConfig::LoadFineStages() {
    // Fine-grained: individual passes
    // Stages will be added dynamically as passes are registered
    // This provides a base list, and additional stages can be added at runtime
    stage_set_.clear();
}

void StageConfig::LoadCustomStages(const std::vector<std::string>& stage_names) {
    if (stage_names.empty()) {
        ALOG_WARN("Custom stages list is empty, loading default coarse stages");
        LoadCoarseStages();
        return;
    }

    for (const auto& name : stage_names) {
        if (name.empty()) {
            ALOG_WARN("Skipping empty stage name in custom stages");
            continue;
        }
        stages_.emplace_back(name, 0);
        stage_set_[name] = true;
    }
}

bool StageConfig::IsValidStage(const std::string& name) const {
    if (mode_ == Mode::FINE) {
        // In fine-grained mode, accept any stage name
        // Passes are registered dynamically
        return !name.empty();
    }

    auto it = stage_set_.find(name);
    return it != stage_set_.end();
}

const char* GetModeString(StageConfig::Mode mode) {
    switch (mode) {
        case StageConfig::Mode::COARSE:
            return "COARSE";
        case StageConfig::Mode::FINE:
            return "FINE";
        case StageConfig::Mode::CUSTOM:
            return "CUSTOM";
        default:
            return "UNKNOWN";
    }
}

} // namespace npu::tile_fwk
