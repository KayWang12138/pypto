/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!\file monitor_config.h
 * \brief Compiler stage configuration for monitor
 */

#ifndef SRC_INTERFACE_COMPILER_MONITOR_MONITOR_CONFIG_H
#define SRC_INTERFACE_COMPILER_MONITOR_MONITOR_CONFIG_H

#include <string>
#include <vector>
#include <unordered_map>

namespace npu::tile_fwk {

/**
 * @brief Represents a compilation stage with its hierarchy level
 *
 * The level indicates the nesting depth of stages during compilation.
 * For example:
 * - Level 0: Top-level stages (FrontendParser, TensorGraphPass, etc.)
 * - Level 1: Nested stages within top-level stages (individual Passes)
 */
struct Stage {
    std::string name;   // Stage name
    int level;          // Nesting level (0 for top-level)

    Stage(const std::string& n, int l) : name(n), level(l) {}
};

/**
 * @brief Manages the configuration of compilation stages
 *
 * This class provides three modes of stage granularity:
 * - COARSE: 6 major high-level stages
 * - FINE: Individual pass-level monitoring
 * - CUSTOM: User-defined stage names
 *
 * Example usage:
 * @code
 * StageConfig::Instance().LoadConfig(StageConfig::Mode::COARSE);
 * // or
 * StageConfig::Instance().LoadConfig(StageConfig::Mode::CUSTOM, {"Parse", "Optimize", "CodeGen"});
 * @endcode
 */
class StageConfig {
public:
    /**
     * @brief Stage granularity modes
     */
    enum class Mode {
        COARSE = 0,  // Coarse-grained: 6 major stages
        FINE = 1,    // Fine-grained: individual passes
        CUSTOM = 2   // User-defined stages
    };

    /**
     * @brief Get singleton instance
     * @return Reference to the singleton StageConfig instance
     */
    static StageConfig& Instance();

    /**
     * @brief Load stage configuration based on mode
     * @param mode The granularity mode to use
     * @param custom_stages Optional list of custom stage names (required when mode=CUSTOM)
     */
    void LoadConfig(Mode mode, const std::vector<std::string>& custom_stages = {});

    /**
     * @brief Get the configured list of stages
     * @return Reference to the vector of Stage objects
     */
    const std::vector<Stage>& GetStages() const { return stages_; }

    /**
     * @brief Get the current mode
     * @return The currently configured Mode
     */
    Mode GetMode() const { return mode_; }

    /**
     * @brief Check if a stage name is valid
     * @param name The stage name to check
     * @return true if the stage name is in the configured list
     */
    bool IsValidStage(const std::string& name) const;

private:
    StageConfig();
    ~StageConfig() = default;
    StageConfig(const StageConfig&) = delete;
    StageConfig& operator=(const StageConfig&) = delete;

    // Load predefined coarse-grained stages (6 major stages)
    void LoadCoarseStages();

    // Load fine-grained stages (individual passes will be added dynamically)
    void LoadFineStages();

    // Load custom stages from user-provided list
    void LoadCustomStages(const std::vector<std::string>& stage_names);

    Mode mode_;
    std::vector<Stage> stages_;
    std::unordered_map<std::string, bool> stage_set_;  // For fast lookup
};

/**
 * @brief Get mode string for debugging/logging
 * @param mode The Mode value
 * @return String representation of the mode
 */
const char* GetModeString(StageConfig::Mode mode);

} // namespace npu::tile_fwk

#endif // SRC_INTERFACE_COMPILER_MONITOR_MONITOR_CONFIG_H
