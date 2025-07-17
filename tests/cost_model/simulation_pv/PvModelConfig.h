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
 * \file PvModelConfig.h
 * \brief
 */

#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include <tuple>

namespace CostModel {
    class PvModelCaseConfigBase {
    protected:
        using ArgPack = std::tuple<uint64_t, uint64_t, std::string>;

        std::string title_;
        uint64_t binAddr_;
        std::string binPath_;
        std::vector<ArgPack> inputArgs_;
        std::vector<ArgPack> outputArgs_;

    public:
        virtual ~PvModelCaseConfigBase() = default;
        PvModelCaseConfigBase() = default;
        void SetTitle(std::string title);
        void SetBin(uint64_t addr, std::string path);
        void AddInputArg(uint64_t addr, uint64_t size, std::string path);
        void AddOutputArg(uint64_t addr, uint64_t size, std::string path);
        virtual void Dump(std::string path) = 0;
    };

    // A2A3
    class PvModelSystemConfigA2A3 {
    public:
        void Dump(std::string path);
    };

    class PvModelCaseConfigA2A3 : public PvModelCaseConfigBase {
    public:
        PvModelCaseConfigA2A3() = default;
        void Dump(std::string path);
    };
} // namespace CostModel
