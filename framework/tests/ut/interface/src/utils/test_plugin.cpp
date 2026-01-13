/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
* \file test_plugin.cpp
* \brief
*/

#include <gtest/gtest.h>
#include <string>

#include "tilefwk/function.h"
#include "tilefwk/tilefwk_op.h"
#include "tilefwk/tilefwk.h"

#include "interface/plugin/plugin.h"
#include "interface/utils/string_utils.h"

using namespace npu::tile_fwk;

TEST(PluginTest, Basic) {
    struct Compute {
        static std::string ComputeAdd(const std::string &filepath, const std::string &source) {
            return source + "Add" + filepath;
        }
        static std::string ComputeSub(const std::string &filepath, const std::string &source) {
            return source + "Sub" + filepath;
        }
    };
    PluginManager &manager = PluginManager::GetInstance();

    EXPECT_TRUE(manager.AddPluginCodegenSrc("add", Compute::ComputeAdd));
    EXPECT_FALSE(manager.AddPluginCodegenSrc("add", Compute::ComputeSub));
    EXPECT_TRUE(manager.AddPluginCodegenSrc("sub", Compute::ComputeSub));

    EXPECT_EQ(2, manager.GetPlugin<PluginCodegenSrc>().size());

    std::string code = manager.RunPluginCodegenSrc("1", "2");
    EXPECT_EQ("2Add1Sub1", code);

    manager.ClearPlugin();
    EXPECT_EQ(0, manager.GetPlugin<PluginCodegenSrc>().size());
<<<<<<< HEAD

    std::string code2 = manager.RunPluginCodegenSrc("1", "2");
    EXPECT_EQ("2", code2);
}
=======
}

TEST(PluginTest, Codegen) {
    TileShape::Current().SetVecTile(32, 32);
    TileShape::Current().SetCubeTile({32, 32}, {32, 32}, {32, 32});

    std::vector<int> opList;
    auto addLinePrefix = [&](const std::string &filepath, const std::string &source) {
        (void)filepath;
        opList.push_back(0);
        return source;
    };
    PluginManager::GetInstance().AddPluginCodegenSrc("AddPrefix", addLinePrefix);

    int n = 8;
    int s = 32;

    Tensor t0(DT_FP32, {n * s, s}, "t0");  // [32*8, 32]
    Tensor t1(DT_FP32, {n * s, s}, "t1");  // [32, 32]
    Tensor t2(DT_FP32, {s, s}, "t2");  // [32, 32]
    Tensor out(DT_FP32, {n * s, s}, "out");

    FUNCTION("main", {t0, t1, t2}, {out}) {
        Tensor s0Out;
        LOOP("L0", FunctionType::DYNAMIC_LOOP, _, LoopRange(1)) {
            (void)_;
            s0Out = Sub(t1, t0);
        }
        LOOP("L1", FunctionType::DYNAMIC_LOOP, i, LoopRange(n)) {
            Tensor t0s = View(s0Out, {s, s}, {i * s, 0});
            Tensor t3 = Add(t0s, t2);
            Assemble(t3, {i * s, 0}, out);
        }
    }
    EXPECT_EQ(2, opList.size());
    PluginManager::GetInstance().ClearPlugin();
}
>>>>>>> e93b3b6... feat(codegen): Add source_cpp ir serializer as codegen
