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
 * \file opcode.cpp
 * \brief
 */

#include <vector>
#include <string>
#include <map>
#include <unordered_set>
#include "ir/opcode.h"

namespace pto {

static std::unordered_map<Opcode, std::string> opcodeNameDict = {

// #define DEF_OP(name, inherit, opcode, ...) DEF_##opcode
// #define DEF_OPCODE(...) __VA_ARGS__

// #include "operation.def"
// #include "tile_graph.def"

// #undef DEF_OPCODE
// #undef DEF_OP

};

std::string GetOpcodeName(Opcode opcode) {
    if (opcodeNameDict.count(opcode)) {
        return opcodeNameDict[opcode];
    } else {
        return "";
    }
}

} // namespace pto
