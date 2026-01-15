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
 * \file program.cpp
 * \brief
 */

#include "ir/program.h"
#include "ir/utils.h"

#include "ir/function.h"

namespace pto {
    
namespace {
// Attribute registrar for ProgramModule (defined in program.cpp).
struct ProgramAttrRegistrar {
    ProgramAttrRegistrar() {
        RegisterAttrKey<ProgramModule>("arch");
        RegisterAttrKey<ProgramModule>("tile_default");
        RegisterAttrKey<ProgramModule>("enable_debug");
        RegisterAttrKey<ProgramModule>("test_type");
    }
};
static ProgramAttrRegistrar g_programAttrRegistrar;
} // namespace

ProgramModule::ProgramModule(std::string name)
    : Object(ObjectType::Program, std::move(name)) {}

void ProgramModule::SetProgramEntry(const std::shared_ptr<Function>& programEntry) {
    programEntry_ = programEntry;
}

void ProgramModule::AddFunction(const std::shared_ptr<Function>& function) {
    functions_.push_back(function);
}

void ProgramModule::Print(std::ostream& os, int indent) const {
    // Print indentation for the module header.
    PrintIndent(os, indent);
    os << "program.module " << GetPrefixedName() << " {\n";

    // Entrypoints
    PrintIndent(os, indent + 1);
    os << "program.entry " << programEntry_->GetPrefixedName() << "\n";

    // Program-level attributes.
    auto attrKeys = GetSetAttrKeys();
    for (const auto& key : attrKeys) {
        PrintIndent(os, indent + 1);
        os << "attr " << key << " = " << GetAttr(key) << "\n";
    }

    // Functions.
    for (const auto& f : functions_) {
        f->Print(os, indent + 1);
        os << "\n";
    }

    // Closing brace.
    for (int i = 0; i < indent; ++i) {
        os << "  ";
    }
    os << "}\n";
}

std::ostream& operator<<(std::ostream& os, const ProgramModule& module) {
    module.Print(os, 0);
    return os;
}

} // namespace pto


