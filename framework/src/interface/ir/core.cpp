/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ir/core.h"

#include <sstream>
#include <string>
#include <utility>

namespace pypto {
namespace ir {

Span::Span(std::string filename, int beginLine, int beginColumn, int endLine, int endColumn)
    : filename_(std::move(filename)),
      beginLine_(beginLine),
      beginColumn_(beginColumn),
      endLine_(endLine),
      endColumn_(endColumn)
{}

std::string Span::ToString() const
{
    std::ostringstream oss;
    oss << filename_ << ":" << beginLine_ << ":" << beginColumn_;
    return oss.str();
}

bool Span::IsValid() const
{
    if (beginLine_ <= 0 || (beginColumn_ <= 0 && beginColumn_ != -1)) {
        return false;
    }
    if (endLine_ == -1 || endColumn_ == -1) {
        return true;
    }
    if (endLine_ <= 0 || (endColumn_ <= 0 && endColumn_ != -1)) {
        return false;
    }
    if (beginColumn_ == -1 || endColumn_ == -1) {
        return endLine_ >= beginLine_;
    }
    return endLine_ >= beginLine_ && (endLine_ > beginLine_ || endColumn_ >= beginColumn_);
}

Span Span::Unknown() { return Span("", -1, -1, -1, -1); }

} // namespace ir
} // namespace pypto
