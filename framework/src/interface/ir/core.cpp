/*
 * Copyright (c) PyPTO Contributors.
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * -----------------------------------------------------------------------------------------------------------
 */

#include "ir/core.h"

#include <sstream>
#include <string>
#include <utility>

namespace pypto {
namespace ir {

Span::Span(std::string filename, int begin_line, int begin_column, int end_line, int end_column)
    : filename_(std::move(filename)),
      begin_line_(begin_line),
      begin_column_(begin_column),
      end_line_(end_line),
      end_column_(end_column)
{}

std::string Span::to_string() const
{
    std::ostringstream oss;
    oss << filename_ << ":" << begin_line_ << ":" << begin_column_;
    return oss.str();
}

static Span kUnknownSpan = Span("", -1, -1, -1, -1);

bool Span::is_valid() const
{
    if (begin_line_ <= 0 || (begin_column_ <= 0 && begin_column_ != -1)) {
        return false;
    }
    if (end_line_ == -1 || end_column_ == -1) {
        return true;
    }
    if (end_line_ <= 0 || (end_column_ <= 0 && end_column_ != -1)) {
        return false;
    }
    if (begin_column_ == -1 || end_column_ == -1) {
        return end_line_ >= begin_line_;
    }
    return end_line_ >= begin_line_ && (end_line_ > begin_line_ || end_column_ >= begin_column_);
}

bool Span::is_unknown(const Span& span)
{
    return span.filename_.empty() && span.begin_line_ == -1 && span.begin_column_ == -1 && span.end_line_ == -1 &&
           span.end_column_ == -1;
}

Span Span::unknown() { return Unknown(); }

std::string Span::ToString() const { return to_string(); }

bool Span::IsUnknown(const Span& span) { return is_unknown(span); }

Span& Span::Unknown() { return kUnknownSpan; }

} // namespace ir
} // namespace pypto
