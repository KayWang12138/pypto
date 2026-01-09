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
 * \file serializer.h
 * \brief
 */

#include "ir/function.h"
#include "ir/program.h"

namespace pto {

class IRBuffer {
public:
    virtual ~IRBuffer() = default;
    virtual void Append(const std::string &data) = 0;
    virtual void Append(const std::vector<uint8_t> &data) = 0;
};

class IRMemoryBuffer : IRBuffer {
public:
    virtual void Append(const std::string &data);
    virtual void Append(const std::vector<uint8_t> &data);

    std::string &GetRawBuffer() { return buffer_; }
private:
    std::string buffer_;
};

class IRSerializer {
public:
    enum SerializerKind {
        /* assemble style */
        SOURCE_ASM,
        /* cplusplus style which is for codegen. Serializer is not responsible for whether the serialized code is further compiled
        * as AscendC or pure C++ code. */
        SOURCE_CPP,
    };
public:
    virtual ~IRSerializer() = default;
    virtual void Serialize(IRBuffer &buffer, ProgramModulePtr module) = 0;

    virtual ProgramModulePtr Deserialize(IRBuffer &buffer) = 0;
private:
    SerializerKind kind_;
};

}
