#pragma once

#include <cstdint>
#include <initializer_list>
#include <vector>

#include "gert_ge_minimal.hpp"

namespace ge {

// Use the real CANN enum tag so ``ge::FORMAT_ND`` references from generated
// code keep their semantics and `ge::Format` doesn't clash with anything.
enum Format {
    FORMAT_ND = 20,
};

}  // namespace ge

namespace ops {

// Real header lives at $ASCEND_HOME_PATH/<arch>/include/register/op_def.h
// with ``ops::OpDef``, ``ops::Option``, and the ``OP_ADD`` macro. Fixture
// mirrors the top-level ``ops::`` namespace so generated
// ``class X : public OpDef`` inside ``namespace ops`` resolves here.

enum Option {
    IGNORE = 0,
    OPTIONAL = 1,
    REQUIRED = 2,
    DYNAMIC = 3,
    VIRTUAL = 4,
};

class OpDef {
public:
    explicit OpDef(const char *) {}
    virtual ~OpDef() = default;

    OpDef &Input(const char *) { return *this; }
    OpDef &Output(const char *) { return *this; }
    OpDef &ParamType(Option) { return *this; }
    OpDef &DataType(std::vector<ge::DataType>) { return *this; }
    OpDef &Format(std::vector<ge::Format>) { return *this; }
    // Match ``gert::OpImplRegisterV2::InferShapeKernelFunc`` = ``UINT32 (*)(...)``;
    // ``ge::graphStatus`` is ``uint32_t`` in real Ascend (see ``gert_ge_minimal.hpp``).
    OpDef &SetInferShape(ge::graphStatus (*)(gert::InferShapeContext *)) { return *this; }
    OpDef &SetInferDataType(ge::graphStatus (*)(gert::InferDataTypeContext *)) { return *this; }

    class AICoreConfig {
    public:
        AICoreConfig &AddConfig(const char *) { return *this; }
    };

    AICoreConfig &AICore() { return ai_core_; }

private:
    AICoreConfig ai_core_{};
};

}  // namespace ops

#define OP_ADD(name) /* test stub */
