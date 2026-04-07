#pragma once

#include <initializer_list>

#include "gert_ge_minimal.hpp"

namespace ge {

inline constexpr int FORMAT_ND = 20;
inline constexpr int REQUIRED = 21;

class OpDef {
public:
    explicit OpDef(const char *) {}
    virtual ~OpDef() = default;

    OpDef &Input(const char *) { return *this; }
    OpDef &Output(const char *) { return *this; }
    OpDef &ParamType(int) { return *this; }
    OpDef &DataType(std::initializer_list<DataType>) { return *this; }
    OpDef &Format(std::initializer_list<int>) { return *this; }
    OpDef &SetInferShape(graphStatus (*)(gert::InferShapeContext *)) { return *this; }
    OpDef &SetInferDataType(graphStatus (*)(gert::InferDataTypeContext *)) { return *this; }

    class AICoreConfig {
    public:
        void AddConfig(const char *) {}
    };

    AICoreConfig &AICore() { return ai_core_; }

private:
    AICoreConfig ai_core_{};
};

}  // namespace ge

#define OP_ADD(name) /* test stub */
