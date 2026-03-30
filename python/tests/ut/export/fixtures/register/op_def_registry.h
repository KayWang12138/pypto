#pragma once

#include <initializer_list>

#include "gert_ge_minimal.hpp"

namespace ge {

// Minimal GE dtype / format constants used by export codegen tests.
inline constexpr int DT_FLOAT = 0;
inline constexpr int DT_FLOAT16 = 1;
inline constexpr int DT_BF16 = 2;
inline constexpr int DT_INT8 = 3;
inline constexpr int DT_INT16 = 4;
inline constexpr int DT_INT32 = 5;
inline constexpr int DT_INT64 = 6;
inline constexpr int DT_UINT8 = 7;
inline constexpr int DT_UINT16 = 8;
inline constexpr int DT_UINT32 = 9;
inline constexpr int DT_UINT64 = 10;
inline constexpr int DT_BOOL = 11;

inline constexpr int FORMAT_ND = 20;
inline constexpr int REQUIRED = 21;

class OpDef {
public:
    explicit OpDef(const char*) {}
    virtual ~OpDef() = default;

    OpDef& Input(const char*) { return *this; }
    OpDef& Output(const char*) { return *this; }
    OpDef& ParamType(int) { return *this; }
    OpDef& DataType(std::initializer_list<int>) { return *this; }
    OpDef& Format(std::initializer_list<int>) { return *this; }
    OpDef& SetInferShape(graphStatus (*)(gert::InferShapeContext*)) { return *this; }
    OpDef& SetInferDataType(graphStatus (*)(gert::InferDataTypeContext*)) { return *this; }

    class AICoreConfig {
    public:
        void AddConfig(const char*) {}
    };

    AICoreConfig& AICore() { return ai_core_; }

private:
    AICoreConfig ai_core_{};
};

}  // namespace ge

#define OP_ADD(name) /* test stub */
