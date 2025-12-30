// PTO-IR prototype: utility functions implementation.

#include "ir/utils.h"
#include "ir/type.h"

#include <ostream>

namespace pto {

// Initialize static member
std::map<ObjectType, int> IDGen::counters_;

int IDGen::NextID(ObjectType type) {
    return ++counters_[type];
}

void IDGen::Reset(ObjectType type) {
    counters_[type] = 0;
}

void IDGen::ResetAll() {
    counters_.clear();
}

void PrintIndent(std::ostream& os, int indent) {
    for (int i = 0; i < indent; ++i) {
        os << "  ";
    }
}

std::string DataTypeToString(DataType type) {
    switch (type) {
    case DataType::BOOL:
        return "bool";
    case DataType::INT4:
        return "int4";
    case DataType::INT8:
        return "int8";
    case DataType::INT16:
        return "int16";
    case DataType::INT32:
        return "int32";
    case DataType::INT64:
        return "int64";
    case DataType::UINT8:
        return "uint8";
    case DataType::UINT16:
        return "uint16";
    case DataType::UINT32:
        return "uint32";
    case DataType::UINT64:
        return "uint64";
    case DataType::FP8:
        return "fp8";
    case DataType::FP16:
        return "fp16";
    case DataType::BF16:
        return "bf16";
    case DataType::FP32:
        return "fp32";
    case DataType::FP64:
        return "fp64";
    case DataType::HF4:
        return "hf4";
    case DataType::HF8:
        return "hf8";
    case DataType::BOTTOM:
        return "bottom";
    case DataType::UNKNOWN:
        return "unknown";
    default:
        return "unknown";
    }
    return "unknown";
}

DataType StringToValueType(const std::string& name) {
    if (name == "bool") return DataType::BOOL;
    if (name == "int4") return DataType::INT4;
    if (name == "int8") return DataType::INT8;
    if (name == "i8") return DataType::INT8;
    if (name == "int16") return DataType::INT16;
    if (name == "i16") return DataType::INT16;
    if (name == "int32") return DataType::INT32;
    if (name == "i32") return DataType::INT32;
    if (name == "int64") return DataType::INT64;
    if (name == "i64") return DataType::INT64;
    if (name == "uint8") return DataType::UINT8;
    if (name == "u8") return DataType::UINT8;
    if (name == "uint16") return DataType::UINT16;
    if (name == "u16") return DataType::UINT16;
    if (name == "uint32") return DataType::UINT32;
    if (name == "u32") return DataType::UINT32;
    if (name == "uint64") return DataType::UINT64;
    if (name == "u64") return DataType::UINT64;
    if (name == "fp8") return DataType::FP8;
    if (name == "fp16") return DataType::FP16;
    if (name == "f16") return DataType::FP16;
    if (name == "bf16") return DataType::BF16;
    if (name == "fp32") return DataType::FP32;
    if (name == "f32") return DataType::FP32;
    if (name == "fp64") return DataType::FP64;
    if (name == "f64") return DataType::FP64;
    if (name == "hf4") return DataType::HF4;
    if (name == "hf8") return DataType::HF8;
    if (name == "bottom") return DataType::BOTTOM;
    if (name == "unknown") return DataType::UNKNOWN;
    // Default to INT32 if unknown
    return DataType::INT32;
}

} // namespace pto

